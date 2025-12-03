import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
import os, cv2, time, torch, warnings, argparse
warnings.filterwarnings("ignore",category=UserWarning)

from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale,
    )
from torchvision.transforms._functional_video import normalize
from metrics import get_metrics
from utils import cal_filter

def read_filter(filter_path):
    with open(filter_path, 'r') as f:
        lines = f.readlines()
        values = [float(entry.split(': ')[1]) for entry in lines]
        values = np.array(values)
        values = values.reshape(1, values.shape[0])
    return values

class MyVideoCapture:
    
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.idx = -1
        self.end = False
        self.stack = []
        self.width, self.height = int(self.cap.get(3)), int(self.cap.get(4))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_wh(self):
        return self.width, self.height
    
    def read(self):
        self.idx += 1
        ret, img = self.cap.read()
        if ret:
            self.stack.append(img)
        else:
            self.end = True
        return ret, img
    
    def to_tensor(self, img):
        img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img.unsqueeze(0)
        
    def get_video_clip(self):
        assert len(self.stack) > 0, "clip length must large than 0 !"
        self.stack = [self.to_tensor(img) for img in self.stack]
        clip = torch.cat(self.stack).permute(-1, 0, 1, 2)
        del self.stack
        self.stack = []
        return clip
    
    def release(self):
        self.cap.release()

def inference_transform(
    clip, 
    num_frames = 32, #if using slowfast_r50_detection, change this to 32, 4 for slow 
    crop_size = 256, #640, 
    data_mean = [0.432, 0.432,  0.432], # aver
    data_std = [0.291, 0.291, 0.291], # aver
    slow_fast_alpha = 4, #if using slowfast_r50_detection, change this to 4, None for slow
):
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0
    clip = short_side_scale(clip,size=crop_size)
    clip = normalize(clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),) 
    if slow_fast_alpha is not None:
        fast_pathway = clip
        slow_pathway = torch.index_select(clip,1,
            torch.linspace(0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha).long())
        clip = [slow_pathway, fast_pathway]

    return clip

# TODO
def cls_list(classes_dir): 
    cls_file = os.path.join(classes_dir, 'classes.txt')
    with open(cls_file) as f:
        classes = f.readlines()
    classes_list = {}
    for c in classes:
        c = c.strip('\n').split(': ')
        id_ = c[0].replace(' ', '')
        cls_ = c[1].replace('\'', '').replace(',', '')
        classes_list[int(id_)] = cls_

    idx_list = {}
    for idx, item in classes_list.items():
        idx_list[item] = int(idx)

    return classes_list, idx_list

def get_filter(input="input/videos/demo.mp4", output="temp_train/demo/temp-filter", reference="temp_train/demo", action_weights='models/init.pth', save='models/filter/filer.txt', interval=30, imsize=256, device=0):
    '''
    args:
        input: 輸入影片檔
        output: 結果存放之資料夾路徑
        reference: 參照資料夾, 包含 [classes.txt] 和 [labels-action.csv]
        action_weights: 動作辨識權重檔路徑(slowfast)
        save: filter權重儲存路徑, 
        interval: inference interval (frames), 預設為25
        imsize: inference size (pixels), 預設為256
        device: cuda device, i.e. 0 or 0,1,2,3 or cpu
    '''

    # input, device, action_weights, imsize, reference, interval = args.input, args.device, args.action_weights, args.imsize, args.reference, args.interval
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # -------------------------
    # 載入SlowFast模型
    # -------------------------
    print(f'video_model = torch.load({action_weights}')
    video_model = torch.load(action_weights)
    video_model = video_model.eval().to(device)
    labelnames, idxnames = cls_list(reference)

    # -------------------------
    # 設定輸出路徑
    # -------------------------
    save_path = output
    csv_path = os.path.join(save_path, 'results')
    os.makedirs(csv_path, exist_ok=True)
    csv_save_path = os.path.join(csv_path, f'{os.path.basename(input).replace(".mp4", "")}.csv')
    
    cap = MyVideoCapture(input)
    print("processing...")

    # -------------------------
    # 設定變數
    # -------------------------   
    other = idxnames['Other']
    prob_slowfast = None
    pred_slowfast_idx = other
    
    columns = ['slowfast', 'prob_slowfast']
    df = pd.DataFrame(columns=columns)

    a = time.time()

    total_frames = cap.total_frames
    print(f"total frames: {total_frames}")
    wrapped_cap = tqdm(cap, desc="Processing Video", unit="frame")
    while not wrapped_cap.n >= total_frames:
        ret, img = cap.read()
        if not ret:
            continue
            
        if len(cap.stack) == interval:
            clip = cap.get_video_clip()
            inputs = inference_transform(clip,  crop_size=imsize)

            if isinstance(inputs, list):
                inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
            else:
                inputs = inputs.unsqueeze(0).to(device)

            with torch.no_grad():                
                prob_slowfast = video_model(inputs)
                prob_slowfast = prob_slowfast.cpu()
                prob_slowfast = F.softmax(prob_slowfast, dim=1)
                prob_slowfast = prob_slowfast.detach().numpy()
            
            for label in np.argmax(prob_slowfast, axis=1).tolist():
                pred_slowfast = labelnames[label]
                pred_slowfast_idx = label

        wrapped_cap.update()

        df = pd.concat([df, pd.DataFrame([{'slowfast': pred_slowfast_idx, 'prob_slowfast': prob_slowfast}])], ignore_index=True)
        df.to_csv(csv_save_path)
    print("total cost: {:.3f} s, video length: {} s".format(time.time()-a, cap.idx / 30))
    wrapped_cap.close()
    cap.release()

    result_path = os.path.join(output, 'results')
    results = get_metrics(input_path=result_path, output_path=output, ref_path=reference, interval=interval)
    cal_filter(results, save)

    shutil.rmtree(output)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="input/videos/demo.mp4", help='輸入影片檔')
    parser.add_argument('--output', type=str, default="temp_train/demo/temp-filter", help='結果存放之資料夾路徑')
    parser.add_argument('--reference', type=str, default="temp_train/demo", help='參照資料夾, 包含 [classes.txt] 和 [labels-action.csv]')
    parser.add_argument('--action-weights', type=str, default='models/init.pth', help='動作辨識權重檔路徑(slowfast)')
    parser.add_argument('--save', type=str, default='models/filter/filer.txt', help='filter權重儲存路徑')
    parser.add_argument('--interval', type=int, default=25, help='inference interval (frames)')
    parser.add_argument('--imsize', type=int, default=256, help='inference size (pixels)')
    parser.add_argument('--device', type=int, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    args = parser.parse_args()
    main(args)
