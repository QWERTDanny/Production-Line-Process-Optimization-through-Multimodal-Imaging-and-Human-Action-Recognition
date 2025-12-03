import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os, cv2, time, torch, random, pytorchvideo, warnings, argparse, math
warnings.filterwarnings("ignore",category=UserWarning)

from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale,
    clip_boxes_to_image,)
from torchvision.transforms._functional_video import normalize
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


# ----- yolov7 ----- #
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

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

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        # 改為在bbox的左上方顯示label
        # c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        # 改為在bbox的右方顯示label
        c1 = (c2[0] - t_size[0], c2[1])
        c2 = (c1[0] + t_size[0], c1[1] + t_size[1])
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1]), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def save_yolopreds_tovideo(im, pred, output_video, vis=False):
    # for i, (im, pred) in enumerate(zip(img, pred_label)):
    text = 'Predict: {}'.format(pred)
    if pred == []:
        text = 'predicting...'
    
    cv2.rectangle(im, (0, 0), (600, 150), (0, 0, 0), -1)
    cv2.putText(im, text, (20, 100), cv2.FONT_HERSHEY_TRIPLEX, 1.5, [255,255,255], 3)
    im = im.astype(np.uint8)
    output_video.write(im)
    if vis:
        im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        cv2.imshow("demo", im)

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

def main(args):
    # -------------------------
    # Set Parameters
    # -------------------------
    input, device, source, detect_weights, action_weights, filter_weights, view_img, save_txt, imgsz, trace, classes_path, interval, imsize = args.input, args.device, args.source, args.detect_weights, args.action_weights, args.filter_weights, args.view_img, args.save_txt, args.img_size, not args.no_trace, args.classes_path, args.interval, args.imsize
    
    save_img = not args.nosave and not source.endswith('.txt')  # save inference images
    # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    #     ('rtsp://', 'rtmp://', 'http://', 'https://'))
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # -------------------------
    # Load Model: YOLOv7
    # -------------------------
    model = attempt_load(detect_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, args.img_size)
    if half:
        model.half()  # to FP16
    
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    # Run inference (attempt)
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    # -------------------------
    # Set Path
    # -------------------------
    save_path = args.output
    csv_path = os.path.join(save_path, 'results')
    vide_path = os.path.join(save_path, 'videos')
    os.makedirs(csv_path, exist_ok=True)
    os.makedirs(vide_path, exist_ok=True)
    csv_save_path = os.path.join(csv_path, f'{os.path.basename(input).replace(".mp4", "")}.csv')
    vide_save_path = os.path.join(vide_path, f'{os.path.basename(input)}')

    # -------------------------
    # Set Variables
    # -------------------------   
    labelnames, idxnames = cls_list(classes_path)
    yolo_init_labels = {0: 'operator'}
    yolo_init_idx = {k: idxnames.get(v, 99) for k, v in yolo_init_labels.items()}

    other = idxnames['Other']
    pred_slowfast, pred_yolo, pred_label = 'Predicting...', 'Predicting...', 'Predicting...'
    pred_slowfast_idx, pred_yolo_idx, pred_label_idx = other, other, other
    
    conf_yolo, conf_slowfast = 0, 0
    prob_slowfast = None
    
    columns = ['yolo', 'conf_yolo', 'slowfast', 'conf_slowfast']
    df = pd.DataFrame(columns=columns)

    a = time.time()

    # -------------------------
    # Load Model: SlowFast
    # -------------------------
    print(f'video_model = torch.load({action_weights}')
    video_model = torch.load(action_weights)
    video_model = video_model.eval().to(device)

    # deepsort_tracker = DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    # -------------------------
    # Load Filter
    # -------------------------
    filter = read_filter(filter_weights)

    # -------------------------
    # Load Video
    # -------------------------
    cap = MyVideoCapture(input)
    width, height = cap.get_wh()
    outputvideo = cv2.VideoWriter(vide_save_path, 
                                  cv2.VideoWriter_fourcc(*'mp4v'), 
                                  30, (width, height))
    
    print("processing...")
    while not cap.end:
        ret, img = cap.read()
        if not ret:
            continue
## 模型判斷
        # -----------------------------
        # Operator Detection (YOLOv7)
        # -----------------------------
        # Transform Data
        im0 = letterbox(img, 640, stride=stride)[0]
        im0 = im0[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        im0 = np.ascontiguousarray(im0)
        im0 = torch.from_numpy(im0).to(device)
        im0 = im0.half() if half else im0.float()
        im0 /= 255.0  # 0 - 255 to 0.0 - 1.0
        if im0.ndimension() == 3:
            im0 = im0.unsqueeze(0)
        # im0 = im0.permute(0, 3, 1, 2)

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            preds = model(im0, augment=args.augment)[0]

        # Apply NMS
        preds = non_max_suppression(preds, 
                                   args.conf_thres, 
                                   args.iou_thres, 
                                   classes=args.classes, 
                                   agnostic=args.agnostic_nms)

        # Process Detections
        for i, det in enumerate(preds):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im0.shape[2:], det[:, :4], img.shape).round()

                # Write results
                # TODO: softmax for yolo's action label prediction & keep only the center operator
                for *xyxy, conf, cls in reversed(det):
                    if save_img or view_img:  # Add bbox to image
                        pred_yolo_ = f'{names[int(cls)]} {conf:.2f}'
                        pred_yolo, conf_yolo = pred_yolo_.split(' ')
                        conf_yolo = float(conf_yolo)
                        pred_yolo_idx = 0
                        plot_one_box(xyxy, img, label=pred_yolo_, color=colors[int(cls)], line_thickness=3)
            else:
                pred_yolo, conf_yolo, pred_yolo_idx = 'Other', 0, 1
        
        # -----------------------------
        # Action Detection (SlowFast)
        # -----------------------------
        if len(cap.stack) == interval:
            print(f"processing {cap.idx // 30}th second clips")
            clip = cap.get_video_clip()

            if not(pred_yolo == 'Other'):

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
                    prob_slowfast += filter
                    conf_slowfast = np.max(prob_slowfast)
                
                for label in np.argmax(prob_slowfast, axis=1).tolist():
                    pred_slowfast = labelnames[label]
                    pred_slowfast_idx = label

            else:
                pred_slowfast = 'Other'
                pred_slowfast_idx = idxnames[pred_slowfast]
        
        # -----------------------------
        # Save Results as CSV File
        # -----------------------------
        current_data = {'yolo': pred_yolo_idx, 
                        'conf_yolo': conf_yolo, 
                        'slowfast': pred_slowfast_idx, 
                        'conf_slowfast': conf_slowfast}
        df = pd.concat([df, pd.DataFrame([current_data])], ignore_index=True)
        df.to_csv(csv_save_path)
        save_yolopreds_tovideo(img, pred_slowfast, outputvideo, args.show)
    print("total cost: {:.3f} s, video length: {} s".format(time.time()-a, cap.idx / 30))
    
    cap.release()
    outputvideo.release()
    print('saved video to:', vide_save_path)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="../input/videos/demo.mp4", help='輸入影片檔')
    parser.add_argument('--output', type=str, default="../outputs/demo", help='結果存放之資料夾路徑')
    parser.add_argument('--classes-path', type=str, default="../temp_train/demo", help='參照資料夾, 包含 [classes.txt]')
    parser.add_argument('--action-weights', type=str, default='../models/action/init.pth', help='slowfast(model.pth)')
    parser.add_argument('--detect-weights', type=str, default='../model/operator/init.pt', help='yolo(model.pt)')
    parser.add_argument('--filter-weights', type=str, default='../model/filter/init.txt', help='filter(weights.txt)')
    parser.add_argument('--interval', type=int, default=25, help='inference interval (frames)')
    parser.add_argument('--imsize', type=int, default=256, help='Slowfast inference size (pixels)')
    parser.add_argument('--img-size', type=int, default=640, help='YOLO inference size (pixels)')

    # YOLO (not suggested to change)
    # parser.add_argument('--conf', type=float, default=0.4, help='object confidence threshold')
    # parser.add_argument('--iou', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true', help='show img')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')

    args = parser.parse_args()    
    # if args.input.isdigit():
    #     print("using local camera.")
    #     args.input = int(args.input)
        
    print(args)
    main(args)
