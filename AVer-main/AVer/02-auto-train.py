import os
import subprocess
from glob import glob
from datetime import datetime
from utils import cal_filter
from dataPrepare import data_prepare
from train import train
from getFilter import get_filter
import argparse

# -------------------------
# 參數設定
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default="Demo", help="專案名稱")
parser.add_argument("--operator-weights", type=str, default="models/operator/init.pt", help="預設使用yolo模型")
parser.add_argument("--action-weights", type=str, default="models/action/init.pth", help="預設使用slowfast模型")
args = parser.parse_args()

now = datetime.now()
series = now.strftime("%Y%m%d%H")

projects = [args.project]

# models output path
path_ = os.getcwd()
input_path = os.path.join(path_, "input")
temp_path = os.path.join(path_, "temp_train")
model_path = os.path.join(path_, "models")
operator_path = os.path.join(model_path, "operator")
action_path = os.path.join(model_path, "action")
filter_path = os.path.join(model_path, "filter")

for path in [temp_path, operator_path, action_path, filter_path]:
    os.makedirs(path, exist_ok=True)
    
for project_ in projects:
    print(project_)

    # -------------------------
    # initialize
    # -------------------------
    print("------------initialize------------")
    project_name = f"{project_}_{series}"
    print(project_name)
    input = os.path.join(input_path, project_)
    temp_train = os.path.join(temp_path, project_name)
    temp_filter = os.path.join(temp_train, 'temp-filter')
    os.makedirs(temp_filter, exist_ok=True)
    init_operator_weights = args.operator_weights #os.path.join(operator_path, "init.pt")
    init_action_weights = args.action_weights #os.path.join(action_path, "init.pth")
    trained_action_weights = os.path.join(action_path, f"{project_name}.pth")
    trained_filter_weights = os.path.join(filter_path, f"{project_name}.txt")

    # -------------------------
    # 01. get dataloader
    # -------------------------
    '''
    args:
        input: 輸入資料夾, 包含labels和videos
        output: 輸出資料夾
        fps: 影片fps, 預設為30fps
        interval: 每段影片的時間長度(單位:幀), 預設為20幀
        train_data: 每個影片的訓練資料數量, 預設為600筆
        test: 是否僅生成測試資料; 預設為訓練資料
    '''
    print("------------get dataloader------------")
    data_prepare(input=input, 
                output=temp_train, 
                fps=30, 
                interval=20, 
                train_data=600, 
                test=False)
#要改
    # -------------------------
    # training - slowfast
    # -------------------------
    '''
    args:
        project: 專案名稱
        input: 資料集路徑
        output: 儲存權重路徑,  預設為models
        weight: 初始模型權重, 預設為models/action/init.pth
        test_ratio: 訓練-驗證比例, 預設為0.2
        batch_size: batch size, 預設為2
        lr: learning rate, 預設為0.0001
        epoch: epoch, 預設為20
        seed: fix random seed, 預設為666
    '''

    print("------------training slowfast model------------")
    train(project=project_name, 
                input=temp_train, 
                output=action_path, 
                weight=init_action_weights, 
                test_ratio=0.2, 
                batch_size=2, 
                lr=0.0001, 
                epoch=20, 
                seed=666)

    # -------------------------
    # training - filter: detect
    # -------------------------
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
    print("------------calculating filter------------")
    files = glob(os.path.join(input, "videos", "*.mp4"))
    print(f"files: {files}")
    for file_ in files:
        get_filter(input=file_, 
                    output=temp_filter, 
                    reference=temp_train, 
                    action_weights=trained_action_weights, 
                    save=trained_filter_weights)
