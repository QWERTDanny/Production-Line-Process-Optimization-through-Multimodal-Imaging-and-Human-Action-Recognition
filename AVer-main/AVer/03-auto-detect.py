import os
import subprocess
from glob import glob
from datetime import datetime
import argparse


# 參數設定
parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default="project", help="專案名稱")
parser.add_argument("--input", type=str, default="../input/videos", help="輸入資料夾, 包含欲預測之影片檔")
parser.add_argument("--output", type=str, default="../outputs", help="輸出資料夾, 根據專案名稱自動生成")
parser.add_argument("--operator-weights", type=str, default="../models/operator/init.pt", help="預設使用yolo模型")
parser.add_argument("--action-weights", type=str, default="", help="預設使用slowfast模型, 若無則根據專案名稱自動抓取")
parser.add_argument("--filter-weights", type=str, default="", help="預設使用filter, 若無則根據專案名稱自動抓取")
parser.add_argument("--classes", type=str, default="", help="類別檔路徑, 包含classes.txt, 類別數量需相符, 若無則根據專案名稱自動抓取")
parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
args = parser.parse_args()

input, output = args.input, args.output
project, operator_weights, action_weights, filter_weights, classes = \
    args.project, args.operator_weights, args.action_weights, args.filter_weights, args.classes

path_ = os.getcwd()

if action_weights == "":
    action_weights = os.path.join(path_, 'models', 'action', f'{project}.pth')
if filter_weights == "":
    filter_weights = os.path.join(path_, 'models', 'filter', f'{project}.txt')
if classes == "":
    classes_path = os.path.join(path_, 'temp_train' , project)
else:
    classes_path = os.path.join(path_, classes)

# 辨識動作
os.chdir("yolov7")
print(os.getcwd())




videos = glob(os.path.join(path_, input, "*.mp4"))
print(f"files: {videos}")

detect = "slowfast-yolo-detect.py"

for video in videos:
    args = [
        "--input", video,
        "--output", output,
        "--action-weights", action_weights,  
        "--detect-weights", operator_weights,
        "--filter-weights", filter_weights,
        "--classes-path", classes_path,
        '--conf-thres', "0.5"
    ]
    print(args)
    result = subprocess.run(
        ["python", detect] + args, capture_output=True, text=True
    )

    if result.returncode == 0:
        print("程式執行成功！")
        print("輸出訊息：", result.stdout)
    else:
        print("程式執行失敗！")
        print("錯誤訊息：", result.stderr)

os.chdir("..")
print(os.getcwd())

