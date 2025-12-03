# -------------------------
# 錄製影片，並分別存入訓練資料與所有資料
# -------------------------
import cv2
import os
import time
import argparse

def get_next_filename(base_filename):
    """
    如果檔案已存在，則回傳下一個可用的檔案名稱，包含編號。
    """
    if not os.path.exists(base_filename):
        return base_filename

    filename, file_extension = os.path.splitext(base_filename)
    index = 1

    while True:
        new_filename = f"{filename}_{index}{file_extension}"
        if not os.path.exists(new_filename):
            return new_filename
        index += 1

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default="new_project", help="專案名稱")
args = parser.parse_args()

project = args.project

# 訓練資料儲存路徑
save_path_train = os.path.join("input", project, "videos")
os.makedirs(save_path_train, exist_ok=True)

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 取得影像寬度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 取得影像高度
fourcc = cv2.VideoWriter_fourcc("M", "P", "4", "V")  # 表示 MP4 編碼格示，副檔名為 .mp4

filename = get_next_filename(os.path.join(save_path_train, f"{project}.mp4"))
out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
print(f"Saving video to {filename}")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("The video does not exist, or your video is already being generated.")
        break

    # saving the video without skeleton
    out.write(frame)
    cv2.imshow("video", frame)

    # 按下 'ESC' 鍵結束錄製
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
