import warnings

warnings.filterwarnings("ignore")
import pandas as pd
from glob import glob
import os
from tqdm import tqdm
from pytorchvideo.data.encoded_video import EncodedVideo
import numpy as np
import torch
import argparse

def data_prepare(input="input", output="temp_train", fps=30, interval=20, train_data=600, test=False):
    '''
    args:
        input: 輸入資料夾, 包含labels和videos
        output: 輸出資料夾
        fps: 影片fps, 預設為30fps
        interval: 每段影片的時間長度(單位:幀), 預設為20幀
        train_data: 每個影片的訓練資料數量, 預設為600筆
        test: 是否僅生成測試資料; 預設為訓練資料
    '''
    if test:
        folder = "temp_test"
        os.makedirs(folder, exist_ok=True)
    else:
        folder = output

    # -------------------------
    # 01. 讀取標籤檔
    # -------------------------
    label_path = os.path.join(input, "labels")
    save_path = os.path.join(folder, "labels-action.csv")
    print(f"saving label file to {save_path}...")

    # -------------------------
    # 02. 合併label檔 (labels-action.csv)
    # -------------------------

    # 合併所有label檔
    labels = glob(os.path.join(label_path, "*.csv"))
    merge = pd.DataFrame()
    for label in labels:
        file_name = os.path.basename(label).replace(".csv", "")
        file = pd.read_csv(
            label, encoding="latin1", names=["1", "2", "frame", "action", "3"]
        )
        file.drop(["1", "2", "3"], axis=1, inplace=True)
        file.drop(0, axis=0, inplace=True)
        file["video"] = file_name
        file["frame_name"] = file["video"] + "_" + file["frame"]

        merge = pd.concat([merge, file], axis=0)

    # 顯示動作標籤資訊
    labels = sorted(merge.action.unique())
    labels.remove("Other")
    labels = {v: k for k, v in enumerate(labels)}
    labels["Other"] = len(labels)
    print(f"The number of data: {merge.shape[0]}.")
    print(f"The number of labels: {len(labels)}.")
    print(f"The label list: {labels}")

    # 將標籤存入txt檔 --> classes.txt
    file_name = os.path.join(folder, "classes.txt")
    with open(file_name, "w") as file:
        for label, index in labels.items():
            line = f"{index}: {label}\n"
            file.write(line)

    # 對動作標籤進行編號
    merge["label"] = merge["action"].map(labels)
    merge["label"] = merge["label"].astype("int")

    # 儲存合併後的label檔 --> labels-action.csv
    merge.to_csv(save_path, index=False)

    # -------------------------
    # 以下TRAINING DATA的處理
    # -------------------------

    if test != True:
        merge = merge[merge.action != "Other"]

        # -------------------------
        # 03. 將標記檔轉換為成訓練資料格式 --> labels-generate.csv
        # -------------------------
        # 設定每段影片資料的開始時間與結束時間 = interval/fps秒 (預設20/30s)
        save_path = os.path.join(folder, "labels-generate.csv")
        print(f"saving file to {save_path}...")

        merge["start"] = merge["frame"].apply(lambda x: round(int(x) / fps, 2))
        merge["end"] = merge["start"].apply(lambda x: round(x + (interval / fps), 2))
        videos = merge.video.unique()

        # 設定可使用的資料：在一段區間內的動作標籤必須相同，若超出區間則不使用該幀資料
        used = []
        for video in videos:
            temp = merge[merge.video == video]
            labels = temp.label.values

            for i in range(len(temp) - interval):
                current_label = labels[i]
                compare_label = labels[i + interval]
                used.append(1) if current_label == compare_label else used.append(0)

            used.extend([0] * interval)
            del temp
        merge["used"] = used
        merge["used"] = np.where(merge.index % (interval // 4) == 0, merge.used, 0)
        merge.reset_index(drop=True, inplace=True)
        merge = merge[merge.used == 1]

        # -------------------------
        # 04. 依照每個工作站的資料量進行資料分割 (預設為600筆[train_data])
        # -------------------------
        used = []
        for video in videos:
            labels_ = merge[(merge.video == video)].label.unique()
            num_label = int(train_data / len(labels_))
            print(f"{video} has {num_label} data for each label.")

            for label in labels_:
                temp = (
                    merge[(merge.video == video) & (merge.label == label)]
                    .head(num_label)
                    .index.values
                )
                used.extend(temp)

        merge.loc[used, "train"] = 1
        merge.loc[merge[merge.train != 1].index, "train"] = 0
        merge["train"] = merge["train"].astype(int)

        # 儲存處理後的label檔
        merge.to_csv(save_path, index=False)

        # 顯示數據分布
        pd.set_option("display.max_rows", 40)
        pd.set_option("display.max_columns", 40)
        # merge[merge.train==1].pivot_table(index='label', columns='video', values='frame', aggfunc='count', fill_value=0)
        merge[merge.train == 1].groupby("video").size()

        # -------------------------
        # 05. 讀取標籤檔與影片檔並儲存至建立之暫存資料夾 --> train/..data.pt
        # -------------------------
        # 建立路徑
        save_path = os.path.join(folder, "train")
        os.makedirs(save_path, exist_ok=True)
        print(f"saving file to {save_path}...")

        # 讀取標籤檔
        dataset = merge[merge.train == 1]
        print(f"The number of dataset: {len(dataset)}")

        # 讀取並解碼影片檔
        video_path = [
            os.path.join(input, "videos", f"{video}.mp4")
            for video in dataset.video.unique()
        ]
        for video_ in video_path:
            print(f"Processing the video file [{video_}]...")
            encoded_video = EncodedVideo.from_path(video_)
            video_name = os.path.basename(video_).replace(".mp4", "")
            video_segment = dataset[dataset.video == video_name]

            start = video_segment["start"].values
            end = video_segment["end"].values
            labels = video_segment["label"].values
            frames = video_segment["frame"].values
            series = list(video_segment.index)

            # 儲存影片片段為tensor檔(.pt)
            for i in tqdm(range(len(start))):
                clip = encoded_video.get_clip(start_sec=start[i], end_sec=end[i])
                clip = clip["video"]
                torch.save(
                    clip,
                    os.path.join(
                        save_path,
                        f"S{str(series[i]).zfill(6)}-{video_name}-A{labels[i]}.pt",
                    ),
                )
            del encoded_video

        print("Completed processing and saving the label file!")

        # -------------------------
        # 06. 根據資料夾中的tensor檔(影片解碼檔)，生成訓練資料集並儲存 --> labels-train.csv
        # -------------------------

        # 讀取tensor檔
        dataset = glob(os.path.join(save_path, "*.pt"))

        save_path = os.path.join(folder, "labels-train.csv")
        print(f"saving training data to the dir: {save_path}...")

        # 將tensor檔的資訊存入dataframe
        file = pd.DataFrame(dataset, columns=["path"])
        file["series"] = file["path"].apply(
            lambda x: os.path.basename(x).replace(".pt", "").split("-")[0][1:]
        )
        file["video"] = file["path"].apply(
            lambda x: os.path.basename(x).replace(".pt", "").split("-")[1]
        )
        file["label"] = file["path"].apply(
            lambda x: os.path.basename(x).replace(".pt", "").split("-")[2][1:]
        )

        # 顯示訓練資料分布
        print(
            file.pivot_table(
                index="video",
                columns="label",
                values="series",
                aggfunc="count",
                fill_value=0,
            )
        )

        # 儲存訓練資料集
        file.to_csv(save_path, index=False)

    print(
        f"Data preparation completed and was saved to the folder {folder} and will be removed after finishing training."
    )


if __name__ == "__main__":
    # -------------------------
    # 參數設定
    # -------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=30, help="影片fps, 預設為30fps")
    parser.add_argument("--interval", type=int, default=20, help="每段影片的時間長度(單位:幀), 預設為20幀")
    parser.add_argument("--train-data", type=int, default=600, help="每個影片的訓練資料數量, 預設為600筆")
    parser.add_argument("--input", type=str, default="input", help="輸入資料夾, 包含labels和videos")
    parser.add_argument("--output", type=str, default="temp_train", help="輸出資料夾")
    parser.add_argument("--test", action="store_true", help="是否僅生成測試資料; 預設為訓練資料")
    args = parser.parse_args()
    
    data_prepare(args.input, args.output, args.fps, args.interval, args.train_data, args.test)