import warnings

warnings.filterwarnings("ignore")
import os
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
import random

from utils import *
from dataloader import get_dataloader

from sklearn.metrics import accuracy_score
import shutil
import argparse
from utils import get_logger

def train(project="AVer-exp", input="temp_train", output="models", weight="models/action/init.pth", test_ratio=0.2, batch_size=2, lr=0.0001, epoch=20, seed=666):
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

    # -------------------------
    # 初始設定
    # -------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    project_name, input, output, init_weights = project, input, output, weight
    action_weights = os.path.join(output, f"{project_name}.pth") # 儲存路徑

    test_ratio = test_ratio
    min_batch_size = batch_size
    ttl_batch_size = 128
    batch_size = ttl_batch_size // min_batch_size
    learning_rate = lr
    num_epoch = epoch

    # 固定隨機種子
    seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 建立記錄檔
    logger = get_logger(f"{os.path.join(input, 'log-train.txt')}")
    logger.info(f"project: {project_name}\n input: {input}\n output: {output}\n init_weights: {init_weights}\n test_ratio: {test_ratio}\n batch_size: {batch_size}\n lr: {learning_rate}\n epoch: {num_epoch}\n seed: {seed}\n device: {device}")

    # -------------------------
    # 載入資料集
    # -------------------------
    action_list, cls_wgt, trainloader, validloader = get_dataloader(
        input, min_batch_size, test_ratio, seed
    )
    num_act = len(action_list) - 1 # remove Other

    # -------------------------
    # 載入模型 `slowfast_r50`
    # -------------------------
    logger.info("Loading `slowfast_r50` model...")
    try:
        model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)
    except:
        model = torch.load(weight)
    model.blocks[6].proj = nn.Linear(in_features=2304, out_features=num_act, bias=True)
    model = model.to(device)

    # criterion = nn.CrossEntropyLoss(weight=cls_wgt).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

    logger.info(f"action_list: {action_list}")
    logger.info(f"num_act: {num_act}")
    logger.info(f"trainloader: {len(trainloader)}, validloader: {len(validloader)}")

    # -------------------------
    # 訓練與驗證模型
    # -------------------------
    best_ = 99.0
    for epoch in tqdm(range(num_epoch)):
        logger.info("===Epoch %d===" % (epoch + 1))
        
        # training step
        running_loss, train_loss, train_acc, total, step = 0.0, 0.0, 0.0, 0, 0
        model.train()
        optimizer.zero_grad()
        for data in trainloader:
            inputs, labels = data
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            train_loss += loss.item()
            total += labels.size(0)
            loss.backward()

            # update weights every 2 steps -> total batch size  = 2 x mini batch_size
            if step % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

            # TODO: 計算accuracy
            logger.info("Epoch: %d, Step: %d, loss: %.6f" % (epoch + 1, step + 1, running_loss))
            running_loss = 0.0
            step += 1

        average_train_loss = train_loss / step
        logger.info("Training loss after epoch %d: %.6f" % (epoch + 1, average_train_loss))

        # Validation step
        model.eval()
        val_loss, total, step = 0.0, 0, 0

        with torch.no_grad():
            for data in validloader:
                val_inputs, val_labels = data
                val_labels = val_labels.to(device)

                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()
                total += val_labels.size(0)
                step += 1

            average_val_loss = val_loss / step  # total * 100

            # TODO: 計算accuracy
            if average_val_loss < best_:
                best_loss = average_val_loss
                torch.save(model, action_weights)
                logger.info("Best model updated at epoch %d | loss: %.6f" % (epoch + 1, best_loss))

        model.train()
    logger.info(f"Finished Training. The final model weight has been saved to {action_weights}")

    shutil.rmtree(os.path.join(input, 'train'))
    logger.info('Files in "train" have been removed.')


if __name__ == "__main__":
    # -------------------------
    # 參數設定
    # -------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="AVer-exp", help="專案名稱")
    parser.add_argument("--input", type=str, default="temp_train", help="資料集路徑")
    parser.add_argument("--output", type=str, default="models", help="儲存權重路徑")
    parser.add_argument("--weight", type=str, default="models/action/init.pth", help="初始模型權重")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="訓練-驗證比例")
    parser.add_argument("--batch-size", type=int, default=2, help="batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--epoch", type=int, default=20, help="epoch")
    parser.add_argument("--seed", type=int, default=666, help="fix random seed")
    args = parser.parse_args()

    model_train(args)