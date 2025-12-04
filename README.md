[English] | [中文](#繁體中文)

---

## English

# Production Line Optimization via Multi-Modal Imaging and Human Action Recognition

### Project Overview

This project leverages deep learning techniques, combining multi-modal image processing and human action recognition to optimize production line analysis.  
By integrating the YOLOv7 object detection model with the SlowFast action recognition model, the system can automatically analyze operator behavior, recognize assembly actions, and assist in production flow optimization.

#### Research Objectives

1. **Operator Detection**: Detect operator positions on the production line using YOLOv7.
2. **Action Recognition**: Recognize operators' assembly actions using the SlowFast model.
3. **Process Analysis**: Analyze assembly action cycles to optimize production efficiency.
4. **Performance Evaluation**: Provide evaluation metrics such as precision and recall.

### System Architecture

#### Deep Learning Models

- **YOLOv7**: For operator detection.
- **SlowFast Networks**: For action recognition.

#### System Pipeline

```
Data Collection → Data Preprocessing → Model Training → Action Detection → Performance Evaluation → Process Analysis
```

### Project Structure

```
main/
├── code/
│   ├── input/                      # Raw training data
│   ├── temp_train/                 # Temporary training data (auto-generated)
│   ├── models/                     # Model weights
│   ├── outputs/                    # Output results
│   ├── yolov7/                     # YOLOv7 implementation
│   ├── 01-data-collect.py          # Data collection: record training videos
│   ├── 02-auto-train.py            # Automated training pipeline
│   ├── 03-auto-detect.py           # Automated detection pipeline
│   ├── 04-auto-metrics.py          # Automated evaluation pipeline
```

> **Important**: Filenames inside `input/{project}/labels/` and `input/{project}/videos/` must be exactly the same (1-to-1 mapping).

## Environment Setup

### System Requirements

- Python 3.7+
- CUDA 11.6+ (GPU recommended)
- Docker (optional but recommended)

### Option 1: Using Docker (Recommended)

1. **Build Docker Image**
```bash
cd main
docker build -t aver-demo .
```

2. **Run Container**
```bash
docker run -it --gpus all --name aver-demo -v /path/to/your-project:/A aver-demo bash
```

3. **Download Pretrained Weights**
   Download pretrained model weights and place them in the `models/` folder inside the container:
   - Download link: https://drive.google.com/drive/folders/1NUEQX7uN9KJFT1U_a6uFMBWL73afhXgX?usp=drive_link
   - Required files:
     - `models/operator/init.pt` - operator detection model
     - `models/action/init.pth` - action recognition model

### Option 2: Local Installation

1. **Install PyTorch**
```bash
# Choose the correct PyTorch version for your CUDA
# Example: CUDA 11.6
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

2. **Install Other Dependencies**
```bash
cd main/code
pip install -r yolov7/requirements.txt
pip install pytorchvideo
```

## Usage

### Step 1: Data Collection

Use a camera to record training videos:

```bash
cd main/code
python 01-data-collect.py --project Demo
```

### Step 2: Data Annotation

Use `ActionLabel.exe` to manually annotate videos and generate CSV label files:
- Save label files under `input/{project}/labels/`
- The label filename must be identical to the corresponding video filename

### Step 3: Automatic Training

Run the full automatic training pipeline:

```bash
python 02-auto-train.py \
    --project Demo \
    --operator-weights models/operator/init.pt \
    --action-weights models/action/init.pth
```

### Step 4: Automatic Detection

Run action recognition on new videos:

```bash
python 03-auto-detect.py \
    --project Demo \
    --input input/Demo/videos \
    --output outputs \
    --operator-weights models/operator/init.pt \
    --conf-thres 0.5
```

### Step 5: Automatic Evaluation

Compute model performance metrics:

```bash
python 04-auto-metrics.py \
    --input outputs/results/Demo.csv \
    --reference input/Demo
```

## Results and Performance

The system achieves good performance on the action recognition task:

- **Precision**: Accurately recognizes various assembly actions.
- **Recall**: Effectively detects when actions occur.
- **Confusion Matrix**: Provides detailed analysis of per-class recognition performance.

---

## 繁體中文

# 基於多模態影像及人體動作辨識技術優化產線流程分析

### 專題簡介

本專題旨在透過深度學習技術，結合多模態影像處理與人體動作辨識，優化產線流程分析。系統整合 YOLOv7 物件偵測模型與 SlowFast 動作識別模型，實現自動化的操作員動作分析、組裝動作識別與產線流程優化。

#### 研究目標

1. **操作員偵測**：使用 YOLOv7 模型偵測產線上的操作員位置
2. **動作識別**：使用 SlowFast 模型識別操作員的組裝動作
3. **流程分析**：分析組裝動作循環週期，優化產線效率
4. **效能評估**：提供精確率、召回率等評估指標

### 技術架構

#### 使用的深度學習模型

- **YOLOv7**：用於操作員偵測（Operator Detection）

- **SlowFast Networks**：用於動作識別（Action Recognition）

#### 系統架構流程

```
資料收集 → 資料預處理 → 模型訓練 → 動作偵測 → 效能評估 → 流程分析
```

## 專案結構

```
main/
├── code/
│   ├── input/                      # 訓練資料來源
│   ├── temp_train/                 # 訓練資料暫存資料夾（自動生成，訓練後自動刪除）
│   ├── models/                     # 模型權重儲存
│   ├── outputs/                    # 輸出結果
│   ├── yolov7/                     # YOLOv7 模型實作
│   ├── 01-data-collect.py          # 資料收集：錄製訓練影片
│   ├── 02-auto-train.py            # 自動化訓練流程
│   ├── 03-auto-detect.py           # 自動化偵測流程
│   ├── 04-auto-metrics.py          # 自動化評估流程
```

> **重要提醒**：`input` 資料夾中的 `labels` 和 `videos` 資料夾內的檔案名稱必須完全相同。

## 環境設定

### 系統需求

- Python 3.7+
- CUDA 11.6+ (建議使用 GPU)
- Docker (可選，建議使用)

### 方法一：使用 Docker（建議）

1. **建立 Docker 映像檔**
```bash
cd main
docker build -t aver-demo .
```

2. **建立並執行容器**
```bash
docker run -it --gpus all --name aver-demo -v /path/to/your-project:/A aver-demo bash
```

3. **下載模型權重**
   將預訓練模型權重下載並放置於容器內的 `models/` 資料夾中：
   - 下載連結：https://drive.google.com/drive/folders/1NUEQX7uN9KJFT1U_a6uFMBWL73afhXgX?usp=drive_link
   - 需要下載的檔案：
     - `models/operator/init.pt` - 操作員偵測模型
     - `models/action/init.pth` - 動作識別模型

### 方法二：本地環境安裝

1. **安裝 PyTorch**
```bash
# 根據您的 CUDA 版本選擇對應的 PyTorch
# 範例：CUDA 11.6
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

2. **安裝其他依賴套件**
```bash
cd main/code
pip install -r yolov7/requirements.txt
pip install pytorchvideo
```

## 使用說明

### 步驟 1：資料收集

使用攝影機錄製訓練影片：

```bash
cd main/code
python 01-data-collect.py --project Demo
```

### 步驟 2：資料標註

使用 `ActionLabel.exe` 手動標註影片，生成 CSV 標籤檔案：
- 標籤檔案需儲存在 `input/{project}/labels/`
- 標籤檔案名稱需與對應的影片檔案名稱完全相同

### 步驟 3：自動訓練

執行完整的自動化訓練流程：

```bash
python 02-auto-train.py \
    --project Demo \
    --operator-weights models/operator/init.pt \
    --action-weights models/action/init.pth
```

### 步驟 4：自動偵測

對新影片進行動作識別：

```bash
python 03-auto-detect.py \
    --project Demo \
    --input input/Demo/videos \
    --output outputs \
    --operator-weights models/operator/init.pt \
    --conf-thres 0.5
```

### 步驟 5：自動評估

計算模型效能指標：

```bash
python 04-auto-metrics.py \
    --input outputs/results/Demo.csv \
    --reference input/Demo
```

## 研究結果與效能

根據研究結果，系統在動作識別任務上達到良好的效能表現：

- **精確率（Precision）**：系統能夠準確識別各種組裝動作
- **召回率（Recall）**：能夠有效偵測動作的發生
- **混淆矩陣**：提供詳細的類別間識別結果分析
