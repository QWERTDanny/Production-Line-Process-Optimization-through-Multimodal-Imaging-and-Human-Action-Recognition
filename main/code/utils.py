import logging
import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm


#### utils ####
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def evaluation(preds, labels):
    # 儲存動作預測結果

    test_result = pd.DataFrame()
    test_result["pred"] = preds.astype(int)
    test_result["label"] = labels.astype(int)
    test_result["correct"] = preds == labels
    acc = accuracy_score(labels, preds)
    print(f"Accuracy on validation data: {acc}")
    return test_result, acc


def plot_metrics(preds, labels, project_dir, class_list):
    # 儲存混淆矩陣   
    target_names = [class_list[str(int(n))] for n in set(labels.tolist()+preds.tolist())]
    print('classification_report:\n', classification_report(labels, preds, target_names=target_names))

    # 繪製混淆矩陣
    cm = confusion_matrix(labels, preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.round(cm, 2)
    plt.figure(figsize=(12, 12))
    sns.heatmap(cm, annot=True, xticklabels=target_names, yticklabels=target_names ,cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predict')
    plt.ylabel('True')
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    file_name = os.path.join(project_dir, "results", "matrix_test.png")
    plt.savefig(file_name)

def plot_cm_recall(results, labels, save_path, title):
    cm = confusion_matrix(results.trues, results.preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.round(cm, 2)

    sns.set(font_scale=1.3)
    plt.figure(figsize=(12, 12))
    sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels ,cmap="Blues", annot_kws={"size": 16})
    plt.title(f'Confusion Matrix (Recall) {title}', fontsize=20)
    plt.xlabel('Predict')
    plt.ylabel('True')
    plt.xticks(rotation=75)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'cm(Recall){title}.png'))
    print(f'saving confusion matrix to {save_path} as cm(Recall){title}.png')

def plot_cm_precision(results, labels, save_path, title):

    cm = confusion_matrix(results.trues, results.preds)
    cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    cm = np.round(cm, 2)

    sns.set(font_scale=1.3)
    plt.figure(figsize=(12, 12))
    sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels ,cmap='Blues', annot_kws={"size": 16})
    plt.title(f'Confusion Matrix (Precision) {title}', fontsize=20)
    plt.xlabel('Predict')
    plt.ylabel('True')
    plt.xticks(rotation=75)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'cm(Precision){title}.png'))
    print(f'saving confusion matrix to {save_path} as cm(Precision){title}.png')

def cal_filter(results, save_path):
    # results = pd.read_csv(results_path)
    result_cm = classification_report(results.trues, results.preds, output_dict=True)
    with open(save_path, "w") as file:
        for label, metrics in result_cm.items():
            if label.isdigit():
                precision = metrics['precision']
                recall = metrics['recall']
                
                if precision < 0.6 and recall > 0.85:
                    line = f"{label}: -0.6\n"
                elif precision > 0.85 and recall < 0.6:
                    line = f"{label}: 0.2\n"
                else:
                    line = f"{label}: 0.0\n"
                file.write(line)
