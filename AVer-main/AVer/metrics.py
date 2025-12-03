import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from utils import get_logger, plot_cm_precision, plot_cm_recall, cal_filter
import argparse

                       
def cls_list(classes_dir): 
    cls_file = os.path.join(classes_dir, 'classes.txt')
    with open(cls_file) as f:
        classes = f.readlines()
    classes_list = {}
    for c in classes:
        c = c.strip('\n').split(': ')
        id_ = c[0].replace(' ', '')
        cls_ = c[1].replace('\'', '').replace(',', '')
        classes_list[id_] = cls_

    idx_list = {}
    for idx, item in classes_list.items():
        idx_list[item] = idx

    return classes_list, idx_list

def idx2cls(classes_list, cls2idx_list):
    classes_result = []
    for idx in classes_list:
        classes_result.append(cls2idx_list[str(idx)])
    return classes_result

def get_trues(classes_dir):
    return pd.read_csv(os.path.join(classes_dir, 'labels-action.csv'))

def get_metrics(input_path='input', 
                output_path='output', 
                ref_path='temp_train', 
                interval=25):
    """
    args:
        input_path (預設值: 'input'): 輸入資料的路徑。CSV檔案應位於此路徑下。
        output_path (預設值: 'output'): 輸出結果的路徑。包括 'result.csv' 和 'result_filter.csv' 等檔案。
        ref_path (預設值: 'temp_train'): 參考資料的路徑，用於取得類別標籤、真實標籤等。
        interval (預設值: 25): 用於控制計算結果時取樣的間隔，例如每 N 幀計算一次。
        no_log (預設值: False): 一個布林值，如果設定為 True，則不會記錄任何日誌。
    """
    
    # -------------------------
    # 定義資料標籤
    # -------------------------
    # get cls
    cls2idx_list, idx2cls_list = cls_list(ref_path)
    other_idx = idx2cls_list['Other']
    # true label
    file_label = get_trues(ref_path)
    # test file
    files = glob(os.path.join(input_path, '*.csv'))

    # -------------------------
    # 合併預測結果與真實標籤
    # -------------------------
    results = pd.DataFrame()

    for file_ in files:
        file_pred = pd.read_csv(file_)
        video_name = os.path.basename(file_).replace('.csv', '')

        try:
            preds = file_pred.loc[:, 'slowfast'].values
        except:
            preds = file_pred.loc[:, 'preds'].values
        trues = file_label[file_label.video==video_name].label.values
        result_ = pd.DataFrame({'video': video_name, 
                            'frame': file_pred.index,
                            'preds': preds, 
                            'trues': trues, 
                            'correct': preds==trues, 
                            }) 
        results = pd.concat([results, result_], axis=0)

    # -------------------------
    # 確認標籤是否對應
    # -------------------------
    label_trues = sorted(results.trues.unique())
    labels_preds = sorted(results.preds.unique())
    labels = sorted(list(set(list(results.preds.unique()) + list(results.trues.unique()))))
    labels = idx2cls(labels, cls2idx_list)

    # -------------------------
    # 存檔
    # -------------------------
    results.to_csv(os.path.join(output_path, f'result.csv'), index=False)

    # -------------------------
    # 計算結果
    # -------------------------
    results['isCount'] = results.frame
    results['isCount'] = results.isCount.apply(lambda x: 1 if (x+1) % interval == 0 else 0) #TODO:hyperparameter
    results_n = results[results.isCount==1]
    results_wo = results_n[results_n.trues!=int(other_idx)]
    results_wo.to_csv(os.path.join(output_path, f'result_filter.csv'), index=False)
    
    return results_wo

def metrics(input_path='input', 
            ref_path='temp_train', 
            interval=25, 
            no_log=True):
    """
    args:
        input_path (預設值: 'input'): 輸入資料的路徑。CSV檔案應位於此路徑下。
        ref_path (預設值: 'temp_train'): 參考資料的路徑，用於取得類別標籤、真實標籤等。
        interval (預設值: 25): 用於控制計算結果時取樣的間隔，例如每 N 幀計算一次。
        no_log (預設值: False): 一個布林值，如果設定為 True，則不會記錄任何日誌。
    """
    exp_name = os.path.basename(input_path).replace('.csv', '')
    output_path = os.path.join('outputs', 'metrics', exp_name)
    os.makedirs(output_path, exist_ok=True)
    
    # 定義資料標籤
    cls2idx_list, idx2cls_list = cls_list(ref_path)
    other_idx = idx2cls_list['Other']

    # 合併預測結果與真實標籤
    results = pd.DataFrame()
    
    # true label
    file_label = get_trues(ref_path)
    file_pred = pd.read_csv(input_path)
    video_name = os.path.basename(input_path).replace('.csv', '')

    preds = file_pred.loc[:, 'slowfast'].values
    trues = file_label.label.values
    result_ = pd.DataFrame({'video': video_name, 
                        'frame': file_pred.index,
                        'preds': preds, 
                        'trues': trues, 
                        'correct': preds==trues, 
                        'isCount':True}) 
    results = pd.concat([results, result_], axis=0)

    # 確認標籤是否對應
    label_trues = sorted(results.trues.unique())
    labels_preds = sorted(results.preds.unique())
    labels = sorted(list(set(list(results.preds.unique()) + list(results.trues.unique()))))
    labels = idx2cls(labels, cls2idx_list)

    # 存檔
    results.to_csv(os.path.join(output_path, f'result.csv'), index=False)

    # -------------------------
    # 計算結果
    # -------------------------
    results['isCount'] = results.frame
    results['isCount'] = results.isCount.apply(lambda x: 1 if (x+1) % interval == 0 else 0)
    results_n = results[results.isCount==1]
    results_wo = results_n[results_n.trues!=int(other_idx)]
    # results_wo.to_csv(os.path.join(output_path, f'{exp_name}_result_interval.csv'), index=False)
    
    if no_log:
        # -------------------------
        # 畫混淆矩陣
        # -------------------------
        plot_cm_recall(results_n, labels, output_path, title='Other')
        plot_cm_precision(results_n, labels, output_path, title='Other')

        # -------------------------
        # 紀錄log
        # -------------------------
        logger = get_logger(os.path.join(output_path, 'log-test.txt'))
        logger.info(f'cls2idx_list: {cls2idx_list}')
        logger.info(f'idx2cls_list: {idx2cls_list}')
        logger.info(f'label_trues: {label_trues}')
        logger.info(f'labels_preds: {labels_preds}')
        logger.info(f'labels: {labels}')
        
        logger.info(f'data size: {results_n.shape[0]}')
        logger.info(results_n.correct.sum() / results_n.shape[0])
        logger.info(classification_report(results_n.trues, results_n.preds))
        print(f'log file saved to {os.path.join(output_path, "log-test.txt")}')

    return results_wo

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="input", help="輸入路徑")
    parser.add_argument("--reference", type=str, default="temp_train", help="參照資料路徑(包含classes.txt, labels-action.csv)")
    parser.add_argument("--interval", type=int, default="25", help="辨識間隔幀數")
    parser.add_argument("--no-log", action="store_true", help="是否儲存log和畫圖")
    args = parser.parse_args()

    input, ref, interval, no_log = args.input, args.reference, args.interval, args.no_log
    
    metrics(input_path=input, 
                ref_path=ref,
                interval=25, 
                no_log=True)
    
