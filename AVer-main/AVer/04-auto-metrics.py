import os
import subprocess
from glob import glob
import argparse
from metrics import metrics
from dataPrepare import data_prepare

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="outputs/results/0912_am_3th.csv", help="輸入檔案")
    parser.add_argument("--reference", type=str, default="input/Demo", help="輸入資料夾")
    args = parser.parse_args()

    input, reference = args.input, args.reference

    metrics(input_path=input, 
            ref_path='temp_test', 
            interval=25, 
            no_log=True)
    


    
    data_prepare(input=reference, 
                test=True)


    