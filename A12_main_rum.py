import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from config import ModelArgs
from models.A12_ExperimentBuilder import ExperimentBuilder
from models import utils

def main():
    args = ModelArgs()

    args.ablation_experiment = 'proposed'
    args.experiment_name = 'stratified_kfold'
    args.data_id = 0
    
    # 初始化模型
    mfn = ExperimentBuilder(args)
    mfn.build_model()
    mfn.build_data()

    # 模型训练和测试
    mfn.run_training_epochs()  # 训练
    out_label = mfn.run_testing_epoch()  # 测试

    # 评估
    eval_values = utils.eval_accuracy(y_pred=out_label[:, 1], y_true=out_label[:, 0])
    
    print(eval_values)



if __name__ == '__main__':
    main()
