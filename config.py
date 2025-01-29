import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd



class ModelArgs:
    def __init__(self):
        self.debug = False

        if self.debug:
            self.n_splits = 2
            self.epochs = 5  # 迭代次数
            self.test_times = 3  # 测试集重复测试的次数
        else:
            self.n_splits = 2
            self.epochs = 300  # 迭代次数
            self.test_times = 5  # 测试集重复测试的次数

        self.kfold_index = None
        self.experiment_name = None

        self.current_path = os.path.split(os.path.realpath(__file__))[0]

        self.data_name = 'German'

        if self.data_name == 'German':
            self.dim_data = 20
            self.num_bins = 5
            self.d = 100  # 每个任务中的样本的个数
        elif self.data_name == 'Taiwan':
            self.dim_data = 23
            self.num_bins = 10
            self.d = 3000  # 每个任务中的样本的个数
        elif self.data_name == 'GiveMeSomeCredit':
            self.dim_data = 10
            self.num_bins = 10
            self.d = 1000  # 10800  # 每个任务中的样本的个数
        elif self.data_name == 'LendingClub_part_2015_Q1':
            self.dim_data = 65
            self.num_bins = 10
            self.d = 2598  # 10800  # 每个任务中的样本的个数
        elif self.data_name == 'LendingClub_part_2015_Q2':
            self.dim_data = 65
            self.num_bins = 10
            self.d = 2598  # 每个任务中的样本的个数
        elif self.data_name == 'LendingClub_part_2015_Q3':
            self.dim_data = 65
            self.num_bins = 10
            self.d = 2800  # 每个任务中的样本的个数
        elif self.data_name == 'LendingClub_part_2015_Q3':
            self.dim_data = 65
            self.num_bins = 10
            self.d = 2285  # 每个任务中的样本的个数

        elif self.data_name == 'Vesta_part_1':
            self.dim_data = 676
            self.num_bins = 10
            self.d = 11400  # 每个任务中的样本的个数

        self.dim_embed = 128
        self.dim_out = 16
        self.num_heads = 2  # 必须要能整除dim_hid
        self.classes_list = [0, 1]
        self.classe_tpye_list = [0, 1]  # 类别代码
        self.cluster_num_list = [5, 5]  # 每一类里面有多少簇
        self.cluster_num_range = [3, 15]  # 每一类里面有多少簇
        self.n_way = sum(self.cluster_num_list)  # Number of classes in the few shot classification task

        self.depth_sdt = 5
        self.batch_size = 500
        self.num_tasks = 10  # 跟batch_size一个意思。Number of k-shot tasks to group into a single batch
        self.dropout = 0.1

        self.feature_dropout = 0.1  # 0.5,
        self.decay_rate = 0.995
        self.l2_regularization = 0.1
        self.output_regularization = 0.1
        self.hidden_sizes = [16, 16]  # [64, 32],

        self.input_shape = 1

        self.activation = "exu"

        self.batch_size = 128
        self.shuffle = True
        self.drop_last = True

        self.scheduler = False

        self.overlap_ratio = 1 / 2
        # 采样器参数

        self.dropout = 0

        self.fm = True
        self.EAttention = True
        # self.fce = True  # Whether to us fully conditional embeddings
        # self.classify = True
        self.WeighLossesUseUncertainty = False
        self.clip_grad = True

        self.optimizer = 'adam'  # 'adam' 或者 'sgd'

        self.incremental_learning = False  # 是否是在原来模型基础上进行增量训练

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = 'cpu'

        self.metric_loss_coef = 0.5  # 度量损失的折算率
        self.query_loss_coef = 0.5  # 度量损失的折算率

        self.batch_stratified = True

        self.imbalance_ratio = [1, 5, 10, 20, 30, 40, 50]  # 1, 5, 10, 20, 30, 40, 50

        # 分类方法

        self.classify_method = 'logr'  # 分类方法可选：'knn', 或者 'proto', logr
        self.knn_num = 5  # knn中的k，K for K-nearest neighbors

        self.ablation_experiment = None
        self.ablation_list = [
            'proposed',  # 完整的模型
            # 'without_sampler',    # 删除抽样模块
            # 'without_sigmoid',  # 删除对交叉熵的处理模块
            # 'without_class',  # 删除对类别的平衡
            # 'with_num',  # 按照每个分箱的数量抽样
            # 'with_hard',  # 按照每个分箱的平均难度抽样
            # 'without_circle1_to_hard',  # 按照从简单到困难的顺序学习
            # 'without_circle1_to_easy',  # 按照从困难到简单的顺序学习
            # 'with_equal_num',  # 每个分箱抽相同个数
            # 'without_circle1_to_equal_num',  # 按照从侧重数量到等量的顺序学习，不对难度加强
            # 'without_fm',
            # 'without_dense',
            # 'without_fm_dense',
        ]
        self.splits_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # 'proposed',
        # 'without_sampler', 'without_class', 'without_num', 'without_hard',
        # 'without_circle1_to_hard', 'without_circle1_to_easy', 'without_circle1_to_equal_num',
        # 'without_fm', 'without_dense', 'without_fm_dense'

        self.learning_rate = 0.0001  # 学习率

        self.metric_loss_beta = 0.1
        self.support_loss_beta = 0.1
        self.query_loss_beta = 0.1

        self.taiwan = True

        # self.k_list = [2, 10]    # 每一类里面分簇个数的范围

        # self.k_shot = 200    # Number of examples per class in the support set
        # self.q_query = 10   # Number of examples per class in the query set

        self.m = sum(self.cluster_num_list)  # 每个batch中的簇的个数，Number of clusters present in a mini-batch

        self.support_ratio = 0.5  # D中，支持集比例

        self.metric_loss_fn = 'magnet'  # 'magnet' 或者 'triplet'
        self.proto_loss_fn = 'distance'  # 'softmax' 或者 'distance'

        self.wd = 1e-4  # 优化器中的权值衰减，weight_decay
        self.distance = 'l2'  # 'cosine' 或者 'l2'

        self.lstm_num_layers = 1  # Number of LSTM layers in the bidrectional LSTM g that embeds the support set

        self.unrolling_steps = 2  # Number of unrolling steps to run the Attention LSTM

        self.encoder_num_layers = 2

        # 采样器参数
        # self.num_tasks = 1  # 跟batch_size一个意思。Number of k-shot tasks to group into a single batch

        self.alpha = 1  # Margin alpha for magnet loss

        self.epsilon = 1e-8  # 为了避免分母为0

        self.wandb_key = '8d20af76ce543018197b017d3be5feccd40e83c7'

        self.evaluation_index = ['ACC',
                                 'Recall_1_recall', 'Precision_1',
                                 'Recall_0_specificity', 'Precision_0',
                                 'Gmean',
                                 'F_measure_1', 'F_measure_0',
                                 'AUC', 'KS']
        # self.evaluation_index = ['Brier', 'KS', 'AUC',
        #                          'AUPR', 'F_measure_1', 'F_measure_0',
        #                          'Gmean',
        #                          'ACC',
        #                          'Recall_1_recall', 'Precision_1',
        #                          'Recall_0_specificity', 'Precision_0',
        #                          ]
