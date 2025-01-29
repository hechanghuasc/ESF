import os
import random
import numpy as np
import pandas as pd
import sqlite3
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from models import utils
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import models.loss_function as lf


class LoadData(object):
    def __init__(self, args):
        self.db = None

        self.args = args
        self.scaler_col_norm = MinMaxScaler()

        self.data_array = {}

    def get_data_loader(self):
        train_set, test_set = self.get_data_set()

        train_loader = DataLoader(train_set,
                                  num_workers=10,
                                  batch_sampler=utils.NShotTaskSampler(train_set,
                                                                       self.args.episodes_per_epoch,
                                                                       self.args.n_train,
                                                                       self.args.k_train,
                                                                       self.args.q_train,
                                                                       ),
                                  )
        test_loader = DataLoader(test_set,
                                 num_workers=10,
                                 batch_sampler=utils.NShotTaskSampler(test_set,
                                                                      self.args.episodes_per_epoch,
                                                                      self.args.n_train,
                                                                      self.args.k_train,
                                                                      self.args.q_train,
                                                                      ),
                                 )

        return train_loader, test_loader

    def get_data_set(self):
        # 连接到db数据库
        self.connect_db()

        # 获取数据
        train_x, train_y, test_x, test_y = self.get_raw_data_from_db(self.args.kfold_index)

        # 数据归一化
        train_x, test_x = self.get_scaler(train_x, test_x)

        return train_x, train_y, test_x, test_y

    def get_scaler(self, train, test):
        # 数据归一化
        train_x = self.scaler_col_norm.fit_transform(train)
        test_x = self.scaler_col_norm.transform(test)

        return train_x, test_x

    def get_kf_xy(self, data_dist, data_name):
        idx = data_dist['stratified_kfold_10_id'][data_name]

        data = data_dist['raw_data'][data_dist['raw_data']['ID'].isin(idx)]

        y = data['y'].values
        x = data.drop(columns=['ID', 'y']).values

        return x, y

    def get_kfold_data_from_pkl(self, kfold_index):
        data_dist = self.load_pkl()

        # 读取原始训练数据
        train_x, train_y = self.get_kf_xy(data_dist, 'train_' + str(kfold_index))
        test_x, test_y = self.get_kf_xy(data_dist, 'test_' + str(kfold_index))

        train_x, test_x = self.get_scaler(train_x, test_x)

        return train_x, train_y, test_x, test_y

    def load_pkl(self):
        pkl_path = os.path.join(self.args.current_path, "data", self.args.data_name + ".pkl")
        # with open(pkl_path, 'rb') as f:
        #     data_dist = pickle.load(f)
        data_dist = pd.read_pickle(pkl_path)

        return data_dist

    def get_imbalance_xy(self, data_dist, data_name):
        idx = data_dist['imbalance_kfold_10_id'][data_name]

        data = data_dist['raw_data'][data_dist['raw_data']['ID'].isin(idx)]

        y = data['y'].values
        x = data.drop(columns=['ID', 'y']).values

        return x, y

    def get_imbalance_data_from_pkl(self, bad_ratio):
        data_dist = self.load_pkl()

        # 读取原始训练数据
        train_x, train_y = self.get_imbalance_xy(data_dist, 'train_' + str(bad_ratio) + '_0')
        test_x, test_y = self.get_imbalance_xy(data_dist, 'test_' + str(bad_ratio) + '_0')

        train_x, test_x = self.get_scaler(train_x, test_x)

        return train_x, train_y, test_x, test_y

    def connect_db(self):
        if self.args.data_name:
            self.db = sqlite3.connect(os.path.join(self.args.current_path, "data", self.args.data_name + ".db"))


class DataBatchSamplerValue(object):
    def __init__(self, args):
        self.test_x_n = None
        self.test_x_p = None
        self.train_x_n = None
        self.train_x_p = None
        self.test_y_n = None
        self.train_y_n = None
        self.test_y_p = None
        self.train_y_p = None
        self.args = args
        self.scaler_col_norm = MinMaxScaler()

    def load_pkl(self):
        pkl_path = os.path.join(self.args.current_path, "data", self.args.data_name + ".pkl")
        f = open(pkl_path, 'rb')
        data_dist = pickle.load(f)

        return data_dist

    def get_kf_xy(self, data_dist, data_name):
        idx = data_dist['stratified_kfold_10_id'][data_name]

        data = data_dist['raw_data'][data_dist['raw_data']['ID'].isin(idx)]

        y = data['y'].values
        x = data.drop(columns=['ID', 'y']).values

        idx = np.where(y == 1)[0]
        y_p = y[idx]
        x_p = x[idx, :]

        idx = np.where(y == 0)[0]
        y_n = y[idx]
        x_n = x[idx, :]

        return x_p, y_p, x_n, y_n

    def get_imbalance_xy(self, data_dist, data_name):
        idx = data_dist['imbalance_kfold_10_id'][data_name]

        data = data_dist['raw_data'][data_dist['raw_data']['ID'].isin(idx)]

        y = data['y'].values
        x = data.drop(columns=['ID', 'y']).values

        idx = np.where(y == 1)[0]
        y_p = y[idx]
        x_p = x[idx, :]

        idx = np.where(y == 0)[0]
        y_n = y[idx]
        x_n = x[idx, :]

        return x_p, y_p, x_n, y_n

    def get_scaler(self, train_x_p, train_x_n, test_x):
        x = np.concatenate([train_x_p, train_x_n], axis=0)
        # 数据归一化
        self.scaler_col_norm.fit(x)
        train_x_p = self.scaler_col_norm.transform(train_x_p)
        train_x_n = self.scaler_col_norm.transform(train_x_n)
        test_x = self.scaler_col_norm.transform(test_x)

        return train_x_p, train_x_n, test_x

    def get_data_from_pkl(self, experiment_name, data_id):
        global test_x_p, test_x_n, test_y_p, test_y_n

        data_dist = self.load_pkl()

        if experiment_name.split('_')[0] == 'stratified':
            self.train_x_p, self.train_y_p, self.train_x_n, self.train_y_n = self.get_kf_xy(data_dist,
                                                                                            'train_{}'.format(
                                                                                                str(data_id)))
            test_x_p, test_y_p, test_x_n, test_y_n = self.get_kf_xy(data_dist, 'test_{}'.format(str(data_id)))

        elif experiment_name.split('_')[0] == 'imbalance':
            self.train_x_p, self.train_y_p, self.train_x_n, self.train_y_n = self.get_imbalance_xy(data_dist,
                                                                                                   'train_{}_{}'.format(
                                                                                                       str(
                                                                                                           experiment_name.split(
                                                                                                               '_')[1]),
                                                                                                       str(data_id)))
            test_x_p, test_y_p, test_x_n, test_y_n = self.get_imbalance_xy(data_dist,
                                                                           'test_{}_{}'.format(
                                                                               str(experiment_name.split('_')[1]),
                                                                               str(data_id)))

        test_x = np.concatenate([test_x_p, test_x_n], axis=0)
        test_y = np.concatenate([test_y_p, test_y_n], axis=0)

        self.train_x_p, self.train_x_n, test_x = self.get_scaler(self.train_x_p, self.train_x_n, test_x)

        self.test_dl = self.get_dataloader(test_x, test_y)

    def get_dataloader(self, x, y):
        x = torch.from_numpy(x).to(self.args.device, dtype=torch.double)
        y = torch.from_numpy(y).to(self.args.device, dtype=torch.double)

        data_dl = DataLoader(dataset=TensorDataset(x, y), batch_size=self.args.batch_size)

        return data_dl

    def get_test_dataloader(self):
        test_x = np.concatenate([self.train_x_p, self.train_x_n], axis=0)
        test_y = np.concatenate([self.train_y_p, self.train_y_n], axis=0)

        idx_list = list(range(test_x.shape[0]))
        random.shuffle(idx_list)

        test_x = test_x[idx_list, :]
        test_y = test_y[idx_list]
        dl = self.get_dataloader(test_x, test_y)

        return dl

    def cal_sample_values(self, model):
        # 计算样本价值
        model.eval()

        # 少数类
        p_pred = model(torch.from_numpy(self.train_x_p).to(self.args.device))
        p_loss = lf.binary_ce_loss(p_pred.view(-1), torch.from_numpy(self.train_y_p).to(self.args.device))
        # loss = lf.f1_loss(pred.view(-1), torch.from_numpy(self.train_y_p).to(self.args.device), sample_type='p')
        # self.sample_value_p = torch.sigmoid(loss + torch.log(torch.tensor(0.5))).cpu().detach().numpy()
        if self.args.ablation_experiment == 'without_sigmoid':
            self.sample_value_p = p_loss.cpu().detach().numpy()
        else:
            self.sample_value_p = torch.sigmoid(p_loss - torch.mean(p_loss)).cpu().detach().numpy()
        # self.sample_value_p = torch.sigmoid(loss).cpu().detach().numpy()
        # self.sample_value_p = loss.cpu().detach().numpy()

        # 多数类
        n_pred = model(torch.from_numpy(self.train_x_n).to(self.args.device))
        n_loss = lf.binary_ce_loss(n_pred.view(-1), torch.from_numpy(self.train_y_n).to(self.args.device))
        # loss = lf.f1_loss(pred.view(-1), torch.from_numpy(self.train_y_n).to(self.args.device), sample_type='n')
        # self.sample_value_n = torch.sigmoid(loss + torch.log(torch.tensor(0.5))).cpu().detach().numpy()
        if self.args.ablation_experiment == 'without_sigmoid':
            self.sample_value_n = n_loss.cpu().detach().numpy()
        else:
            self.sample_value_n = torch.sigmoid(n_loss - torch.mean(n_loss)).cpu().detach().numpy()
        # self.sample_value_n = torch.sigmoid(loss).cpu().detach().numpy()
        # self.sample_value_n = loss.cpu().detach().numpy()

        p = np.exp(np.mean(self.sample_value_p))
        n = np.exp(np.mean(self.sample_value_n))

        self.p = p / (p + n + 1e-8)
        self.n = n / (p + n + 1e-8)

        pred = torch.concat([p_pred, n_pred], dim=0).view(-1)
        label = torch.concat([torch.from_numpy(self.train_y_p).to(self.args.device),
                             torch.from_numpy(self.train_y_n).to(self.args.device)], dim=0)

        tp = torch.sum(pred * label).cpu().detach().numpy()
        fp = torch.sum(pred * (1 - label)).cpu().detach().numpy()
        fn = torch.sum((1 - pred) * label).cpu().detach().numpy()
        tn = torch.sum((1 - pred) * (1 - label)).cpu().detach().numpy()

        p_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
        n_f1 = 2 * tn / (2 * tn + fn + fp)

        self.p_f1_cost = (1 - p_f1)/((1 - p_f1)+(1 - n_f1))
        self.n_f1_cost = (1 - n_f1)/((1 - p_f1)+(1 - n_f1))

        # 构建样本分箱
        # 列含义：id, sample_value, cumsum, bin(分箱编号)
        self.sample_new_labels_p = self.cut_sample_bins(np.zeros((self.train_y_p.shape[0], 4)), self.sample_value_p)
        self.sample_new_labels_n = self.cut_sample_bins(np.zeros((self.train_y_n.shape[0], 4)), self.sample_value_n)

        # 存储每个分箱中样本个数和平均样本价值
        self.bin_dict = {}
        self.bin_dict['p'] = self.get_bin_dict(self.sample_new_labels_p)
        self.bin_dict['n'] = self.get_bin_dict(self.sample_new_labels_n)

        z = 0

    def plt_sample_value(self, epoch):

        sns.displot(self.sample_value_p)
        path = os.path.join(self.args.current_path, 'jpg', 'sample_value_p_{}.jpg'.format(epoch))
        plt.savefig(path)

        sns.displot(self.sample_value_n)
        path = os.path.join(self.args.current_path, 'jpg', 'sample_value_n_{}.jpg'.format(epoch))
        plt.savefig(path)

    def cut_sample_bins(self, new_labels, sample_values):
        # 列含义：id, sample_value, cumsum, cluster

        new_labels[:, 0] = np.argsort(sample_values, kind='quicksort')  # 获取从小到大的索引
        new_labels[:, 1] = sample_values[new_labels[:, 0].astype(int)]  # 获取样本价值得分

        new_labels[:, 2] = np.cumsum(new_labels[:, 1])  # 计算样本价值得分累加
        # bin_values = new_labels[-int(new_labels.shape[0]*0.1), 2] / self.args.num_bins  # 计算分箱长度
        bin_values = (new_labels[-1, 2] - new_labels[0, 2]) / self.args.num_bins  # 计算分箱长度

        for i in range(self.args.num_bins):
            idx = np.where(new_labels[:, 2] >= new_labels[0, 2] + bin_values * i)
            new_labels[idx, 3] = i

        return new_labels

    def get_bin_dict(self, new_labels):
        bin_dict = {'count_importance': {}, 'value_importance': {}}

        dict_temp = {'count': {}, 'value': {}}
        for i in range(self.args.num_bins):
            idx = np.where(new_labels[:, 3] == i)[0]
            dict_temp['count'][i] = len(idx)
            dict_temp['value'][i] = np.mean(new_labels[idx, 1])

        for i in range(self.args.num_bins):
            bin_dict['count_importance'][i] = dict_temp['count'][i] / (sum(dict_temp['count'].values()) + 1e-8)
            bin_dict['value_importance'][i] = dict_temp['value'][i] / (sum(dict_temp['value'].values()) + 1e-8)

        return bin_dict

    def get_batch_samples(self):
        xx = {}
        yy = {}
        sample_values = {}

        for task_id in range(self.args.num_tasks + 1):
            bin_num_p = self.get_batch_num(task_id, self.bin_dict['p'], self.p_f1_cost)
            # bin_num_p = [100,100,100,100,100,100,100,100,100,100]
            bin_num_n = self.get_batch_num(task_id, self.bin_dict['n'], self.n_f1_cost)
            # bin_num_n = [100,100,100,100,100,100,100,100,100,100]

            idx_p_list = []
            idx_n_list = []

            for bin_id in range(self.args.num_bins):
                idx_p = np.where(self.sample_new_labels_p[:, 3] == bin_id)[0].tolist()
                # print('{}-{}'.format(len(idx_p), bin_num_p[bin_id]))
                idx_n = np.where(self.sample_new_labels_n[:, 3] == bin_id)[0].tolist()
                # print('{}-{}'.format(len(idx_n), bin_num_n[bin_id]))

                bin_num = min(len(idx_p), len(idx_n), abs(bin_num_p[bin_id]), abs(bin_num_n[bin_id]))
                # print(bin_num)
                # print(len(idx_p), len(idx_n), abs(bin_num_p[bin_id]), abs(bin_num_n[bin_id]), bin_num)

                idx_p_list.extend(random.sample(self.sample_new_labels_p[idx_p, 0].astype(int).tolist(), bin_num))
                idx_n_list.extend(random.sample(self.sample_new_labels_n[idx_n, 0].astype(int).tolist(), bin_num))

            x = np.concatenate([self.train_x_p[idx_p_list], self.train_x_n[idx_n_list]], axis=0)
            y = np.concatenate([self.train_y_p[idx_p_list], self.train_y_n[idx_n_list]], axis=0)

            x = torch.from_numpy(x).to(self.args.device, dtype=torch.double)
            y = torch.from_numpy(y).to(self.args.device, dtype=torch.double)

            xx[task_id] = x
            yy[task_id] = y

            sample_value = np.concatenate([self.sample_value_p[idx_p_list], self.sample_value_n[idx_n_list]])
            sample_value = torch.from_numpy(sample_value).to(self.args.device, dtype=torch.double)
            sample_values[task_id] = sample_value

        return xx, yy, sample_values

    def get_all_train_data(self):
        train_x = np.concatenate([self.train_x_p, self.train_x_n], axis=0)
        train_y = np.concatenate([self.train_y_p, self.train_y_n], axis=0)

        train_dl = self.get_dataloader(train_x, train_y)

        return train_dl

    def get_batch_num(self, task_id, importance_dict, f1_cost):
        bin_ratio = None

        # 计算每个bin选多少个样本
        count_importance = np.array(list(importance_dict['count_importance'].values()))
        value_importance = np.array(list(importance_dict['value_importance'].values()))

        if self.args.ablation_experiment == 'without_class':
            n_ratio = abs((self.args.num_tasks / 2) - task_id) / (self.args.num_tasks / 2)
            h_ratio = abs((self.args.num_tasks / 2) - abs((self.args.num_tasks / 2) - task_id)) / (
                        self.args.num_tasks / 2)

            bin_ratio = (n_ratio * count_importance) + (h_ratio * value_importance)
        elif self.args.ablation_experiment == 'with_num':
            bin_ratio = count_importance

        elif self.args.ablation_experiment == 'with_hard':
            bin_ratio = value_importance

        elif self.args.ablation_experiment == 'without_circle1_to_hard':
            n_ratio = (self.args.num_tasks - task_id) / self.args.num_tasks
            h_ratio = task_id / self.args.num_tasks

            bin_ratio = (n_ratio * count_importance) + (h_ratio * value_importance)

        elif self.args.ablation_experiment == 'without_circle1_to_easy':
            n_ratio = task_id / self.args.num_tasks
            h_ratio = (self.args.num_tasks - task_id) / self.args.num_tasks

            bin_ratio = (n_ratio * count_importance) + (h_ratio * value_importance)

        elif self.args.ablation_experiment == 'with_equal_num':
            bin_ratio = np.ones_like(value_importance)

        elif self.args.ablation_experiment == 'without_circle1_to_equal_num':
            equal_value = np.ones_like(value_importance)

            n_ratio = (self.args.num_tasks - task_id) / self.args.num_tasks
            h_ratio = task_id / self.args.num_tasks

            bin_ratio = (n_ratio * count_importance) + (h_ratio * equal_value)

        else:
            n_ratio = abs((self.args.num_tasks / 2) - task_id) / (self.args.num_tasks / 2) * (1 - f1_cost)
            h_ratio = abs((self.args.num_tasks / 2) - abs((self.args.num_tasks / 2) - task_id)) / (
                        self.args.num_tasks / 2) * f1_cost

            bin_ratio = (n_ratio * count_importance) + (h_ratio * value_importance)

        bin_ratio = bin_ratio / (sum(bin_ratio) + 1e-8)

        bin_num = (bin_ratio * self.args.d).astype(int)

        return bin_num
