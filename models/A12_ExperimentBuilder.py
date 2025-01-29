import os
from copy import copy
import numpy as np
import torch
from torch import nn, optim, tensor
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from models.load_dataset import DataBatchSamplerValue
from models import utils
from models.A12 import DAnet, DAnet_rubust_test, DAnet_wo_fm, DAnet_wo_dense, DAnet_wo_fm_dense
import models.loss_function as lf


class ExperimentBuilder(nn.Module):
    def __init__(self, args):
        super(ExperimentBuilder, self).__init__()
        # 初始化参数
        self.model = None
        self.args = args

    def build_model(self):
        # 模型--------------------------------------------------------

        if self.args.ablation_experiment == 'without_fm':
            self.model = DAnet_wo_fm(self.args).to(self.args.device, dtype=torch.double)

        elif self.args.ablation_experiment == 'without_dense':
            self.model = DAnet_wo_dense(self.args).to(self.args.device, dtype=torch.double)

        elif self.args.ablation_experiment == 'without_fm_dense':
            self.model = DAnet_wo_fm_dense(self.args).to(self.args.device, dtype=torch.double)

        else:
            # self.args.ablation_experiment == 'proposed':
            self.model = DAnet(self.args).to(self.args.device, dtype=torch.double)
            # self.model = DAnet_rubust_test(self.args).to(self.args.device, dtype=torch.double)
            # self._load_model()

        # 优化器--------------------------------------------------------
        self.params = ([p for p in self.model.parameters()])
        self.optimizer = self._create_optimizer(self.params)

    def build_data(self):
        # 读取数据
        self.data_sampler = DataBatchSamplerValue(self.args)
        self.data_sampler.get_data_from_pkl(experiment_name=self.args.experiment_name, data_id=self.args.data_id)

    def run_training_epochs(self):
        self.model.train()

        train_bar = tqdm(range(self.args.epochs))
        for i, epoch in enumerate(train_bar):

            # 不抽样
            if self.args.ablation_experiment == 'without_sampler':
                train_dl = self.data_sampler.get_all_train_data()

                loss = torch.Tensor([0]).to(self.args.device, dtype=torch.double)

                for j, (x, y) in enumerate(train_dl):
                    y_pred = self.model(x)

                    lo_c = lf.binary_ce_loss(y_pred.view(-1), y)
                    lo_c = torch.mean(lo_c)
                    lo_c.backward()

                    loss = loss + lo_c

                    # 训练不稳定，将梯度的范数最多限制为1
                    if self.args.clip_grad:
                        clip_grad_norm_(self.params, 1)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                train_bar.set_description('loss: {:.4f}'.format(loss.item() / self.args.num_tasks))

            # 执行课程学习抽样
            else:
                self.data_sampler.cal_sample_values(self.model)

                # if epoch % 50 == 0:
                #     self.data_sampler.plt_sample_value(epoch)

                x_dict, y_dict, sample_values_dict = self.data_sampler.get_batch_samples()

                loss = torch.Tensor([0]).to(self.args.device, dtype=torch.double)

                for task_id in range(self.args.num_tasks):
                    # 选择训练集中的支持集和查询集
                    x = x_dict[task_id]
                    y_true = y_dict[task_id]
                    sample_values = sample_values_dict[task_id]

                    y_pred = self.model(x)
                    # 计算损失, 并更新样本价值
                    lo_c = lf.binary_ce_loss(y_pred.view(-1), y_true)
                    # lo_c = lf.binary_ce_loss(y_pred.view(-1), y_true, sample_values)
                    lo_c = torch.mean(lo_c)

                    lo_c.backward()

                    loss = loss + lo_c

                    # 训练不稳定，将梯度的范数最多限制为1
                    if self.args.clip_grad:
                        clip_grad_norm_(self.params, 1)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                train_bar.set_description('loss: {:.4f}'.format(loss.item()/self.args.num_tasks))

        self._save_model()

        # self.data_sampler.cal_sample_values(self.model)
        # self.data_sampler.plt_sample_value(epoch)

    def run_testing_epoch(self):
        self.model.eval()

        # 推理测试集
        y_pred = None
        y_true = None
        with torch.no_grad():
            for j, (x, y) in enumerate(self.data_sampler.test_dl):
                y_hat = self.model(x)
                y_hat = y_hat.view(-1).cpu().detach().numpy()
                y_pred = y_hat if y_pred is None else np.concatenate((y_pred, y_hat), axis=0)

                y = y.cpu().detach().numpy()
                y_true = y if y_true is None else np.concatenate((y_true, y), axis=0)

        # eval_values = utils.eval_accuracy(y_pred, y_true)

        out_label = np.concatenate((y_true.reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1)

        return out_label#eval_values

    def _create_optimizer(self, params):

        # setup optimizer
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.args.learning_rate, momentum=0.9)

            # if self.args.scheduler:
            #     scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=self.args.learning_rate,
            #                                                   step_size_up=50, mode="triangular2")
            # else:
            #     scheduler = None

        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.args.learning_rate)

            # if self.args.scheduler:
            #     scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=self.args.learning_rate,
            #                                                   step_size_up=50, mode="triangular2", cycle_momentum=False)
            # else:
            #     scheduler = None

        else:
            raise Exception('Not supported optimizer: {0}'.format(self.args.optimizer))

        return optimizer#, scheduler

    def _save_model(self):
        print('正在保存模型。。。')
        torch.save({'model': self.model.state_dict()},
                   os.path.join(self.args.current_path, 'checkpoint', '{}_model_dict.pth'.format(self.args.data_name)))

    def _load_model(self):
        pth = os.path.join(self.args.current_path, 'checkpoint', '{}_model_dict.pth'.format(self.args.data_name))

        if os.path.exists(pth):
            print('正在加载模型。。。')

            state_dict = torch.load(pth)
            self.model.load_state_dict(state_dict['model'])


