import random
import numpy as np
import csv
import torch
import torch.nn.functional as F
from copy import copy
from torch import nn, optim, tensor
# from pytorch_metric_learning import distances, losses, miners, reducers, testers


def binary_ce_loss(y_pred, y_true, weight=None):
    if weight is not None:
        loss = - weight * (y_true * torch.log(y_pred + 1e-8) + (1 - y_true) * torch.log(1 - y_pred + 1e-8))
    else:
        loss = - y_true * torch.log(y_pred + 1e-8) - (1 - y_true) * torch.log(1 - y_pred + 1e-8)

    return loss


class penalized_loss(nn.Module):
    def __init__(self, args):
        super(penalized_loss, self).__init__()
        self.args = args

    def features_loss(self, per_feature_outputs: torch.Tensor) -> torch.Tensor:
        """Penalizes the L2 norm of the prediction of each feature net."""
        per_feature_norm = [torch.mean(torch.square(outputs)) for outputs in per_feature_outputs]
        return sum(per_feature_norm) / len(per_feature_norm)

    def weight_decay(self, model: nn.Module) -> torch.Tensor:
        """Penalizes the L2 norm of weights in each feature net."""
        num_networks = len(model.feature_nns) + 1
        l2_losses = [(x**2).sum() for x in model.parameters()]
        return sum(l2_losses) / num_networks

    def forward(self, logits, targets, fnn_out, model):
        """Computes penalized loss with L2 regularization and output penalty.

        Args:
          args: Global config.
          model: Neural network model.
          inputs: Input values to be fed into the model for computing predictions.
          targets: Target values containing either real values or binary labels.

        Returns:
          The penalized loss.
        """

        loss = F.binary_cross_entropy_with_logits(logits.view(-1), targets.view(-1))

        # reg_loss = 0.0
        # if self.args.output_regularization > 0:
        #     reg_loss += self.args.output_regularization * self.features_loss(fnn_out)
        #
        # if self.args.l2_regularization > 0:
        #     reg_loss += self.args.l2_regularization * self.weight_decay(model)
        #
        # loss = loss + reg_loss

        return loss


def macro_double_soft_f1(y_pred, y_true):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.

    Args:
        y_true (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_pred (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y_true = y_true.type(torch.float32)
    y_pred = y_pred.type(torch.float32)

    tp = torch.sum(y_pred * y_true, dim=0)
    fp = torch.sum(y_pred * (1 - y_true), dim=0)
    tn = torch.sum((1 - y_pred) * (1 - y_true), dim=0)
    fn = torch.sum((1 - y_pred) * y_true, dim=0)

    soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)

    cost_class1 = 1 - soft_f1_class1  # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0  # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = (cost_class1 + cost_class0)  # take into account both class 1 and class 0
    # cost = cost_class1
    macro_cost = torch.sum(cost)  # average on all labels
    return macro_cost


def f1_loss(y_pred, y_true, sample_type):
    y_true = y_true.type(torch.float32)
    y_pred = y_pred.type(torch.float32)

    tp = y_pred * y_true
    fp = y_pred * (1 - y_true)
    tn = (1 - y_pred) * (1 - y_true)
    fn = (1 - y_pred) * y_true

    soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)
    if sample_type == 'p':
        return soft_f1_class1
    if sample_type == 'n':
        return soft_f1_class0

def ce_f1(y_pred, y_true):
    # 少数类用f1
    y_true = y_true.type(torch.float32)
    y_pred = y_pred.type(torch.float32)

    tp = torch.sum(y_pred * y_true, dim=0)
    fp = torch.sum(y_pred * (1 - y_true), dim=0)
    tn = torch.sum((1 - y_pred) * (1 - y_true), dim=0)
    fn = torch.sum((1 - y_pred) * y_true, dim=0)

    soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)

    cost_class1 = 1 - soft_f1_class1  # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0  # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0)  # take into account both class 1 and class 0
    # cost = cost_class1
    macro_cost = torch.sum(cost)  # average on all labels

    # 多数类交叉熵
    ce = torch.mean(
        - torch.log(y_pred) * y_true
        - torch.log(1 - y_pred) * (1 - y_true)
    )

    # loss = ce + macro_cost
    loss = ce
    # loss = macro_cost
    return loss























class MagnetLoss(nn.Module):
    def __init__(self, args):
        super(MagnetLoss, self).__init__()
        self.args = args
        self.alpha = self.args.alpha
        self.M = self.args.m
        self.D = self.args.d
        self.p2dist = nn.PairwiseDistance(p=2)  # p=2就是计算欧氏距离，p=1就是曼哈顿距离

        self.epsilon = self.args.epsilon

    def forward(self, x_embedd, y_true):
        clusters_centers, clusters_sigma = self._compute_center_sigma(x_embedd, y_true)
        clusters_sigma = torch.sum(clusters_sigma)

        num_samples = x_embedd.shape[0]

        loss = torch.zeros(num_samples).to(device=self.args.device, dtype=torch.double)

        # 公式5
        for s in range(num_samples):
            dis = -(1 / (2 * clusters_sigma.pow(2))) * self.p2dist(x_embedd[s], clusters_centers).pow(2)

            inds1 = torch.where((self.cluster_labels[:, 2] == y_true[s, 2]))[0]  # 同一簇
            inds2 = torch.where(self.cluster_labels[:, 0] != y_true[s, 0])[0]  # 不是同一类
            # inds2 = torch.where(clusters_labels[:, 2] != y_true[s, 2])[0]  # 不是同一簇

            numer = torch.exp(dis[inds1] + self.args.alpha)  # 分子项，表示样本到所在簇中心的距离

            # denum = torch.exp(dis[inds2]).sum()  # 分母项， 表示样本到其他类所有簇中心的距离和
            # denum = torch.exp(dis[inds2]).mean()  # 分母项， 表示样本到其他类所有簇中心的距离和
            # denum = torch.exp(dis[inds2]).max()  # 分母项， 表示样本到其他类所有簇中心的距离最大值
            denum = torch.exp(dis[inds2]).min()  # 分母项， 表示样本到其他类所有簇中心的距离最大值

            lo = -torch.log(numer / (denum + self.args.epsilon) + self.args.epsilon)

            loss[s] = lo

        loss = torch.clamp(loss, min=0.0)
        loss = loss.mean(0)

        return loss

    def _compute_center_sigma(self, x, y):
        self.cluster_labels = torch.unique(y, dim=0, sorted=True)

        cluster_centers = torch.zeros((self.args.n_way, self.args.dim_out)).to(self.args.device, dtype=torch.double)
        sigmas = torch.zeros(self.args.n_way).to(self.args.device, dtype=torch.double)

        for cluster_id, cluster_label in enumerate(self.cluster_labels):
            idx = torch.where(y[:, 2] == cluster_label[2])[0]
            select_x = x[idx, :]

            mean_vector = torch.mean(select_x, dim=0)
            cluster_centers[cluster_id, :] = mean_vector
            sigmas[cluster_id] = self.p2dist(select_x, mean_vector).pow(2).sum(0)

        return cluster_centers, sigmas


class GaussianLoss(nn.Module):
    def __init__(self, args):
        super(GaussianLoss, self).__init__()
        self.args = args
        self.args = args
        self.alpha = self.args.alpha
        self.M = self.args.m
        self.D = self.args.d
        self.p2dist = nn.PairwiseDistance(p=2)  # p=2就是计算欧氏距离，p=1就是曼哈顿距离

        self.epsilon = self.args.epsilon

    def forward(self, x, y):
        centers, sigmas = self._compute_center_sigma(x, y)

        num_overlab = int(self.args.d * self.args.overlap_ratio)

        loss = torch.Tensor([0]).to(self.args.device, dtype=torch.double)

        for cluster_id_1, cluster_label_1 in enumerate(self.cluster_labels):
            idx1 = torch.where((cluster_label_1[2] == y[:, 2]))[0]  # 同一簇

            prob1_a = 1 / torch.sqrt(2 * torch.pi * sigmas[cluster_id_1] ** 2)
            prob1_b = torch.exp(-0.5 * ((x[idx1] - centers[cluster_id_1]) / sigmas[cluster_id_1]) ** 2)
            prob1 = torch.sum(torch.log(prob1_a * prob1_b), dim=1)

            prob2_sum = torch.Tensor([0]).to(self.args.device, dtype=torch.double)
            for cluster_id_2, cluster_label_2 in enumerate(self.cluster_labels):
                if cluster_id_1 == cluster_id_2:
                    pass
                else:
                    # idx2 = torch.where((cluster_label_2[2] == y[:, 2]))[0]  # 同一簇

                    prob2_a = 1 / torch.sqrt(2 * torch.pi * sigmas[cluster_id_2] ** 2)
                    prob2_b = torch.exp(-0.5 * ((x[idx1] - centers[cluster_id_2]) / sigmas[cluster_id_2]) ** 2)
                    prob2 = torch.sum(torch.log(prob2_a * prob2_b), dim=1)

                    prob2_sum = prob2_sum + prob2

            lo = torch.exp(-1 * (prob1 - prob2_sum))  # / idx1.shape[0]
            loss = loss + lo

        return loss

    def _compute_center_sigma(self, x, y):
        self.cluster_labels = torch.unique(y, dim=0, sorted=True)

        cluster_centers = torch.zeros((self.args.n_way, self.args.dim_embed)).to(self.args.device, dtype=torch.double)
        # sigmas = torch.zeros(self.args.n_way).to(self.args.device, dtype=torch.double)
        sigmas = torch.zeros((self.args.n_way, self.args.dim_embed)).to(self.args.device, dtype=torch.double)

        for cluster_id, cluster_label in enumerate(self.cluster_labels):
            idx = torch.where(y[:, 2] == cluster_label[2])[0]
            select_x = x[idx, :]

            # mean_vector = torch.mean(select_x, dim=0)
            cluster_centers[cluster_id, :] = torch.mean(select_x, dim=0)
            sigmas[cluster_id] = torch.std(select_x, dim=0)
            # sigmas[cluster_id] = self.p2dist(select_x, mean_vector).pow(2).sum(0)

        return cluster_centers, sigmas


class ClassifyLoss(nn.Module):
    def __init__(self, args):
        super(ClassifyLoss, self).__init__()
        self.args = args
        self.alpha = self.args.alpha
        self.M = self.args.m
        self.D = self.args.d
        self.p2dist = nn.PairwiseDistance(p=2)  # p=2就是计算欧氏距离，p=1就是曼哈顿距离

        self.loss_bce = nn.BCELoss()
        self.loss_cel = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):

        if y_pred.shape[1] == 1:
            lo = self.loss_bce(y_pred, y_true[0].long())
        else:
            lo = self.loss_cel(y_pred, y_true)

        loss = lo

        return loss.mean()


class MinorityLoss(nn.Module):
    def __init__(self, args):
        super(MinorityLoss, self).__init__()
        self.args = args

    def forward(self, y_pred, y_true, beta=1):
        tp = sum(torch.logical_and(y_pred >= 0.5, y_true == 1))

        tp1 = y_true * y_pred  # 大
        fn1 = y_true * y_pred # 小
        fp1 = (1 - y_true) * y_pred # 大






        fp = sum(torch.logical_and(y_pred >= 0.5, y_true == 0))
        tn = sum(torch.logical_and(y_pred < 0.5, y_true == 0))
        fn = sum(torch.logical_and(y_pred < 0.5, y_true == 1))

        error_rate = fp / (fp + tp + 1E-10)
        leak_rate = fn / (fn + tp + 1E-10)

        loss = (1 + beta ** 2) * ((error_rate * leak_rate) / (beta ** 2 * leak_rate + error_rate))

        return loss


# class TripletLoss(nn.Module):
#     def __init__(self, args):
#         super(TripletLoss, self).__init__()
#         self.args = args
#         self.miner_func = miners.BatchEasyHardMiner()
#         self.loss_metric_func = losses.TripletMarginLoss(margin=self.args.alpha, swap=True)

#     def forward(self, x, y):
#         y = y[:, 0]
#         hard_tuples = self.miner_func(x, y)
#         loss = self.loss_metric_func(x, y, hard_tuples)

#         return loss


class torchTripLoss(nn.Module):
    def __init__(self, args):
        super(torchTripLoss, self).__init__()
        self.args = args

        self.p2dist = nn.PairwiseDistance(p=2)  # p=2就是计算欧氏距离，p=1就是曼哈顿距离
        self.lossfn = nn.TripletMarginLoss(margin=1.0, p=2)

        self.num_overlab = int(self.args.d * self.args.overlap_ratio)

    def forward(self, x, y):
        self.class_types, count = torch.unique(y[:, 2], return_counts=True)
        self.centers = self.get_centers(x, y)

        loss = torch.Tensor([0]).to(self.args.device, dtype=torch.double)

        for class_id, class_type in enumerate(self.class_types):
            center = self.centers[class_id, :]  # 锚点中心
            # a = torch.expand_copy(center.reshape(1, -1), self.num_overlab, center.shape[1])
            a = center.reshape(1, -1).repeat(self.num_overlab, 1)
            p, n = self.get_hard_sample(class_type, center, x, y)
            lo = self.lossfn(a, p, n)
            loss = loss + lo

        return loss / self.class_types.shape[0]

    def get_centers(self, x, y):

        centers = torch.zeros((len(self.class_types), self.args.dim_out)).to(self.args.device, dtype=torch.double)

        for class_id, class_type in enumerate(self.class_types):
            idx = torch.where(y[:, 2] == class_type)[0]

            select_x = x[idx, :]
            centers[class_id, :] = torch.mean(select_x, dim=0)

        return centers

    def get_hard_sample(self, class_type, center, x, y):
        # 同簇样本
        idx = torch.where(y[:, 2] == class_type)[0]  # 同簇样本
        x_data = x[idx, :]
        pos_dist = self.p2dist(center, x_data)
        pos_id = torch.argsort(pos_dist, dim=0, descending=True)[:self.num_overlab]
        # pos_dist = torch.sort(pos_dist, dim=0, descending=True)[0][:self.num_overlab]  # 距离最远的样本
        pos_x = x[idx[pos_id], :]

        # 不同类样本
        idx = torch.where(y[:, 0] != class_type)[0]  # 同簇样本
        x_data = x[idx, :]
        neg_dist = self.p2dist(center, x_data)
        neg_id = torch.argsort(neg_dist, dim=0, descending=False)[:self.num_overlab]
        # neg_dist = torch.sort(neg_dist, dim=0, descending=True)[0][:self.num_overlab]  # 距离最进的样本
        neg_x = x[idx[neg_id], :]

        return pos_x, neg_x


# class MarginalTripLoss(nn.Module):
#     def __init__(self, args, cluster_labels):
#         super(MarginalTripLoss, self).__init__()
#         self.args = args
#         self.cluster_types = cluster_labels
#         self.p2dist = nn.PairwiseDistance(p=2)  # p=2就是计算欧氏距离，p=1就是曼哈顿距离
#         self.lossfn = nn.TripletMarginLoss(margin=1.0, p=2)

#         self.num_overlab = int(self.args.d * self.args.overlap_ratio)

#         self.miner_func = miners.BatchEasyHardMiner()
#         self.loss_metric_func = losses.TripletMarginLoss(margin=self.args.alpha, swap=True)

#     def forward(self, x, y):
#         # self.cluster_types, count = torch.unique(y, return_counts=True)
#         self.centers = self.get_centers(x, y)

#         loss = torch.Tensor([0]).to(self.args.device, dtype=torch.double)

#         for cluster_id, cluster_type in enumerate(self.cluster_types):
#             center = self.centers[cluster_id, :]  # 锚点中心
#             # a = torch.expand_copy(center.reshape(1, -1), self.num_overlab, center.shape[1])
#             # a = center.reshape(1, -1).repeat(self.num_overlab, 1)

#             hard_x, hard_y = self.get_hard_sample(cluster_type, center, x, y)

#             hard_tuples = self.miner_func(hard_x, hard_y)
#             lo = self.loss_metric_func(hard_x, hard_y, hard_tuples)

#             loss = loss + lo

#         return loss / self.cluster_types.shape[0]

#     def get_centers(self, x, y):

#         centers = torch.zeros((len(self.cluster_types), self.args.dim_out)).to(self.args.device, dtype=torch.double)

#         for class_id, class_type in enumerate(self.cluster_types):
#             idx = torch.where(y[:, 2] == class_type[2])[0]

#             select_x = x[idx, :]
#             centers[class_id, :] = torch.mean(select_x, dim=0)

#         return centers

#     def get_hard_sample(self, cluster_type, center, x, y):
#         # 同簇样本
#         idx = torch.where(y[:, 2] == cluster_type[2])[0]  # 同簇样本
#         x_data = x[idx, :]
#         pos_dist = self.p2dist(center, x_data)
#         pos_id = torch.argsort(pos_dist, dim=0, descending=True)[:self.num_overlab]
#         # pos_dist = torch.sort(pos_dist, dim=0, descending=True)[0][:self.num_overlab]  # 距离最远的样本
#         pos_id = copy(idx[pos_id])

#         # 不同类样本
#         idx = torch.where(y[:, 0] != cluster_type[0])[0]  # 同簇样本
#         x_data = x[idx, :]
#         neg_dist = self.p2dist(center, x_data)
#         neg_id = torch.argsort(neg_dist, dim=0, descending=False)[:self.num_overlab]
#         # neg_dist = torch.sort(neg_dist, dim=0, descending=True)[0][:self.num_overlab]  # 距离最进的样本
#         neg_id = copy(idx[neg_id])

#         idx = torch.concat([pos_id, neg_id], dim=0)
#         hard_x = x[idx, :]
#         hard_y = y[idx, 0]

#         return hard_x, hard_y


# class ClusterClassTripLoss(nn.Module):
#     def __init__(self, args, cluster_labels):
#         super(ClusterClassTripLoss, self).__init__()
#         self.args = args
#         self.cluster_types = cluster_labels

#         self.miner_func = miners.BatchEasyHardMiner()
#         self.loss_metric_func = losses.TripletMarginLoss(margin=self.args.alpha, swap=True)

#     def forward(self, x, y):
#         loss = torch.Tensor([0]).to(self.args.device, dtype=torch.double)

#         for cluster_id, cluster_type in enumerate(self.cluster_types):
#             idx1 = torch.where(y[:, 2] == cluster_type[2])[0]  # 同簇样本
#             idx2 = torch.where(y[:, 0] != cluster_type[0])[0]  # 不同类样本

#             idx = torch.concat([idx1, idx2], dim=0)
#             trip_x = x[idx, :]
#             trip_y = y[idx, 0]

#             hard_tuples = self.miner_func(trip_x, trip_y)
#             lo = self.loss_metric_func(trip_x, trip_y, hard_tuples)

#             loss = loss + lo

#         return loss / self.cluster_types.shape[0]


class OverlapTripletLoss1(nn.Module):
    def __init__(self, args):
        super(OverlapTripletLoss1, self).__init__()
        self.args = args
        self.p2dist = nn.PairwiseDistance(p=2)  # p=2就是计算欧氏距离，p=1就是曼哈顿距离

    def forward(self, x, y):
        # y = y[:, 2]

        # self.class_types = torch.unique(y, dim=0, sorted=True)
        self.class_types, count = torch.unique(y, return_counts=True)
        # num_overlab = int(min(count) * self.args.overlap_ratio)
        num_overlab = int(self.args.d * self.args.overlap_ratio)

        self.centers = self.get_centers(x, y)

        loss = torch.Tensor([0]).to(self.args.device, dtype=torch.double)

        # 遍历锚点簇
        for class_id1, class_type1 in enumerate(self.class_types):
            center = self.centers[class_id1, :]  # 锚点中心

            idx = torch.where(y == class_type1)[0]  # 同簇样本
            x_data = x[idx, :]

            pos_dist = self.p2dist(center, x_data)
            pos_dist = torch.sort(pos_dist, dim=0, descending=True)[0][:num_overlab]  # 距离最远的样本
            # pos_dist = torch.sort(pos_dist, dim=0, descending=True)[0][0]  # 距离最远的样本
            pos_mean = pos_dist.mean(0)

            idx = torch.where(y != class_type1)[0]  # 同簇样本
            x_data = x[idx, :]

            neg_dist = self.p2dist(center, x_data)
            neg_dist = torch.sort(neg_dist, dim=0, descending=False)[0][:num_overlab]  # 距离最近的样本
            # neg_dist = torch.sort(neg_dist, dim=0, descending=False)[0][0]  # 距离最近的样本
            neg_mean = neg_dist.mean(0)
            neg_std = neg_dist.std(0)

            lo = torch.clamp(self.args.alpha + pos_mean - neg_mean, min=0.0)
            # lo = torch.clamp((self.args.alpha + pos_mean) / neg_mean, min=0.0)
            # lo = torch.clamp(
            #     (self.args.alpha + pos_mean + 1.96 * pos_std) - (neg_mean - 1.96 * neg_std),
            #     min=0.0)

            loss = loss + lo

        # loss = loss / ((len(self.class_types) + 1) * (len(self.class_types) / 2))
        loss = loss / (len(self.class_types) * num_overlab)
        return loss

    def get_centers(self, x, y):

        centers = torch.zeros((len(self.class_types), self.args.dim_embed)).to(self.args.device, dtype=torch.double)

        for class_id, class_type in enumerate(self.class_types):
            idx = torch.where(y == class_type)[0]

            select_x = x[idx, :]
            centers[class_id, :] = torch.mean(select_x, dim=0)

        return centers


class OverlapTripletLoss(nn.Module):
    def __init__(self, args):
        super(OverlapTripletLoss, self).__init__()
        self.args = args
        self.p2dist = nn.PairwiseDistance(p=2)  # p=2就是计算欧氏距离，p=1就是曼哈顿距离

    def forward(self, x, y):
        # y = y[:, 2]

        # self.class_types = torch.unique(y, dim=0, sorted=True)
        self.class_types, count = torch.unique(y, return_counts=True)
        # num_overlab = int(min(count) * self.args.overlap_ratio)
        num_overlab = int(self.args.d * self.args.overlap_ratio)

        self.centers = self.get_centers(x, y)

        loss = torch.Tensor([0]).to(self.args.device, dtype=torch.double)

        # 遍历锚点簇
        for class_id1, class_type1 in enumerate(self.class_types):
            center = self.centers[class_id1, :]  # 锚点中心

            idx = torch.where(y == class_type1)[0]  # 同簇样本
            x_data = x[idx, :]

            pos_dist = self.p2dist(center, x_data)
            pos_dist = torch.sort(pos_dist, dim=0, descending=True)[0][:num_overlab]  # 距离最远的样本
            # pos_dist = torch.sort(pos_dist, dim=0, descending=True)[0][0]  # 距离最远的样本
            pos_mean = pos_dist.mean(0)
            pos_std = pos_dist.std(0)

            # 遍历非锚点簇
            for class_id2, class_type2 in enumerate(self.class_types):
                if class_type2 == class_type1:
                    pass

                else:
                    idx = torch.where(y == class_type2)[0]  # 同簇样本
                    x_data = x[idx, :]

                    neg_dist = self.p2dist(center, x_data)
                    neg_dist = torch.sort(neg_dist, dim=0, descending=False)[0][:num_overlab]  # 距离最近的样本
                    # neg_dist = torch.sort(neg_dist, dim=0, descending=False)[0][0]  # 距离最近的样本
                    neg_mean = neg_dist.mean(0)
                    neg_std = neg_dist.std(0)

                    lo = torch.clamp(self.args.alpha + pos_mean - neg_mean, min=0.0)
                    # lo = torch.clamp((self.args.alpha + pos_mean) / neg_mean, min=0.0)
                    # lo = torch.clamp(
                    #     (self.args.alpha + pos_mean + 1.96 * pos_std) - (neg_mean - 1.96 * neg_std),
                    #     min=0.0)

                    loss = loss + lo

        loss = loss / ((len(self.class_types) + 1) * (len(self.class_types) / 2))

        return loss

    def get_centers(self, x, y):

        centers = torch.zeros((len(self.class_types), self.args.dim_out)).to(self.args.device, dtype=torch.double)

        for class_id, class_type in enumerate(self.class_types):
            idx = torch.where(y == class_type)[0]

            select_x = x[idx, :]
            centers[class_id, :] = torch.mean(select_x, dim=0)

        return centers


class OverlapMeanStdTripletLoss(nn.Module):
    def __init__(self, args):
        super(OverlapMeanStdTripletLoss, self).__init__()
        self.args = args
        self.p2dist = nn.PairwiseDistance(p=2)  # p=2就是计算欧氏距离，p=1就是曼哈顿距离

    def forward(self, x, y):
        y = y[:, 2]

        self.class_types = torch.unique(y, dim=0, sorted=True)

        self.centers = self.get_centers(x, y)

        loss = torch.Tensor([0]).to(self.args.device, dtype=torch.double)

        # 遍历锚点簇
        for class_id1, class_type1 in enumerate(self.class_types):
            center = self.centers[class_id1, :]  # 锚点中心

            pos_idx = torch.where(y == class_type1)[0]  # 同簇样本
            pos_x = x[pos_idx, :]

            pos_dist = self.p2dist(center, pos_x)
            pos_dist = torch.sort(pos_dist, dim=0, descending=True)[0][
                       :int(self.args.d * self.args.overlap_ratio)]  # 距离最远的样本

            pos_mean = pos_dist.mean(0)
            pos_var = pos_dist.var(0)

            # 遍历非锚点簇
            for class_id2, class_type2 in enumerate(self.class_types):
                if class_type2 == class_type1:
                    pass

                else:
                    neg_idx = torch.where(y == class_type2)[0]  # 同簇样本
                    neg_x = x[neg_idx, :]

                    neg_dist = self.p2dist(center, neg_x)
                    neg_dist = torch.sort(neg_dist, dim=0, descending=False)[0][
                               :int(self.args.d * self.args.overlap_ratio)]  # 距离最近的样本

                    neg_mean = neg_dist.mean(0)
                    neg_var = neg_dist.var(0)

                    g_loss2 = torch.clamp(self.args.alpha +
                                          (pos_mean + 1.96 * (pos_var ** 0.5)) -
                                          (neg_mean - 1.96 * (neg_var ** 0.5)), min=0.0)

                    loss = loss + g_loss2
                    # loss = loss + torch.clamp(self.args.alpha + pos_mean - neg_mean, min=0.0)

        return loss

    def get_centers(self, x, y):

        centers = torch.zeros((len(self.class_types), self.args.dim_embed)).to(self.args.device, dtype=torch.double)

        for class_id, class_type in enumerate(self.class_types):
            idx = torch.where(y == class_type)[0]

            select_x = x[idx, :]
            centers[class_id, :] = torch.mean(select_x, dim=0)

        return centers


class OverlapTripletLoss05(nn.Module):
    def __init__(self, args):
        super(OverlapTripletLoss05, self).__init__()
        self.args = args
        self.p2dist = nn.PairwiseDistance(p=2)  # p=2就是计算欧氏距离，p=1就是曼哈顿距离

    def forward(self, x, y, mu):
        y = y[:, 2]

        self.class_types = torch.unique(y, dim=0, sorted=True)

        self.centers = mu

        loss = torch.Tensor([0]).to(self.args.device, dtype=torch.double)

        # 遍历锚点簇
        for class_id1, class_type1 in enumerate(self.class_types):
            center = self.centers[class_id1, :]  # 锚点中心

            idx = torch.where(y == class_type1)[0]  # 同簇样本
            x_data = x[idx, :]

            pos_dist = self.p2dist(center, x_data)
            pos_dist = torch.sort(pos_dist, dim=0, descending=True)[0][
                       :int(self.args.d * self.args.overlap_ratio)]  # 距离最远的样本

            pos_mean = pos_dist.mean(0)
            pos_std = pos_dist.std(0)

            # 遍历非锚点簇
            for class_id2, class_type2 in enumerate(self.class_types):
                if class_type2 == class_type1:
                    pass

                else:
                    idx = torch.where(y == class_type2)[0]  # 同簇样本
                    x_data = x[idx, :]

                    neg_dist = self.p2dist(center, x_data)
                    neg_dist = torch.sort(neg_dist, dim=0, descending=False)[0][
                               :int(self.args.d * self.args.overlap_ratio)]  # 距离最近的样本

                    neg_mean = neg_dist.mean(0)
                    neg_std = neg_dist.std(0)

                    # g_loss1 = pos_var + neg_var
                    # g_loss2 = torch.clamp(2 * pos_mean - neg_mean, min=0.0)

                    # loss = loss + g_loss2
                    loss = loss + torch.clamp(self.args.alpha + pos_mean - neg_mean, min=0.0)

                    # loss = loss + torch.clamp((self.args.alpha + pos_mean + 1.96 * pos_std) ** 2 / (neg_mean - 1.96 * neg_std + 1E-10) ** 2, min=0.0)

        return loss / x.shape[0]

    def get_centers(self, x, y):

        centers = torch.zeros((len(self.class_types), self.args.dim_embed)).to(self.args.device, dtype=torch.double)

        for class_id, class_type in enumerate(self.class_types):
            idx = torch.where(y == class_type)[0]

            select_x = x[idx, :]
            centers[class_id, :] = torch.mean(select_x, dim=0)

        return centers




