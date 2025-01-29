import torch
import random
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn import metrics

from torch.utils.data import Sampler
from typing import List, Iterable, Callable, Tuple
# from imblearn.metrics import geometric_mean_score
from sklearn.metrics import classification_report


def eval_accuracy(y_pred, y_true, threshold=0.5, epsilon=1e-8):
    # -------------------------------------------------------------------------------
    # 区分度

    fpr, tpr, roc_thresholds = metrics.roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)
    ks = max(abs(fpr - tpr))

    precision, recall, pr_thresholds = metrics.precision_recall_curve(y_true, y_pred)
    aupr = metrics.auc(recall, precision)

    if threshold == 'ROC':
        maxindex = np.argmax(tpr + (1 - fpr))
        threshold = roc_thresholds[maxindex]
        print('threshold:{}'.format(threshold))

    elif threshold == 'KS':
        maxindex = np.argmax(abs(fpr - tpr))
        threshold = roc_thresholds[maxindex]
        print('threshold:{}'.format(threshold))

    elif threshold == 'PR':
        maxindex = np.argmax(2 * (precision * recall) / (precision + recall + epsilon))
        threshold = pr_thresholds[maxindex]
        print('threshold:{}'.format(threshold))

    else:
        print('threshold:{}'.format(threshold))

    tp = sum(np.logical_and(y_pred > threshold, y_true == 1))
    fp = sum(np.logical_and(y_pred > threshold, y_true == 0))

    tn = sum(np.logical_and(y_pred <= threshold, y_true == 0))
    fn = sum(np.logical_and(y_pred <= threshold, y_true == 1))

    acc = (tp + tn) / (tp + tn + fp + fn + epsilon)

    recall_1_recall = tp / (tp + fn + epsilon)
    precision_1 = tp / (tp + fp + epsilon)

    recall_0_specificity = tn / (tn + fp + epsilon)
    precision_0 = tn / (tn + fn + epsilon)

    gmean = (recall_1_recall * recall_0_specificity + epsilon) ** 0.5

    f_measure_1 = (2 * recall_1_recall * precision_1) / (precision_1 + recall_1_recall + epsilon)
    f_measure_0 = (2 * recall_0_specificity * precision_0) / (precision_0 + recall_0_specificity + epsilon)

    # -------------------------------------------------------------------------------
    # 校准度
    brier = metrics.brier_score_loss(y_true, y_pred)

    # return [brier, ks, auc,
    #         aupr, f_measure_1, f_measure_0,
    #         gmean,
    #         acc,
    #         recall_1_recall, precision_1,
    #         recall_0_specificity, precision_0,
    #         ]
    return [
            acc,
            recall_1_recall, precision_1,
            recall_0_specificity, precision_0,
            gmean,
            f_measure_1, f_measure_0,
            auc, ks,
            ]

def pairwise_distances(x: torch.Tensor, y: torch.Tensor, matching_fn: str, EPSILON=1e-8) -> torch.Tensor:
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.
    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        distances = (x.unsqueeze(1).expand(n_x, n_y, -1) - y.unsqueeze(0).expand(n_x, n_y, -1)).pow(2).sum(dim=2)
        return distances

    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities

    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)
        return -(expanded_x * expanded_y).sum(dim=2)

    else:
        raise (ValueError('Unsupported similarity function'))
