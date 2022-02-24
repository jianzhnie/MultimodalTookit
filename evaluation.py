'''
Author: jianzhnie
Date: 2021-11-15 18:31:40
LastEditTime: 2022-02-24 12:10:09
LastEditors: jianzhnie
Description:

'''
import math
from typing import Callable, Dict

import numpy as np
from scipy.special import softmax
from sklearn.metrics import auc, confusion_matrix, f1_score, matthews_corrcoef, mean_absolute_error, mean_squared_error, precision_recall_curve, roc_auc_score
from transformers import EvalPrediction


def build_compute_metrics_fn(
        task_name: str) -> Callable[[EvalPrediction], Dict]:

    def compute_metrics_fn(p: EvalPrediction):
        if task_name == 'classification':
            preds_labels = np.argmax(p.predictions, axis=1)
            if p.predictions.shape[-1] == 2:
                pred_scores = softmax(p.predictions, axis=1)[:, 1]
            else:
                pred_scores = softmax(p.predictions, axis=1)
            return calc_classification_metrics(pred_scores, preds_labels,
                                               p.label_ids)
        elif task_name == 'regression':
            preds = np.squeeze(p.predictions)
            return calc_regression_metrics(preds, p.label_ids)
        else:
            return {}

    return compute_metrics_fn


def calc_classification_metrics(pred_scores, pred_labels, labels):
    if len(np.unique(labels)) == 2:  # binary classification
        roc_auc_pred_score = roc_auc_score(labels, pred_scores)
        precisions, recalls, thresholds = precision_recall_curve(
            labels, pred_scores)
        fscore = (2 * precisions * recalls) / (precisions + recalls)
        fscore[np.isnan(fscore)] = 0
        ix = np.argmax(fscore)
        threshold = thresholds[ix].item()
        pr_auc = auc(recalls, precisions)
        tn, fp, fn, tp = confusion_matrix(
            labels, pred_labels, labels=[0, 1]).ravel()
        result = {
            'roc_auc': roc_auc_pred_score,
            'threshold': threshold,
            'pr_auc': pr_auc,
            'recall': recalls[ix].item(),
            'precision': precisions[ix].item(),
            'f1': fscore[ix].item(),
            'tn': tn.item(),
            'fp': fp.item(),
            'fn': fn.item(),
            'tp': tp.item()
        }
    else:
        acc = (pred_labels == labels).mean()
        f1_micro = f1_score(y_true=labels, y_pred=pred_labels, average='micro')
        f1_macro = f1_score(y_true=labels, y_pred=pred_labels, average='macro')
        f1_weighted = f1_score(
            y_true=labels, y_pred=pred_labels, average='weighted')

        result = {
            'acc': acc,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'mcc': matthews_corrcoef(labels, pred_labels),
        }

    return result


def calc_regression_metrics(preds, labels):
    mse = mean_squared_error(labels, preds)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(labels, preds)
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
    }
