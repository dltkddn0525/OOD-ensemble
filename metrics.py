from __future__ import print_function
from sklearn.metrics import roc_auc_score
import sklearn
import numpy as np

def get_auroc(cifar,other):
    scores = np.concatenate([cifar,other])
    labels = np.zeros(len(cifar)+len(other))
    labels[:len(cifar)]=1

    return roc_auc_score(labels,scores)


def tpr95(X1, Y1,diff):
    # calculate the falsepositive error when tpr is 95%
    total = 0.0
    fpr = 0.0
    for delta in diff:
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 0.9549 and tpr >= 0.9450:
            fpr += error2
            total += 1

    fprBase = fpr / total

    return fprBase


def auroc(X1, Y1,diff):
    # calculate the AUROC
    aurocBase = 0.0
    fprTemp = 1.0
    for delta in diff:
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        aurocBase += (-fpr + fprTemp) * tpr
        fprTemp = fpr
    aurocBase += fpr * tpr

    return aurocBase


def auprIn(X1, Y1,diff):
    # calculate the AUPR
    precisionVec = []
    recallVec = []
    auprBase = 0.0
    recallTemp = 1.0
    for delta in diff:
        tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        precisionVec.append(precision)
        recallVec.append(recall)
        auprBase += (recallTemp - recall) * precision
        recallTemp = recall
    auprBase += recall * precision

    return auprBase


def auprOut(X1, Y1,diff):
    # calculate the AUPR
    auprBase = 0.0
    recallTemp = 1.0
    for delta in diff[::-1]:
        fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        auprBase += (recallTemp - recall) * precision
        recallTemp = recall
    auprBase += recall * precision

    return auprBase


def detection(X1, Y1,diff):
    # calculate the minimum detection error
    errorBase = 1.0
    threshold = 0.0
    for delta in diff:
        tpr_ = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        #errorBase = np.minimum(errorBase, (tpr_ + error2) / 2.0)
        if errorBase > ((tpr_ + error2) / 2.0):
            errorBase = ((tpr_ + error2) / 2.0)
            threshold = delta
    return errorBase, threshold

def confusion_matrix(y_pred,target,label):
    result = np.zeros((max(target)+1,2))
    for i in range(max(target)+1):
        y_pred_temp = y_pred[np.nonzero(target==i)]
        y_true_temp = np.ones_like(y_pred_temp) * label
        confusion = sklearn.metrics.confusion_matrix(y_true_temp,y_pred_temp)
        result[i]=confusion[label]

    return result