import numpy as np
import pandas as pd
import os
from collections import defaultdict
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt


def get_label_frequency(ontology):
    col_sums = ontology.sum(0)
    index_1_10 = np.where((col_sums >= 1) & (col_sums <= 10))[0]
    index_11_30 = np.where((col_sums>=10) & (col_sums<=30))[0]
    index_31_75 = np.where((col_sums>=31) & (col_sums<=75))[0]
    index_larger_75 = np.where(col_sums >= 101)[0]
    return  index_1_10, index_11_30, index_31_75, index_larger_75

def calculate_fmax(preds, labels):
    preds = np.round(preds, 2)
    labels = labels.astype(np.int32)
    f_max = 0
    p_max = 0
    r_max = 0
    sp_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        tp = np.sum(predictions * labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp
        sn = tp / (1.0 * np.sum(labels))
        sp = np.sum((predictions ^ 1) * (labels ^ 1))
        sp /= 1.0 * np.sum(labels ^ 1)
        fpr = 1 - sp
        precision = tp / (1.0 * (tp + fp))
        recall = tp / (1.0 * (tp + fn))
        f = 2 * precision * recall / (precision + recall)
        if f_max < f:
            f_max = f
            p_max = precision
            r_max = recall
            sp_max = sp
            t_max = threshold
    return f_max,t_max,p_max,r_max

def calculate_f1_score(preds, labels):
    preds = np.round(preds, 2)
    labels = labels.astype(np.int32)
    threshold = 0.5
    predictions = (preds > threshold).astype(np.int32)
    p0 = (preds < threshold).astype(np.int32)
    tp = np.sum(predictions * labels)
    fp = np.sum(predictions) - tp
    fn = np.sum(labels) - tp
    tn = np.sum(p0) - fn
    sn = tp / (1.0 * np.sum(labels))
    sp = np.sum((predictions ^ 1) * (labels ^ 1))
    sp /= 1.0 * np.sum(labels ^ 1)
    fpr = 1 - sp
    precision = tp / (1.0 * (tp + fp))
    recall = tp / (1.0 * (tp + fn))
    f = 2 * precision * recall / (precision + recall)
    acc = (tp + tn) / (tp + fp + tn + fn)
    return f,acc,precision,recall

def evaluate_performance(y_test, y_score):
    """Evaluate performance"""
    n_classes = y_test.shape[1]
    perf = dict()

    perf["M-aupr"] = 0.0
    perf["M-auc"] = 0.0
    n = 0
    aupr_list = []
    auc_list = []
    num_pos_list = []
    for i in range(n_classes):
        num_pos = sum(y_test[:, i])
        num_pos = num_pos.astype(float)
        if num_pos > 0:
            ap = average_precision_score(y_test[:, i], y_score[:, i])
            auc = roc_auc_score(y_test[:, i], y_score[:, i])
            n += 1
            perf["M-aupr"] += ap
            perf["M-auc"] += auc
            aupr_list.append(ap)
            auc_list.append(auc)
            num_pos_list.append(num_pos)
    perf["M-aupr"] /= n
    perf['aupr_list'] = aupr_list
    perf['num_pos_list'] = num_pos_list
    perf["M-auc"] /= n
    perf['auc_list'] = auc_list

    # Compute micro-averaged AUPR
    perf['m-aupr'] = average_precision_score(y_test.ravel(), y_score.ravel())
    perf['m-auc'] = roc_auc_score(y_test.ravel(), y_score.ravel())

    perf['F-max'],perf['t_max'],perf['p_max'],perf['r_max'] = calculate_fmax(y_score, y_test)
    perf['F1-score'], perf['accuracy'], perf['precision'], perf['recall'] = calculate_f1_score(y_score, y_test)


    return perf

def get_results(ontology, Y_test, y_score):
    perf = defaultdict(dict) 
    index_1_10,index_11_30, index_31_75, index_101 = get_label_frequency(ontology)

    perf['1-10'] = evaluate_performance(Y_test[:,index_1_10], y_score[:,index_1_10])
    perf['11-30'] = evaluate_performance(Y_test[:,index_11_30], y_score[:,index_11_30])
    perf['31-75'] = evaluate_performance(Y_test[:,index_31_75], y_score[:,index_31_75])
    # perf['101-'] = evaluate_performance(Y_test[:,index_101], y_score[:,index_101])
    perf['all'] = evaluate_performance(Y_test, y_score)

    # plot_PRCurve(Y_test, y_score)
    return perf

def plot_roc(Y_label, y_pred,str,path):
    fpr, tpr, threshold = roc_curve(Y_label, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, color='red',lw=3.5, label='AUC = %0.4f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve ('+str+')')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(path,str+".jpg"))
    plt.show()

    return roc_auc
