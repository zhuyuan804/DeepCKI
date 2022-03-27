# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
import pandas as pd

#DeepDSI
plt.figure(figsize=(6, 6))
lw = 3.5
# def plot_roc(label,score,model,str,):
#     fpr, tpr, threshold = roc_curve(label,score)
#     auc_value = auc(fpr, tpr)
#     plt.figure(figsize=(4.5, 4.5))
#     plt.plot(fpr, tpr, color='red', lw=lw, label= model+' \n(AUC = %0.2f)' % auc_value)  ###假正率为横坐标，真正率为纵坐标做曲线

def plot_roc(model,color):
    pred_score = pd.read_pickle(model+'/Y_pred_str.pkl')
    label = pd.read_pickle(model+'/Y_label_str.pkl')
    CV_fpr, CV_tpr, threshold = roc_curve(label, pred_score)
    auc_value = auc(CV_fpr, CV_tpr)
    plt.plot(CV_fpr, CV_tpr, color=color, lw=lw, label= model+' (AUC = %0.4f)' % auc_value)  ###假正率为横坐标，真正率为纵坐标做曲线

plot_roc('VAE','blue')
plot_roc('GAE','navy')
plot_roc('VGAE','red')


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve (5-fold cross-validation)')
plt.legend(loc="lower right")
plt.savefig('picture/compare 5-CV.png')
plt.savefig('picture/compare 5-CV.pdf')
plt.show()
# plt.plot(CV_fpr_VAE, CV_tpr_VAE, color='blue', lw=lw, label='VAE AUC = %0.2f' % auc_VAE)  ###假正率为横坐标，真正率为纵坐标做曲线
