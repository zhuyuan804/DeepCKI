import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # 创建解析器
# global parameters,  添加参数
# parser.add_argument('--data_path', type=str, default="../../data/", help="path storing data.")
parser.add_argument('--model', type=str, default="VAE", help="which species to use.")
parser.add_argument('--supervised', type=str, default="nn", help="neural networks or svm")
parser.add_argument('--data_path', type=str, default="../evaluation_index", help="neural networks or svm")
parser.add_argument('--ppi_attributes', type=int, default=1, help="types of attributes used by ppi.")
parser.add_argument('--thr_ppi', type=float, default=0.3, help="threshold for combiend ppi network.")
parser.add_argument('--epochs_ppi', type=int, default=60, help="Number of epochs to train ppi.")
args = parser.parse_args()  # 解析参数


df = pd.DataFrame()

model = ['cell-cytokine','cell-cytokine+','cytokine-cell+','cytokine-cell-','cytokine-cell']


for i in model:
    save_path = os.path.join(i , "results_graph2go_" + args.supervised +"_" + str(args.thr_ppi) + "_" + str(args.epochs_ppi))
    with open(save_path + "_5cytokine_cell.csv", "r") as f:
        file = pd.read_csv(f)
        file = file.set_index(['Unnamed: 0'])
        file = file.reindex(["M-aupr","M-auc","m-aupr","m-auc","precision","recall","F1-score"])
        roc = file['all']
        df = pd.concat([df,roc],axis=1)

df.columns = model

df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)

df2.to_csv(os.path.join("value.csv"))
