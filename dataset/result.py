import numpy as np
import pandas as pd
import os
import argparse
import datacompy

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # 创建解析器
# global parameters,  添加参数
# parser.add_argument('--data_path', type=str, default="../../data/", help="path storing data.")
parser.add_argument('--species', type=str, default="cytokine-cell", help="which species to use.")
parser.add_argument('--supervised', type=str, default="nn", help="neural networks or svm")
args = parser.parse_args()  # 解析参数

path = args.species+"/output/VGAE/"
df1 = pd.read_csv(path + "cell-cytokine_true.csv")
df1 = df1.set_index("Unnamed: 0")
df2 = pd.read_csv(path + "cell-cytokine_pred01.csv")
df2 = df2.iloc[:122,:]
df2 = df2.set_index("Unnamed: 0")
df3 = pd.read_csv(path+"cell-cytokine_pred.csv")
df3 = df3.set_index("Unnamed: 0")

# compare = datacompy.Compare(df1, df2, join_columns='key')

index = df1.index.values.tolist()
columns = df1.columns.values.tolist()

df = pd.DataFrame(columns= columns,index= index)

cytokines = []
cells = []
score = []

for i in range(df1.shape[0]):
    for j in range(df1.shape[1]):
        a = df1.iloc[i,j]
        b = df2.iloc[i,j]
        c = df3.iloc[i,j]
        if a == b:
            df.iloc[i,j] = -1
        else:
            df.iloc[i,j] = b
            if b == 0:
                pass
            else:
                cytokines.append(index[i])
                cells.append(columns[j])
                score.append(c)
df_pred = pd.DataFrame.from_dict({"cytokines": cytokines, "cell": cells, "score": score})
df_pred = df_pred.sort_values(by = "score", ascending=False)

df_pred.to_csv("pred_cell_cytokine/"+args.species+"_pred.csv")


# def list_txt(path,name,list):
#     with open(path+name,"w") as f:
#         for i in range(len(list)):
#             f.write(str(list[i])+"\n")
#
# list_txt(path,"pred_mistake.txt",pred_mistake)
# list_txt(path,"pred.txt",pred)

