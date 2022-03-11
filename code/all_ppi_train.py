import pandas as pd
import os,json
import argparse
import numpy as np
from numpy import *
import pickle as pkl
from trainNN import train_nn
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score,roc_auc_score,roc_curve,auc
from evaluation import get_results


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default="../", help="path storing data.")
parser.add_argument('--species', type=str, default="cytokine-cell-", help="which species to use.")
parser.add_argument('--node_attributes', type=int, default=0, help="types of attributes used by ppi.")
parser.add_argument('--model', type=str, default="gcn_vae", help="Model string.")
parser.add_argument('--supervised', type=str, default="nn", help="neural networks or svm")
parser.add_argument('--graph', type=str, default="combined", help="lists of graphs to use.")
parser.add_argument('--thr_ppi', type=float, default=0.3, help="threshold for combiend ppi network.")
parser.add_argument('--epochs_ppi', type=int, default=80, help="Number of epochs to train ppi.")
parser.add_argument('--save_results', type=int, default=1, help="whether to save the performance results")
parser.add_argument('--data_result', type=str, default="networks/五倍交叉验证/ppi+seq(200vae)/", help="path storing data.")
args = parser.parse_args()


uniprot = pd.read_pickle(os.path.join(args.data_path,args.species,"features.pkl"))
cytokine = pd.read_table(os.path.join(args.data_path, args.species + "/cytokine_list.txt"),header = 0)#immport https://www.immport.org/shared/genelists
uniprot['entrez-gene-id'] = uniprot['entrez-gene-id'].astype(int)
feature_pre = pd.merge(cytokine['ID'], uniprot, how="inner",left_on='ID',right_on = 'entrez-gene-id')
feature_pre.drop(['ID'],axis=1,inplace=True)
cytokine_IX = pd.read_table(os.path.join(args.data_path, args.species + "/ix_cytokine_list.txt"),header = 0)#string  map工具
feature_label = pd.merge(cytokine_IX[['ID','Cell Ontology ID']], uniprot, how="inner",left_on='ID',right_on = 'entrez-gene-id' )
feature_label.drop(['ID'],axis=1,inplace=True)

print("encode protein domains...")

def process_cell(x):
    temp = x.split(";")
    return temp

feature_label['Cell'] = feature_label['Cell Ontology ID'].apply(process_cell)
items = [item for sublist in feature_label['Cell'] for item in sublist]
unique_elements, counts_elements = np.unique(items, return_counts=True)
items = unique_elements[np.where(counts_elements > 2)]
pro_mapping = dict(zip(list(items),range(len(items))))
pro_encoding = [[0]*len(items) for i in range(len(feature_label))]

for i,row in feature_label.iterrows():
    for fam in row['Cell']:
        if fam in pro_mapping:
            pro_encoding[i][pro_mapping[fam] ] = 1

feature_label['cell_encoding'] = pro_encoding


def write_cell_list(ontology,ll):
    filename = os.path.join(args.data_path, args.species, ontology+"_list.txt")
    with open(filename,'w') as f:
        for x in ll:
            f.write(x + '\n')
print("writing cell term list...")
write_cell_list('cell',items)

feature_label.to_pickle(os.path.join(args.data_path, args.species, "features_label.pkl"))
feature_pre.to_pickle(os.path.join(args.data_path, args.species, "features_pre.pkl"))


# feature_label = pd.read_pickle(os.path.join(args.data_path, args.species + "/features_label.pkl"))
# feature_pre = pd.read_pickle(os.path.join(args.data_path, args.species + "/features_pre.pkl"))
idx = uniprot[uniprot['entrez-gene-id'].isin(feature_label['entrez-gene-id'])].index.tolist()
uniprot_pre_index = uniprot[uniprot['entrez-gene-id'].isin(feature_pre['entrez-gene-id'])].index.tolist()

cell = feature_label['cell_encoding'].values
cell = np.hstack(cell).reshape((len(cell), len(cell[0])))
np.random.seed(5000)


embedding = pd.read_pickle(
    os.path.join(args.data_path, args.species + "/" + args.model + "_" + args.graph + "_" + str(
        args.node_attributes) + "_embeddings_2019_" + str(args.epochs_ppi)+ ".pkl"))
embeddings = embedding[idx, :]

# split data into train and test
all_idx = list(range(cell.shape[0]))
np.random.shuffle(all_idx)

X_hp_data = embeddings[all_idx]
kf = KFold(n_splits=5)
y_score_hp_list = []
Y_test_hp_list = []
n =0
for train, test in kf.split(X_hp_data):
    Y_train_hp = cell[train]
    Y_test_hp = cell[test]
    X_train = embeddings[train]
    X_test = embeddings[test]
    n = n+1
    # Y_val_cell = cell[val_idx]
    y_score_hp = train_nn(X_train, Y_train_hp, X_test, Y_test_hp)
    y_score_hp_list.append(y_score_hp)
    Y_test_hp_list.append(Y_test_hp)
    y_hp_data = np.vstack(y_score_hp_list)
    Y_hp_data = np.vstack(Y_test_hp_list)
#
print("Start running supervised model...")
rand_str = np.random.randint(10)
save_path = os.path.join(args.data_path,
                         args.species + "/results_" + args.supervised + "_" +
                         args.graph + "_" + str(args.node_attributes) + "_" + str(args.thr_ppi) + "_" + str(
                             args.epochs_ppi))

print("###################################")
print('----------------------------------')

#保留预测打分
cell_label_list = pd.read_table(os.path.join(args.data_path, args.species + "/cell_list.txt"),header = None)
pd.DataFrame(y_hp_data).to_csv(os.path.join(args.data_path, args.species, args.data_result,"gcn_vae_1_phe_pre.txt"), index=True,
                         sep = '\t',index_label = 'id', header=[x[0] for x in cell_label_list.values])
pd.DataFrame(Y_hp_data).to_csv(os.path.join(args.data_path, args.species, args.data_result,"gcn_vae_1_phe_true.txt"), index=True,
                         sep = '\t',index_label = 'id', header=[x[0] for x in cell_label_list.values])

perf_hp_all = get_results(cell, Y_hp_data, y_hp_data)
if args.save_results:
    with open(save_path + "_5.json", "w") as f:
        json.dump(perf_hp_all, f)