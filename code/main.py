from input_data import load_data,load_labels
import numpy as np
import pandas as pd
import argparse
import os
from trainGcn import train_gcn,train_vae
from trainNN import train_nn
import pickle as pkl
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve,auc
from evaluation import get_results,plot_roc

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def reshape(features):
    return np.hstack(features).reshape((len(features),len(features[0])))

def train(args):
    # load feature dataframe
    print("loading features...") 
    uniprot = pd.read_pickle(os.path.join(args.data_path, args.species ,"features.pkl"))

    print("#############################")
    adj, features = load_data(uniprot,0, args)
    embeddings = train_gcn(features, adj, args) #VGAE
    # embeddings = train_vae(features, args) #VAE

    path = os.path.join(args.data_path, args.species,"output",args.model, "embeddings_" + args.species + "_" + str(args.epochs_ppi) + ".pkl")
    with open(path, 'wb') as file:
        pkl.dump(embeddings, file)  #Save the embedding vector
    
    #load labels
    cc,mf,bp, cytokine_cell = load_labels(uniprot)  #GO Term feature
    np.random.seed(5000)
    # split data into train and test
    all_idx = list(range(cc.shape[0]))
    np.random.shuffle(all_idx)

    #5 fold cross
    X_cytokine_cell_data = embeddings[all_idx]
    kf = KFold(n_splits=5)
    y_score_cytokine_cell_list = [] #predicted
    Y_test_cytokine_cell_list = [] #known
    for train, test in kf.split(X_cytokine_cell_data):
        Y_train_cytokine_cell = cytokine_cell[train]
        Y_test_cytokine_cell = cytokine_cell[test]
        X_train = embeddings[train]
        X_test = embeddings[test]
        y_score_cytokine_cell = train_nn(X_train, Y_train_cytokine_cell, X_test, Y_test_cytokine_cell)  #分类器
        y_score_cytokine_cell_list.append(y_score_cytokine_cell)
        Y_test_cytokine_cell_list.append(Y_test_cytokine_cell)
        y_cytokine_cell_data = np.vstack(y_score_cytokine_cell_list)
        Y_cytokine_cell_data = np.vstack(Y_test_cytokine_cell_list)

    print("Start running supervised model...")

    save_path = os.path.join(args.data_path,args.species,"output",args.model,"results_graph2go_" + args.supervised + "_" +
                             str(args.thr_ppi) + "_" + str(args.epochs_ppi))

    print("###################################")
    print('----------------------------------')
    print('5cytokine_cell')

    perf_cytokine_cell_all = get_results(cytokine_cell, Y_cytokine_cell_data, y_cytokine_cell_data)
    perf_cytokine_cell_all = pd.DataFrame.from_dict(perf_cytokine_cell_all)
    if args.save_results:
        perf_cytokine_cell_all.to_csv(save_path + "_5cytokine_cell.csv")  # Save various evaluation indicators

    cytokines = [x for x in uniprot['cytokine'].values]
    cytokines = [x for x in uniprot['Cytokine Ontology Label'].values]
    #save score
    cell_name = pd.read_excel("../data/cell name.xlsx")
    Cell_Ontology_ID = cell_name['Cell Ontology ID'].values
    Cell_Ontology_Label = cell_name['Cell Ontology Label'].values
    name_map = dict(zip(Cell_Ontology_ID, Cell_Ontology_Label))
    cytokine_cell_label_list = pd.read_table(os.path.join(args.data_path, args.species + "/cell_list.txt"),
                                             header=None)
    cytokine_cell_label_list = [x[0] for x in cytokine_cell_label_list.values]
    cytokine_cell_label_list = [name_map[i] for i in cytokine_cell_label_list]

    #ROC  截断值
    def Find_Optimal_Cutoff(target, predicted):
        fpr, tpr, threshold = roc_curve(target, predicted)
        roc_auc = auc(fpr, tpr)
        i = np.arange(len(tpr))
        roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'auc': pd.Series(roc_auc, index=i),
                            'threshold': pd.Series(threshold, index=i)})
        roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
        return list(roc_t['threshold']), list(roc_t['auc'])

    # Find optimal probability threshold
    def roc_thr(label, predict):
        roc_thr_list = []
        roc_auc_list = []
        for i in range(predict.shape[1]):
            threshold, roc_auc = Find_Optimal_Cutoff(label[:, i], predict[:, i])
            predict[:, i][predict[:, i] >= threshold] = 1
            predict[:, i][predict[:, i] < threshold] = 0
            roc_thr_list.append(threshold)
            roc_auc_list.append(roc_auc)
        return predict, roc_thr_list, roc_auc_list
    #frequency of cytokine_cell
    perf = dict()
    perf['fre'] = []
    for i in range(Y_cytokine_cell_data.shape[1]):
        fre = dict(pd.DataFrame(Y_cytokine_cell_data).iloc[:, i].value_counts()) #每个细胞连接细胞因子的个数
        perf['fre'].append(fre[1])

    def array_to_score(array,name):
        df = pd.DataFrame(array)
        df.index = cytokines
        df.columns = cytokine_cell_label_list
        df.to_csv(os.path.join(args.data_path, args.species,name))
        return df

    pre = array_to_score(y_cytokine_cell_data, "output/"+args.model+"/cell-cytokine_pred.csv")
    known = array_to_score(Y_cytokine_cell_data, "output/"+args.model+"/cell-cytokine_true.csv")

    Y_label = Y_cytokine_cell_data.reshape(-1, 1)
    Y_pred = y_cytokine_cell_data.reshape(-1, 1)

    plot_roc(Y_label, Y_pred, '5-fold cross-validation',os.path.join(args.data_path, args.species, "output", args.model))

    def output_data(Y_label,Y_pred,model):
        with open(os.path.join(args.data_path, args.species,"output",model,'Y_label_str.pkl'), 'wb') as file:
            pkl.dump(Y_label, file)
        with open(os.path.join(args.data_path, args.species,"output",model,'Y_pred_str.pkl'), 'wb') as file:
            pkl.dump(Y_pred, file)

    output_data(Y_label, Y_pred, args.model)

    predict, roc_thr_list, roc_auc_list = roc_thr(Y_cytokine_cell_data, y_cytokine_cell_data)

    perf['roc_auc'] = [x[0] for x in roc_auc_list]
    perf['roc_thr'] = [x[0] for x in roc_thr_list]
    f = pd.DataFrame.from_dict(perf, orient='index', columns=None)
    # f.columns = cytokine_cell_label_list
    f.to_csv(os.path.join(args.data_path, args.species,"output",args.model, "fre_roc.csv"))

    def array_to_df(array,name,evaluation_index):
        df = pd.DataFrame(array)
        df.index = cytokines
        df.columns = cytokine_cell_label_list
        df = pd.concat([df, evaluation_index], axis=0)
        df.to_csv(os.path.join(args.data_path, args.species,"output",args.model,name))
        return df

    pred_label = array_to_df(predict, "cell-cytokine_pred01.csv", f)

    print("The End")


if __name__ == "__main__":
    parser = argparse.ArgumentParser( formatter_class=argparse.ArgumentDefaultsHelpFormatter)#创建解析器
    #global parameters
    parser.add_argument('--dataset_type', type=str, default="cell-cytokine_all", help="which species to use.")
    parser.add_argument('--data_path', type=str, default="../dataset/", help="path storing data.")
    parser.add_argument('--thr_ppi', type=float, default=0.3, help="threshold for combiend ppi network.")
    parser.add_argument('--supervised', type=str, default="nn", help="neural networks or svm")
    parser.add_argument('--only_gcn', type=int, default=0, help="0 for training all, 1 for only embeddings.")
    parser.add_argument('--save_results', type=int, default=1, help="whether to save the performance results")
    
    #parameters for traing GCN
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument('--epochs_ppi', type=int, default=60, help="Number of epochs to train ppi.")
    parser.add_argument('--hidden1', type=int, default=800, help="Number of units in hidden layer 1.")
    parser.add_argument('--hidden2', type=int, default=400, help="Number of units in hidden layer 2.")
    parser.add_argument('--weight_decay', type=float, default=0., help="Weight for L2 loss on embedding matrix.")
    parser.add_argument('--dropout', type=float, default=0., help="Dropout rate (1 - keep probability).")
    parser.add_argument('--model', type=str, default="VGAE", help="Model string.")

    args = parser.parse_args() #Analytical parameters
    print(args)
    train(args)

