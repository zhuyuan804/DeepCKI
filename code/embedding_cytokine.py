from input_data import load_data,load_labels
import numpy as np
import pandas as pd
import argparse
import os
from trainGcn import train_gcn
import pickle as pkl



def reshape(features):
    return np.hstack(features).reshape((len(features),len(features[0])))


def train(args):
    # load feature dataframe
    print("loading features...") 
    uniprot = pd.read_pickle(os.path.join(args.data_path,  "2019/features.pkl"))
    # for graph in args.graphs:
    #     adj, features = load_data(graph, uniprot, args)
    #     embeddings = train_gcn(features, adj, args, graph)


    print("#############################")
    print("Training",args.graph)
    adj, features = load_data(args.graph, uniprot, args)
    embeddings = train_gcn(features, adj, args, args.graph)


    path = os.path.join(args.data_path, args.species + "/" + args.model + "_" + str(args.lr) + "_" + args.graph + "_" + str(args.node_attributes) +"_embeddings_" + args.species + "_" + str(args.epochs_ppi) + ".pkl")
    with open(path, 'wb') as file:
        pkl.dump(embeddings, file)
    file.close()


    if args.only_gcn == 1:
        return
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser( formatter_class=argparse.ArgumentDefaultsHelpFormatter)#创建解析器
    #global parameters,  添加参数
    parser.add_argument('--node_attributes', type=int, default=1, help="types of attributes used by ppi.")
    parser.add_argument('--simi_attributes', type=int, default=0, help="types of attributes used by simi.")
    parser.add_argument('--graph', type=str, default="combined", help="lists of graphs to use.")
    parser.add_argument('--species', type=str, default="2019", help="which species to use.")
    parser.add_argument('--data_path', type=str, default="../../data/", help="path storing data.")
    parser.add_argument('--thr_ppi', type=float, default=0.3, help="threshold for combiend ppi network.")
    parser.add_argument('--thr_evalue', type=float, default=1e-4, help="threshold for similarity network.")
    parser.add_argument('--supervised', type=str, default="nn", help="neural networks or svm")
    parser.add_argument('--only_gcn', type=int, default=1, help="0 for training all, 1 for only embeddings.")
    parser.add_argument('--save_results', type=int, default=1, help="whether to save the performance results")
    
    #parameters for traing GCN
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument('--epochs_ppi', type=int, default=80, help="Number of epochs to train ppi.")
    parser.add_argument('--epochs_simi', type=int, default=300, help="Number of epochs to train similarity network.")
    parser.add_argument('--hidden1', type=int, default=800, help="Number of units in hidden layer 1.")
    parser.add_argument('--hidden2', type=int, default=400, help="Number of units in hidden layer 2.")
    parser.add_argument('--weight_decay', type=float, default=0., help="Weight for L2 loss on embedding matrix.")
    parser.add_argument('--dropout', type=float, default=0., help="Dropout rate (1 - keep probability).")
    parser.add_argument('--model', type=str, default="gcn_vae", help="Model string.")

    args = parser.parse_args()#解析参数
    print(args)
    train(args)

