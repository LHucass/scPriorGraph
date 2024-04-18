import os
import sys
import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch
import snf
from sklearn.neighbors import kneighbors_graph
from utils import sparse_mx_to_torch_sparse_tensor, normalize, encode_onehot, prepare_data, diffusion_fun_improved_ppmi_dynamic_sparsity
from runR2 import get_mat_path
from scipy.sparse import csr_matrix


def get_knn_graph(arr, knn):
    m, n = arr.shape
    out = np.zeros((m, n))
    for i in range(m):
        row = arr[i]
        top_k = np.argpartition(-row, knn)[:knn]
        out[i, top_k] = 1
    for i in range(min(m, n)):
        out[i, i] = 1
    return out


def get_similarity_matrix(matrix):
    non_zero = np.where(matrix != 0)
    matrix[non_zero] = 1 / (1 + matrix[non_zero])
    return matrix


def get_snf_matrix(features_graph_pathway, features_graph, knn, args):
    features_graph_pathway = features_graph_pathway.T
    features_graph = features_graph.T
    print('SNF begin')
    print(features_graph_pathway.shape)
    print(features_graph.shape)

    if args.enable_weights == True:
        graph1 = kneighbors_graph(features_graph.values, knn, mode='distance', include_self=True)
    else:
        graph1 = kneighbors_graph(features_graph.values, knn, mode='connectivity', include_self=True)

    affinity_network1 = get_similarity_matrix(graph1.toarray())

    if args.enable_weights == True:
        graph2 = kneighbors_graph(features_graph_pathway.values, knn, mode='distance', include_self=True)
    else:
        graph2 = kneighbors_graph(features_graph_pathway.values, knn, mode='connectivity', include_self=True)

    affinity_network2 = get_similarity_matrix(graph2.toarray())
    affinity_networks = [affinity_network1, affinity_network2]
    fused_network = snf.snf(affinity_networks, K=20, t=1)
    fused_network = get_knn_graph(fused_network, knn)
    sparse_matrix = csr_matrix(fused_network)

    gene_graph = affinity_network1
    gene_graph = csr_matrix(gene_graph)
    return sparse_matrix, gene_graph


def cut_label(labels, train_num, test_num):
    labels = np.array_split(labels, [0, train_num], axis=0)
    labels_train = labels[1]
    labels_test = labels[2]
    return labels_train, labels_test

def lower_matrix(df):
    df = df.T
    index = df.index
    index = list(index)
    index2 = []
    for x in index:
        index2.append(x.lower())
    df.index = index2
    return df


def data_loader(args):

    pathway = args.pathway
    knnr = args.knnr

    data_list = prepare_data(args)
    DR_r = data_list[0]
    DR_q = data_list[1]
    HVG_r = data_list[2]
    HVG_q = data_list[3]
    L_r = data_list[4]
    L_q = data_list[5]

    query_index = HVG_q.index

    if (pathway == 'KEGGHuman'):
        pa_path1 = './data/pathway/KEGG_human_2.csv'
    elif (pathway == 'KEGGMouse'):
        pa_path1 = './data/pathway/KEGG_mouse_2.csv'
    elif (pathway == 'ReactomeHuman'):
        pa_path1 = './data/pathway/Reactome_human_2.csv'
    elif (pathway == 'ReactomeMouse'):
        pa_path1 = './data/pathway/Reactome_mouse_2.csv'
    elif (pathway == 'WikiHuman'):
        pa_path1 = './data/pathway/Wikipathways_human_2.csv'
    elif (pathway == 'WikiMouse'):
        pa_path1 = './data/pathway/Wikipathways_mouse_2.csv'
    else:
        sys.exit("Error! Please check the pathway name and path. ")

    pa_path2 = './data/pathway/output_path4.txt'

    # label
    train_label = pd.DataFrame(L_r)
    test_label = pd.DataFrame(L_q)
    label = pd.concat([train_label, test_label])
    labels = label['celltype']

    labels = labels.values
    labels, type_df = encode_onehot(labels)
    class_num = labels.shape[1]
    train_num = train_label.shape[0]
    test_num = test_label.shape[0]
    labels_train, labels_test = cut_label(labels, train_num, test_num)
    print("Label matrix:" + str(labels.shape))
    print("Class num:" + str(class_num))
    print('ref num:' + str(train_num))
    print('query num:' + str(test_num))

    # feature matrix used for refer
    features_train = pd.DataFrame(DR_r)
    features_train = features_train.astype(float)
    print("Feature matrix train:" + str(features_train.shape))

    # feature matrix used for query
    features_test = pd.DataFrame(DR_q)
    features_test = features_test.astype(float)
    print("Feature matrix test:" + str(features_test.shape))

    # adjacency matrix A1 A2 for refer
    knn_train = int((train_num) * knnr)
    features_graph_train = HVG_r
    features_graph_train = lower_matrix(features_graph_train)
    features_graph_train.to_csv("./data/temp/features_graph_train.csv", sep=',')
    output_path1 = f"./data/temp/r_A1_{args.r_name}_{args.q_name}_{args.pathway}_{args.num_hvg}.csv"
    output_path2 = f"./data/temp/r_A2_{args.r_name}_{args.q_name}_{args.pathway2}_{args.num_hvg}.csv"
    sc_path = "./data/temp/features_graph_train.csv"

    if os.path.exists(output_path1):
        print("scroed files exist !")
    else:
        get_mat_path(sc_path, pa_path1, output_path1)
    if os.path.exists(output_path2):
        print("scroed files exist !")
    else:
        get_mat_path(sc_path, pa_path2, output_path2)
    features_graphA1_pathway_train = pd.read_csv(output_path1, sep=',', header=0, index_col=0)
    features_graphA2_pathway_train = pd.read_csv(output_path2, sep=',', header=0, index_col=0)
    A1_train, _ = get_snf_matrix(features_graphA1_pathway_train, features_graph_train, knn_train, args)
    A2_train, _ = get_snf_matrix(features_graphA2_pathway_train, features_graph_train, knn_train, args)
    A1_train = normalize(A1_train)
    A2_train = normalize(A2_train)
    print("Adjcent matrix A1 for train:" + str(A1_train.shape))
    print("Adjcent matrix A2 for train:" + str(A2_train.shape))

    # adjacency matrix P1 P2 for refer
    P1_train = diffusion_fun_improved_ppmi_dynamic_sparsity(A1_train, path_len=2, k=1.0)
    P2_train = diffusion_fun_improved_ppmi_dynamic_sparsity(A2_train, path_len=2, k=1.0)
    P1_train = normalize(P1_train + sp.eye(P1_train.shape[0]))
    P2_train = normalize(P2_train + sp.eye(P2_train.shape[0]))
    print("Adjcent matrix P1 for train:" + str(P1_train.shape))
    print("Adjcent matrix P2 for train:" + str(P2_train.shape))

    # adjacency matrix A1 A2 for query
    knn_test = int((test_num) * knnr)
    features_graph_test = HVG_q
    features_graph_test = lower_matrix(features_graph_test)
    features_graph_test.to_csv("./data/temp/features_graph_test.csv", sep=',')
    sc_path = "./data/temp/features_graph_test.csv"
    output_path3 = f"./data/temp/q_A1_{args.r_name}_{args.q_name}_{args.pathway}_{args.num_hvg}.csv"
    output_path4 = f"./data/temp/q_A2_{args.r_name}_{args.q_name}_{args.pathway2}_{args.num_hvg}.csv"
    if os.path.exists(output_path3):
        print("scroed files exist !")
    else:
        get_mat_path(sc_path, pa_path1, output_path3)

    if os.path.exists(output_path4):
        print("scroed files exist !")
    else:
        get_mat_path(sc_path, pa_path2, output_path4)

    features_graphA1_pathway_test = pd.read_csv(output_path3, sep=',', header=0, index_col=0)
    features_graphA2_pathway_test = pd.read_csv(output_path4, sep=',', header=0, index_col=0)
    A1_test, _ = get_snf_matrix(features_graphA1_pathway_test, features_graph_test, knn_test, args)
    A2_test, _ = get_snf_matrix(features_graphA2_pathway_test, features_graph_test, knn_test, args)
    A1_test = normalize(A1_test)
    A2_test = normalize(A2_test)
    print("Adjcent matrix A1 for test:" + str(A1_test.shape))
    print("Adjcent matrix A2 for test:" + str(A2_test.shape))

    # adjacency matrix P1 P2 for query
    P1_test = diffusion_fun_improved_ppmi_dynamic_sparsity(A1_test, path_len=2, k=1.0)
    P2_test = diffusion_fun_improved_ppmi_dynamic_sparsity(A2_test, path_len=2, k=1.0)
    P1_test = normalize(P1_test + sp.eye(P1_test.shape[0]))
    P2_test = normalize(P2_test + sp.eye(P2_test.shape[0]))
    print("Adjcent matrix P1 for test:" + str(P1_test.shape))
    print("Adjcent matrix P2 for test:" + str(P2_test.shape))

    features_train = torch.FloatTensor(np.array(features_train))
    features_test = torch.FloatTensor(np.array(features_test))
    labels_train = torch.LongTensor(np.where(labels_train)[1])
    labels_test = torch.LongTensor(np.where(labels_test)[1])
    A1_train0 = A1_train
    A2_train0 = A2_train
    A1_test0 = A1_test
    A2_test0 = A2_test
    A1_train = sparse_mx_to_torch_sparse_tensor(A1_train)
    A1_test = sparse_mx_to_torch_sparse_tensor(A1_test)
    A2_train = sparse_mx_to_torch_sparse_tensor(A2_train)
    A2_test = sparse_mx_to_torch_sparse_tensor(A2_test)
    P1_train = sparse_mx_to_torch_sparse_tensor(P1_train)
    P1_test = sparse_mx_to_torch_sparse_tensor(P1_test)
    P2_train = sparse_mx_to_torch_sparse_tensor(P2_train)
    P2_test = sparse_mx_to_torch_sparse_tensor(P2_test)
    adjs = [A1_train0, A2_train0, A1_test0, A2_test0, A1_train, A1_test, A2_train, A2_test, P1_train, P1_test, P2_train,
            P2_test]

    return adjs, features_train, features_test, labels_train, labels_test, class_num, knn_train, knn_test, type_df, query_index
