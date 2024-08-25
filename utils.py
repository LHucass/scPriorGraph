import sys
import os
import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import scanpy as sc
import anndata
import io
import random
import itertools
import scipy
import anndata as ad
from numpy import inf
from scipy import sparse
import math
import anndata as ad
from sklearn.metrics import accuracy_score, confusion_matrix


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)

    acc = accuracy_score(y_true=preds.cpu().numpy(), y_pred=labels.cpu().numpy())

    return acc


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    df = pd.DataFrame(classes_dict)
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot, df


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def cell_fliter(label1,threshold):
    count_dict = label1.value_counts().to_dict()
    keep_categories = [key for key in count_dict.keys() if count_dict[key] >= threshold]
    keep_rows = label1.isin(keep_categories)
    label1 = label1[keep_rows]
    return label1


def lower_matrix(df):
    index = df.index
    index = list(index)
    index2 = []
    for x in index:
        index2.append(x.lower())
    df.index = index2
    return df


def get_adjs(adjs):
    A1_train0 = adjs[0]
    A2_train0 = adjs[1]
    A1_test0 = adjs[2]
    A2_test0 = adjs[3]
    A1_train = adjs[4]
    A1_test = adjs[5]
    A2_train = adjs[6]
    A2_test = adjs[7]
    P1_train = adjs[8]
    P1_test = adjs[9]
    P2_train = adjs[10]
    P2_test = adjs[11]
    return A1_train0, A2_train0, A1_test0, A2_test0, A1_train, A1_test, A2_train, A2_test, P1_train, P1_test, P2_train, P2_test


def prepare_data(args):
    """Get HVG expression information and aligned dimensionality reduction expression matrix from reference and query matrices"""

    refer_M = pd.read_csv(args.refer_M_path, header=0, index_col=0)
    refer_L = pd.read_csv(args.refer_L_path, header=0, index_col=0)
    query_M = pd.read_csv(args.query_M_path, header=0, index_col=0)
    query_L = pd.read_csv(args.query_L_path, header=0, index_col=0)

    adata_r = ad.AnnData(X=refer_M)
    adata_r.obs["celltype"] = refer_L.iloc[:, 0]
    sc.pp.filter_cells(adata_r, min_genes=args.min_genes)
    sc.pp.filter_genes(adata_r, min_cells=args.min_cells)
    sc.pp.filter_genes(adata_r, min_counts=args.min_counts)

    adata_q = ad.AnnData(X=query_M)
    adata_q.obs["celltype"] = query_L.iloc[:, 0]
    sc.pp.filter_cells(adata_q, min_genes=args.min_genes)
    sc.pp.filter_genes(adata_q, min_cells=args.min_cells)
    sc.pp.filter_genes(adata_q, min_counts=args.min_counts)

    temp_Lr = adata_r.obs["celltype"]
    temp_Lr = cell_fliter(temp_Lr, args.min_cells2)
    common_type = set(set(temp_Lr) & set(adata_q.obs["celltype"]))
    common_type.discard('Other/Doublet')
    print(common_type)
    mask_r = adata_r.obs["celltype"].isin(common_type)
    mask_q = adata_q.obs["celltype"].isin(common_type)
    adata_r = adata_r[mask_r, :].copy()
    adata_q = adata_q[mask_q, :].copy()

    sc.pp.normalize_total(adata_r, target_sum=1e4)
    sc.pp.log1p(adata_r)
    sc.pp.highly_variable_genes(adata_r, n_top_genes=args.num_hvg)
    hvg_r = adata_r.var["highly_variable"]

    sc.pp.normalize_total(adata_q, target_sum=1e4)
    sc.pp.log1p(adata_q)
    sc.pp.highly_variable_genes(adata_q, n_top_genes=args.num_hvg)
    hvg_q = adata_q.var["highly_variable"]

    hvg_common = list(adata_r.var_names[hvg_r].intersection(adata_q.var_names[hvg_q]))
    adata_r = adata_r[:, adata_r.var_names.isin(hvg_common)]
    adata_q = adata_q[:, adata_q.var_names.isin(hvg_common)]

    adata_all = sc.concat([adata_r, adata_q], join='inner', label='study', keys=['refer', 'query'], index_unique=None)

    sc.pp.pca(adata_all, n_comps=args.dim_reduction)
    sc.external.pp.harmony_integrate(adata_all, key="study", verbose=False)
    adata_r_split = adata_all[adata_all.obs["study"] == "refer", :]
    adata_q_split = adata_all[adata_all.obs["study"] == "query", :]
    DR_r = adata_r_split.obsm["X_pca_harmony"]
    DR_q = adata_q_split.obsm["X_pca_harmony"]
    HVG_r = pd.DataFrame(adata_r_split.X, index=adata_r_split.obs_names, columns=adata_r_split.var_names)
    HVG_q = pd.DataFrame(adata_q_split.X, index=adata_q_split.obs_names, columns=adata_q_split.var_names)
    L_r = adata_r_split.obs['celltype']
    L_q = adata_q_split.obs['celltype']

    data_list = [DR_r, DR_q, HVG_r, HVG_q, L_r, L_q]

    return data_list


def diffusion_fun_sparse(A):
    n, m = A.shape
    A_with_selfloop = A + sp.identity(n, format='csc')
    diags = A_with_selfloop.sum(axis=1).flatten()
    with np.errstate(divide='ignore'):
        diags_sqrt = 1.0 / np.sqrt(diags)
    diags_sqrt[np.isinf(diags_sqrt)] = 0
    DH = sp.spdiags(diags_sqrt, [0], m, n, format='csc')
    d = DH.dot(A_with_selfloop.dot(DH))
    return d


def _normalize_diffusion_matrix(A):
    n, m = A.shape
    A_with_selfloop = A
    diags = A_with_selfloop.sum(axis=1).flatten()

    with np.errstate(divide='ignore'):
        diags_sqrt = 1.0 / np.sqrt(diags)
    diags_sqrt[np.isinf(diags_sqrt)] = 0
    DH = sp.spdiags(diags_sqrt, [0], m, n, format='csc')
    d = DH.dot(A_with_selfloop.dot(DH))
    return d


def diffusion_fun_improved(A, sampling_num=100, path_len=3,
                           self_loop=True, spars=False):
    """Return normalized adjcent matrix plus PPMI"""
    shape = A.shape
    print("Do the sampling...")
    mat = _diffusion_fun_sampling(
        A, sampling_num=sampling_num, path_len=path_len,
        self_loop=self_loop, spars=spars)
    print("Calculating the PPMI...")

    pmi = None
    if spars:
        pmi = _PPMI_sparse(mat)
    else:
        pmi = _PPMI(mat)
    A_with_selfloop = A + pmi
    dig = np.sum(A_with_selfloop, axis=1)
    dig = np.squeeze(np.asarray(dig))
    Degree = np.diag(dig)
    Degree_normalized = Degree ** (-0.5)
    Degree_normalized[Degree_normalized == inf] = 0.0
    Diffusion = np.dot(
        np.dot(Degree_normalized, A_with_selfloop), Degree_normalized)
    return Diffusion


def diffusion_fun_improved_ppmi_dynamic_sparsity(A, sampling_num=100, path_len=2,
                                                 self_loop=True, spars=True, k=1.0):
    print("Do the sampling...")
    mat = _diffusion_fun_sampling(
        A, sampling_num=sampling_num, path_len=path_len,
        self_loop=self_loop, spars=spars)
    print("Calculating the PPMI...")

    if spars:
        pmi = _PPMI_sparse(mat)
    else:
        pmi = _PPMI(mat)

    pmi = _shift(pmi, k)
    ans = _normalize_diffusion_matrix(pmi.tocsc())

    return ans


def _shift(mat, k):
    print(k)
    r, c = mat.shape
    x, y = mat.nonzero()
    mat = mat.todok()
    offset = np.log(k)
    print("Offset: " + str(offset))
    for i, j in zip(x, y):
        mat[i, j] = max(mat[i, j] - offset, 0)

    x, y = mat.nonzero()
    sparsity = 1.0 - len(x) / float(r * c)
    print("Sparsity: " + str(sparsity))
    return mat


def _diffusion_fun_sampling(A, sampling_num=100, path_len=3, self_loop=True, spars=False):
    """Return diffusion matrix"""
    re = None
    if not spars:
        re = np.zeros(A.shape)
    else:
        re = sparse.dok_matrix(A.shape, dtype=np.float32)

    if self_loop:
        A_with_selfloop = A + sparse.identity(A.shape[0], format="csr")
    else:
        A_with_selfloop = A

    # record each node's neignbors
    dict_nid_neighbors = {}
    for nid in range(A.shape[0]):
        neighbors = np.nonzero(A_with_selfloop[nid])[1]
        dict_nid_neighbors[nid] = neighbors

    # for each node
    for i in range(A.shape[0]):
        # for each sampling iter
        for j in range(sampling_num):
            _generate_path(i, dict_nid_neighbors, re, path_len)
    return re


def _generate_path(node_id, dict_nid_neighbors, re, path_len):
    path_node_list = [node_id]
    for i in range(path_len - 1):
        temp = dict_nid_neighbors.get(path_node_list[-1])
        if len(temp) < 1:
            break
        else:
            path_node_list.append(random.choice(temp))
    # update difussion matrix re
    for pair in itertools.combinations(path_node_list, 2):
        if pair[0] == pair[1]:
            re[pair[0], pair[1]] += 1.0
        else:
            re[pair[0], pair[1]] += 1.0
            re[pair[1], pair[0]] += 1.0


def _PPMI(mat):
    (nrows, ncols) = mat.shape
    colTotals = mat.sum(axis=0)
    rowTotals = mat.sum(axis=1).T
    N = np.sum(rowTotals)
    rowMat = np.ones((nrows, ncols), dtype=np.float32)
    for i in range(nrows):
        rowMat[i, :] = 0 if rowTotals[i] == 0 else rowMat[i, :] * (1.0 / rowTotals[i])
    colMat = np.ones((nrows, ncols), dtype=np.float)
    for j in range(ncols):
        colMat[:, j] = 0 if colTotals[j] == 0 else colMat[:, j] * (1.0 / colTotals[j])
    P = N * mat * rowMat * colMat
    P = np.fmax(np.zeros((nrows, ncols), dtype=np.float32), np.log(P))
    return P


def _PPMI_sparse(mat):
    # mat is a sparse dok_matrix
    nrows, ncols = mat.shape
    colTotals = mat.sum(axis=0)
    rowTotals = mat.sum(axis=1).T

    N = float(np.sum(rowTotals))
    rows, cols = mat.nonzero()

    p = sp.dok_matrix((nrows, ncols))
    for i, j in zip(rows, cols):
        _under = rowTotals[0, i] * colTotals[0, j]
        if _under != 0.0:
            log_r = np.log((N * mat[i, j]) / _under)
            if log_r > 0:
                p[i, j] = log_r
    return p

from sklearn.neighbors import kneighbors_graph


def model_loss(output_za, output_zp1, output_zp2, embeddings_train, knn_train, A1_train0, labels_train, args):
    args.weight_ce = 0.5
    args.weight_reg = 0.3
    args.weight_rec = 0.2
    loss_CE = torch.nn.CrossEntropyLoss()
    loss_MSE = torch.nn.MSELoss()
    embeddings_train = embeddings_train.cpu().detach().numpy()
    re_graph_train = kneighbors_graph(embeddings_train, knn_train, mode='connectivity', include_self=True)
    re_graph_train = normalize(re_graph_train)
    A1_train_dense = torch.tensor(A1_train0.toarray())
    re_graph_train = torch.tensor(re_graph_train.toarray())
    loss_ce = loss_CE(output_za, labels_train)
    loss_reg1 = loss_MSE(output_za, output_zp1)
    loss_reg2 = loss_MSE(output_za, output_zp2)
    loss_rec = loss_MSE(re_graph_train, A1_train_dense)
    loss_train = args.weight_ce * loss_ce + args.weight_reg * (loss_reg1 + loss_reg2) + args.weight_rec * loss_rec
    return loss_train
