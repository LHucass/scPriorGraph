from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from utils import accuracy, get_adjs, model_loss
from models import GCN_plus
from preprocess import data_loader
import sys
from datetime import datetime

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--refer_M_path', type=str, help='Path to r_path_M csv file', required=True)
parser.add_argument('--refer_L_path', type=str, help='Path to r_path_L csv file', required=True)
parser.add_argument('--query_M_path', type=str, help='Path to q_path_M csv file', required=True)
parser.add_argument('--query_L_path', type=str, help='Path to q_path_L csv file', required=True)
parser.add_argument('--pathway', type=str, required=True)
parser.add_argument('--r_name', type=str, default="Reference", required=False)
parser.add_argument('--q_name', type=str, default="Query", required=False)
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=50, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, required=True)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--hidden1', type=int, default=128, help='Size of hidden layer.')
parser.add_argument('--hidden2', type=int, default=64, help='Size of hidden layer.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--knnr', type=float, default=0.05, help='Graph density.')
parser.add_argument('--pathway2', type=str, default="lr")
parser.add_argument('--dim_reduction', type=float, default=50, help='dim size after reduction')
parser.add_argument('--enable_weights', type=bool, default=False, help='Graph Type')
parser.add_argument('--min_genes', type=int, default=50, help='Parameters in scanpy.pp.filter_cells')
parser.add_argument('--min_cells', type=int, default=50, help='Parameters in scanpy.pp.filter_genes')
parser.add_argument('--min_counts', type=int, default=100, help='Parameters in scanpy.pp.filter_genes')
parser.add_argument('--num_hvg', type=int, default=5000, help='Number of highly variable genes extracted.')
parser.add_argument('--min_cells2', type=int, default=1, help='Cell count threshold to filter out types')
parser.add_argument('--weight_ce', type=float, default=0.5)
parser.add_argument('--weight_reg', type=float, default=0.3)
parser.add_argument('--weight_rec', type=float, default=0.2)
parser.add_argument('--layers', type=int, default=3)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

pathway_list = ["KEGGHuman", "KEGGMouse", "ReactomeHuman", "ReactomeMouse", "WikiHuman", "WikiMouse"]
if args.pathway in pathway_list:
    print('Valid Pathways:' + args.pathway)
else:
    sys.exit('Error! Invalid pathway name.')


adjs, feature_train, feature_test, labels_train, labels_test, \
    class_num, knn_train, knn_test, type_df, query_index = data_loader(args=args)


A1_train0, A2_train0, A1_test0, A2_test0, A1_train, A1_test, A2_train, A2_test, P1_train, \
    P1_test, P2_train, P2_test = get_adjs(adjs)

# Model and optimizer
model = GCN_plus(nfeat=feature_train.shape[1],
                 nhid=args.hidden1,
                 nclass=class_num,
                 nlayer=args.layers,
                 dropout=args.dropout)

if args.cuda:
    print("Use GPU")
    model.cuda()
    feature_train = feature_train.cuda()
    feature_test = feature_test.cuda()
    A1_train = A1_train.cuda()
    A1_test = A1_test.cuda()
    A2_train = A2_train.cuda()
    A2_test = A2_test.cuda()
    P1_train = P1_train.cuda()
    P1_test = P1_test.cuda()
    P2_train = P2_train.cuda()
    P2_test = P2_test.cuda()
    labels_train = labels_train.cuda()
    labels_test = labels_test.cuda()
else:
    print("Use CPU")

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def train(epoch):
    model.train()
    optimizer.zero_grad()
    output_za, output_zp1, output_zp2, embeddings_train = model(feature_train, A1_train, P1_train, A2_train, P2_train)
    loss_train = model_loss(output_za, output_zp1, output_zp2, embeddings_train, knn_train, A1_train0, labels_train, args)
    acc_train = accuracy(output_za, labels_train)

    loss_train.backward()
    optimizer.step()

    model.eval()
    optimizer.zero_grad()
    output_za_q, output_zp1_q, output_zp2_q, embeddings_test = model(feature_test, A1_test, P1_test, A2_test, P2_test)
    loss_test = model_loss(output_za_q, output_zp1_q, output_zp2_q, embeddings_test, knn_test, A1_test0, labels_test, args)
    acc_test = accuracy(output_za_q, labels_test)

    if epoch % 1 == 0:
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train),
              'loss_test: {:.4f}'.format(loss_test.item()))

def evaluation():
    model.eval()
    optimizer.zero_grad()
    output_za_q, output_zp1_q, output_zp2_q, embeddings_test = model(feature_test, A1_test, P1_test, A2_test, P2_test)
    acc_test = accuracy(output_za_q, labels_test)
    print('\nAccuracy of test dataset: {:.4f}'.format(acc_test.item()))
    save_pred(output_za_q, labels_test)

def save_pred(output_test, labels_test):
    output_test = output_test.cpu()
    labels_test = labels_test.cpu()
    pred = output_test.max(1)[1].type_as(labels_test)
    pred_df = pd.DataFrame(pred.cpu())
    mapping_dict = {i: col for i, col in enumerate(type_df.columns) if 1 in type_df[col].tolist()}
    result_dict = {key: value for key, value in mapping_dict.items()}
    pred_df[pred_df.columns[0]] = pred_df[pred_df.columns[0]].replace(result_dict)
    pred_df.columns = ['pred_type']
    pred_df.index = query_index
    pred_df.to_csv(f'./result/pred.csv')

# Train model
t_total = time.time()

for epoch in range(args.epochs):
    train(epoch)

evaluation()
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
