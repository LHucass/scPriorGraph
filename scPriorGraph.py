from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.neighbors import kneighbors_graph
from utils import accuracy, normalize, get_adjs
from models import GCN
from preprocess import data_loader
import sys

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=50, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.05,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden1', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--hidden2', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--ref', type=str, default='bh', required=True,
                    help='Reference dataset.')
parser.add_argument('--query', type=str, default='se', required=True,
                    help='Query dataset.')
parser.add_argument('--knnr', type=float, default=0.01,
                    help='The proportion of neighboring nodes in the KNN graph.')
parser.add_argument('--pathway', type=str, required=True,
                    help='The pathways you need to use in the task.')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Loss function weight')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

epochs = parser.parse_args().epochs-1
q = parser.parse_args().query
r = parser.parse_args().ref
knnr = parser.parse_args().knnr
pathway = parser.parse_args().pathway


print('Refer dataset:' + r)
print('Query dataset:' + q)
pathway_list = ["KEGGHuman", "ReactomeMouse"]
if pathway in pathway_list:
    print('Pathways:' + pathway)
else:
    print('Error! Invalid pathway name.')
    sys.exit()


adjs, feature_train, feature_test, labels_train, labels_test, \
    class_num, knn_train, knn_test, type_df, query_index = data_loader(q=q, r=r, knnr=knnr, pathway=pathway)


A1_train0, A2_train0, A1_test0, A2_test0, A1_train, A1_test, A2_train, A2_test, P1_train, \
    P1_test, P2_train, P2_test = get_adjs(adjs)

# Model and optimizer
model = GCN(nfeat=feature_train.shape[1],
             nhid1=args.hidden1,
             nhid2=args.hidden2,
             nclass=class_num,
             dropout=args.dropout)

if args.cuda:
    print("GPU")
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

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train(epoch):
    a = parser.parse_args().alpha
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output_za, output_zp1, output_zp2, embeddings_train = model(feature_train, A1_train, P1_train, A2_train, P2_train)
    loss_CE = torch.nn.CrossEntropyLoss()
    loss_MSE = torch.nn.MSELoss()
    embeddings_train = embeddings_train.cpu()
    embeddings_train = embeddings_train.detach().numpy()
    re_graph_train = kneighbors_graph(embeddings_train, knn_train, mode='connectivity', include_self=True)
    re_graph_train = normalize(re_graph_train)
    A1_train_dense = torch.tensor(A1_train0.toarray())
    re_graph_train = torch.tensor(re_graph_train.toarray())
    loss_train = loss_CE(output_za, labels_train)+loss_MSE(output_za,output_zp1)+loss_MSE(output_za,output_zp2)
    loss_re_train = loss_MSE(re_graph_train, A1_train_dense)
    loss_train = a * loss_train + (1 - a) * loss_re_train
    acc_train = accuracy(output_za, labels_train)
    loss_train.backward()
    optimizer.step()
    if epoch % 1 == 0:
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'time: {:.4f}s'.format(time.time() - t))

def evaluation():
    model.eval()
    a = parser.parse_args().alpha
    loss_CE = torch.nn.CrossEntropyLoss()
    loss_MSE = torch.nn.MSELoss()
    output_za, output_zp1, output_zp2, embeddings_test = model(feature_test, A1_test, P1_test, A2_test, P2_test)
    embeddings_test = embeddings_test.cpu()
    embeddings_test = embeddings_test.detach().numpy()
    re_graph_test = kneighbors_graph(embeddings_test, knn_test, mode='connectivity', include_self=True)
    re_graph_test = normalize(re_graph_test)
    A1_test_dense = torch.tensor(A1_test0.toarray())
    re_graph_test = torch.tensor(re_graph_test.toarray())
    loss_test = loss_CE(output_za, labels_test)+loss_MSE(output_za,output_zp1)+loss_MSE(output_za,output_zp2)
    loss_re_test = loss_MSE(re_graph_test, A1_test_dense)
    loss_test = a * loss_test + (1 - a) * loss_re_test
    acc_test = accuracy(output_za, labels_test)
    print('loss_test: {:.4f}'.format(loss_test.item()),
          'acc_test: {:.4f}'.format(acc_test.item()))

    pred = output_za.max(1)[1].type_as(labels_test)
    pred_df = pd.DataFrame(pred)
    mapping_dict = {i: col for i, col in enumerate(type_df.columns) if 1 in type_df[col].tolist()}
    result_dict = {key: value for key, value in mapping_dict.items()}
    pred_df[pred_df.columns[0]] = pred_df[pred_df.columns[0]].replace(result_dict)
    pred_df.columns = ['pred_type']
    pred_df.index = query_index

    print('The prediction results are saved to '+'./result/pred.csv')
    pred_df.to_csv('./result/pred.csv')


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
evaluation()
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
