import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader
from model_main import GCN
import argparse
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score
from utils import *
import math
# import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from data import GraphAdjSampler
from data import Graph_load_batch
import networkx as nx
# import matplotlib.pyplot as plt
import numpy as np

def random_miss(dim,missing_r):
    a = np.random.rand(29*dim)
    a[a >= missing_r] = 1.0
    a[a < missing_r] = 0.0
    mask = a.reshape(dim,29)
    return mask


parser = argparse.ArgumentParser(description='META arguments.')
parser.add_argument('--lr', dest='lr', type=float, help='Learning rate.',default=0.01)
parser.add_argument('--missing_r', dest='missing_r', type=float, help='Learning rate.',default=0.15)
parser.add_argument('--batch_size', dest='batch_size', type=int, help='Batch size.',default=1)
parser.add_argument('--epochsize', dest='epochsize',type=int, help='epoch for training',default=50)
parser.add_argument('--gamma', dest='gamma', type=float,help='Number of workers to load data.',default=0.1)
parser.add_argument('--max_num_nodes', dest='max_num_nodes', type=int,help='Predefined maximum number of nodes in train/test graphs. -1 if determined by training data.',default=1000)
parser.add_argument('--feature_dim', dest='feature_dim',help='input feature dim',default=1)
parser.add_argument('--feature_dim_c', dest='feature_dim_c',help='encoder dim c',default=29)
parser.add_argument('--feature_dim_d', dest='feature_dim_d',help='encoder dim d',default=29)
parser.add_argument('--feature_dim_e', dest='feature_dim_e',help='encoder dim e',default=20)
args = parser.parse_args()
learning_rate = args.lr
missing_r = args.missing_r
batch_size = args.batch_size
gamma = args.gamma
max_num_nodes = args.max_num_nodes
feature_dim = args.feature_dim
c = args.feature_dim_c
d = args.feature_dim_d
e = args.feature_dim_e
epochsize = args.epochsize
def mask_set(missing_r_test):
    mask_set = []
    for batchidx, real_adj_f in enumerate(graph_test):
        ori_num = len(graphs_test[batchidx].nodes)
        realadj_ori = real_adj_f['adj'].float()
        realfeatures_ori = real_adj_f['features'].float()
        realadj_ori = torch.squeeze(realadj_ori, 0)
        realfeatures_ori = torch.squeeze(realfeatures_ori, 0)
        realadj = realadj_ori[:ori_num,:ori_num]
        realfeatures = realfeatures_ori[:ori_num,]

        # realfeatures = torch.squeeze(realfeatures, 0)
        realadj_norm = normalize_adj(realadj)
        realadj_input = preprocess_adj(realadj_norm)
        realadj_input = torch.tensor(realadj_input)
        realadj_input = realadj_input.float()

        realadj_inverse = preprocess_adj(realadj_norm)
        realadj_inverse = torch.tensor(realadj_inverse)
        realadj_inverse = realadj_inverse.float()


        dim_sample = realfeatures.size(0)
        sub_mask = random_miss(dim_sample, missing_r_test)
        sub_mask = sub_mask.tolist()
        # sub_mask = sub_mask.astype(np.float32)
        mask_set.append(sub_mask)
    return mask_set

def entropy(C0,C1,C2,C3,C4):
    entro_all = 0
    avoid_0 = 0.000001
    P0 = torch.sum(C0*C0,0)
    P1 = torch.sum(C1*C1, 0)
    P2 = torch.sum(C2*C2, 0)
    P3 = torch.sum(C3*C3, 0)
    P4 = torch.sum(C4*C4, 0)
    P = P0 + P1 + P2 + P3 + P4
    P0 = P0/P
    P1 = P1/P
    P2 = P2/P
    P3 = P3/P
    P4 = P4/P
    P0 = P0 + avoid_0
    P1 = P1 + avoid_0
    P2 = P2 + avoid_0
    P3 = P3 + avoid_0
    P4 = P4 + avoid_0
    for i in range(29):
        ent = P0[i]*math.log(P0[i])+P1[i]*math.log(P1[i])+P2[i]*math.log(P2[i])+P3[i]*math.log(P3[i])+P4[i]*math.log(P4[i])
        entro_all = ent + entro_all
    entro_all = -entro_all/29
    return entro_all

def missing_MSE(mask_set):
    mse_all = 0
    a = 0
    for batchidx, real_adj_f in enumerate(graph_test):

        ori_num = len(graphs_test[batchidx].nodes)
        realadj_ori = real_adj_f['adj'].float()
        realfeatures_ori = real_adj_f['features'].float()
        realadj_ori = torch.squeeze(realadj_ori, 0)
        realfeatures_ori = torch.squeeze(realfeatures_ori, 0)
        realadj = realadj_ori[:ori_num,:ori_num]
        realfeatures = realfeatures_ori[:ori_num,]

        realadj_norm = normalize_adj(realadj)
        realadj_input = preprocess_adj(realadj_norm)

        A_WL1 = torch.tensor(preprocess_adj_gw1(realadj_norm)).float()
        A_WL2 = torch.tensor(preprocess_adj_gw2(realadj_norm)).float()
        A_WL3 = torch.tensor(preprocess_adj_gw3(realadj_norm)).float()
        A_WL4 = torch.tensor(preprocess_adj_gw4(realadj_norm)).float()
        A_WL_I_1 = torch.tensor(np.linalg.inv(preprocess_adj_gw1(realadj_norm))).float()
        A_WL_I_2 = torch.tensor(np.linalg.inv(preprocess_adj_gw2(realadj_norm))).float()
        A_WL_I_3 = torch.tensor(np.linalg.inv(preprocess_adj_gw3(realadj_norm))).float()
        A_WL_I_4 = torch.tensor(np.linalg.inv(preprocess_adj_gw4(realadj_norm))).float()
        A_WL_I_0 = torch.tensor(preprocess_inverse_adj(realadj_norm)).float()

        realadj_input = torch.tensor(realadj_input)
        realadj_input = realadj_input.float()

        realadj_inverse = preprocess_adj(realadj_norm)
        realadj_inverse = torch.tensor(realadj_inverse)
        realadj_inverse = realadj_inverse.float()

        realadj = torch.squeeze(realadj,0)
        realfeatures = torch.squeeze(realfeatures,0)

        dim_sample = realfeatures.size(0)
        mask = mask_set[batchidx]
        mask = np.array(mask)
        mask = mask.astype(np.float32)
        mask_np = mask
        missing_num = np.sum(mask == 0)
        mask = torch.tensor(mask)
        realfeatures_input = mask * realfeatures
        realfeatures_input1 = realfeatures_input.numpy()
        fakefeatures_mean = np.zeros((dim_sample,29))
        realfeatures_input1[realfeatures_input1==0] = np.nan
        realfeatures_input = np.nan_to_num(realfeatures_input)
        realfeatures_input = torch.tensor(realfeatures_input)
        fakefeatures,C0,C1,C2,C3,C4 = gcn(realadj_input, realfeatures_input, A_WL1, A_WL2, A_WL3, A_WL4, A_WL_I_1, A_WL_I_2, A_WL_I_3, A_WL_I_4, A_WL_I_0)
        mse = nn.MSELoss()
        "mean"
        for i in range(dim_sample):
            for j in range(29):
                if mask_np[i,j] == 0:
                    fakefeatures_mean[i,j] = np.nanmean(realfeatures_input1[:,j])
        "mean"
        fakefeatures_mean = np.nan_to_num(fakefeatures_mean)
        fakefeatures_mean = torch.tensor(fakefeatures_mean)
        mse_batch_mean = torch.sqrt(torch.sum(torch.pow((1 - mask) * fakefeatures_mean - (1 - mask) * realfeatures, 2)) / missing_num)
        # mse_batch = mse(fakefeatures, realfeatures)
        mse_batch = torch.sqrt(torch.sum(torch.pow((1-mask) * fakefeatures - (1-mask) * realfeatures, 2)) / missing_num)
        mse_all = mse_all + mse_batch
        a = a + mse_batch_mean
    mse_final = mse_all/batchidx
    b = a/batchidx
    # return print('missing_MSE', mse_final), print('missing_MSE_mean', b)
    return mse_final, b


# prog_args = arg_parse()

graphs= Graph_load_batch(min_num_nodes=1, name='PROTEINS_full')
num_graphs_raw = len(graphs)

 # remove graphs with number of nodes greater than max_num_nodes
graphs = [g for g in graphs if g.number_of_nodes() <= max_num_nodes]

graphs_len = len(graphs)
print('Number of graphs removed due to upper-limit of number of nodes: ',
        num_graphs_raw - graphs_len)
graphs_test = graphs[:int(0.2 * graphs_len)]
graphs_test_len = len(graphs_test)
    #graphs_train = graphs[0:int(0.8*graphs_len)]
graphs_train = graphs[int(0.2 * graphs_len):]
graphs_test_MF1 = graphs_test[int(0.2 * graphs_test_len):]
print('total graph num: {}, training set: {}'.format(len(graphs),len(graphs_train)))
print('max number node: {}'.format(max_num_nodes))
print('total graph num: {}, testing set: {}'.format(len(graphs),graphs_test_len))

dataset = GraphAdjSampler(graphs_train, max_num_nodes)
dataset_test = GraphAdjSampler(graphs_test, max_num_nodes)

dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False)

dataset_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        num_workers=0,
        shuffle=False)

dataset_loader_test_MF1 = torch.utils.data.DataLoader(
        graphs_test_MF1,
        batch_size=1,
        num_workers=0,
        shuffle=False)

graph_train = dataset_loader
graph_test = dataset_loader_test
maskset = mask_set(0.1)
"With generated maskset to miss"
maskset = np.load('missing.npy', allow_pickle = True)
maskset = maskset.tolist()

# feature_dim = 1

gcn = GCN(feature_dim, max_num_nodes,c,d,e)
print(gcn)
# criteon = nn.MSELoss()
optimizer = optim.Adam(gcn.parameters(), lr=learning_rate, betas=(0.9,0.99), eps=1e-08)
# optimizer = optim.Adam(gcn.parameters(), lr=learning_rate)
# optimizer = optim.SGD(gcn.parameters(), lr=learning_rate)
testMSE = []
print("start train...")
testMSE = np.zeros([epochsize+10], dtype=float, order='C')
loss_GDN = 0
loss_mean = 0
num_ok = 0
best_aa = 1
best_bb = 1
#train process


for epoch in range(epochsize):
    losstrain = 0
    # training
    for batchidx, real_adj_f in enumerate(graph_train):
        ori_num = len(graphs_train[batchidx].nodes)
        realadj = real_adj_f['adj'].float()
        realfeatures = real_adj_f['features'].float()
        # realfeatures = torch.squeeze(realfeatures, 0)
        adjmed = torch.squeeze(realadj, 0)
        realadj_norm = normalize_adj(adjmed[:ori_num, :ori_num])

        realadj_input = preprocess_adj(realadj_norm)

        A_WL1 = torch.tensor(preprocess_adj_gw1(realadj_norm)).float()
        A_WL2 = torch.tensor(preprocess_adj_gw2(realadj_norm)).float()
        A_WL3 = torch.tensor(preprocess_adj_gw3(realadj_norm)).float()
        A_WL4 = torch.tensor(preprocess_adj_gw4(realadj_norm)).float()
        A_WL_I_1 = torch.tensor(np.linalg.inv(preprocess_adj_gw1(realadj_norm) + 0.0001)).float()
        A_WL_I_2 = torch.tensor(np.linalg.inv(preprocess_adj_gw2(realadj_norm) + 0.0001)).float()
        A_WL_I_3 = torch.tensor(np.linalg.inv(preprocess_adj_gw3(realadj_norm) + 0.0001)).float()
        A_WL_I_4 = torch.tensor(np.linalg.inv(preprocess_adj_gw4(realadj_norm) + 0.0001)).float()
        A_WL_I_0 = torch.tensor(preprocess_inverse_adj(realadj_norm)).float()
        realadj_input = torch.tensor(realadj_input)
        realadj_input = realadj_input.float()

        realadj_inverse = preprocess_inverse_adj(realadj_norm)
        realadj_inverse = torch.tensor(realadj_inverse)
        realadj_inverse = realadj_inverse.float()

        realadj = torch.squeeze(realadj, 0)
        realfeatures = torch.squeeze(realfeatures, 0)
        realfeatures = realfeatures[:ori_num, :]
        # if ori_num > 400:
        #     np.savetxt('power1.txt',np.array(realfeatures))
        realadj = realadj[:ori_num, :ori_num]
        realfeatures_ = realfeatures
        dim_sample = realfeatures.size(0)
        "mask for training"
        mask = random_miss(dim_sample, missing_r)
        mask = mask.astype(np.float32)
        mask = torch.tensor(mask)
        realfeatures_input = mask * realfeatures
        fakefeatures, C0, C1, C2, C3, C4 = gcn(realadj_input, realfeatures_input, A_WL1, A_WL2, A_WL3, A_WL4,
                                               A_WL_I_1, A_WL_I_2, A_WL_I_3, A_WL_I_4, A_WL_I_0)
        # fakefeatures = torch.squeeze(fakefeatures, 0)
        real_node_num = len(graphs_train[batchidx].nodes)
        # MSE_A = torch.sum(torch.pow(fakeadj - torch.squeeze(realadj, 0), 2)) / (realadj.size(0) * 4096)
        MSE_X = torch.sqrt(
            torch.sum(torch.pow(fakefeatures - realfeatures, 2)) / (realfeatures.size(0) * realfeatures.size(0)))
        # CE = nn.CrossEntropyLoss()
        # MSE_X = CELOSS(fakefeatures, realfeatures_)
        # entroy = nn.BCEWithLogitsLoss()
        # w = [0.1, 0.9]
        # weight = torch.zeros(fakeadj.shape)
        # for i in range(fakeadj.shape[0]):
        #     for j in range(fakeadj.shape[1]):
        #         weight[i][j] = w[int(fakeadj[i][j])]
        spe_ent = entropy(C0, C1, C2, C3, C4)
        # entroy = nn.BCELoss()
        mse = nn.MSELoss()
        # MSE_A = entroy(fakeadj, torch.reshape(realadj,(1,ori_num * ori_num)))
        # MSE_A = entroy(fakeadj, realadj_norm)
        # loss = MSE_A + MSE_X
        # loss = MSE_X
        loss = MSE_X - torch.abs(spe_ent) * gamma
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losstrain = losstrain + loss

    print("train_loss__________________________________________:{}"
          .format(losstrain / batchidx))

    test_a, mean_a = missing_MSE(maskset)
    print('test_RMSE', test_a)
    print('mean_RMSE', mean_a)
    if test_a < best_aa:
        best_aa = test_a
    if mean_a < best_bb:
        best_bb = mean_a
print('mean', best_bb)
print('test', best_aa)



