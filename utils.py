import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

def CELOSS(a, b):
    loss_all = 0
    CE = nn.CrossEntropyLoss()
    dim = a.size(0)
    for i in range(dim):
        c = a[i,:]
        c = c.unsqueeze(0)
        c = c.to(torch.float32)
        d = b[i,:].int().long()
        loss = CE(c, d)
        loss_all = loss_all + loss
    return loss_all/dim


def one_hot(input):
    input = input.long()
    class_num = 18
    batch_size = input.size(0)
    label = torch.LongTensor(input)

    one_hot = torch.zeros(batch_size, class_num).scatter_(1, label, 1)
    return one_hot

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo().toarray()

# def normalize_adj(adj):
#     """Symmetrically normalize adjacency matrix."""
#     # rows, cols = adj.nonzero()
#     # adj[cols, rows] = adj[rows, cols]
#     adj = sp.coo_matrix(adj)
#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -1).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     return d_mat_inv_sqrt.dot(adj).tocoo().toarray()

def preprocess_adj(adj):

    #adj_normalized = normalize_adj(adj)
    # adj_normalized = normalize_adj(adj + 1 * sp.eye(adj.shape[0]))
    adj_normalized = normalize_adj(adj)
    #return sparse_to_tuple(adj_normalized)
    sym_l = sp.eye(adj_normalized.shape[0]) - adj_normalized
    #true_a = adj_normalized + 0.5 * sym_l.dot(sym_l)
    true_a = 1 * adj_normalized + 0.5 * sym_l.dot(sym_l) - 1.0/6 * sym_l.dot(sym_l).dot(sym_l) + 1.0/24 * sym_l.dot(sym_l).dot(sym_l).dot(sym_l)
    return true_a

def preprocess_inverse_adj(adj):
    #adj_normalized = normalize_adj(adj)
    adj_normalized = normalize_adj(adj + 1 * sp.eye(adj.shape[0]))
    sym_l = sp.eye(adj_normalized.shape[0]) - adj_normalized
    #true_a = sp.eye(adj_normalized.shape[0]) +sym_l + 0.5 * sym_l.dot(sym_l)
    true_a = sp.eye(adj_normalized.shape[0]) +sym_l + 0.5 * sym_l.dot(sym_l) + 1.0/6 * sym_l.dot(sym_l).dot(sym_l) + 1.0/24 * sym_l.dot(sym_l).dot(sym_l).dot(sym_l)
    return true_a

def preprocess_adj_gw1(adj):

    #adj_normalized = normalize_adj(adj)
    # adj_normalized = normalize_adj(adj + 1 * sp.eye(adj.shape[0]))
    adj_normalized = normalize_adj(adj)
    #return sparse_to_tuple(adj_normalized)
    sym_l = sp.eye(adj_normalized.shape[0]) - adj_normalized
    #true_a = adj_normalized + 0.5 * sym_l.dot(sym_l)
    true_a = 3 * adj_normalized + 0.75 * sym_l.dot(sym_l)
    return true_a


def preprocess_adj_gw2(adj):

    #adj_normalized = normalize_adj(adj)
    # adj_normalized = normalize_adj(adj + 1 * sp.eye(adj.shape[0]))
    adj_normalized = normalize_adj(adj)
    #return sparse_to_tuple(adj_normalized)
    sym_l = sp.eye(adj_normalized.shape[0]) - adj_normalized
    #true_a = adj_normalized + 0.5 * sym_l.dot(sym_l)
    true_a = 1 * sp.eye(adj_normalized.shape[0]) + 3 * sym_l - 1.5 * sym_l.dot(sym_l)
    return true_a

def preprocess_adj_gw3(adj):

    #adj_normalized = normalize_adj(adj)
    # adj_normalized = normalize_adj(adj + 1 * sp.eye(adj.shape[0]))
    adj_normalized = normalize_adj(adj)
    #return sparse_to_tuple(adj_normalized)
    sym_l = sp.eye(adj_normalized.shape[0]) - adj_normalized
    #true_a = adj_normalized + 0.5 * sym_l.dot(sym_l)
    true_a = 1 * sp.eye(adj_normalized.shape[0]) + 0.75 * sym_l.dot(sym_l)
    return true_a

def preprocess_adj_gw4(adj):

    #adj_normalized = normalize_adj(adj)
    # adj_normalized = normalize_adj(adj + 1 * sp.eye(adj.shape[0]))
    adj_normalized = normalize_adj(adj)
    #return sparse_to_tuple(adj_normalized)
    sym_l = sp.eye(adj_normalized.shape[0]) - adj_normalized
    #true_a = adj_normalized + 0.5 * sym_l.dot(sym_l)
    true_a = 1 * sp.eye(adj_normalized.shape[0]) + 0.5 * sym_l.dot(sym_l)
    return true_a





def preprocess_adj_gw1_inverse(adj):

    #adj_normalized = normalize_adj(adj)
    # adj_normalized = normalize_adj(adj + 1 * sp.eye(adj.shape[0]))
    adj_normalized = normalize_adj(adj)
    #return sparse_to_tuple(adj_normalized)
    sym_l = sp.eye(adj_normalized.shape[0]) - adj_normalized
    #true_a = adj_normalized + 0.5 * sym_l.dot(sym_l)
    true_a = 3 * adj_normalized + 0.75 * sym_l.dot(sym_l)
    return true_a


def preprocess_adj_gw2_inverse(adj):

    #adj_normalized = normalize_adj(adj)
    # adj_normalized = normalize_adj(adj + 1 * sp.eye(adj.shape[0]))
    adj_normalized = normalize_adj(adj)
    #return sparse_to_tuple(adj_normalized)
    sym_l = sp.eye(adj_normalized.shape[0]) - adj_normalized
    #true_a = adj_normalized + 0.5 * sym_l.dot(sym_l)
    true_a = 3 * sym_l - 1.5 * sym_l.dot(sym_l)
    return true_a

def preprocess_adj_gw3_inverse(adj):

    #adj_normalized = normalize_adj(adj)
    # adj_normalized = normalize_adj(adj + 1 * sp.eye(adj.shape[0]))
    adj_normalized = normalize_adj(adj)
    #return sparse_to_tuple(adj_normalized)
    sym_l = sp.eye(adj_normalized.shape[0]) - adj_normalized
    #true_a = adj_normalized + 0.5 * sym_l.dot(sym_l)
    true_a = 0.75 * sym_l.dot(sym_l)
    return true_a

def preprocess_adj_gw4_inverse(adj):

    #adj_normalized = normalize_adj(adj)
    # adj_normalized = normalize_adj(adj + 1 * sp.eye(adj.shape[0]))
    adj_normalized = normalize_adj(adj)
    #return sparse_to_tuple(adj_normalized)
    sym_l = sp.eye(adj_normalized.shape[0]) - adj_normalized
    #true_a = adj_normalized + 0.5 * sym_l.dot(sym_l)
    true_a = 0.75 * sym_l.dot(sym_l)
    return true_a


