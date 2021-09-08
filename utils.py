#!/usr/local/bin/python
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)





def load_data(data_adj, data_features, data_dis, geo_adj, data_labels):
    adj = data_adj
    features = data_features

    pattern_dis = []
    for item in data_dis:
        aaa = np.power(item, -1)
        aaa[np.isinf(aaa)] = 0.
        pattern_dis.append(aaa)



    idx_train = range(1794)
    idx_val = range(1794, 1794+384)
    idx_test = range(1794+384, 1794+384+384)


    train_mask = sample_mask(idx_train, data_labels.shape[0])
    val_mask = sample_mask(idx_val, data_labels.shape[0])
    test_mask = sample_mask(idx_test, data_labels.shape[0])


    y_train = np.zeros(data_labels.shape)
    y_val = np.zeros(data_labels.shape)
    y_test = np.zeros(data_labels.shape)
    y_train[train_mask, :] = data_labels[train_mask, :]
    y_val[val_mask, :] = data_labels[val_mask, :]
    y_test[test_mask, :] = data_labels[test_mask, :]

    return adj, features, pattern_dis, geo_adj, y_train, y_val, y_test, train_mask, val_mask, test_mask






def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    features = sp.csr_matrix(features)
    return sparse_to_tuple(features)


def preprocess_features_all(features):
    rowsum = np.array(features.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features



def construct_feed_dict(features, support, pattern_dis, tuopu, labels, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['tuopu_adj']: tuopu})
    feed_dict.update({placeholders['pattern_dis']: pattern_dis})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict



def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()




def preprocess_geo(geo_adj):
    geo_adj_normalized = normalize_adj(geo_adj + sp.eye(geo_adj.shape[0]))
    A_geo = geo_adj_normalized.dot(geo_adj_normalized)

    return sparse_to_tuple(A_geo)


def normalize_gcn(adj):

    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def preprocess_gcn(adj):
    adj_normalized = normalize_gcn(adj + sp.eye(adj.shape[0]))
    A_2 = adj_normalized.dot(adj_normalized)

    return A_2




def z_score(x):
    mu = np.average(x)
    sigma = np.std(x)
    if sigma == 0:
        x = x - mu
    else:
        x = (x - mu) / sigma
    return x

def z_inverse(x, mean ,std):
    return x * std + mean