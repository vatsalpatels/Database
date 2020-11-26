import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import dgl
import networkx as nx
from dgl import DGLGraph, transform
import csv


class EthereumDataset(object):
    """Cora citation network dataset. Nodes mean author and edges mean citation
    relationships.
    """
    def __init__(self):
        self.name = 'ethereum'
        self.dir = "./data/"
        
        self._load()

    def _load(self):
        idx_features_labels = np.genfromtxt("{}ethereum/ethereum1.txt".format(self.dir),
                                            dtype=np.dtype(str), delimiter=",")
        # print("idx_features_labels:",idx_features_labels.shape)

        features = sp.csr_matrix(idx_features_labels[:, 1:-1],
                                 dtype=np.float32)
        print("features:",features.shape)

        labels = encode_onehot(idx_features_labels[:, -1])
        # print(labels)
        self.num_labels = labels.shape[1]
        # print("labels:",labels)

        # build graph
        idx = np.asarray(idx_features_labels[:, 0], dtype=np.int64)
        # print("idx:",idx.shape)

        idx_map = {j: i for i, j in enumerate(idx)}
        # print(idx_map)

        edges_unordered = np.genfromtxt("{}ethereum/edges1.txt".format(self.dir),
                                        dtype=np.int64, delimiter=",")
        # print("edges_unordered",edges_unordered)

        edges = np.asarray(list(map(idx_map.get, edges_unordered.flatten())),
                           dtype=np.int64).reshape(edges_unordered.shape)
        # print("edges:",edges)

        adj = sp.coo_matrix((np.ones(edges.shape[0]),
                             (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        self.graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())

        features = normalize(features)
        self.features = np.asarray(features.todense())
        self.labels = np.where(labels)[1]


        self.train_mask = _sample_mask(range(5000), labels.shape[0])
        # print("Train_mask",self.train_mask.sum().item())
        self.val_mask = _sample_mask(range(5000, 15000), labels.shape[0])
        self.test_mask = _sample_mask(range(15000, 25257), labels.shape[0])

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        g = DGLGraph(self.graph)
        g.ndata['train_mask'] = self.train_mask
        g.ndata['val_mask'] = self.val_mask
        g.ndata['test_mask'] = self.test_mask
        g.ndata['label'] = self.labels
        g.ndata['feat'] = self.features
        return g
    
    def __len__(self):
        return 1
   

def load_ethereum(args):
    data = EthereumDataset()
    return data

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int64)
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def _sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask


#def load_ether(path="/Users/patel/OneDrive/Documents/github/repos/OCGNN/data/", dataset="ethereum"):
#    """Load Ethereum Transaction Network"""
#    print('Loading {} dataset...'.format(dataset))
    
#    idx_features_labels = np.genfromtxt("{}{}/ethereum.txt".format(path,dataset), dtype=np.dtype(str), delimiter=",")
#    print("idx_features_labels Shape:",idx_features_labels.shape)
#    print("idx_features_labels:",idx_features_labels.dtype)
    
#    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
#    print("Features Shape:",features.shape)
#    print("Features:",features.dtype)
    
#    labels = encode_onehot(idx_features_labels[:,-1])
#    np.savetxt("/Users/patel/OneDrive/Documents/vatsal/v_label.csv", labels, delimiter=",")
#    print("labels Shape:",labels.shape)
#    print("labels:",labels.dtype)
    
#    idx = np.array(idx_features_labels[:, 0], dtype=np.int64)
#    np.savetxt("/Users/patel/OneDrive/Documents/vatsal/v_idx.csv", idx, delimiter=",")
#    print("idx Shape:",idx.shape)
#    print("idx :",idx.dtype)
#    idx_map = {j: i for i, j in enumerate(idx)}

#    with open('/Users/patel/OneDrive/Documents/vatsal/v_idx_map.csv', 'w') as csv_file:  
#        writer = csv.writer(csv_file)
#        for key, value in idx_map.items():
#            writer.writerow([key, value])
    
#    edges_unordered = np.genfromtxt("{}{}/edges.txt".format(path,dataset), dtype=np.int64, delimiter=",")
#    np.savetxt("/Users/patel/OneDrive/Documents/vatsal/v_edges_un.csv", edges_unordered, delimiter=",")
#    print("edges unordered Shape:",edges_unordered.shape)
#    print("edges unordered:",edges_unordered.dtype)

#    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=np.int64).reshape(edges_unordered.shape)
#    np.savetxt("/Users/patel/OneDrive/Documents/vatsal/v_edges.csv", edges, delimiter=",")
#    print("edges Shape:",edges.shape)
#    print("edges:",edges.dtype)

#    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#                        shape=(labels.shape[0], labels.shape[0]),
#                        dtype=np.float32)

#    # build symmetric adjacency matrix
#    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

#    features = normalize(features)
#    adj = normalize(adj + sp.eye(adj.shape[0]))

#    idx_train = range(0, 3300)
#    idx_val = range(3300, 7000)
#    idx_test = range(7000, 10700)

#    features = torch.FloatTensor(np.array(features.todense()))
#    labels = torch.LongTensor(np.where(labels)[1])
#    adj = sparse_mx_to_torch_sparse_tensor(adj)

#    idx_train = torch.LongTensor(idx_train)
#    idx_val = torch.LongTensor(idx_val)
#    idx_test = torch.LongTensor(idx_test)
    
#    dfdict={'adj':adj,'features':features,'labels':labels,'train_mask':idx_train,
#        'val_mask':idx_val,'test_mask': idx_test}

#    return dfdict
