import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import numpy as np
import networkx as nx
from sklearn.metrics import auc as auc3
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score
import datetime
from sklearn.metrics.pairwise import cosine_similarity

from torch_geometric.utils import to_dense_adj

def accuracy(output, labels):

    # preds = torch.argmax(torch.softmax(output, dim=1), dim=1)

    # preds = torch.argmax(output, dim=1) # main_e4.py
    # correct = (preds == torch.argmax(labels, dim=1)).float()
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()

    correct = correct.sum()
    # print(f'correct:{correct}')

    return correct / len(labels)


def precision_recall_f1(output, labels):
    """
    output:nx2
    labels:one-hot
    """

    # output = torch.softmax(output, dim=1).cpu()  # main_E4.py时注释这一行
    output = output.cpu()
    labels = labels.cpu()

    # predict = torch.argmax(output, dim=1).numpy().tolist()
    # predict = [int(i) for i in predict]
    # labels = torch.argmax(labels, dim=1).numpy()


    predict = output.max(1)[1].type_as(labels)

    # c = roc_auc_score(labels, predict)

    precision = precision_score(labels, predict)
    recall = recall_score(labels, predict)
    f1 = f1_score(labels, predict)
    
    return precision, recall, f1

def calculate_specificity_and_negative_precision(output, labels):

    # output = torch.softmax(output, dim=1).cpu()   
    output = output.cpu()
    labels = labels.cpu()

    # predict = torch.argmax(output, dim=1).numpy().tolist()
    # output[:, 1].numpy().tolist()
    # predict = np.array([int(i) for i in predict])
    # labels = torch.argmax(labels, dim=1).numpy()

    predict = output.max(1)[1].type_as(labels)
 
    # 计算混淆矩阵的各个元素
    TP = ((predict == 1) & (labels == 1)).sum()
    TN = ((predict == 0) & (labels == 0)).sum()
    FP = ((predict == 1) & (labels == 0)).sum()
    FN = ((predict == 0) & (labels == 1)).sum()

    # 计算特异度
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0.0
    # 计算负精度
    negative_precision = TN / (TN + FN) if (TN + FN) != 0 else 0.0

    return specificity, negative_precision



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def load_data(config, device):
    f = np.loadtxt(config.feature_path, dtype=float)
    l = np.loadtxt(config.label_path, dtype=int)
    test = np.loadtxt(config.test_path, dtype=int)
    train = np.loadtxt(config.train_path, dtype=int)
    # val = np.loadtxt(config.val_path, dtype=int)  # case study, no validation
    features = sp.csr_matrix(f, dtype=np.float32)
    feature_demo = features
    # features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense())).to(device)


    idx_test = test.tolist()
    idx_train = train.tolist()
    # idx_val = val.tolist()

    idx_train = torch.LongTensor(idx_train).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)
    # idx_val = torch.LongTensor(idx_val).to(device)

    label = torch.LongTensor(np.array(l)).to(device)

    return features, feature_demo,  label, idx_train, idx_test




def sample_mask(idx, l):
    """Create mask."""
    # mask = torch.zeros(l, dtype=torch.bool)
    idx = idx.cpu()
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


def ft_load_graph(config, features):
    featuregraph_path = config.featuregraph_path + str(config.k) + '_near.txt'#
    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(config.n, config.n),
                         dtype=np.float32)
    
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)

    fadj1 = fadj #+ sp.eye(fadj.shape[0])
    
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))

    struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    tedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    tadj = sp.coo_matrix((np.ones(tedges.shape[0]), (tedges[:, 0], tedges[:, 1])), shape=(config.n, config.n),
                         dtype=np.float32)
    

    tadj = tadj + tadj.T.multiply(tadj.T > tadj) - tadj.multiply(tadj.T > tadj)

    tadj1 = tadj #+ sp.eye(tadj.shape[0])

    ntadj = normalize(tadj + sp.eye(tadj.shape[0]))

    ft_edges = np.unique(np.sort(np.concatenate((fedges, tedges), axis=0), axis=1), axis=0)

    # np.savetxt('edges.csv', ft_edges, fmt='%d', delimiter=',')

    ft_adj = sp.coo_matrix((np.ones(ft_edges.shape[0]), (ft_edges[:, 0], ft_edges[:, 1])), shape=(config.n, config.n),
                         dtype=np.float32)
    ft_adj = ft_adj + ft_adj.T.multiply(ft_adj.T > ft_adj) - ft_adj.multiply(ft_adj.T > ft_adj)
    
    ft_adj1 = ft_adj #+ sp.eye(ft_adj.shape[0])

    ft_adj = normalize(ft_adj + sp.eye(ft_adj.shape[0]))

    # row, col = tadj1.nonzero()
    row, col = ft_adj1.nonzero()
  

    init_edge_index = np.stack([row, col], axis=0)
    init_edge_index = torch.tensor(init_edge_index, dtype=torch.long)

    min_vals = torch.min(init_edge_index[0], init_edge_index[1])
    max_vals = torch.max(init_edge_index[0], init_edge_index[1])


    init_edge_index = torch.stack([min_vals, max_vals], dim=0)
    init_edge_index = torch.unique(init_edge_index, dim=1)

    num_nodes = ft_adj1.shape[0]  
    
    # num_nodes = tadj1.shape[0]  
    node_degrees = np.bincount(row, minlength=num_nodes)  

    mean_degree = np.mean(node_degrees)  
    low_degree_nodes = np.nonzero(node_degrees < mean_degree)[0]  
    high_degree_nodes = np.nonzero(node_degrees >= mean_degree)[0]  

    edges_add = torch.combinations(torch.arange(ft_adj1.shape[0]), r=2)


    all_edge_index = edges_add.t()

    filtered_low, dropped_adj = drop_edge_weighted(all_edge_index, features, low_degree_nodes, high_degree_nodes, node_degrees, init_edge_index, ft_adj1, config.p, threshold=0.9)
    filtered_high, added_adj = add_edge_weighted(all_edge_index, features, low_degree_nodes, high_degree_nodes, node_degrees, init_edge_index, ft_adj1, config.p, threshold=0.9)

    dropped_adj = normalize(dropped_adj + sp.eye(dropped_adj.shape[0]))
    added_adj = normalize(added_adj + sp.eye(added_adj.shape[0]))

    ntadj = sparse_mx_to_torch_sparse_tensor(ntadj)
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    ft_adj = sparse_mx_to_torch_sparse_tensor(ft_adj)
    
    dropped_adj = sparse_mx_to_torch_sparse_tensor(dropped_adj)
    added_adj = sparse_mx_to_torch_sparse_tensor(added_adj)

    filtered_high_tensor = torch.tensor(filtered_high, dtype = torch.int64)
    
    filtered_low_tensor = torch.tensor(filtered_low, dtype = torch.int64)

    return ntadj, nfadj, ft_adj, ft_adj1, fadj1, tadj1, dropped_adj, added_adj, filtered_high_tensor, filtered_low_tensor#torch.tensor(high_degree_nodes), torch.tensor(low_degree_nodes)


def drop_edge_weighted(all_edge_index, features, low_degree_nodes, high_degree_nodes, node_degrees, init_edge_index, ft_adj1, p: float, threshold: float = 1.):
    edge_drop_weights = edge_prob(all_edge_index, node_degrees, features, "drop")

    edge_drop_weights = edge_drop_weights / edge_drop_weights.mean() * p
   

    edge_drop_weights = edge_drop_weights.where(edge_drop_weights < threshold, torch.ones_like(edge_drop_weights) * threshold)
    sel_mask_drop = torch.bernoulli(edge_drop_weights).to(torch.bool)

    high_degree_edges_mask = (
    torch.isin(all_edge_index[0], torch.tensor(high_degree_nodes)) |  
    torch.isin(all_edge_index[1], torch.tensor(high_degree_nodes))    
    )


    final_mask_high = high_degree_edges_mask & sel_mask_drop
    edge_index_drop_high = all_edge_index[:, final_mask_high]
   
    min_vals = torch.min(edge_index_drop_high[0], edge_index_drop_high[1])
    max_vals = torch.max(edge_index_drop_high[0], edge_index_drop_high[1])

    edge_index_drop_high = torch.stack([min_vals, max_vals], dim=0)
    edge_index_drop_high = torch.unique(edge_index_drop_high, dim=1)  

    edge_index_combined = torch.cat([init_edge_index, edge_index_drop_high], dim=1)
    edge_index_combined = torch.unique(edge_index_combined, dim=1)  

    num_nodes = torch.max(all_edge_index).item() + 1  

    A_sorted, _ = torch.sort(edge_index_combined.t(), dim=1)
    B_sorted, _ = torch.sort(edge_index_drop_high.t(), dim=1)
   
    A_flat = A_sorted[:, 0] * num_nodes + A_sorted[:, 1]
    B_flat = B_sorted[:, 0] * num_nodes + B_sorted[:, 1]
    mask_to_remove = torch.isin(A_flat, B_flat)
    edge_index_updated = edge_index_combined[:, ~mask_to_remove]

    if edge_index_updated.numel() == 0:
        adj_updated = torch.zeros_like(ft_adj1)
        adj_updated[low_degree_nodes, :] = ft_adj1[low_degree_nodes, :]
        adj_updated[:, low_degree_nodes] = ft_adj1[:, low_degree_nodes]
    else:
        
        values = torch.ones(edge_index_updated.size(1), dtype=torch.float32)
        sparse_adj = torch.sparse.FloatTensor(edge_index_updated, values, torch.Size([ft_adj1.shape[0], ft_adj1.shape[0]]))
        adj_updated = sparse_adj.to_dense()


    adj_matrix_sparse = sp.csr_matrix(adj_updated.numpy())  

    if edge_index_updated.numel() != 0:
        edge_index_high = edge_index_updated
     
        degrees_before = torch.bincount(init_edge_index[0], minlength=ft_adj1.shape[0]) + torch.bincount(init_edge_index[1], minlength=ft_adj1.shape[0])
        degrees_after  = torch.bincount(edge_index_high[0], minlength=ft_adj1.shape[0]) + torch.bincount(edge_index_high[1], minlength=ft_adj1.shape[0])
        nodes_with_unchanged_degree = torch.nonzero(degrees_before == degrees_after).squeeze()

        if nodes_with_unchanged_degree.numel() != 0:
            
            nodes_for_contrastive = nodes_with_unchanged_degree
        else:
            nodes_for_contrastive = torch.tensor(low_degree_nodes, device=degrees_before.device)
    else:
        nodes_for_contrastive = torch.tensor(low_degree_nodes, device=degrees_before.device)

    return nodes_for_contrastive, adj_matrix_sparse


def add_edge_weighted(all_edge_index, features, low_degree_nodes, high_degree_nodes, node_degrees, init_edge_index, ft_adj1, p: float, threshold: float = 1.):
   
    edge_add_weights = edge_prob(all_edge_index, node_degrees, features, "add")
    edge_add_weights = edge_add_weights / edge_add_weights.mean() * p
    edge_add_weights = edge_add_weights.where(edge_add_weights < threshold, torch.ones_like(edge_add_weights) * threshold)
    sel_mask_add = torch.bernoulli(edge_add_weights).to(torch.bool)
   
    low_degree_edges_mask = (
        torch.isin(all_edge_index[0], torch.tensor(low_degree_nodes)) | 
        torch.isin(all_edge_index[1], torch.tensor(low_degree_nodes))    
    )


    final_mask_low = low_degree_edges_mask & sel_mask_add 
    edge_index_add_low = all_edge_index[:, final_mask_low] 
  
    min_vals = torch.min(edge_index_add_low[0], edge_index_add_low[1])
    max_vals = torch.max(edge_index_add_low[0], edge_index_add_low[1])

    edge_index_add_low = torch.stack([min_vals, max_vals], dim=0)
    edge_index_add_low = torch.unique(edge_index_add_low, dim=1)

    edge_index_combined = torch.cat([init_edge_index, edge_index_add_low], dim=1)
    edge_index_combined = torch.unique(edge_index_combined, dim=1)
    adj_combined = to_dense_adj(edge_index_combined, batch_size=1)
    adj_combined = adj_combined.squeeze(0)

    new_low_matrix = adj_combined
    adj_matrix_sparse = sp.csr_matrix(new_low_matrix)

    edge_index_low = torch.tensor(edge_index_combined,  dtype=torch.long)
    num_nodes = ft_adj1.shape[0]
    degrees_before = torch.bincount(init_edge_index[0], minlength=num_nodes) + torch.bincount(init_edge_index[1], minlength=num_nodes)
    degrees_after = torch.bincount(edge_index_low[0], minlength=num_nodes) + torch.bincount(edge_index_low[1], minlength=num_nodes)

    nodes_with_unchanged_degree = torch.nonzero(degrees_before == degrees_after, as_tuple=False).squeeze()

    if nodes_with_unchanged_degree.numel() == 0:
        nodes_for_contrastive = torch.tensor(high_degree_nodes, device=degrees_before.device)
    else:
        nodes_for_contrastive = nodes_with_unchanged_degree
    return nodes_for_contrastive, adj_matrix_sparse


def edge_prob(all_edge_index, node_degrees, features, doa):
    features = features.cpu().numpy()   
    similarity_matrix = cosine_similarity(features)
    
    s_col = torch.log(torch.tensor(node_degrees))  

    max_val = torch.max(s_col)  
    mean_val = torch.mean(s_col)  

    sorted_indices = np.argsort(similarity_matrix, axis=1)  
    s_rank = np.argsort(sorted_indices, axis=1)  
    
    s_rank = torch.tensor(s_rank) + 1 
    s_col = s_col.unsqueeze(1) 


    if doa == "drop":

        probs = (torch.log(s_rank)) * ((max_val - s_col) / (max_val - mean_val + 1e-6)) # [num_high_degree_nodes, num_neighbors]

        row_idx = all_edge_index[0]  
        col_idx = all_edge_index[1]

        edge_probs = probs[row_idx, col_idx]  #
        edge_probs = torch.clamp(edge_probs, min=0.0)  
        
    elif doa == "add":
        
        probs = (1 / (torch.log(s_rank) + 1e-6)) * ((max_val - s_col) / (max_val - mean_val + 1e-6)) #（7611，7611）

        row_idx = all_edge_index[0] 
        col_idx = all_edge_index[1]  

        edge_probs = probs[row_idx, col_idx]  
        edge_probs = torch.clamp(edge_probs, min=0.0)  

  
    edge_probs = torch.tensor(edge_probs)
    
    return edge_probs
    

def log(string, output_name, output, f = None, newline=False, timestamp=True):
    if timestamp:
        time = datetime.datetime.now()
        line = '\t'.join([str(time.strftime('%m-%d %H:%M:%S')), string])
    else:
        time = None
        line = string

 
    print(line, file=sys.stderr)

    if newline: print("", file=sys.stderr)

    if not output == sys.stdout:
        print(line, file=output)
        if newline: print("", file=output)


    if f =='None':
        pass
    else:
        with open(output_name, "a") as fs:
            fs.write(line + '\n')
            if newline: fs.write("" + '\n')


    output.flush()
 
    return time

def get_L2reg(parameters):
    reg = 0
    for param in parameters:
        reg += 0.5 * (param ** 2).sum()
    return reg