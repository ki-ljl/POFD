# -*- coding:utf-8 -*-

import copy
import os
import pickle
import random
import sys

sys.path.append("..")

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from torch.optim.lr_scheduler import StepLR
from torch_geometric.utils import to_scipy_sparse_matrix, negative_sampling
import scipy.sparse as sp
from tqdm import tqdm

from pytorchtools import EarlyStopping
from src.models import POFD_LP, POFD_NC, POFD_DBLP


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pickle(file_name):
    f = open(file_name, "rb+")
    dataset = pickle.load(f)
    f.close()
    return dataset


def coo2adj(edge_index, num_nodes=None):
    if num_nodes is None:
        return to_scipy_sparse_matrix(edge_index).toarray()
    else:
        return to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).toarray()


def adj2coo(adj):
    """
        adj: numpy
    """
    edge_index_temp = sp.coo_matrix(adj)
    indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
    edge_index = torch.LongTensor(indices)

    return edge_index


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data_edge_index(
        poc_graph,
        only_user_graph,
        only_poc_graph,
        both_graph,
        homogeneous_graph,
        edge_index):
    # Let the edge_index of all graphs be intersected by the current edge_index
    edge_index = edge_index.T.cpu().numpy().tolist()
    edge_index = [(x[0], x[1]) for x in edge_index]
    # 1. poc_graph
    poc_edge_index = poc_graph.edge_index.T.cpu().numpy().tolist()
    rev_dict = {v: k for k, v in poc_graph.global_to_local_nodes_dict.items()}
    # local index to global index
    poc_edge_index = [(rev_dict[x[0]], rev_dict[x[1]]) for x in
                      poc_edge_index]
    poc_edge_index = list(set(poc_edge_index) & set(edge_index))
    poc_edge_index = [[poc_graph.global_to_local_nodes_dict[x[0]],
                       poc_graph.global_to_local_nodes_dict[x[1]]] for x in
                      poc_edge_index]
    poc_edge_index = torch.tensor(poc_edge_index).T
    poc_graph.edge_index = poc_edge_index

    # 2. only_user_graph
    if only_user_graph.edge_index is not None:
        only_user_edge_index = only_user_graph.edge_index.T.numpy().tolist()
        rev_dict = {v: k for k, v in only_user_graph.global_to_local_nodes_dict.items()}
        only_user_edge_index = [(rev_dict[x[0]], rev_dict[x[1]]) for x in
                                only_user_edge_index]
        only_user_edge_index = list(set(only_user_edge_index) & set(edge_index))
        only_user_edge_index = [[only_user_graph.global_to_local_nodes_dict[x[0]],
                                 only_user_graph.global_to_local_nodes_dict[x[1]]] for x in
                                only_user_edge_index]
        only_user_edge_index = torch.tensor(only_user_edge_index).T
        only_user_graph.edge_index = only_user_edge_index

    # 3. only_poc_graph
    if only_poc_graph.edge_index is not None:
        only_poc_edge_index = only_poc_graph.edge_index.T.cpu().numpy().tolist()
        rev_dict = {v: k for k, v in only_poc_graph.global_to_local_nodes_dict.items()}
        only_poc_edge_index = [(rev_dict[x[0]], rev_dict[x[1]]) for x in
                               only_poc_edge_index]
        only_poc_edge_index = list(set(only_poc_edge_index) & set(edge_index))
        only_poc_edge_index = [[only_poc_graph.global_to_local_nodes_dict[x[0]],
                                only_poc_graph.global_to_local_nodes_dict[x[1]]] for x in
                               only_poc_edge_index]
        only_poc_edge_index = torch.tensor(only_poc_edge_index).T
        only_poc_graph.edge_index = only_poc_edge_index

    # 4. both graph
    if both_graph.edge_index is not None:
        both_edge_index = both_graph.edge_index.T.cpu().numpy().tolist()
        rev_dict = {v: k for k, v in both_graph.global_to_local_nodes_dict.items()}
        both_edge_index = [(rev_dict[x[0]], rev_dict[x[1]]) for x in
                           both_edge_index]
        both_edge_index = list(set(both_edge_index) & set(edge_index))
        both_edge_index = [[both_graph.global_to_local_nodes_dict[x[0]],
                            both_graph.global_to_local_nodes_dict[x[1]]] for x in
                           both_edge_index]
        both_edge_index = torch.tensor(both_edge_index).T
        both_graph.edge_index = both_edge_index

    # 5. global
    global_edge_index = homogeneous_graph.edge_index
    global_edge_index = global_edge_index.T.cpu().numpy().tolist()
    global_edge_index = [(x[0], x[1]) for x in global_edge_index]
    global_edge_index = list(set(global_edge_index) & set(edge_index))
    global_edge_index = torch.tensor(global_edge_index).t()

    return poc_graph, only_user_graph, only_poc_graph, both_graph, global_edge_index


def negative_sample(train_data):
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')
    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    return edge_label, edge_label_index


def get_lp_metrics(out, edge_label):
    # out = [0.1, 0.2, ..., ] # cuda tensor
    # edge_label = [0, 1, 0, 1, ...]  cuda tensor
    edge_label = edge_label.cpu().numpy()
    out = out.cpu().numpy()
    pred = (out > 0.5).astype(int)
    auc = roc_auc_score(edge_label, out)
    f1 = f1_score(edge_label, pred)
    ap = average_precision_score(edge_label, out)

    return auc, f1, ap


@torch.no_grad()
def lp_test(args, model, init_x,
            val_graph, val_datas,
            test_graph, test_datas):
    val_poc_graph, val_only_user_graph, val_only_poc_graph, val_both_graph, val_edge_index = val_datas
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    out, s_loss = model(init_x,
                        val_edge_index,
                        val_poc_graph,
                        val_only_user_graph,
                        val_only_poc_graph,
                        val_both_graph,
                        val_graph.edge_label_index)
    out = out.view(-1)
    val_loss = criterion(out, val_graph.edge_label) + args.mu * s_loss
    # cal metric
    test_poc_graph, test_only_user_graph, test_only_poc_graph, test_both_graph, test_edge_index = test_datas
    out, _ = model(init_x,
                   test_edge_index,
                   test_poc_graph,
                   test_only_user_graph,
                   test_only_poc_graph,
                   test_both_graph,
                   test_graph.edge_label_index)
    out = out.view(-1).sigmoid()
    model.train()

    auc, f1, ap = get_lp_metrics(out, test_graph.edge_label)
    return val_loss, auc, ap


def lp_train(args, root_path,
             init_x, data_dict,
             graphs, datas):
    model_path = root_path + '/src/checkpoints/' + args.dataset + '_lp_model.pt'
    model = POFD_LP(args.in_feats, args.hidden_feats, args.out_feats,
                    data_dict, heads=args.heads, activation_str='tanh').to(device)
    # data
    train_graph, val_graph, test_graph = graphs
    train_datas, val_datas, test_datas = datas
    train_poc_graph, train_only_user_graph, train_only_poc_graph, train_both_graph, train_edge_index = train_datas

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    min_epochs = args.min_epochs
    best_model = None
    min_val_loss = np.Inf
    final_test_auc = 0
    final_test_ap = 0
    epochs = args.epochs
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        # edge_label, edge_label_index = negative_sample()
        edge_label, edge_label_index = train_graph.edge_label, train_graph.edge_label_index
        out, s_loss = model(init_x,
                            train_edge_index,
                            train_poc_graph,
                            train_only_user_graph,
                            train_only_poc_graph,
                            train_both_graph,
                            train_graph.edge_label_index)
        out = out.view(-1)
        loss = criterion(out, edge_label) + args.mu * s_loss
        loss.backward()
        optimizer.step()
        # validation
        val_loss, test_auc, test_ap = lp_test(args, model, init_x,
                                              val_graph, val_datas,
                                              test_graph, test_datas)
        if epoch + 1 > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            final_test_auc = test_auc
            final_test_ap = test_ap
            best_model = copy.deepcopy(model)
            # save model
            state = {'model': best_model.state_dict()}
            torch.save(state, model_path)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print('epoch {:03d} train_loss {:.8f} val_loss {:.4f} test_auc {:.4f} test_ap {:.4f}'
              .format(epoch, loss.item(), val_loss, test_auc, test_ap))

    return final_test_auc, final_test_ap


def get_nc_metrics(nc_data, out):
    label = nc_data.y[nc_data.test_mask].cpu().numpy()
    out = out[nc_data.test_mask].cpu().numpy()
    pred = (out >= 0.5).astype(int)
    acc = accuracy_score(label, pred)
    f1 = f1_score(label, pred)
    return acc, f1


@torch.no_grad()
def nc_test(model, init_x, graphs, global_edge_index, nc_data):
    poc_graph, only_user_graph, only_poc_graph, both_graph = graphs
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    model.eval()
    out, s_loss = model(init_x,
                        global_edge_index,
                        poc_graph,
                        only_user_graph,
                        only_poc_graph,
                        both_graph)
    out = out.view(-1)
    val_loss = criterion(out[nc_data.val_mask], nc_data.y[nc_data.val_mask].float())
    acc, f1 = get_nc_metrics(nc_data, out.sigmoid())

    return val_loss, acc, f1


def nc_train(args, root_path,
             init_x, data_dict,
             global_edge_index, nc_data, graphs,
             is_pon=True):
    poc_graph, only_user_graph, only_poc_graph, both_graph = graphs
    model_path = root_path + '/src/checkpoints/' + args.dataset + '_nc_model.pt'
    model = POFD_NC(args.in_feats, args.hidden_feats, args.out_feats,
                    data_dict, heads=args.heads, activation_str='tanh', is_pon=is_pon).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    min_epochs = args.min_epochs
    best_model = None
    min_val_loss = np.Inf
    final_test_acc = 0
    final_test_f1 = 0
    epochs = args.epochs
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        out, s_loss = model(init_x,
                            global_edge_index,
                            poc_graph,
                            only_user_graph,
                            only_poc_graph,
                            both_graph)
        out = out.view(-1)
        loss = criterion(out[nc_data.train_mask], nc_data.y[nc_data.train_mask].float()) + args.mu * s_loss
        loss.backward()
        optimizer.step()
        # validation
        val_loss, test_acc, test_f1 = nc_test(model, init_x, graphs,
                                              global_edge_index, nc_data)
        val_loss += args.mu * s_loss
        val_losses.append(val_loss.item())
        train_losses.append(loss.item())
        # tr
        if epoch + 1 > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            final_test_acc = test_acc
            final_test_f1 = test_f1
            best_model = copy.deepcopy(model)
            # save model
            state = {'model': best_model.state_dict()}
            torch.save(state, model_path)

        # scheduler.step()
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print('epoch {:03d} train_loss {:.8f} val_loss {:.4f} test_acc {:.4f} test_f1 {:.4f}'
              .format(epoch, loss.item(), val_loss, test_acc, test_f1))

    return final_test_acc, final_test_f1


def get_nc_dblp_metrics(graph, nc_data, out):
    label = graph['author'].y[nc_data.test_mask].cpu().numpy()
    _, pred = out[nc_data.test_mask].max(dim=1)
    pred = pred.cpu().numpy()
    acc = accuracy_score(label, pred)
    ma_f1 = f1_score(label, pred, average='macro')
    mi_f1 = f1_score(label, pred, average='micro')

    return acc, ma_f1, mi_f1


@torch.no_grad()
def nc_dblp_test(model, init_x, global_edge_index,
                 graph, graphs, nc_data):
    model.eval()
    poc_graph, only_user_graph, only_poc_graph, both_graph = graphs
    out, _ = model(init_x,
                   global_edge_index,
                   poc_graph,
                   only_user_graph,
                   only_poc_graph,
                   both_graph)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    val_loss = criterion(out[nc_data.val_mask], graph['author'].y[nc_data.val_mask])

    acc, ma_f1, mi_f1 = get_nc_dblp_metrics(graph, nc_data, out)

    return val_loss, acc, ma_f1, mi_f1


def nc_dblp_train(args, root_path,
                  init_x, data_dict,
                  global_edge_index, graph,
                  nc_data, graphs,
                  is_pon=True):
    poc_graph, only_user_graph, only_poc_graph, both_graph = graphs
    model_path = root_path + '/src/checkpoints/' + args.dataset + '_nc_model.pt'
    model = POFD_DBLP(args.in_feats, args.hidden_feats, args.out_feats,
                      data_dict, heads=args.heads, activation_str='tanh',
                      is_pon=is_pon).to(args.device)
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    min_epochs = args.min_epochs
    best_model = None
    min_val_loss = np.Inf
    final_test_acc = 0
    final_test_ma_f1 = 0
    final_test_mi_f1 = 0
    epochs = args.epochs
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        out, s_loss = model(init_x,
                            global_edge_index,
                            poc_graph,
                            only_user_graph,
                            only_poc_graph,
                            both_graph)
        loss = criterion(out[nc_data.train_mask], graph['author'].y[nc_data.train_mask])
        loss = loss + args.mu * s_loss
        loss.backward()
        optimizer.step()
        # validation
        val_loss, test_acc, test_ma_f1, test_mi_f1 = nc_dblp_test(model, init_x, global_edge_index,
                                                                  graph, graphs, nc_data)
        val_loss = val_loss + args.mu * s_loss
        val_losses.append(val_loss.item())
        train_losses.append(loss.item())
        if epoch + 1 > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            final_test_acc = test_acc
            final_test_ma_f1 = test_ma_f1
            final_test_mi_f1 = test_mi_f1
            best_model = copy.deepcopy(model)
            # save model
            state = {'model': best_model.state_dict()}
            torch.save(state, model_path)

        scheduler.step()
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print('epoch {:03d} train_loss {:.8f} val_loss {:.4f} test_acc {:.4f} test_ma_f1 {:.4f} test_mi_f1 {:.4f}'
              .format(epoch, loss.item(), val_loss, test_acc, test_ma_f1, test_mi_f1))

    return final_test_acc, final_test_ma_f1, final_test_mi_f1
