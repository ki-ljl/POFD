# -*- coding:utf-8 -*-

import argparse
import os
import sys

import torch

from util import device, load_pickle, get_data_edge_index, lp_train

root_path = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_path)
data_path = root_path + '/data/'


def main():
    print('Handling edge_index to prevent data leakage...')
    print('train data...')
    train_datas = get_data_edge_index(
        poc_graph,
        only_user_graph,
        only_poc_graph,
        both_graph,
        homogeneous_graph,
        train_graph.edge_index
    )
    train_datas = list(train_datas)
    train_datas = [x.to(device) for x in train_datas]

    print('val data...')
    val_datas = get_data_edge_index(
        poc_graph,
        only_user_graph,
        only_poc_graph,
        both_graph,
        homogeneous_graph,
        val_graph.edge_index
    )
    val_datas = list(val_datas)
    val_datas = [x.to(device) for x in val_datas]

    print('test data...')
    test_datas = get_data_edge_index(
        poc_graph,
        only_user_graph,
        only_poc_graph,
        both_graph,
        homogeneous_graph,
        test_graph.edge_index
    )
    test_datas = list(test_datas)
    test_datas = [x.to(device) for x in test_datas]

    test_auc, test_ap = lp_train(
        args,
        root_path,
        init_x,
        data_dict,
        graphs=(train_graph.to(device), val_graph.to(device), test_graph.to(device)),
        datas=(train_datas, val_datas, test_datas)
    )
    print('final best auc:', test_auc)
    print('final best ap:', test_ap)


if __name__ == '__main__':
    # add args
    parser = argparse.ArgumentParser(description='pofd public opinion concern prediction')

    parser.add_argument('--dataset', type=str, default='BuzzFeed',
                        help='dataset name', choices=['BuzzFeed', 'PolitiFact'])
    parser.add_argument('--epochs', type=int, default=200, help='training epochs')
    parser.add_argument('--min_epochs', type=int, default=10, help='min training epochs')
    parser.add_argument('--in_feats', type=int, default=128, help='input dimension')
    parser.add_argument('--hidden_feats', type=int, default=256, help='hidden dimension')
    parser.add_argument('--out_feats', type=int, default=128, help='output dimension')
    parser.add_argument('--heads', type=int, default=4, help='attention heads')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--mu', type=float, default=0.1, help='control MI')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--step_size', type=int, default=50, help='step size')
    parser.add_argument('--patience', type=int, default=50, help='early stopping patience')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')

    args = parser.parse_args()
    # load data
    graph = load_pickle(data_path + args.dataset + '/hetero_graph.pkl')
    homogeneous_graph = graph.to_homogeneous()

    poc_graph = load_pickle(data_path + args.dataset + '/poc_graph.pkl')
    only_user_graph = load_pickle(data_path + args.dataset + '/only_user_graph.pkl')
    only_poc_graph = load_pickle(data_path + args.dataset + '/only_poc_graph.pkl')
    both_graph = load_pickle(data_path + args.dataset + '/both_graph.pkl')

    train_graph = load_pickle(data_path + args.dataset + '/train_data.pkl')
    val_graph = load_pickle(data_path + args.dataset + '/val_data.pkl')
    test_graph = load_pickle(data_path + args.dataset + '/test_data.pkl')
    data_dict = load_pickle(data_path + args.dataset + '/data_dict.pkl')

    node_types = data_dict['node_types']
    init_x = [graph[x].x.to(device) for x in node_types]
    # run exp
    main()
