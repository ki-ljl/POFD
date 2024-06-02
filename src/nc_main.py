# -*- coding:utf-8 -*-
import argparse
import os
import sys

import torch

from util import load_pickle, nc_train

root_path = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_path)
data_path = root_path + '/data/'


def main():
    test_acc, test_f1 = nc_train(
        args,
        root_path,
        init_x,
        data_dict,
        global_edge_index=homogeneous_graph.edge_index,
        nc_data=nc_data.to(args.device),
        graphs=(poc_graph, only_user_graph, only_poc_graph, both_graph)
    )
    print('final best acc:', test_acc)
    print('final best f1:', test_f1)


if __name__ == '__main__':
    # add args
    parser = argparse.ArgumentParser(description='pofd event classification')

    parser.add_argument('--dataset', type=str, default='BuzzFeed',
                        help='dataset name', choices=['BuzzFeed', 'PolitiFact'])
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--min_epochs', type=int, default=10, help='min training epochs')
    parser.add_argument('--in_feats', type=int, default=128, help='input dimension')
    parser.add_argument('--hidden_feats', type=int, default=128, help='hidden dimension')
    parser.add_argument('--out_feats', type=int, default=128, help='output dimension')
    parser.add_argument('--heads', type=int, default=4, help='attention heads')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--mu', type=float, default=0.01, help='control MI')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--step_size', type=int, default=50, help='step size')
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')

    args = parser.parse_args()
    # load data
    graph = load_pickle(data_path + args.dataset + '/hetero_graph.pkl')
    homogeneous_graph = graph.to_homogeneous().to(args.device)

    poc_graph = load_pickle(data_path + args.dataset + '/poc_graph.pkl').to(args.device)
    only_user_graph = load_pickle(data_path + args.dataset + '/only_user_graph.pkl').to(args.device)
    only_poc_graph = load_pickle(data_path + args.dataset + '/only_poc_graph.pkl').to(args.device)
    both_graph = load_pickle(data_path + args.dataset + '/both_graph.pkl').to(args.device)
    nc_data = load_pickle(data_path + args.dataset + '/nc_data.pkl').to(args.device)

    data_dict = load_pickle(data_path + args.dataset + '/data_dict.pkl')
    node_types = data_dict['node_types']
    init_x = [graph[x].x.to(args.device) for x in node_types]
    # run exp
    main()
