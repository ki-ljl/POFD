# -*- coding:utf-8 -*-

import pickle

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, k_hop_subgraph, to_undirected, add_self_loops


def save_pickle(dataset, file_name):
    f = open(file_name, "wb")
    pickle.dump(dataset, f, protocol=4)
    f.close()


def load_pickle(file_name):
    f = open(file_name, "rb+")
    dataset = pickle.load(f)
    f.close()
    return dataset


def get_data(path):
    graph = load_pickle(path)
    return graph


def get_sub_graph(node_idx, num_nodes,
                  global_edge_index,
                  poc_dict=None, poc_nodes=None,
                  flag=None):
    if len(node_idx) == 0:
        return Data(nodes_idx=None, edge_index=None)
    #
    subset, graph_edge_index, _, _ = k_hop_subgraph(
        node_idx=node_idx, num_hops=1,
        edge_index=global_edge_index,
        num_nodes=num_nodes,
        relabel_nodes=False
    )
    subset = subset.numpy()
    # reset index
    # poc_graph: poc + user_neighbors
    # only_user_graph: user + user_neighbors
    # only_poc_graph: user + poc_neighbors
    # both_graph: user + poc_neighbors + user_neighbors
    if flag == "only_poc_graph":
        graph_nodes = node_idx + list(set(subset) - set(node_idx))
        global_poc_neighbor_idx = torch.tensor(graph_nodes[len(node_idx):])
        poc_idx = [poc_dict[x] for x in graph_nodes[len(node_idx):]]
        poc_idx = torch.tensor(poc_idx)
    elif flag == "both_graph":
        neighbor_nodes = list(set(subset) - set(node_idx))
        poc_neighbors = list(set(neighbor_nodes) & set(poc_nodes))
        user_neighbors = list(set(neighbor_nodes) - set(poc_neighbors))
        graph_nodes = node_idx + poc_neighbors + user_neighbors
        global_poc_neighbor_idx = torch.tensor(poc_neighbors)
        poc_idx = [poc_dict[x] for x in poc_neighbors]
        poc_idx = torch.tensor(poc_idx)
    else:
        graph_nodes = node_idx + list(set(subset) - set(node_idx))
        poc_idx = None
        global_poc_neighbor_idx = None
    # global index to local index
    global_to_local_nodes_dict = dict(zip(graph_nodes, [x for x in range(len(graph_nodes))]))
    graph_edge_index = graph_edge_index.T.cpu().numpy().tolist()
    graph_edge_index = [
        [global_to_local_nodes_dict[x[0]],
         global_to_local_nodes_dict[x[1]]] for x in graph_edge_index
    ]
    graph_edge_index = torch.tensor(graph_edge_index).T

    sub_graph = Data(
        nodes_idx=torch.tensor(graph_nodes),
        edge_index=graph_edge_index,
        global_to_local_nodes_dict=global_to_local_nodes_dict,
        nodes_len=len(node_idx),
        poc_idx=poc_idx,
        global_poc_neighbor_idx=global_poc_neighbor_idx
    )
    return sub_graph


def generate_datas(path, dataset):
    # 1. load hetero graph
    graph = load_pickle(path)
    print(graph)
    num_news = graph['news'].x.shape[0]
    num_users = graph['user'].x.shape[0]
    num_sources = graph['source'].x.shape[0]
    num_nodes = num_news + num_users + num_sources
    news_nodes = [x for x in range(num_news)]
    user_nodes = [x for x in range(num_news, num_news + num_users)]
    source_nodes = [x for x in range(num_news + num_users, num_news + num_users + num_sources)]
    # 2. get poc_nodes
    poc_nodes = news_nodes + source_nodes
    # high influential users
    homogeneous_graph = graph.to_homogeneous()
    G = to_networkx(homogeneous_graph)
    degrees = [G.degree(idx) for idx in range(num_news, num_news + num_users)]
    nums = 500
    high_degree_users = np.argsort(-np.array(degrees))[:nums]
    high_degree_users = [x + num_news for x in high_degree_users]
    poc_nodes.extend(high_degree_users)
    user_nodes = list((set(user_nodes) - set(high_degree_users)))
    # 3. categorize the nodes
    only_user_neighbor_users = []
    only_poc_neighbor_users = []
    both_poc_and_user_neighbor_users = []
    for user_id in user_nodes:
        neighbor = list(nx.neighbors(G, user_id))
        if set(neighbor) < set(user_nodes):
            only_user_neighbor_users.append(user_id)
        elif set(neighbor) < set(poc_nodes):
            only_poc_neighbor_users.append(user_id)
        else:
            both_poc_and_user_neighbor_users.append(user_id)

    print(len(user_nodes), len(only_user_neighbor_users),
          len(only_poc_neighbor_users), len(both_poc_and_user_neighbor_users))
    # 4. get sub_graph
    global_edge_index = to_undirected(homogeneous_graph.edge_index, num_nodes=num_nodes)
    global_edge_index, _ = add_self_loops(global_edge_index, num_nodes=num_nodes)
    # 4.1. poc graph
    poc_graph = get_sub_graph(node_idx=poc_nodes,
                              num_nodes=num_nodes,
                              global_edge_index=global_edge_index,
                              flag="poc_graph")
    # 4.2. only_user_graph
    only_user_graph = get_sub_graph(node_idx=only_user_neighbor_users,
                                    num_nodes=num_nodes,
                                    global_edge_index=global_edge_index,
                                    flag="only_user_graph")
    # 4.3. only_poc_graph
    only_poc_graph = get_sub_graph(node_idx=only_poc_neighbor_users,
                                   num_nodes=num_nodes,
                                   global_edge_index=global_edge_index,
                                   poc_dict=poc_graph.global_to_local_nodes_dict,
                                   poc_nodes=poc_nodes,
                                   flag="only_poc_graph")
    # 4.4. both_graph
    both_graph = get_sub_graph(node_idx=both_poc_and_user_neighbor_users,
                               num_nodes=num_nodes,
                               global_edge_index=global_edge_index,
                               poc_dict=poc_graph.global_to_local_nodes_dict,
                               poc_nodes=poc_nodes,
                               flag="both_graph")
    # 5. get data_dict
    node_types, edge_types = graph.metadata()
    num_relations = len(edge_types)
    init_sizes = [graph[x].x.shape[1] for x in node_types]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # negative_news_index: negative event position index in poc_nodes
    data_dict = {
        "num_news": num_news,
        "num_sources": num_sources,
        "num_users": num_users,
        "num_nodes": num_nodes,
        "sort_index": torch.tensor(poc_nodes + only_user_neighbor_users
                                   + only_poc_neighbor_users
                                   + both_poc_and_user_neighbor_users),
        "negative_news_index": torch.tensor([x for x in range(int(num_news / 2), num_news)]),
        "init_sizes": init_sizes,
        "node_types": node_types,
        "ns_edge_index": graph['news', 'to', 'source'].edge_index.to(device)
    }
    # 6. saving datasets
    save_pickle(poc_graph, 'data/' + dataset + '/poc_graph.pkl')
    save_pickle(only_user_graph, 'data/' + dataset + '/only_user_graph.pkl')
    save_pickle(only_poc_graph, 'data/' + dataset + '/only_poc_graph.pkl')
    save_pickle(both_graph, 'data/' + dataset + '/both_graph.pkl')
    save_pickle(data_dict, 'data/' + dataset + '/data_dict.pkl')


def generate_dblp_datas(path, dataset="DBLP"):
    # 1. load hetero graph
    graph = load_pickle(path)
    print(graph)
    num_authors = graph['author'].x.shape[0]
    num_papers = graph['paper'].x.shape[0]
    num_terms = graph['term'].x.shape[0]
    num_confs = graph['conference'].num_nodes
    num_nodes = num_authors + num_papers + num_terms + num_confs
    author_nodes = [x for x in range(num_authors)]
    paper_nodes = [x for x in range(num_authors, num_authors + num_papers)]
    term_nodes = [x for x in range(num_authors + num_papers, num_authors + num_papers + num_terms)]
    conf_nodes = [x for x in range(num_authors + num_papers + num_terms, num_nodes)]
    # 2. get poc_nodes
    poc_nodes = conf_nodes   # poc nodes = conference nodes
    user_nodes = author_nodes + paper_nodes + term_nodes
    homogeneous_graph = graph.to_homogeneous()
    G = to_networkx(homogeneous_graph)
    # 3. categorize the nodes
    only_user_neighbor_users = []
    only_poc_neighbor_users = []
    both_poc_and_user_neighbor_users = []
    for user_id in user_nodes:
        neighbor = list(nx.neighbors(G, user_id))
        if set(neighbor) < set(user_nodes):
            only_user_neighbor_users.append(user_id)
        elif set(neighbor) < set(poc_nodes):
            only_poc_neighbor_users.append(user_id)
        else:
            both_poc_and_user_neighbor_users.append(user_id)

    print(len(user_nodes), len(only_user_neighbor_users),
          len(only_poc_neighbor_users), len(both_poc_and_user_neighbor_users))
    # 4. get sub_graph
    global_edge_index = to_undirected(homogeneous_graph.edge_index, num_nodes=num_nodes)
    global_edge_index, _ = add_self_loops(global_edge_index, num_nodes=num_nodes)
    # 4.1. poc graph
    poc_graph = get_sub_graph(node_idx=poc_nodes,
                              num_nodes=num_nodes,
                              global_edge_index=global_edge_index,
                              flag="poc_graph")
    # 4.2. only_user_graph
    only_user_graph = get_sub_graph(node_idx=only_user_neighbor_users,
                                    num_nodes=num_nodes,
                                    global_edge_index=global_edge_index,
                                    flag="only_user_graph")
    # 4.3. only_poc_graph
    only_poc_graph = get_sub_graph(node_idx=only_poc_neighbor_users,
                                   num_nodes=num_nodes,
                                   global_edge_index=global_edge_index,
                                   poc_dict=poc_graph.global_to_local_nodes_dict,
                                   poc_nodes=poc_nodes,
                                   flag="only_poc_graph")
    # 4.4. both_graph
    both_graph = get_sub_graph(node_idx=both_poc_and_user_neighbor_users,
                               num_nodes=num_nodes,
                               global_edge_index=global_edge_index,
                               poc_dict=poc_graph.global_to_local_nodes_dict,
                               poc_nodes=poc_nodes,
                               flag="both_graph")
    # 5. get data_dict
    node_types, edge_types = graph.metadata()
    num_relations = len(edge_types)
    graph['conference'].x = torch.randn((graph['conference'].num_nodes, 128))
    init_sizes = [graph[x].x.shape[1] for x in node_types]
    data_dict = {
        "num_authors": num_authors,
        "num_papers": num_papers,
        "num_terms": num_terms,
        "num_confs": num_confs,
        "num_nodes": num_nodes,
        "sort_index": torch.tensor(poc_nodes + only_user_neighbor_users
                                   + only_poc_neighbor_users
                                   + both_poc_and_user_neighbor_users),
        "init_sizes": init_sizes,
        "node_types": node_types
    }
    # 6. saving datasets
    save_pickle(poc_graph, 'data/' + dataset + '/poc_graph.pkl')
    save_pickle(only_user_graph, 'data/' + dataset + '/only_user_graph.pkl')
    save_pickle(only_poc_graph, 'data/' + dataset + '/only_poc_graph.pkl')
    save_pickle(both_graph, 'data/' + dataset + '/both_graph.pkl')
    save_pickle(data_dict, 'data/' + dataset + '/data_dict.pkl')


if __name__ == '__main__':
    generate_datas(path="data/BuzzFeed/hetero_graph.pkl", dataset="Buzzfeed")
