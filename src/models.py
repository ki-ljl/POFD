# -*- coding:utf-8 -*-

import torch.sparse as sparse
from torch_geometric.utils import add_self_loops, to_undirected

from infomax import Infomax

from torch_scatter import scatter

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.dropout = dropout

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  # FxF'
        self.attn = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))  # 2F'

        nn.init.xavier_normal_(self.W, gain=1.414)
        nn.init.xavier_normal_(self.attn, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def sparse_dropout(self, x: torch.Tensor, p: float, training: bool):
        x = x.coalesce()
        return torch.sparse_coo_tensor(x.indices(),
                                       F.dropout(x.values(),
                                                 p=p,
                                                 training=training), size=x.size())

    def forward(self, input, edge):
        """
        input: NxF
        edge: 2xE
        """
        N = input.size()[0]
        if input.is_sparse:
            h = torch.sparse.mm(input, self.W)  # (NxF) * (FxF') = NxF'
        else:
            h = torch.mm(input, self.W)

        # Self-attention (because including self edges) on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()  # Ex2F'.t() = 2F'xE
        values = edge_h.T.mm(self.attn).squeeze()  # E
        sp_edge_h = torch.sparse_coo_tensor(edge, -self.leakyrelu(values), size=(N, N))  # values() = E

        sp_edge_h = sparse.softmax(sp_edge_h, dim=1)
        # apply attention
        h_prime = torch.sparse.mm(sp_edge_h, h)  # (NxN) * (NxF') = (NxF')

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SparseGATConv(nn.Module):
    def __init__(self, in_features, out_features, heads,
                 dropout=.5, alpha=0.01, concat=True):
        super(SparseGATConv, self).__init__()
        self.attentions = torch.nn.ModuleList()
        for _ in range(heads):
            self.attentions.append(SpGraphAttentionLayer(in_features,
                                                         out_features,
                                                         dropout=dropout,
                                                         alpha=alpha,
                                                         concat=concat))
        self.out_att = SpGraphAttentionLayer(out_features * heads,
                                             out_features,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, edge_index):
        x = torch.cat([att(x, edge_index) for att in self.attentions], dim=1)
        x = self.out_att(x, edge_index)

        return x


def get_activation(activation_str):
    if activation_str == 'relu':
        return nn.ReLU()
    elif activation_str == 'sigmoid':
        return nn.Sigmoid()
    elif activation_str == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation_str == 'elu':
        return nn.ELU()
    elif activation_str == 'prelu':
        return nn.PReLU()
    elif activation_str == 'silu':
        return nn.SiLU()
    elif activation_str == 'gelu':
        return nn.GELU()
    elif activation_str == 'tanh':
        return nn.Tanh()
    elif activation_str == 'softplus':
        return nn.Softplus()
    elif activation_str == 'softsign':
        return nn.Softsign()
    else:
        raise ValueError("Unsupported activation function: " + activation_str)


class POFDConv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            data_dict,
            n_heads: int = 4,
            bias: bool = True,
            use_pofe_amplification=True,
            activation_str='relu',
            is_pon=True,
            **kwargs,
    ):
        """
        Args:
            in_channels (int): in_channels.
            out_channels (int): out_channels.
            data_dict (dict): data_dict.
            n_heads (int): gat attention heads.
            bias (bool): nn.Linear bias.
                            Default: True
            use_pofe_amplification (bool): whether to amplify pofe, used only in link prediction tasks
                            Default: True
            activation_str (str): type of activation function.
                            Default: 'relu'
            is_pon (bool): Is it a Public Opinion Network? Not applicable in the DBLP task.
                            Default: True
        """
        super(POFDConv, self).__init__()

        self.data_dict = data_dict
        self.use_pofe_amplification = use_pofe_amplification
        self.is_pon = is_pon
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.proj = nn.Linear(in_channels, out_channels, bias=bias)

        self.wij_att = nn.Parameter(torch.zeros(size=(1, 2 * self.in_channels)))
        self.only_poc_att = nn.Parameter(torch.zeros(size=(2 * self.out_channels, 1)))
        self.both_att_1 = nn.Parameter(torch.zeros(size=(2 * self.out_channels, 1)))
        self.both_att_2 = nn.Parameter(torch.zeros(size=(2 * self.out_channels, 1)))

        self.inf_w = nn.Linear(2 * out_channels, out_channels)

        self.pofe_amplification_factor = 3.0

        self.poc_gat_conv = SparseGATConv(in_channels, out_channels, heads=n_heads)
        self.only_user_gat_conv = SparseGATConv(out_channels, out_channels, heads=n_heads)
        self.social_gat_conv = SparseGATConv(out_channels, out_channels, heads=n_heads)

        self.activation = get_activation(activation_str)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Parameter):
                nn.init.xavier_uniform_(m.data, gain=1.414)

    def social_circle(self, x, global_edge_index):
        u_x = self.social_gat_conv(x, global_edge_index)
        u_x = torch.cat([u_x, x], dim=-1)
        u_x = self.inf_w(u_x)
        if self.is_pon:
            num_news = self.data_dict['num_news']
            num_users = self.data_dict['num_users']
            r = torch.cat((x[:num_news],
                           u_x[num_news:num_news + num_users],
                           x[num_news + num_users:]), dim=0)
            return r
        else:
            return u_x

    def poc_conv(self, global_x, poc_graph):
        # get poc nodes features
        x = global_x[poc_graph.nodes_idx]
        # \tau_{ij}
        e = torch.sigmoid(torch.cov(x))
        edge_index = poc_graph.edge_index
        covariance_vector = e[edge_index[0], edge_index[1]]
        # gat_conv w_ij
        edge_h = torch.cat([x[edge_index[0, :], :], x[edge_index[1, :], :]],
                           dim=1).t()
        e_w = torch.sigmoid(self.wij_att.mm(edge_h).squeeze())  #
        # \lambda_{ij}
        degree = scatter(torch.ones(edge_index.shape[1]).to(device), edge_index[0, :], dim=0,
                         reduce='sum')
        degree = torch.index_select(degree, 0, edge_index[0])

        e_w = e_w * degree * covariance_vector
        # scatter
        e_w = scatter(e_w, edge_index[1, :], dim=0, reduce='sum')
        # get E(P_i)
        pofe = e_w[:poc_graph.nodes_len]
        # poc_nodes = new_nodes + source_nodes + high_degree_user_nodes
        if self.use_pofe_amplification:
            pofe[self.data_dict["negative_news_index"]] = \
                pofe[self.data_dict["negative_news_index"]] * self.pofe_amplification_factor
        # normalization
        pofe = pofe / torch.sum(pofe, dim=-1)
        #
        x = self.poc_gat_conv(x, edge_index)
        x = self.activation(x)
        poc_x = x[:poc_graph.nodes_len]

        return poc_x, pofe, x, e_w

    def only_user_conv(self, global_x, only_user_graph):
        if only_user_graph.edge_index is None:
            return torch.randn(0, self.out_channels).to(device)
        else:
            x = global_x[only_user_graph.nodes_idx]
            x = self.only_user_gat_conv(x, only_user_graph.edge_index)
            x = self.activation(x)
            return x[:only_user_graph.nodes_len]

    def only_poc_conv(self, global_x, only_poc_graph, all_pofe, all_poc_x):
        if only_poc_graph.edge_index is None:
            return torch.randn(0, self.out_channels).to(device)
        # pofe
        poc_neighbor_pofe = all_pofe[only_poc_graph.poc_idx]  #
        poc_neighbor_x = all_poc_x[only_poc_graph.poc_idx]  #
        # init poc features
        init_poc_neighbor_x = global_x[only_poc_graph.global_poc_neighbor_idx]
        user_x = global_x[only_poc_graph.nodes_idx][:only_poc_graph.nodes_len]
        x = torch.cat([user_x, init_poc_neighbor_x], dim=0)  # user + poc
        # pofe normalization
        poc_neighbor_pofe = poc_neighbor_pofe / torch.sum(poc_neighbor_pofe, dim=-1)
        weight_poc_x = init_poc_neighbor_x + poc_neighbor_x * poc_neighbor_pofe.view(-1, 1)
        # sparse version gat
        N = x.size(0)
        edge_h = torch.cat((x[only_poc_graph.edge_index[0, :], :],
                            x[only_poc_graph.edge_index[1, :], :]), dim=1).t()  # Ex2F'.t() = 2F'xE
        values = edge_h.T.mm(self.only_poc_att).squeeze()  # E
        sp_edge_h = torch.sparse_coo_tensor(only_poc_graph.edge_index, -F.leaky_relu(values),
                                            size=(N, N))  # values() = E
        sp_edge_h = sparse.softmax(sp_edge_h, dim=1)
        # apply attention
        h_prime = torch.sparse.mm(sp_edge_h, torch.cat([user_x, weight_poc_x], dim=0))  # (NxN) * (NxF') = (NxF')
        h_prime = self.activation(h_prime)

        return h_prime[:only_poc_graph.nodes_len]

    def both_conv(self, global_x, both_graph, all_pofe, all_poc_x):
        if both_graph.edge_index is None:
            return torch.randn(0, self.out_channels).to(device)
        # user + poc + user
        poc_neighbor_len = both_graph.poc_idx.size(0)
        x = global_x[both_graph.nodes_idx]
        edge_index = both_graph.edge_index
        N = x.size(0)
        edge_h = torch.cat((x[edge_index[0, :], :],
                            x[edge_index[1, :], :]), dim=1).t()  # Ex2F'.t() = 2F'xE
        values_1 = edge_h.T.mm(self.both_att_1).squeeze()  # E
        values_2 = edge_h.T.mm(self.both_att_2).squeeze()  # E
        #
        # user-->poc
        poc_idx = (edge_index[0] < both_graph.nodes_len) & \
                  (both_graph.nodes_len <= edge_index[1]) & (edge_index[1] < both_graph.nodes_len + poc_neighbor_len)
        poc_idx = poc_idx.nonzero().flatten()
        values_1[poc_idx] = values_2[poc_idx]
        sp_edge_h = torch.sparse_coo_tensor(edge_index, -F.leaky_relu(values_1), size=(N, N))  # values() = E
        sp_edge_h = sparse.softmax(sp_edge_h, dim=1)
        # apply attention
        pofe = all_pofe[both_graph.poc_idx]
        pofe = pofe / torch.sum(pofe, dim=-1)
        init_poc_x = x[both_graph.nodes_len:both_graph.nodes_len + poc_neighbor_len, :]  #
        updated_poc_x = all_poc_x[both_graph.poc_idx]
        poc_x = init_poc_x + updated_poc_x * pofe.view(-1, 1)
        concat_x = torch.cat(
            [x[:both_graph.nodes_len], poc_x, x[both_graph.nodes_len + poc_neighbor_len:]],
            dim=0)
        h_prime = torch.sparse.mm(sp_edge_h, concat_x)
        h_prime = self.activation(h_prime)

        return h_prime[:both_graph.nodes_len]

    def forward(self,
                x,
                global_edge_index,
                poc_graph,
                only_user_graph,
                only_poc_graph,
                both_graph):
        # 1. projection
        init_x = self.proj(x)
        # 2. update poc nodes
        poc_x, pofe, all_poc_x, all_pofe = self.poc_conv(x, poc_graph)
        # ---------------------------------------------------------------------------
        # 3. update only_user_neighbor_users
        only_user_x = self.only_user_conv(init_x, only_user_graph)
        # --------------------------------------
        # 4. update only_poc_neighbor_users
        only_poc_x = self.only_poc_conv(init_x, only_poc_graph, all_pofe, all_poc_x)
        # -----------------------------
        # 5. update both_users
        both_x = self.both_conv(init_x, both_graph, all_pofe, all_poc_x)
        # 6. index
        update_x = torch.cat([poc_x, only_user_x, only_poc_x, both_x], dim=0)
        update_x = update_x[np.argsort(self.data_dict['sort_index'])]
        # 7. social circle influence
        update_x = self.social_circle(update_x, global_edge_index)

        return update_x


# POFD for link prediction task
class POFD_LP(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 data_dict,
                 heads=4,
                 is_pon=True,
                 activation_str='tanh'):
        super(POFD_LP, self).__init__()
        self.data_dict = data_dict
        self.conv1 = POFDConv(in_channels, hidden_channels, data_dict,
                              n_heads=heads, is_pon=is_pon, activation_str=activation_str)
        self.conv2 = POFDConv(hidden_channels, out_channels, data_dict,
                              n_heads=heads, is_pon=is_pon, activation_str=activation_str)
        self.sigmoid = nn.Sigmoid()
        self.info_max = Infomax(out_channels, out_channels)
        self.lins = torch.nn.ModuleList()
        for i in range(len(data_dict['node_types'])):
            lin = nn.Linear(data_dict['init_sizes'][i], in_channels)
            self.lins.append(lin)

        self.classifier = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)
        )

    def trans_dimensions(self, xs):
        res = []
        for x, lin in zip(xs, self.lins):
            res.append(lin(x))

        return torch.cat(res, dim=0)

    def encode(self,
               init_x,
               global_edge_index,
               poc_graph,
               only_user_graph,
               only_poc_graph,
               both_graph):
        x_0 = self.trans_dimensions(init_x)
        x = self.conv1(x_0,
                       global_edge_index,
                       poc_graph,
                       only_user_graph,
                       only_poc_graph,
                       both_graph)
        x = F.relu(x)
        x = self.conv2(x,
                       global_edge_index,
                       poc_graph,
                       only_user_graph,
                       only_poc_graph,
                       both_graph)
        # get MI Loss
        poc_graph_x = x[poc_graph.nodes_idx]
        poc_x = poc_graph_x[:poc_graph.nodes_len]
        edge_index, _ = add_self_loops(poc_graph.edge_index, num_nodes=poc_x.shape[0])
        edge_index = to_undirected(edge_index, num_nodes=poc_x.shape[0])
        row, col = edge_index
        new_poc_x = scatter(poc_graph_x[row], col,
                            dim=0, dim_size=poc_graph_x.shape[0],
                            reduce='mean')[:poc_graph.nodes_len]
        s_loss = self.info_max.get_loss(poc_x, new_poc_x)

        return x, s_loss

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        r = torch.cat((src, dst), dim=-1)
        r = self.classifier(r)
        return r

    def forward(self,
                init_x,
                global_edge_index,
                poc_graph,
                only_user_graph,
                only_poc_graph,
                both_graph,
                edge_label_index):
        z, s_loss = self.encode(init_x,
                                global_edge_index,
                                poc_graph,
                                only_user_graph,
                                only_poc_graph,
                                both_graph)

        return self.decode(z, edge_label_index), s_loss


# POFD for link node classification task
class POFD_NC(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 data_dict,
                 heads=4,
                 is_pon=True,
                 activation_str='relu',
                 reduce='max'):
        super(POFD_NC, self).__init__()
        self.is_pon = is_pon
        self.data_dict = data_dict
        self.reduce = reduce
        self.conv1 = POFDConv(in_channels, hidden_channels, data_dict,
                              n_heads=heads, use_pofe_amplification=False,
                              is_pon=is_pon, activation_str=activation_str)
        self.conv2 = POFDConv(hidden_channels, out_channels, data_dict,
                              n_heads=heads, use_pofe_amplification=False,
                              is_pon=is_pon, activation_str=activation_str)
        self.info_max = Infomax(out_channels, out_channels)

        self.lins = torch.nn.ModuleList()
        for i in range(len(data_dict['node_types'])):
            lin = nn.Linear(data_dict['init_sizes'][i], in_channels)
            self.lins.append(lin)

        self.classifier = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(out_channels, 1)
        )

    def trans_dimensions(self, xs):
        res = []
        for x, lin in zip(xs, self.lins):
            res.append(lin(x))

        return torch.cat(res, dim=0)

    def forward(self,
                init_x,
                global_edge_index,
                poc_graph,
                only_user_graph,
                only_poc_graph,
                both_graph):
        x_0 = self.trans_dimensions(init_x).to(device)
        x = self.conv1(x_0,
                       global_edge_index,
                       poc_graph,
                       only_user_graph,
                       only_poc_graph,
                       both_graph)
        x = F.elu(x)
        x = self.conv2(x,
                       global_edge_index,
                       poc_graph,
                       only_user_graph,
                       only_poc_graph,
                       both_graph)
        # get MI Loss
        poc_graph_x = x[poc_graph.nodes_idx]
        poc_x = poc_graph_x[:poc_graph.nodes_len]
        edge_index, _ = add_self_loops(poc_graph.edge_index, num_nodes=poc_x.shape[0])
        edge_index = to_undirected(edge_index, num_nodes=poc_x.shape[0])
        row, col = edge_index
        new_poc_x = scatter(poc_graph_x[row], col,
                            dim=0, dim_size=poc_graph_x.shape[0],
                            reduce=self.reduce)[:poc_graph.nodes_len]

        s_loss = self.info_max.get_loss(poc_x, new_poc_x)
        # classification
        x = F.elu(x)
        x_news = x[:self.data_dict['num_news']]  # all news
        # x_source = x[self.data_dict['num_news'] + self.data_dict['num_users']:]  # all source
        # c_x = x_news.clone()
        # c_x[self.data_dict['ns_edge_index'][0, :]] = x_source[self.data_dict['ns_edge_index'][1, :]]
        # x_news = torch.cat([x_news, c_x], dim=-1)
        x = self.classifier(x_news)

        return x, s_loss


# POFD for DBLP node classification task
class POFD_DBLP(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 data_dict,
                 heads=4,
                 is_pon=False,
                 activation_str='relu',
                 reduce='max'):
        super(POFD_DBLP, self).__init__()
        self.is_pon = is_pon
        self.data_dict = data_dict
        self.reduce = reduce
        self.conv1 = POFDConv(in_channels, hidden_channels, data_dict,
                              n_heads=heads, use_pofe_amplification=False,
                              is_pon=is_pon, activation_str=activation_str)
        self.conv2 = POFDConv(hidden_channels, out_channels, data_dict,
                              n_heads=heads, use_pofe_amplification=False,
                              is_pon=is_pon, activation_str=activation_str)
        self.info_max = Infomax(out_channels, out_channels)

        self.lins = torch.nn.ModuleList()
        for i in range(len(data_dict['node_types'])):
            lin = nn.Linear(data_dict['init_sizes'][i], in_channels)
            self.lins.append(lin)

        self.classifier = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2),
            nn.ReLU(),
            nn.Linear(out_channels // 2, 4)
        )

    def trans_dimensions(self, xs):
        res = []
        for x, lin in zip(xs, self.lins):
            res.append(lin(x))

        return torch.cat(res, dim=0)

    def forward(self,
                init_x,
                global_edge_index,
                poc_graph,
                only_user_graph,
                only_poc_graph,
                both_graph):
        x_0 = self.trans_dimensions(init_x)
        x = self.conv1(x_0,
                       global_edge_index,
                       poc_graph,
                       only_user_graph,
                       only_poc_graph,
                       both_graph)
        x = F.elu(x)
        x = self.conv2(x,
                       global_edge_index,
                       poc_graph,
                       only_user_graph,
                       only_poc_graph,
                       both_graph)
        # get MI Loss
        poc_graph_x = x[poc_graph.nodes_idx]
        poc_x = poc_graph_x[:poc_graph.nodes_len]
        edge_index, _ = add_self_loops(poc_graph.edge_index, num_nodes=poc_x.shape[0])
        edge_index = to_undirected(edge_index, num_nodes=poc_x.shape[0])
        row, col = edge_index
        new_poc_x = scatter(poc_graph_x[row], col,
                            dim=0, dim_size=poc_graph_x.shape[0],
                            reduce=self.reduce)[:poc_graph.nodes_len]
        s_loss = self.info_max.get_loss(poc_x, new_poc_x)
        # classification
        x = F.elu(x)
        x_authors = x[:self.data_dict['num_authors']]
        x_authors = self.classifier(x_authors)

        return x_authors, s_loss
