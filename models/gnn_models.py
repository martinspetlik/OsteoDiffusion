import copy

import torch
import torch.nn as nn
import dgl
import dgl.sparse as dglsp
from components import SinusoidalPosEmb


class GNNLayer(nn.Module):
    """
    Graph Neural Network (GNN) / Message Passing Neural Network (MPNN) Layer.
    """

    def __init__(self,
                 hidden_X,
                 hidden_t,
                 dropout):
        super().__init__()

        #print('hidden_X: {}, hidden_t: {}'.format(hidden_X, hidden_t))

        self.update_X = nn.Sequential(
            nn.Linear(hidden_X + hidden_t, hidden_X),
            nn.ReLU(),
            #nn.LayerNorm(hidden_X),
            #nn.Dropout(dropout)
        )

    def forward(self, A, h_X, h_t):
        """
        Parameters
        ----------
        A : dglsp.SparseMatrix
            Adjacency matrix.
        h_X : torch.Tensor of shape (|V|, hidden_X)
            Hidden representations for the node attributes.
        h_t : torch.Tensor of shape (|V|, hidden_t)
            Hidden representations for the normalized time step.

        Returns
        -------
        h_X : torch.Tensor of shape (|V|, hidden_X)
            Updated hidden representations for the node attributes.
        """
        # print("GNN Layer forward")
        #
        # print("A shape ", A.shape)
        # print("h_X shape ", h_X.shape)
        # print("h_t ", h_t.shape)

        h_aggr_X = dglsp.bspmm(A, h_X)  # A @ h_X for batch

        #h_aggr_X = A @ h_X
        #h_aggr_Y = A @ h_Y

        #print("h_aggr_X ", h_aggr_X.shape)

        num_nodes = h_X.size(0)
        h_t_expand = h_t.expand(num_nodes, -1)
        #print("h t expand shape", h_t_expand.shape)

        h_t_expand = h_t_expand.unsqueeze(-1).expand(*h_t_expand.shape, h_X.shape[-1])
        #print("h_t_expand ", h_t_expand)

        h_aggr_X = torch.cat([h_aggr_X, h_t_expand], dim=1).permute(2, 0, 1)

        h_X = self.update_X(h_aggr_X)
        #h_Y = self.update_Y(h_aggr_Y)

        h_X = h_X.permute(1,2,0)

        return h_X


class GNNTower(nn.Module):
    """Graph Neural Network (GNN) / Message Passing Neural Network (MPNN).

    Parameters
    ----------
    num_attrs_X : int
        Number of node attributes.
    hidden_t_embd : int
        Hidden size for the normalized time step.
    hidden_v_embd : int
        Hidden size for the node attributes.
    num_gnn_layers : int
        Number of GNN/MPNN layers.
    dropout : float
        Dropout rate.
    """
    def __init__(self,
                 num_attrs_X,
                 adj_matrix,
                 hidden_t_embd,
                 hidden_v_embd,
                 num_gnn_layers,
                 dropout,
                 sin_emb_theta=10000,
                 sin_emb_dim=64):
        super().__init__()

        #print("hidden_t_embd: {},  hidden_v_embd: {}".format(hidden_t_embd, hidden_v_embd))

        #in_X = num_attrs_X * adj_matrix.shape[0] #* num_classes_X
        self.num_attrs_X = num_attrs_X
        self.adj_matrix = adj_matrix
        self.orig_adj_matrix = adj_matrix

        sinu_pos_emb = SinusoidalPosEmb(sin_emb_dim, theta=sin_emb_theta)
        time_dim = hidden_t_embd #sin_emb_dim * 4
        self.mlp_in_t = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(sin_emb_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # self.mlp_in_t = nn.Sequential(
        #     nn.Linear(1, hidden_t_embd),
        #     nn.ReLU(),
        #     nn.Linear(hidden_t_embd, hidden_t_embd),
        #     nn.ReLU())

        # print("in X ", in_X)
        # self.mlp_in_X = nn.Sequential(
        #     nn.Linear(in_X, hidden_v_embd),
        #     nn.ReLU(),
        #     nn.Linear(hidden_v_embd, hidden_v_embd),
        #     nn.ReLU()
        # )

        #self.emb_Y = nn.Embedding(num_classes_Y, hidden_Y)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_v_embd,
                     hidden_t_embd,
                     dropout)
            for _ in range(num_gnn_layers)])

        # # +1 for the input attributes
        hidden_cat = (num_gnn_layers + 1) * (hidden_v_embd) + hidden_t_embd

        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_cat, hidden_cat),
            nn.ReLU(),
            nn.Linear(hidden_cat, num_attrs_X)
        )

    def forward(self, x, t):
        # print("GNNTower forward")
        # print("x ", x.shape)
        # print("t shape ", t.shape)

        batch_size, num_vertices, num_node_attrs = x.shape
        x = x.permute(1, 2, 0).float()  # (num_vertices, num_node_attrs, batch_size)

        # Expand adjacency matrix values to the batch size
        # print("self.adj_matrix.val.shape ", self.adj_matrix.val.shape)
        # print("x.shape ", x.shape)
        if len(self.adj_matrix.val.shape) == 1:
            expanded_vals = self.adj_matrix.val.unsqueeze(1).expand(self.adj_matrix.val.shape[0], batch_size).float()
            expanded_vals = expanded_vals.to(torch.device(x.device))
            self.adj_matrix = self.adj_matrix.to(torch.device(x.device))
            self.adj_matrix = dgl.sparse.val_like(self.adj_matrix, expanded_vals)
        elif self.adj_matrix.val.shape[1] != x.shape[-1]:
            new_vals = self.orig_adj_matrix.val.unsqueeze(1).expand(self.orig_adj_matrix.val.shape[0], x.shape[-1]).float()
            new_vals = new_vals.to(torch.device(x.device))

            # print("new vals device ", new_vals.device)
            # print("self.adj_matrix.device ", self.adj_matrix.device)

            # new_vals = self.adj_matrix.val[..., :x.shape[-1]]
            # print("new vals shape ", new_vals.shape)
            #
            # print("orig adj matrix val shape ", self.orig_adj_matrix.val.shape)
            # new_vals = new_vals.to(torch.device(x.device))
            self.adj_matrix = dgl.sparse.val_like(self.adj_matrix, new_vals)

            # print("self.adj_matrix.val.shape ", self.adj_matrix.val.shape)
            # print("x.shape ", x.shape)

        t = t.float()

        h_X = x #self.mlp_in_X(x)
        h_t = self.mlp_in_t(t)#.unsqueeze(0)

        h_X_list = [h_X]
        for gnn in self.gnn_layers:
            h_X = gnn(self.adj_matrix, h_X, h_t)
            h_X_list.append(h_X)


        ###
        # Crucial part - it was not learning properly without output MLP
        ###
        h_t = h_t.expand(h_X.size(0), -1)
        h_t = h_t.unsqueeze(-1).expand(*h_t.shape, batch_size)
        h_cat = torch.cat(h_X_list + [h_t], dim=1)
        h_cat = torch.squeeze(h_cat)

        #return h_X
        return self.mlp_out(h_cat)


class GNN(nn.Module):
    """P(X|Y, X^t, A^t) + P(A|Y, X^t, A^t)

    Parameters
    ----------
    num_attrs_X : int
        Number of node attributes.
    num_classes_X : int
        Number of classes for each node attribute.
    gnn_X_config : dict
        Configuration of the GNN for reconstructing node attributes.
    """
    def __init__(self,
                 num_node_attrs,
                 adj_matrix,
                 gnn_config):
        super().__init__()

        print("gnn config ", gnn_config)
        #
        # self.time_mlp = nn.Sequential(
        #     sinu_pos_emb,
        #     nn.Linear(dim, time_dim),
        #     nn.GELU(),
        #     nn.Linear(time_dim, time_dim),
        # )


        self.num_node_attrs = num_node_attrs
        self.adj_matrix = adj_matrix

        self.gnn_tower = GNNTower(num_node_attrs, adj_matrix, **gnn_config)

    def forward(self, x, t):
        gnn_pred = self.gnn_tower(x, t)
        #print("gnn pred ", gnn_pred.shape)
        return gnn_pred

