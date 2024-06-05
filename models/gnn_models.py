import torch
import torch.nn as nn
# import dgl
# import dgl.sparse as dglsp
from models.components import SinusoidalPosEmb
from torch_geometric.nn import ChebConv
from torch_geometric.nn.models import GraphUNet
from torch_geometric.data import Data


class GNNLayerChebConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 K,
                 hidden_t):
        super().__init__()

        #print('hidden_X: {}, hidden_t: {}'.format(hidden_X, hidden_t))

        self.chebconv = ChebConv(in_channels=in_channels, out_channels=out_channels, K=K)

        #self.edge_indices = edge_indices

        self.update_X = nn.Sequential(
            nn.Linear(out_channels + hidden_t, out_channels),
            nn.ReLU(),
            #nn.LayerNorm(hidden_X),
            #nn.Dropout(dropout)
        )

    def forward(self, A, data, h_t, batch):
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
        # #print("h_X shape ", h_X.shape)
        # print("h_t ", h_t.shape)
        #
        # print("data ", data)
        #
        # #print("A ", type(A))

        h_X = data.x.float()
        #print("h_x.shape ", h_X.shape)
        edge_index = data.edge_index

        h_aggr_X = self.chebconv(h_X, edge_index, batch=batch)
        #print("h_aggr_X.shape ", h_aggr_X.shape)

        #h_aggr_X = dglsp.bspmm(A, h_X)  # A @ h_X for batch

        #h_aggr_X = A @ h_X
        #h_aggr_Y = A @ h_Y

        h_aggr_X = h_aggr_X.unsqueeze(-1)

        #print("h_aggr_X ", h_aggr_X.shape)
        # exit()


        num_nodes = h_X.size(0)
        h_t_expand = h_t.expand(num_nodes, -1)

        h_t_expand = h_t_expand.unsqueeze(-1).expand(*h_t_expand.shape, h_aggr_X.shape[-1])
        #print("h_t_expand ", h_t_expand)
        #print("h t expand shape", h_t_expand.shape)

        h_aggr_X = torch.cat([h_aggr_X, h_t_expand], dim=1).permute(2, 0, 1)

        h_X = self.update_X(h_aggr_X)
        #h_Y = self.update_Y(h_aggr_Y)

        h_X = h_X.permute(1,2,0)

        return Data(x=torch.squeeze(h_X.float(), dim=-1), edge_index=edge_index)
        #return h_X

class GNNLayer(nn.Module):
    """
    Graph Neural Network (GNN) / Message Passing Neural Network (MPNN) Layer.
    """

    def __init__(self,
                 hidden_X,
                 hidden_t,
                 dropout, adj_matrix, edge_indices):
        super().__init__()

        #print('hidden_X: {}, hidden_t: {}'.format(hidden_X, hidden_t))

        self.update_X = nn.Sequential(
            nn.Linear(hidden_X + hidden_t, hidden_X),
            nn.ReLU(),
            #nn.LayerNorm(hidden_X),
            #nn.Dropout(dropout)
        )

    def forward(self, A, data, h_t):
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
        # print("h_X.shape ", h_X.shape)
        # print("A type ", type(A))

        # h_aggr_X = torch.einsum('ij,jfb->ifb', A, h_X)

        h_X = data.x

        h_aggr_X = torch.sparse.mm(A, h_X) #A.matmul(h_X)

        #h_aggr_X = dglsp.bspmm(A, h_X)  # A @ h_X for batch

        #h_aggr_X = A @ h_X
        #h_aggr_Y = A @ h_Y

        #print("h_aggr_X ", h_aggr_X.shape)
        # exit()

        num_nodes = h_X.size(0)
        h_t_expand = h_t.expand(num_nodes, -1)
        #print("h t expand shape", h_t_expand.shape)

        #h_t_expand = h_t_expand.unsqueeze(-1).expand(*h_t_expand.shape, h_X.shape[-1])
        #print("h_t_expand shape final", h_t_expand.shape)

        h_aggr_X = torch.cat([h_aggr_X, h_t_expand], dim=1)#.permute(2, 0, 1)

        #print("h_aggr_X.shape ", h_aggr_X.shape)

        h_X = self.update_X(h_aggr_X)
        #h_Y = self.update_Y(h_aggr_Y)

        #h_X = h_X.permute(1,2,0)

        # print("h_X.shape ", h_X.shape)
        # exit()

        return Data(x=torch.squeeze(h_X.float(), dim=-1), edge_index=data.edge_index)

        #return h_X


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
                 edge_indices,
                 hidden_t_embd,
                 hidden_v_embd,
                 num_gnn_layers,
                 dropout,
                 gnn_layers_config,
                 sin_emb_theta=10000,
                 sin_emb_dim=64):
        super().__init__()

        self.gnn_layers = nn.ModuleList()

        #print("hidden_t_embd: {},  hidden_v_embd: {}".format(hidden_t_embd, hidden_v_embd))

        #in_X = num_attrs_X * adj_matrix.shape[0] #* num_classes_X
        self.num_attrs_X = num_attrs_X
        self.adj_matrix = adj_matrix
        self.edge_indcies = edge_indices
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

        hidden_cat_gnn = num_attrs_X

        for layer_config in gnn_layers_config:
            if layer_config["name"] == "GraphUNet":
                gnn_layer = GraphUNet(in_channels=layer_config["in_channels"],
                                      hidden_channels=layer_config["hidden_channels"],
                                      out_channels=layer_config["out_channels"], depth=layer_config["depth"])
            elif layer_config["name"] == "GNNLayer":
                gnn_layer = GNNLayer(hidden_v_embd, hidden_t_embd, dropout, adj_matrix, edge_indices)
            elif layer_config["name"] == "GNNLayerChebConv":
                gnn_layer = GNNLayerChebConv(in_channels=layer_config["in_channels"],
                                             out_channels=layer_config["out_channels"],
                                             K=layer_config["K"],
                                             hidden_t=hidden_t_embd)

                hidden_cat_gnn += layer_config["out_channels"]

            self.gnn_layers.append(gnn_layer)

            # self.gnn_layers = nn.ModuleList([
            #     gnn_layer_class(hidden_v_embd,
            #              hidden_t_embd,
            #              dropout, adj_matrix, edge_indices)
            #     for _ in range(num_gnn_layers)])

        # # +1 for the input attributes
        #num_gnn_layers = len(gnn_layers_config)
        hidden_cat = hidden_cat_gnn + hidden_t_embd #(num_gnn_layers + 1) * (hidden_v_embd) + hidden_t_embd

        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_cat, hidden_cat),
            nn.ReLU(),
            nn.Linear(hidden_cat, num_attrs_X)
        )

    def forward(self, data, t):
        # print("GNNTower forward")
        t = t.float()
        #h_X = x #self.mlp_in_X(x)
        h_X = Data(x=data.x.float(), edge_index=data.edge_index)
        h_t = self.mlp_in_t(t)#.unsqueeze(0)

        h_X_list = [h_X.x.float()]
        for gnn in self.gnn_layers:
            #print("data.batch ", data.batch)

            #h_X = gnn(h_X.x.float(), h_X.edge_index, batch=data.batch)
            #h_X = Data(x=h_X.float(), edge_index=data.edge_index)

            h_X = gnn(self.adj_matrix, h_X, h_t, batch=data.batch)

            #print("h_X from UNet ", h_X.x.shape)

            #h_X_list.append(h_X.x.float())
            #print("h_X.x.float().shape ", h_X.x.float().shape)
            h_X_list.append(h_X.x.float())


        ###
        # Crucial part - it was not learning properly without output MLP
        ###
        h_t = h_t.expand(h_X.x.size(0), -1)

        #print("h_X_list ", h_X_list)

        #print("h_t.shape ", h_t.shape)
        #h_t = h_t.unsqueeze(-1).expand(*h_t.shape, batch_size)
        #h_t = h_t.unsqueeze(-1).expand(*h_t.shape)
        h_cat = torch.cat(h_X_list + [h_t], dim=1)
        h_cat = torch.squeeze(h_cat)

        #print("h_cat.dtype ", h_cat.dtype)

        #print("h_cat shape ", h_cat.shape)


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
                 edge_indices,
                 gnn_config):
        super().__init__()
        self._name = "GNN_message_passing"

        # self.time_mlp = nn.Sequential(
        #     sinu_pos_emb,
        #     nn.Linear(dim, time_dim),
        #     nn.GELU(),
        #     nn.Linear(time_dim, time_dim),
        # )

        self.num_node_attrs = num_node_attrs
        self.adj_matrix = adj_matrix

        self.gnn_tower = GNNTower(num_node_attrs, adj_matrix, edge_indices, **gnn_config)

    def forward(self, x, t):
        gnn_pred = self.gnn_tower(x, t)
        #print("gnn pred ", gnn_pred.shape)
        return gnn_pred

