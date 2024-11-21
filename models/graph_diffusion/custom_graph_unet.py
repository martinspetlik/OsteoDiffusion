from typing import Callable, List, Union

import torch
from torch import Tensor
import torch.nn as nn
from einops import rearrange


from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_torch_csr_tensor,
)
from torch_geometric.utils.repeat import repeat

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mult=2,
        time_embedding_dim=None,
        norm=True,
        group=8,
    ):
        super().__init__()
        self.mlp = (
            #nn.Sequential(nn.GELU(), nn.Linear(time_embedding_dim, in_channels))
            nn.Sequential(nn.Linear(time_embedding_dim, in_channels))
            if time_embedding_dim
            else None
        )

        print("ConvBlock in channels ", in_channels)


        self.in_conv = GCNConv(in_channels, in_channels, improved=True)

        # self.block = nn.Sequential(
        #     nn.GroupNorm(1, in_channels) if norm else nn.Identity(),
        #     GCNConv(in_channels, hidden_channels, improved=True),
        #     nn.GELU(),
        #     nn.GroupNorm(1, out_channels * mult),
        #     GCNConv(out_channels * mult, out_channels, improved=True),
        # )

        #self.block_groupnorm_1 = nn.GroupNorm(1, in_channels) if norm else nn.Identity()
        self.block_conv_1 = GCNConv(in_channels, out_channels * mult, improved=True)
        self.block_gelu = nn.GELU()
        #self.block_groupnorm_2 = nn.GroupNorm(1, out_channels * mult)
        self.block_conv_2 = GCNConv(out_channels * mult, out_channels, improved=True)

    def forward(self, x, t, edge_index, edge_weight):
        # print("before in conv x shape ", x.shape)
        # print("time embedding shape ", time_embedding.shape)
        # print("in channels : {}, out channels: {}".format(self.in_channels, self.out_channels))
        print("x.shape ", x.shape)
        print("self.in_conv ", self.in_conv.in_channels)
        print("self.block_conv_1.in channels ", self.block_conv_1.in_channels)

        h = self.in_conv(x, edge_index, edge_weight)
        #h = x
        print("h ", h.shape)
        print("h ", h)
        print("t ", t.shape)
        # print("in conv h shape ", h.shape)
        # print("h ", h)
        print("t ", t)

        t = self.mlp(t)

        h = h + t
        #exit()


        # if self.mlp is not None and t is not None:
        #     assert self.mlp is not None, "MLP is None"
        #     print("self.mlp(time_embedding).shape ", self.mlp(t).shape)
        #     print("rearrange(self.mlp(time_embedding).shape ", rearrange(self.mlp(t), "b c -> b c 1 1").shape)
        #     h = h + rearrange(self.mlp(t), "b c -> b c 1 1")
        # print("h shape before block ", h.shape)
        # print("h ", h)

        #h = self.block_groupnorm_1(h)
        print("block groupnorm 1 shape", h.shape)
        h = self.block_conv_1(h, edge_index, edge_weight)
        h = self.block_gelu(h)
        print("block conv 1 shape ", h.shape)
        #h = self.block_groupnorm_2(h)
        print("block groupnorm 2 shape ", h.shape)
        h = self.block_conv_2(h, edge_index, edge_weight)

        print("self.block_conv_2.in channels ", self.block_conv_2.in_channels)
        print("self.block_conv_2.out channels ", self.block_conv_2.out_channels)
        print("block conv 2 shape ", h.shape)

        #print("self.residual_conv(x) ", self.residual_conv(x).shape)


        #h = self.block(h)
        print("after block shape ", h.shape)

        return h #+ self.residual_conv(x)


class GraphUNet(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        depth: int,
        hidden_t_embd: int,
        pool_ratios: Union[float, List[float]] = 0.5,
        sum_res: bool = True,
        act: Union[str, Callable] = 'relu',
    ):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = activation_resolver(act)
        self.sum_res = sum_res

        #channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.up_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()

        #in_out = [(16, 32), (32, 64), (64, 128)]
        in_out = [(16, 16), (16, 16), (16, 16)]

        self.pool_ratios = repeat(pool_ratios, len(in_out))

        self.init_conv = GCNConv(in_channels, hidden_t_embd, improved=True)
        self.final_conv = GCNConv(out_channels, in_channels, improved=True)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            print("down convs channels in: {}, out: {}".format(dim_in, dim_out))
            self.down_convs.append(
                ConvBlock(in_channels=dim_in, out_channels=dim_out, time_embedding_dim=hidden_t_embd))

            self.pools.append(TopKPooling(dim_out, self.pool_ratios[ind]))

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            print("up convs channels in: {}, out: {}".format(dim_out, dim_in))
            self.up_convs.append(
                ConvBlock(in_channels=dim_out, out_channels=dim_in, time_embedding_dim=hidden_t_embd))


        #self.down_convs.append(GCNConv(in_channels, channels, improved=True))



        # self.down_convs.append(ConvBlock(in_channels=hidden_t_embd, hidden_channels=hidden_channels, out_channels=hidden_channels, time_embedding_dim=hidden_t_embd))
        #
        # for i in range(depth):
        #     # self.down_convs.append(GCNConv(channels, channels, improved=True))
        #     self.down_convs.append(ConvBlock(in_channels=hidden_channels, hidden_channels=hidden_channels, out_channels=hidden_channels, time_embedding_dim=hidden_t_embd))
        #     self.pools.append(TopKPooling(hidden_channels, self.pool_ratios[i]))
        #
        # print("len self.down_convs ", len(self.down_convs))
        #
        # in_channels = hidden_channels if sum_res else 2 * hidden_channels
        #
        # self.up_convs = torch.nn.ModuleList()
        # for i in range(depth - 1):
        #     #self.up_convs.append(GCNConv(in_channels, channels, improved=True))
        #     self.up_convs.append(ConvBlock(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=hidden_channels, time_embedding_dim=hidden_t_embd))
        #
        # #self.up_convs.append(GCNConv(in_channels, out_channels, improved=True))
        # self.up_convs.append(ConvBlock(in_channels=hidden_t_embd, hidden_channels=hidden_t_embd, out_channels=out_channels, time_embedding_dim=hidden_t_embd))
        #
        # print("len self.up_convs ", len(self.up_convs))

        #self.final_conv = ConvBlock(in_channels=out_channels * 2, hidden_channels=out_channels*2, out_channels=out_channels, time_embedding_dim=hidden_t_embd)

        #self.reset_parameters()


    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, t: Tensor, edge_index: Tensor,
                batch: OptTensor = None) -> Tensor:
        """"""  # noqa: D419

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        #edge_index = edge_index.to_dense()
        #print("type(edge_index) ", type(edge_index))
        #print("edge_index.size(1) ", type(edge_index.size(1)))
        #print("edge_index.device ",edge_index.device)
        edge_weight = x.new_ones(edge_index.size(1))

        #print("orig x shape ", x.shape)

        print("edge index ", edge_index)

        x = self.init_conv(x, edge_index, edge_weight)

        #print("down convs 0 input shape ", x.shape)
        #x = self.down_convs[0](x, t, edge_index, edge_weight)
        #print("init conv x shape ", x.shape)

        #r = x.clone()

        #x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []
        # xs = []
        # edge_indices = []
        # edge_weights = []
        # perms = []

        for i in range(0, len(self.down_convs)):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))

            print("============================= down convs {} input shape {} =====================================".format(i, x.shape))

            x = self.down_convs[i](x, t, edge_index, edge_weight)
            x = self.act(x)

            #print("after down convs {} input shape {}".format(i, x.shape))

            x, edge_index, edge_weight, batch, perm, _ = self.pools[i](
                x, edge_index, edge_weight, batch)

            if i < len(self.down_convs)-1:
                #print("xs + [x] , x.shape", x.shape)
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

            # print("forward down convs x shape ", x.shape)
            # print("forward down convs edge indices ", edge_index.shape)


        # print("edge indices len ", len(edge_indices))
        #
        # print("len xs ", len(xs))
        # print("xs 0 shape ", xs[0].shape)
        # print("edge indices ", edge_indices)
        # print("edge weight ", edge_weight)
        # print("len perms ", len(perms))

        for i in range(0, len(self.down_convs)):
            j = len(self.down_convs) - 1 - i

            print ("j ", j)

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            # print("res shape ", res.shape)
            # print("x shape ", x.shape)
            # print("perm shape ", perm.shape)

            #up = torch.zeros_like(res)
            up = torch.zeros((res.shape[0], x.shape[1]), device=x.device)
            # print("up .shape ", up.shape)
            # print("perm ", perm)
            # print("up[perm] ", up[perm].shape)
            up[perm] = x

            if self.sum_res:
                x = res + up
            else:
                x = torch.cat((res, up), dim=-1)

            #print("up convs {} input shape {}".format(i, x.shape))

            x = self.up_convs[i](x, t, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x

            #print("forward ups x shape ", x.shape)

        #x = torch.cat((x, r), dim=1)
        #x = self.final_conv(x, t, edge_index, edge_weight)(x, t)

        #print("final conv input shape ", x.shape)

        #x = self.final_conv(x, edge_index, edge_weight)


        #print("return x shape ", x.shape)


        return x

    def augment_adj(self, edge_index: Tensor, edge_weight: Tensor,
                    num_nodes: int) -> PairTensor:
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        adj = to_torch_csr_tensor(edge_index, edge_weight,
                                  size=(num_nodes, num_nodes))
        # print("adj ", adj)
        # print("adj type ", type(adj))
        #adj = adj.to_sparse_coo()

        orig_device = adj.device

        adj = adj.to("cpu")

        print("adj device ", adj.device)

        #adj = adj.to_dense()

        #print("adj type ", type(adj))
        #adj = adj @ adj
        #adj = GraphUNet.chunked_matmul(adj, adj, chunk_size=5000)
        #adj = adj.coalesce()

        print("adj shape ", adj.shape)

        adj = torch.sparse.mm(adj, adj)


        #adj = GraphUNet.block_sparse_matmul_csr(adj, adj, chunk_size=100)

        print("final adj shape ", adj.shape)

        adj = adj.to_sparse_coo()

        #adj = (adj @ adj).to_sparse_coo()

        edge_index, edge_weight = adj.indices(), adj.values()
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index = edge_index.to(orig_device)
        edge_weight = edge_weight.to(orig_device)

        print("edge index device ", edge_index.device)
        return edge_index, edge_weight

    # @staticmethod
    # def chunked_matmul(A, B, chunk_size):
    #     N = A.shape[0]
    #     C = torch.zeros(N, N, device=A.device)
    #     for i in range(0, N, chunk_size):
    #         for j in range(0, N, chunk_size):
    #             for k in range(0, N, chunk_size):
    #                 A_chunk = A[i:i + chunk_size, k:k + chunk_size]
    #                 B_chunk = B[k:k + chunk_size, j:j + chunk_size]
    #                 C[i:i + chunk_size, j:j + chunk_size] += A_chunk @ B_chunk
    #
    #     return C
    #
    @staticmethod
    def block_sparse_matmul_csr(A, B, chunk_size):
        N = A.size(0)

        # Initialize lists to collect indices and values for the sparse result
        result_indices = []
        result_values = []

        device = A.device

        print("A type ", type(A))

        # Iterate over the matrix in chunks
        for i in range(0, N, chunk_size):
            for j in range(0, N, chunk_size):
                # Compute bounds for the block
                row_start, row_end = i, min(i + chunk_size, N)
                col_start, col_end = j, min(j + chunk_size, N)

                # Extract a block of A and B within the current bounds
                A_block = A[row_start:row_end, :]
                B_block = B[:, col_start:col_end]

                # Perform sparse matrix multiplication if A_block and B_block are non-empty
                if A_block._nnz() > 0 and B_block._nnz() > 0:
                    C_block = torch.sparse.mm(A_block, B_block)

                    # Collect non-zero indices and values from C_block
                    if C_block._nnz() > 0:
                        C_indices = C_block._indices() + torch.tensor([[row_start], [col_start]], dtype=torch.long)
                        C_values = C_block._values()

                        result_indices.append(C_indices)
                        result_values.append(C_values)

        # Combine all collected indices and values into a single sparse tensor
        if len(result_indices) > 0:
            result_indices = torch.cat(result_indices, dim=1)
            result_values = torch.cat(result_values)
            C_sparse = torch.sparse_csr_tensor(result_indices, result_values, torch.Size([N, N]))
        else:
            C_sparse = torch.sparse_csr_tensor(torch.empty(2, 0, dtype=torch.long),
                                               torch.empty(0, dtype=torch.float),
                                               torch.Size([N, N]))

        return C_sparse

    # def block_sparse_matmul(A, B, chunk_size):
    #     N = A.size(0)
    #     result_indices = []
    #     result_values = []
    #     for i in range(0, N, chunk_size):
    #         for j in range(0, N, chunk_size):
    #             for k in range(0, N, chunk_size):
    #                 A_chunk = A[i:i + chunk_size, k:k + chunk_size]
    #                 B_chunk = B[k:k + chunk_size, j:j + chunk_size]
    #                 if A_chunk._nnz() > 0 and B_chunk._nnz() > 0:
    #                     C_chunk = torch.sparse.mm(A_chunk, B_chunk)
    #                     if C_chunk._nnz() > 0:
    #                         coalesced = C_chunk.coalesce()
    #                         indices = coalesced.indices()
    #                         values = coalesced.values()
    #                         result_indices.append(indices)
    #                         result_values.append(values)
    #     if len(result_indices) > 0:
    #         result_indices = torch.cat(result_indices, dim=1)
    #         result_values = torch.cat(result_values)
    #         C_sparse = torch.sparse.FloatTensor(result_indices, result_values, torch.Size([N, N])).coalesce()
    #     else:
    #         C_sparse = torch.sparse.FloatTensor(torch.empty(2, 0, dtype=torch.long, device=A.device),
    #                                             torch.empty(0, dtype=torch.float, device=A.device),
    #                                             torch.Size([N, N]))
    #     return C_sparse
    # def block_sparse_matmul(A, B, chunk_size):
    #     N = A.size(0)
    #     C_sparse = torch.zeros(N, N, device=A.device)
    #     for i in range(0, N, chunk_size):
    #         for j in range(0, N, chunk_size):
    #             for k in range(0, N, chunk_size):
    #                 A_chunk = A[i:i + chunk_size, k:k + chunk_size].to_dense()
    #                 B_chunk = B[k:k + chunk_size, j:j + chunk_size].to_dense()
    #                 C_sparse[i:i + chunk_size, j:j + chunk_size] += torch.mm(A_chunk, B_chunk)
    #     return C_sparse

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, {self.out_channels}, '
                f'depth={self.depth}, pool_ratios={self.pool_ratios})')
