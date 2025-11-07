import os
import copy
#os.environ["DGLBACKEND"] = "pytorch"
import dgl
import torch
#from dgl.data import DGLDataset
from torch.utils.data import Dataset
import numpy as np
import pyvista as pv
import dgl.sparse as dglsp


class TestBoneDataset(Dataset):
    def __init__(self, data_dir, init_transform=None, input_transform=None):
        self.data_dir = data_dir  #"/home/martin/Documents/Bones_diff_model/data/L4"
        self.input_transform = input_transform
        self.init_transform = init_transform

        self._graphs = []
        self.num_nodes = 0
        self.process_data()

    def shuffle(self, seed):
        np.random.seed(seed)
        perm = np.random.permutation(len(self._graphs))
        self._graphs = self._graphs[perm]

    def get_vertices_edges(self):
        template_data_path = os.path.join(self.data_dir, 'template.vtk')
        template = pv.read(template_data_path)
        vertices = template.points

        subselect_vertices = True
        if subselect_vertices:
            selected_vertices = set()

        edges_file = os.path.join(self.data_dir, "L4_edges.npz")
        if not subselect_vertices and os.path.exists(edges_file):
            edges = np.load(edges_file)["data"]
        else:
            cells = template.cells.reshape((-1, 5))[:, 1:5]  # no need to store cell type
            edges = []
            for vertex_id in range(len(vertices)):
                connected = TestBoneDataset.find_connected_vertices(vertex_id, cells)
                if subselect_vertices:
                    selected_vertices.add(vertex_id)
                for conected_vertex in connected:
                    edges.append([vertex_id, conected_vertex])
                    if subselect_vertices:
                        selected_vertices.add(conected_vertex)

                if subselect_vertices and len(selected_vertices) > 100:
                    print("selected vertices ", selected_vertices)
                    vertices = selected_vertices
                    break
                print("vertex_id: {}, connected: {} ".format(vertex_id, connected))

        print("len vertices ", len(vertices))
        print("len edges ", len(edges))

        return vertices, edges

    def process_data(self):
        node_features_path = os.path.join(self.data_dir, 'u.npy')
        node_features = np.load(node_features_path)

        #vertices, edges = self.get_vertices_edges()

        num_vertices = 125
        num_edges = 550

        src = torch.randint(0, num_vertices, (num_edges,))
        dst = torch.randint(0, num_vertices, (num_edges,))

        self.num_nodes = num_vertices

        graph = dgl.DGLGraph()
        #graph.add_nodes(num_vertices, {'x': torch.from_numpy(node_features[i][:num_vertices])})
        graph.add_edges(src, dst)
        self.adj_matrix = graph.adj()#(scipy_fmt='coo', etype='develops') # Get a scipy coo sparse matrix.


        #print("self._adj_matrix shape", self._adj_matrix.shape) # Has to be sparse representation

        # print("adj matrix ", self.adj_matrix.val.shape)
        #
        # expanded_tensor = self.adj_matrix.val.unsqueeze(1)
        # print("expanend tensor ", expanded_tensor.shape)
        # expanded_tensor = expanded_tensor.expand(550, 5)
        #
        #
        # batch_adj_matrix = dgl.sparse.val_like(self.adj_matrix, expanded_tensor)
        #
        torch.manual_seed(12345)
        node_features = torch.randn(100, 125, 4)

        # input_mean = torch.mean(node_features, dim=[0, 1])
        # input_std = torch.std(node_features, dim=[0, 1])
        #
        # #standardize_features = node_features - input_mean / input_std
        #
        # print("node features mean ", input_mean)
        # print("node features std ", input_std)
        #
        # print("standardize_features mean", torch.mean(standardize_features, dim=[0, 1]))
        # print("standardize_features std", torch.std(standardize_features, dim=[0, 1]))
        # exit()

        #
        # node_features = node_features.permute(1, 2, 0)
        #
        # #######################
        # indices = torch.tensor([[0, 1, 1], [1, 0, 2]])
        # val = torch.randn(3, 4)
        # A = dglsp.spmatrix(indices, val, shape=(5, 5))
        # X = torch.randn(5, 4, 4)
        #
        # print("A ", A)
        #
        # print("A shape ", A.shape)
        # print("X.shape ", X.shape)
        # result = dglsp.bspmm(A, X)
        # print("result shape ", result.shape)
        # type(result)
        #
        #
        # print("adj matrix shape ", self.adj_matrix.shape)
        # print("node features shape ", node_features.shape)
        #
        #
        # embd = dglsp.bspmm(batch_adj_matrix, node_features)
        #
        #
        #
        # #embd = self.adj_matrix @ node_features
        #
        # print("embd.shape ", embd.shape)
        #
        # exit()


        #self._adj_matrix = self._adj_matrix.to_dense().to_sparse()

        # for i in range(node_features.shape[0]):
        #     graph = dgl.DGLGraph()
        #     graph.add_nodes(num_vertices, {'x': torch.from_numpy(node_features[i][:num_vertices])})
        #     graph.add_edges(src, dst)
        #
        #
        #     print("adj matrix ", self._adj_matrix)
        #
        #     print("graph[x] ", graph.ndata["x"].shape)

            #self._graphs.append(graph)

        self._graphs = node_features

    def __getitem__(self, idx):
        #print("dtype self.graphs ", type(self._graphs))
        graphs = self._graphs[idx]

        # if len(graphs) > 0:
        #     import matplotlib.pyplot as plt
        #     import networkx as nx
        #     G = dgl.to_networkx(graphs[0])
        #     plt.figure(figsize=[15, 7])
        #     nx.draw(G)
        #     exit()
        #graph_signal_dgl = dgl.DGLTensor(self._graphs)


        # reshaped_graphs = self._graphs.permute(1,0)
        # graph_emb = dglsp.matmul(self._adj_matrix, reshaped_graphs)
        # print("graph emb shape ", graph_emb.shape)
        # exit()

        if len(graphs.shape) > 2:
            new_dataset = TestBoneDataset(self.data_dir) #copy.deepcopy(self)
            new_dataset._graphs = graphs
            new_dataset.adj_matrix = self.adj_matrix
            new_dataset.input_transform = self.input_transform
            new_dataset.init_transform = self.init_transform
            return new_dataset

        # print("channel 0 mean: {}, std: {}".format(torch.mean(self._graphs[..., 0]), torch.std(self._graphs[..., 0])))
        # print("channel 1 mean: {}, std: {}".format(torch.mean(self._graphs[..., 1]), torch.std(self._graphs[..., 1])))
        # print("channel 2 mean: {}, std: {}".format(torch.mean(self._graphs[..., 2]), torch.std(self._graphs[..., 2])))
        # print("channel 3 mean: {}, std: {}".format(torch.mean(self._graphs[..., 3]), torch.std(self._graphs[..., 3])))
        #
        # print("self. input transform ", self.input_transform)

        if self.input_transform is not None:
            graphs = self.input_transform(graphs)

        # print("self.graphs shape ", self._graphs.shape)
        # print("channel 0 mean: {}, std: {}".format(torch.mean(self._graphs[..., 0]), torch.std(self._graphs[..., 0])))
        # print("channel 1 mean: {}, std: {}".format(torch.mean(self._graphs[..., 1]), torch.std(self._graphs[..., 1])))
        # print("channel 2 mean: {}, std: {}".format(torch.mean(self._graphs[..., 2]), torch.std(self._graphs[..., 2])))
        # print("channel 3 mean: {}, std: {}".format(torch.mean(self._graphs[..., 3]), torch.std(self._graphs[..., 3])))

        return graphs

    def __len__(self):
        return len(self._graphs)

    @staticmethod
    def find_faces_with_node(index, cells):
        """Pass the index of the node in question.
        Returns the face indices of the faces with that node."""
        return [i for i, cell in enumerate(cells) if index in cell]

    @staticmethod
    def find_connected_vertices(index, cells):
        """Pass the index of the node in question.
        Returns the vertex indices of the vertices connected with that node."""
        cids = TestBoneDataset.find_faces_with_node(index, cells)
        connected = np.unique(cells[cids].ravel())
        return np.delete(connected, np.argwhere(connected == index))


