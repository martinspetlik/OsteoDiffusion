import os
import copy
#os.environ["DGLBACKEND"] = "pytorch"
#import dgl
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
import numpy as np
import pyvista as pv
from torch_geometric.utils import to_scipy_sparse_matrix, to_torch_sparse_tensor


class BoneDataset(Dataset):
    def __init__(self, data_dir, input_transform=None, input_channels=None):
        self.data_dir = data_dir
        self.input_transform = input_transform
        self.input_channels = input_channels
        self._graphs_features = []
        self.num_nodes = 0
        self._adj_matrix = None
        self.process_data()

    def shuffle(self, seed):
        np.random.seed(seed)
        perm = np.random.permutation(len(self._graphs_features))
        self._graphs_features = self._graphs_features[perm]

    def get_graphs(self):
        template_data_path = os.path.join(self.data_dir, 'template.vtk')
        template = pv.read(template_data_path)
        vertices = template.points

        edges_file = os.path.join(self.data_dir, "edges.npz")
        if os.path.exists(edges_file):
            edges = np.load(edges_file)["data"]
        else:
            raise Exception("edges.npz file not found")

        return self.get_graphs_features(template), vertices, edges

    @staticmethod
    def cell2point_data(tvar, data):
        n = len(data)
        m = len(tvar.points)
        new_data = np.zeros((n, m))
        for i in range(n):
            tvar['data'] = data[i]
            new = tvar.cell_data_to_point_data()
            new_data[i] = new['data']
        return new_data

    def get_graphs_features(self, template):
        graphs_features_path = os.path.join(self.data_dir, 'u.npy')
        graphs_features = np.load(graphs_features_path)

        density_data_path = os.path.join(self.data_dir, 'rho.npy')
        density = np.load(density_data_path)

        density = BoneDataset.cell2point_data(template, density)
        density = density[..., np.newaxis]

        graphs_features = torch.tensor(np.concatenate((graphs_features, density), axis=-1))

        if self.input_channels is not None:
            graphs_features = graphs_features[..., self.input_channels]

        return graphs_features


    def process_data(self):
        self._graphs_features, vertices, edges = self.get_graphs()
        self.num_nodes = len(vertices)
        #graph = dgl.DGLGraph()
        src, dst = zip(*edges)
        #graph.add_edges(src, dst)
        self.edge_indices = torch.stack([torch.tensor(src), torch.tensor(dst)], dim=0)
        # Convert to scipy sparse adjacency matrix

        if torch.cuda.is_available():
            self.edge_indices = self.edge_indices.cuda()

        # print("self.edge_indices ", self.edge_indices.device)
        # exit()
        self.adj_matrix = to_torch_sparse_tensor(self.edge_indices)


    def __getitem__(self, idx):
        graphs = self._graphs_features[idx]

        if len(graphs.shape) > 2:
            new_dataset = BoneDataset(self.data_dir)  # copy.deepcopy(self)
            new_dataset._graphs = graphs
            new_dataset.adj_matrix = self.adj_matrix
            new_dataset.input_transform = self.input_transform
            return new_dataset

        if self.input_transform is not None:
            graphs = self.input_transform(graphs)

        graphs = Data(x=graphs.float(), edge_index=self.edge_indices)

        return graphs

    def __len__(self):
        return len(self._graphs_features)
