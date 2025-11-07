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

        local_data_path = "/data"
        split_data_dir = data_dir.split("/")
        self.data_dir = os.path.join(local_data_path, split_data_dir[-1])

        #self.data_dir = data_dir

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

        vertices_file = os.path.join(self.data_dir, 'vertices.npz')
        if os.path.exists(vertices_file):
            vertices = np.load(vertices_file)["data"]

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
        #graphs_features_path = os.path.join(self.data_dir, 'positive_u.npy')
        graphs_features_path = os.path.join(self.data_dir, 'u.npy')
        graphs_features = np.load(graphs_features_path)

        print("graph features shape ", graphs_features.shape)

        # density_data_path = os.path.join(self.data_dir, 'positive_rho.npy')
        density_data_path = os.path.join(self.data_dir, 'rho.npy')
        density = np.load(density_data_path)

        print('density shape ', density.shape)

        for s in range(density.shape[0]):
            data = np.squeeze(density[s])
            has_negative = data[data< 0]

            if len(has_negative) > 0:
                print("has negative ", has_negative)

        if density.shape[1] > graphs_features.shape[1]:
            density = BoneDataset.cell2point_data(template, density)

        print("density.shape ", density.shape)

        # positive_density = []
        # new_graph_features = []
        # for s in range(density.shape[0]):
        #     data = np.squeeze(density[s])
        #     has_negative = data[data< 0]
        #     print("has negative ", has_negative)
        #     if len(has_negative) == 0:
        #         positive_density.append(data)
        #         new_graph_features.append(graphs_features[s])
        #
        # positive_density = np.array(positive_density)
        # new_graph_features = np.array(new_graph_features)
        # print("positive density shape ", positive_density.shape)
        # print("new graph features shape ", new_graph_features.shape)
        # np.save(os.path.join(self.data_dir, 'positive_rho.npy'), positive_density)
        # np.save(os.path.join(self.data_dir, 'positive_u.npy'), new_graph_features)
        #
        # exit()

        density = density[..., np.newaxis]

        graphs_features = torch.tensor(np.concatenate((graphs_features, density), axis=-1))

        if self.input_channels is not None:
            graphs_features = graphs_features[..., self.input_channels]

        print("self.input_channels ", self.input_channels)
        print("graphs_features ", graphs_features.shape)
        return graphs_features

    def process_data(self):
        self._graphs_features, vertices, edges = self.get_graphs()

        print("len(vertices) ", len(vertices))
        print("len(edges) ", len(edges))

        print("vertices ", vertices)
        print("edges ", edges)


        self.num_nodes = len(vertices)
        #graph = dgl.DGLGraph()
        src, dst = zip(*edges)
        #graph.add_edges(src, dst)
        self.edge_indices = torch.stack([torch.tensor(src), torch.tensor(dst)], dim=0)
        # Convert to scipy sparse adjacency matrix


        #############
        # Subgraphs
        #############
        from torch_geometric.utils import subgraph
        sub_vertices = list(range(0, 10))
        subset = torch.tensor(sub_vertices, dtype=torch.long)
        # Generate the subgraph
        sub_edge_index, _ = subgraph(subset, self.edge_indices, relabel_nodes=True)
        # Extract the corresponding node features
        print("self.graph_features.shape ", self._graphs_features.shape)
        self._graphs_features = self._graphs_features[:, subset]
        print("subset self.graph_features.shape ", self._graphs_features.shape)
        #exit()

        self.edge_indices = sub_edge_index

        ###################
        ###################


        if torch.cuda.is_available():
            self.edge_indices = self.edge_indices.cuda()

        # print("self.edge_indices ", self.edge_indices.device)
        # exit()
        self.adj_matrix = to_torch_sparse_tensor(self.edge_indices)


    def __getitem__(self, idx):
        graphs = self._graphs_features[idx]

        if len(graphs.shape) > 2:
            new_dataset = BoneDataset(self.data_dir)  # copy.deepcopy(self)
            new_dataset._graphs_features = graphs
            new_dataset.adj_matrix = self.adj_matrix
            new_dataset.input_transform = self.input_transform
            return new_dataset

        if self.input_transform is not None:
            graphs = self.input_transform(graphs)

        BoneDataset._check_nans(graphs, str_err="Input features contains NaN values, idx: {}".format(idx), idx=idx)
        graphs = Data(x=graphs.float(), edge_index=self.edge_indices)

        return graphs

    def __len__(self):
        return len(self._graphs_features)

    @staticmethod
    def _check_nans(final_features, str_err="Data contains NaN values", idx=None):
        has_nan = torch.any(torch.isnan(final_features))
        if has_nan:
            print("str err ", str_err)
            print("idx ", idx)
            #shutil.rmtree(os.path.dirname(file))
            # raise ValueError(str_err)
