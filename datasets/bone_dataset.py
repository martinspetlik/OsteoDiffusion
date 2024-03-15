import os
import copy
#os.environ["DGLBACKEND"] = "pytorch"
import dgl
import torch
#from dgl.data import DGLDataset
from torch.utils.data import Dataset
import numpy as np
import pyvista as pv


class BoneDataset(Dataset):
    def __init__(self, data_dir, init_transform=None, input_transform=None):
        self.data_dir = data_dir  #"/home/martin/Documents/Bones_diff_model/data/L4"
        self.input_transform = input_transform
        self.init_transform = init_transform

        self._graphs = []
        self.process_data()

    def shuffle(self, seed):
        np.random.seed(seed)
        perm = np.random.permutation(len(self._graphs))
        self._graphs = list(np.array(self._graphs)[perm])

    def get_vertices_edges(self):
        template_data_path = os.path.join(self.data_dir, 'template.vtk')
        template = pv.read(template_data_path)
        vertices = template.points

        edges_file = os.path.join(self.data_dir, "L4_edges.npz")
        if os.path.exists(edges_file):
            edges = np.load(edges_file)["data"]
        else:
            cells = template.cells.reshape((-1, 5))[:, 1:5] # no need to store cell type
            edges = []
            for vertex_id in range(len(vertices)):
                connected = BoneDataset.find_connected_vertices(vertex_id, cells)
                for conected_vertes in connected:
                    edges.append([vertex_id, conected_vertes])
                print("vertex_id: {}, connected: {} ".format(vertex_id, connected))

        return vertices, edges


    def process_data(self):
        node_features_path = os.path.join(self.data_dir, 'u.npy')
        node_features = np.load(node_features_path)

        vertices, edges = self.get_vertices_edges()

        for i in range(node_features.shape[0]):
            graph = dgl.DGLGraph()
            graph.add_nodes(len(vertices), {'x': torch.from_numpy(node_features[i])})
            #print("graph.ndata['x'] ", graph.ndata['x'])

            src, dst = zip(*edges)
            graph.add_edges(src, dst)

            self._graphs.append(graph)

    def __getitem__(self, idx):
        graphs = self._graphs[idx]

        print("graphs ", graphs)

        if isinstance(graphs, (list, np.ndarray)):
            new_dataset = copy.deepcopy(self)
            new_dataset._graphs = graphs
            return new_dataset

        return graphs

    def __len__(self):
        return 1

    @staticmethod
    def find_faces_with_node(index, cells):
        """Pass the index of the node in question.
        Returns the face indices of the faces with that node."""
        return [i for i, cell in enumerate(cells) if index in cell]

    @staticmethod
    def find_connected_vertices(index, cells):
        """Pass the index of the node in question.
        Returns the vertex indices of the vertices connected with that node."""
        cids = BoneDataset.find_faces_with_node(index, cells)
        connected = np.unique(cells[cids].ravel())
        return np.delete(connected, np.argwhere(connected == index))


