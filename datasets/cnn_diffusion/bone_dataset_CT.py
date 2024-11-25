import os
import copy
#os.environ["DGLBACKEND"] = "pytorch"
#import dgl
import glob
import torch
#from torch_geometric.data import Data
from torch.utils.data import Dataset
import numpy as np
#import pyvista as pv
#from torch_geometric.utils import to_scipy_sparse_matrix, to_torch_sparse_tensor


class BoneDatasetCT(Dataset):
    def __init__(self, data_dir, data_file_name, input_transform=None, input_channels=None):

        self.data_dir = data_dir #"/mnt/database/BoneDat/derived/fields"

        if not os.path.exists(self.data_dir):
            raise NotADirectoryError

        #self._data_file_name = "lumbopelvic_masked_normed_resampled_100_100_100.npz"
        #self._data_file_name = "lumbopelvic_masked_normed_resampled_32_32_32.npz"
        self._data_file_name = data_file_name #"lumbopelvic_masked_normed_local_resampled_32_32_32.npz"

        # Values for postprocessing
        self._global_min_value = -1024.00
        self._global_max_value = 2000

        # split_data_dir = data_dir.split("/")
        # self.data_dir = os.path.join(local_data_path, split_data_dir[-1])

        self.input_transform = input_transform

        self._image_file_paths = []
        #self.input_channels = input_channels
        #self._graphs_features = []
        #self.num_nodes = 0
        #self._adj_matrix = None
        #self.process_data()

        self._set_paths_to_samples()

    def shuffle(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        perm = np.random.permutation(len(self._image_file_paths))
        self._image_file_paths = list(np.array(self._image_file_paths)[perm])

    def _set_paths_to_samples(self):
        if self.data_dir is None:
            raise AttributeError

        for data_dir in glob.glob(self.data_dir + '/*'):
            image_file = os.path.join(data_dir, self._data_file_name)
            if os.path.exists(image_file):
                self._image_file_paths.append(image_file)

    def __getitem__(self, idx):
        #print("idx ", idx)
        image_path = self._image_file_paths[idx]
        #print("image path ", image_path)
        if isinstance(image_path, (list, np.ndarray)):
            new_dataset = copy.deepcopy(self)
            new_dataset._image_file_paths = image_path
            return new_dataset

        image_data = np.load(image_path)["data"]

        if self.input_transform is not None:
            image_data = self.input_transform(image_data)

        return image_data.reshape(1, *image_data.shape)

    def __len__(self):
        return len(self._image_file_paths)

    @staticmethod
    def _check_nans(final_features, str_err="Data contains NaN values", idx=None):
        has_nan = torch.any(torch.isnan(final_features))
        if has_nan:
            print("str err ", str_err)
            print("idx ", idx)
            #shutil.rmtree(os.path.dirname(file))
            # raise ValueError(str_err)

