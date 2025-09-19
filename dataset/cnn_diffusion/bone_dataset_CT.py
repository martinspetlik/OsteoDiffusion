import os
import copy
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class BoneDatasetCT(Dataset):
    def __init__(self, data_dir, data_file_name=None, input_transform=None, metadata_dir=None, metadata_file_name="metadata.xlsx"):
        self.data_dir = data_dir

        if metadata_dir is None:
            metadata_dir = self.data_dir
        self.metadata_dir = metadata_dir

        self.data_dir = "/mnt/database/BoneDat/derived/fields" #"/mnt/database/BoneDat/derived/fields"
        #self.metadata_dir = data_dir #"/mnt/database/BoneDat/raw"
        self.metadata_dir = "/mnt/database/BoneDat/raw"
        #
        # self.data_dir = "/test_database"
        # self.metadata_dir = "/test_database"

        if not os.path.exists(self.data_dir):
            raise NotADirectoryError

        self._data_file_name = data_file_name
        self._metadata_file_name = metadata_file_name

        # Values for postprocessing
        self._global_min_value = -1024.00
        self._global_max_value = 1650

        self.input_transform = input_transform
        self._image_file_paths = []
        self._metadata_file_paths = []

        self._set_paths_to_samples()

    def shuffle(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        perm = np.random.permutation(len(self._image_file_paths))
        self._image_file_paths = list(np.array(self._image_file_paths)[perm])
        self._metadata_file_paths = list(np.array(self._metadata_file_paths)[perm])

    def _set_paths_to_samples(self):
        if self.data_dir is None:
            raise AttributeError

        for data_dir in glob.glob(self.data_dir + '/*'):
            image_file = os.path.join(data_dir, self._data_file_name)
            if os.path.exists(image_file):
                self._image_file_paths.append(image_file)

            dir_name = os.path.basename(data_dir)
            metadata_file = os.path.join(self.metadata_dir, os.path.join(dir_name, self._metadata_file_name))
            if not os.path.exists(metadata_file):
                raise FileNotFoundError("Metadata file {} not exists".format(metadata_file))
            self._metadata_file_paths.append(metadata_file)

    def load_metadata(self, metadata_file):
        metadata_dict = pd.read_excel(metadata_file, engine="openpyxl").iloc[0].to_dict()
        return {"sex": metadata_dict["sex"], "age": metadata_dict['CT date'] - metadata_dict['born']}


    def __getitem__(self, idx):
        #print("idx ", idx)
        image_path = self._image_file_paths[idx]
        metadata_path = self._metadata_file_paths[idx]

        #print("image path ", image_path)
        if isinstance(image_path, (list, np.ndarray)):
            new_dataset = copy.deepcopy(self)
            new_dataset._image_file_paths = image_path
            new_dataset._metadata_file_paths = metadata_path
            return new_dataset

        image_data = np.load(image_path)["data"]
        metadata = self.load_metadata(metadata_path)
        sex = 1.0 if metadata["sex"] == "F" else 0.0
        age_norm = metadata["age"] / 100.0

        if self.input_transform is not None:
            image_data = self.input_transform(image_data)

        return image_data.reshape(1, *image_data.shape), (sex, age_norm)

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

    @staticmethod
    def remap_subset_paths(subset):
        if os.path.exists("/scratch/project_465002075/bones_dataset"):
            root = "/scratch/project_465002075/bones_dataset"
        else:
            root = "/home/martin/Documents/Bones_diff_model/test_database"

        ds = subset.dataset
        for idx in subset.indices:
            img = ds._image_file_paths[idx]
            meta = ds._metadata_file_paths[idx]

            rel_img = os.path.basename(os.path.dirname(img)) + "/" + os.path.basename(img)
            rel_meta = os.path.basename(os.path.dirname(meta)) + "/" + os.path.basename(meta)

            ds._image_file_paths[idx] = os.path.join(root, rel_img)
            ds._metadata_file_paths[idx] = os.path.join(root, rel_meta)

