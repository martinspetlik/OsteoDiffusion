import os
import copy
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class BoneDatasetCT(Dataset):
    """
    Dataset for loading bone CT scan volumes and corresponding metadata.
    Each sample contains a 3D CT image (.npz) and a metadata .xlsx file with patient info.
    """
    def __init__(self, data_dir, data_file_name=None, input_transform=None, metadata_dir=None, metadata_file_name="metadata.xlsx"):
        """
        Initialize the dataset, locate all CT image and metadata files.
        :param data_dir: Root directory containing subfolders for each sample.
        :param data_file_name: Name of the .npz file storing scan data.
        :param input_transform: Optional preprocessing transform applied to each image.
        :param metadata_dir: Directory containing metadata subfolders. Defaults to data_dir.
        :param metadata_file_name: Name of metadata Excel file. Default is "metadata.xlsx".
        """
        self.data_dir = data_dir

        if metadata_dir is None:
            metadata_dir = self.data_dir
        self.metadata_dir = metadata_dir

        if not os.path.exists(self.data_dir):
            raise NotADirectoryError

        self._data_file_name = data_file_name
        self._metadata_file_name = metadata_file_name

        # Global limits (for potential normalization)
        self._global_min_value = -1024.0
        self._global_max_value = 1650.0

        self.input_transform = input_transform
        self._image_file_paths = []
        self._metadata_file_paths = []

        self._set_paths_to_samples()

    def shuffle(self, seed=None):
        """
        Shuffle dataset samples.
        :param seed: Optional random seed for reproducibility.
        :return: None
        """
        if seed is not None:
            np.random.seed(seed)
        perm = np.random.permutation(len(self._image_file_paths))
        self._image_file_paths = list(np.array(self._image_file_paths)[perm])
        self._metadata_file_paths = list(np.array(self._metadata_file_paths)[perm])

    def _set_paths_to_samples(self):
        """
        Collect file paths for all samples in the dataset.
        :raises AttributeError: If data_dir is not defined.
        :raises FileNotFoundError: If metadata file for a sample is missing.
        """
        if self.data_dir is None:
            raise AttributeError

        for data_dir in glob.glob(self.data_dir + '/*'):
            image_file = os.path.join(data_dir, self._data_file_name)
            if os.path.exists(image_file):
                self._image_file_paths.append(image_file)

            dir_name = os.path.basename(data_dir)
            metadata_file = os.path.join(self.metadata_dir, os.path.join(dir_name, self._metadata_file_name))
            if not os.path.exists(metadata_file):
                raise FileNotFoundError(f"Metadata file {metadata_file} not exists")
            self._metadata_file_paths.append(metadata_file)

    def load_metadata(self, metadata_file):
        """
        Load metadata from an Excel file and compute derived fields.
        :param metadata_file: Path to metadata Excel file.
        :return: Dictionary containing 'sex' and 'age' fields.
        """
        metadata_dict = pd.read_excel(metadata_file, engine="openpyxl").iloc[0].to_dict()
        return {"sex": metadata_dict["sex"], "age": metadata_dict["CT date"] - metadata_dict["born"]}

    def __getitem__(self, idx):
        """
        Return a single dataset item (CT scan and metadata).
        :param idx: Sample index.
        :return: Tuple (image_tensor, (sex, normalized_age)).
        """
        image_path = self._image_file_paths[idx]
        metadata_path = self._metadata_file_paths[idx]

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
        """
        Return number of available samples in the dataset.
        :return: Integer length of dataset.
        """
        return len(self._image_file_paths)

    @staticmethod
    def remap_subset_paths(subset):
        """
        Remap dataset paths when running in a different environment.
        Automatically replaces dataset root to match local or cluster paths.
        :param subset: torch.utils.data.Subset object referencing BoneDatasetCT.
        :return: None
        """
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
