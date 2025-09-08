import os
import joblib
import numpy as np
import pandas as pd
from dataset.cnn_diffusion.bone_dataset_CT import BoneDatasetCT
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


def _split_dataset(dataset, config, n_train_samples=None):
    """
    Split a dataset into train, validation, and test subsets with stratification.

    :param dataset: full dataset (torch.utils.data.Dataset).
    :param config: dictionary with split configuration. Keys:
        - "train_samples_ratio": fraction of samples to use for training.
        - "val_samples_ratio": fraction of training set to use for validation.
        - "n_test_samples": optional, number of test samples.
        - "seed": random seed for reproducibility.
    :param n_train_samples: optional, fixed number of training samples. If None,
                            determined from config["train_samples_ratio"].
    :return: (train_set, validation_set, test_set), each as torch.utils.data.Subset.
    """
    # Step 1: Extract metadata (sex and denormalized age)
    metadata = []
    for i in range(len(dataset)):
        _, (sex, age_norm) = dataset[i]
        age = age_norm * 100.0  # de-normalize age
        metadata.append((sex, age))

    metadata = pd.DataFrame(metadata, columns=["sex", "age"])
    metadata["sex"] = metadata["sex"].astype(int)

    # Optional: could add age binning here for finer stratification
    # metadata["age_bin"] = pd.cut(metadata["age"], bins=[0, 30, 45, 60, 75, 100], labels=False)
    # metadata["strata"] = metadata["sex"].astype(str) + "_" + metadata["age_bin"].astype(str)

    all_indices = np.arange(len(dataset))

    # Step 2: Determine training sample size
    if n_train_samples is None:
        n_train_samples = int(len(dataset) * config["train_samples_ratio"])
    n_train_samples = min(n_train_samples, int(len(dataset) * config["train_samples_ratio"]))
    print("Number of training samples:", n_train_samples)

    # Step 3: Determine test set size
    if "n_test_samples" in config and config["n_test_samples"] is not None:
        test_size = config["n_test_samples"]
    else:
        test_size = len(dataset) - n_train_samples

    # Step 4: Stratified train/test split
    if test_size > 0:
        train_val_idx, test_idx = train_test_split(
            all_indices,
            test_size=test_size,
            train_size=n_train_samples,
            stratify=metadata["sex"],  # stratify by sex only
            random_state=config["seed"]
        )
    else:
        train_val_idx = all_indices[:n_train_samples]
        test_idx = []

    # Step 5: Validation split (optional)
    if config["val_samples_ratio"] == 0.0:
        train_idx = train_val_idx
        val_idx = []
    else:
        val_size = int(len(train_val_idx) * config["val_samples_ratio"])
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size,
            stratify=metadata.loc[train_val_idx, "sex"],  # stratify by sex only
            random_state=config["seed"]
        )

    # Step 6: Wrap indices into Subsets
    train_set = Subset(dataset, train_idx)
    validation_set = Subset(dataset, val_idx) if len(val_idx) > 0 else []
    test_set = Subset(dataset, test_idx) if len(test_idx) > 0 else []

    return train_set, validation_set, test_set


def prepare_dataset(study, config, data_dir, data_file_name, serialize_path=None, train_dataset=None):
    """
    Prepare dataset and split into train/val/test sets, with optional serialization.

    :param study: Optuna study or object supporting set_user_attr.
    :param config: dictionary with dataset split configuration (see _split_dataset).
    :param data_dir: path to dataset directory.
    :param data_file_name: file name for dataset file (e.g., CSV/NPZ).
    :param serialize_path: optional directory to save pickled subsets.
    :param train_dataset: unused (reserved for future use).
    :return: (train_set, validation_set, test_set), each as torch.utils.data.Subset.
    """
    # Load dataset
    dataset = BoneDatasetCT(data_dir=data_dir, data_file_name=data_file_name)

    # Get train/val/test splits
    n_train_samples = config.get("n_train_samples", None)
    train_set, validation_set, test_set = _split_dataset(dataset, config, n_train_samples)

    # Store dataset attributes in study object
    study.set_user_attr("data_dir", dataset.data_dir)
    study.set_user_attr("global_min_value", dataset._global_min_value)
    study.set_user_attr("global_max_value", dataset._global_max_value)

    # Optionally serialize subsets
    if serialize_path is not None:
        joblib.dump(train_set, os.path.join(serialize_path, "train_dataset.pkl"))
        joblib.dump(validation_set, os.path.join(serialize_path, "val_dataset.pkl"))
        joblib.dump(test_set, os.path.join(serialize_path, "test_dataset.pkl"))

    return train_set, validation_set, test_set
