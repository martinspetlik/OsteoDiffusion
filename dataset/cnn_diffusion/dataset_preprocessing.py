import os
import joblib
import copy
import torch
import numpy as np
import torchvision.transforms as transforms
from dataset.cnn_diffusion.bone_dataset_CT import BoneDatasetCT
from models.auxiliary_functions import get_mean_std, log_data, exp_data, quantile_transform_fit, QuantileTRF, NormalizeData, log_all_data, init_norm, log10_data, log10_all_data, get_loss_fn
#from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import pandas as pd


def get_inverse_transform(study, results_dir):
    transforms_list = []
    if ("input_transform" in study.user_attrs and len(study.user_attrs["input_transform"]) > 0) \
            or os.path.exists(os.path.join(results_dir, "input_transform.pkl")):
        output_transform = joblib.load(os.path.join(results_dir, "input_transform.pkl"))
        quantile_trf_obj = QuantileTRF()
        quantile_trf_obj.quantile_trfs_out = output_transform
        transforms_list.append(quantile_trf_obj.quantile_inv_transform_out)

    print("transforms_list ", transforms_list)

    std = 1 / study.user_attrs["input_std"]
    zeros_mean = np.zeros(len(study.user_attrs["input_mean"]))

    print("input_mean ", study.user_attrs["input_mean"])
    print("input_std ", study.user_attrs["input_std"])

    ones_std = np.ones(len(zeros_mean))
    mean = -study.user_attrs["input_mean"]

    transforms_list = [transforms.Normalize(mean=zeros_mean, std=std),
                       transforms.Normalize(mean=mean, std=ones_std)]

    if "input_log" in study.user_attrs and study.user_attrs["input_log"]:
        print("input log to transform list")
        transforms_list.append(transforms.Lambda(exp_data))

    return transforms.Compose(transforms_list)

def features_transform(config, data_dir, input_transform_list, dataset_for_transform=None, input_channels=None):
    #################################
    ## Data for Quantile Transform ##
    #################################
    quantile_trf_obj = QuantileTRF()
    if dataset_for_transform is None:
        dataset_for_transform = BoneDatasetCT(data_dir=data_dir, input_channels=input_channels)
    input_data = []

    n_data_input = 1000000
    #n_data_output = 300000
    if "input_transform" in config or "output_transform" in config:
        for index, data in enumerate(dataset_for_transform):

            input_data.append(data.x.numpy())

            # input = np.reshape(input, (input.shape[0], input.shape[-1] * input.shape[-2]))
            # if input.shape[-1] * input.shape[-2] * index < n_data_input:
            #     if len(input_data) == 0:
            #         input_data = input
            #     else:
            #         input_data = np.concatenate([input_data, input], axis=1)

            # if output_data.shape[-1] < n_data_output:
            #     output = np.reshape(output, (output.shape[0], 1))
            #     if len(output_data) == 0:
            #         output_data = output
            #     else:
            #         output_data = np.concatenate([output_data, output], axis=1)

        input_data = np.concatenate(input_data, axis=0)


    if "input_transform" in config and len(config["input_transform"]) > 0:
        quantile_trfs = quantile_transform_fit(input_data,
                                                indices=config["input_transform"]["indices"],
                                                transform_type=config["input_transform"]["type"])

        joblib.dump(quantile_trfs, os.path.join(config["output_dir"], "input_transform.pkl"))
        quantile_trf_obj.quantile_trfs_in = quantile_trfs
        input_transform_list.append(quantile_trf_obj.quantile_transform_in)

    # if "output_transform" in config and len(config["output_transform"]) > 0:
    #     quantile_trfs_out = quantile_transform_fit(output_data,
    #                                                indices=config["output_transform"]["indices"],
    #                                                transform_type=config["output_transform"]["type"])
    #     joblib.dump(quantile_trfs_out, os.path.join(config["output_dir"], "output_transform.pkl"))
    #     quantile_trf_obj.quantile_trfs_out = quantile_trfs_out
    #     output_transform_list.append(quantile_trf_obj.quantile_transform_out)
    return input_transform_list#, output_transform_list


def _append_dataset(dataset_1, dataset_2):
    dataset_1._bulk_file_paths.extend(dataset_2._bulk_file_paths)
    dataset_1._fracture_file_paths.extend(dataset_2._fracture_file_paths)
    dataset_1._cross_section_file_paths.extend(dataset_2._cross_section_file_paths)
    dataset_1._output_file_paths.extend(dataset_2._output_file_paths)


def prepare_sub_datasets(study, config, data_dir, serialize_path=None):
    complete_train_set, complete_val_set, complete_test_set = None, None, None
    for key, dset_config in config["sub_datasets"].items():
        prepare_dset_config = copy.deepcopy(config)
        prepare_dset_config["log_input"] = dset_config["log_input"]
        if "init_norm" in dset_config:
            prepare_dset_config["init_norm"] = dset_config["init_norm"]
        if "input_transform" in dset_config:
            prepare_dset_config["input_transform"] = dset_config["input_transform"]
        if "output_transform" in dset_config:
            prepare_dset_config["output_transform"] = dset_config["output_transform"]

        prepare_dset_config["normalize_input"] = dset_config["normalize_input"]
        prepare_dset_config["log_output"] = dset_config["log_output"]
        if "log10_output" in dset_config:
            prepare_dset_config["log10_output"] = dset_config["log10_output"]
        if "log_all_output" in dset_config:
            prepare_dset_config["log_all_output"] = dset_config["log_all_output"]
        if "log10_all_output" in dset_config:
            prepare_dset_config["log10_all_output"] = dset_config["log10_all_output"]
        prepare_dset_config["normalize_output"] = dset_config["normalize_output"]
        prepare_dset_config["n_train_samples"] = dset_config["n_train_samples"]
        prepare_dset_config["n_test_samples"] = dset_config["n_test_samples"]
        prepare_dset_config["val_samples_ratio"] = dset_config["val_samples_ratio"]
        print("prepare_dset_config ", prepare_dset_config)

        sub_train_set, sub_val_set, sub_test_set = prepare_dataset(study, prepare_dset_config, dset_config['dataset_path'])

        if complete_train_set is None:
            complete_train_set = sub_train_set
            complete_val_set = sub_val_set
            complete_test_set = sub_test_set
        else:
            _append_dataset(complete_train_set, sub_train_set)
            if len(sub_val_set) > 0:
                _append_dataset(complete_val_set, sub_val_set)
            _append_dataset(complete_test_set, sub_test_set)

    # print("complete train set len ", len(complete_train_set))
    #
    # dataset_loader = torch.utils.data.DataLoader(complete_train_set, shuffle=False)
    # k_xy = []
    #
    # for input, output in dataset_loader:
    #     output = np.squeeze(output.numpy())
    #     k_xy.append(output[2])
    #
    # np.savez_compressed(os.path.join("/home/martin/Documents/MLMC-DFM", "fr_div_0_k_yy"), data=np.array(k_xy))
    # exit()


    #plot_dataset(torch.utils.data.DataLoader(complete_train_set, shuffle=False))

    data_init_transform, data_input_transform, data_output_transform = prepare_dataset(study, config, data_dir, train_dataset=complete_train_set)
    complete_train_set.init_transform = data_init_transform
    complete_train_set.input_transform = data_input_transform
    complete_train_set.output_transform = data_output_transform
    if len(complete_val_set) > 0:
        complete_val_set.init_transform = data_init_transform
        complete_val_set.input_transform = data_input_transform
        complete_val_set.output_transform = data_output_transform
    complete_test_set.init_transform = data_init_transform
    complete_test_set.input_transform = data_input_transform
    complete_test_set.output_transform = data_output_transform

    dataset = copy.deepcopy(complete_train_set)
    if len(complete_val_set) > 0:
        _append_dataset(dataset, complete_val_set)
    _append_dataset(dataset, complete_test_set)

    if serialize_path is not None:
        joblib.dump(dataset, os.path.join(serialize_path, "dataset.pkl"))

    if study is not None:
        study.set_user_attr("n_train_samples", len(complete_train_set))
        study.set_user_attr("n_val_samples", len(complete_val_set))
        study.set_user_attr("n_test_samples", len(complete_test_set))

    # dataset_loader = torch.utils.data.DataLoader(complete_train_set, shuffle=False)
    #
    # for input, output in dataset_loader:
    #     output = np.squeeze(output.numpy())
    #     print("output shape ", output.shape)
    #     print("type output ", output)
    #     print("output[1] ", output[1])
    #     # print("output squeeze ", np.squeeze(output))
    #     k_xy.append(output[1])
    #
    #     exit()

    return complete_train_set, complete_val_set, complete_test_set


# def _split_dataset(dataset, config, n_train_samples):
#     if n_train_samples is None:
#         n_train_samples = int(len(dataset) * config["train_samples_ratio"])
#
#     n_train_samples = np.min([n_train_samples, int(len(dataset) * config["train_samples_ratio"])])
#
#     print("n train samples ", n_train_samples)
#
#     train_val_set = dataset[:n_train_samples]
#     if config["val_samples_ratio"] == 0.0:
#         train_set = train_val_set
#         validation_set = []
#     else:
#         train_set = train_val_set[:-int(n_train_samples * config["val_samples_ratio"])]
#         validation_set = train_val_set[-int(n_train_samples * config["val_samples_ratio"]):]
#
#     if "n_test_samples" in config and config["n_test_samples"] is not None:
#         n_test_samples = config["n_test_samples"]
#         test_set = dataset[-n_test_samples:]
#     else:
#         test_set = dataset[n_train_samples:]
#
#     return train_set, validation_set, test_set

def _split_dataset(dataset, config, n_train_samples=None):
    # Step 1: Extract metadata (sex and age)
    metadata = []
    for i in range(len(dataset)):
        _, (sex, age_norm) = dataset[i]
        age = age_norm * 100.0  # denormalize
        metadata.append((sex, age))

    #min_samples = 20

    metadata = pd.DataFrame(metadata, columns=["sex", "age"])

    # Step 2: Create stratification column (e.g. "F_2" = Female in age bin 2)
    metadata["sex"] = metadata["sex"].astype(int)

    # # Sort by age
    # metadata_sorted = metadata.sort_values("age").reset_index()
    #
    # print("metadata_sorted ", metadata_sorted)
    #
    # # Assign bin IDs based on min_samples
    # bin_ids = np.repeat(np.arange(len(metadata_sorted) // min_samples + 1), min_samples)[:len(metadata_sorted)]
    # metadata_sorted["age_bin"] = bin_ids
    #
    # # Merge back to original order
    # metadata["age_bin"] = metadata_sorted.set_index("index")["age_bin"]

    #metadata["age_bin"] = pd.cut(metadata["age"], bins=[0, 30, 45, 60, 75, 100], labels=False)
    metadata["strata"] = metadata["sex"].astype(str) #+ "_" + metadata["age_bin"].astype(str)

    all_indices = np.arange(len(dataset))

    # Step 3: How many samples to use?
    if n_train_samples is None:
        n_train_samples = int(len(dataset) * config["train_samples_ratio"])

    n_train_samples = min(n_train_samples, int(len(dataset) * config["train_samples_ratio"]))
    print("n train samples ", n_train_samples)

    # Step 4: Stratified train_val/test split
    if "n_test_samples" in config and config["n_test_samples"] is not None:
        test_size = config["n_test_samples"]
    else:
        test_size = len(dataset) - n_train_samples

    # Step 4: Train/test split
    if test_size > 0:
        train_val_idx, test_idx = train_test_split(
            all_indices,
            test_size=test_size,
            stratify=metadata["sex"],  # stratify by sex only
            random_state=config["seed"]
        )
    else:
        train_val_idx = all_indices
        test_idx = []

    # Step 5: Optional validation split
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

    # Step 6: Return Subsets
    train_set = Subset(dataset, train_idx)
    validation_set = Subset(dataset, val_idx) if len(val_idx) > 0 else []
    test_set = Subset(dataset, test_idx) if len(test_idx) > 0 else []

    return train_set, validation_set, test_set


def prepare_dataset(study, config, data_dir, serialize_path=None, train_dataset=None):
    data_file_name = None
    if "data_file_name" in config:
        data_file_name = config["data_file_name"]

    dataset = BoneDatasetCT(data_dir=data_dir, data_file_name=data_file_name)

    print("len dataset ", len(dataset))

    n_train_samples = None
    if "n_train_samples" in config and config["n_train_samples"] is not None:
        n_train_samples = config["n_train_samples"]

    train_set, validation_set, test_set = _split_dataset(dataset, config, n_train_samples)

    study.set_user_attr("data_dir", dataset.data_dir)
    study.set_user_attr("global_min_value", dataset._global_min_value)
    study.set_user_attr("global_max_value", dataset._global_max_value)

    if serialize_path is not None:
        joblib.dump(train_set, os.path.join(serialize_path, "train_dataset.pkl"))
        joblib.dump(validation_set, os.path.join(serialize_path, "val_dataset.pkl"))
        joblib.dump(test_set, os.path.join(serialize_path, "test_dataset.pkl"))

    return train_set, validation_set, test_set

    # # ===================================
    # # Get mean and std for each channel
    # # ===================================
    # input_mean, input_std = 0, 1
    # data_normalizer = NormalizeData()
    # data_normalizer.input_indices = config["input_channels"]
    #
    # n_train_samples = None
    # if "n_train_samples" in config and config["n_train_samples"] is not None:
    #     n_train_samples = config["n_train_samples"]
    #
    # init_transform = []
    # input_transform_list = []
    #
    # ###########################
    # ## Initial normalization ##
    # ###########################
    #
    # if config["init_norm"]:
    #     init_transform.append(transforms.Lambda(init_norm))
    #
    # ####################
    # ## Log transforms ##
    # ####################
    # if config["log_input"]:
    #     if "log_all_input_channels" in config and config["log_all_input_channels"]:
    #         input_transform_list.append(transforms.Lambda(log_all_data))
    #     else:
    #         input_transform_list.append(transforms.Lambda(log_data))
    #
    # # # ########################
    # # # ## Quantile Transform ##
    # # # ########################
    # input_transform_list = features_transform(config, data_dir, input_transform_list, input_channels=config["input_channels"] if "input_channels" in config else None)
    #
    # input_transform = transforms.Compose(input_transform_list)
    #
    # if len(init_transform) > 0:
    #     init_transform = transforms.Compose(init_transform)
    # else:
    #     init_transform = None
    #
    # if config["normalize_input"]:
    #     if train_dataset is not None:
    #         dataset_for_mean_std = train_dataset
    #         dataset_for_mean_std.init_transform = init_transform
    #         dataset_for_mean_std.input_transform = input_transform
    #     else:
    #         dataset_for_mean_std = BoneDatasetCT(data_dir=data_dir,
    #                                           input_transform=input_transform,
    #                                            input_channels=config["input_channels"] if "input_channels" in config else None,)
    #
    #     dataset_for_mean_std.shuffle(seed=config["seed"])
    #
    #     if n_train_samples is None:
    #         n_train_samples = int(len(dataset_for_mean_std) * config["train_samples_ratio"])
    #
    #     n_train_samples = np.min([n_train_samples, int(len(dataset_for_mean_std) * config["train_samples_ratio"])])
    #
    #     # train_val_set = dataset_for_mean_std[:n_train_samples]
    #     # if config["val_samples_ratio"] == 0:
    #     #     train_set = train_val_set
    #     # else:
    #     #     train_set = train_val_set[:-int(n_train_samples * config["val_samples_ratio"])]
    #
    #     print("len(train_val_set) ", len(dataset_for_mean_std))
    #     train_loader_mean_std = DataLoader(dataset_for_mean_std, batch_size=1, shuffle=False) #@TODO: use train_set AGAIN
    #     iqr = []
    #     if "output_iqr_scale" in config:
    #         iqr = config["output_iqr_scale"]
    #     input_mean, input_std = get_mean_std(train_loader_mean_std, output_iqr=iqr, input_channels=config["input_channels"])
    #     print("input mean: {}, std:{}".format(input_mean, input_std))
    #
    # # =======================
    # # data transforms
    # # =======================
    # input_transformations = []
    # init_transform = []
    #
    # ###########################
    # ## Initial normalization ##
    # ###########################
    # data_init_transform = None
    # if config["init_norm"]:
    #     init_transform.append(transforms.Lambda(init_norm))
    #
    # if len(init_transform) > 0:
    #     data_init_transform = transforms.Compose(init_transform)
    #
    # data_input_transform = None
    # # Standardize input
    # if config["log_input"]:
    #     if "log_all_input_channels" in config and config["log_all_input_channels"]:
    #         input_transformations.append(transforms.Lambda(log_all_data))
    #     else:
    #         input_transformations.append(transforms.Lambda(log_data))
    # if config["normalize_input"]:
    #     if "normalize_input_indices" in config:
    #         data_normalizer.input_indices = config["normalize_input_indices"]
    #     data_normalizer.input_mean = input_mean
    #     data_normalizer.input_std = input_std
    #     input_transformations.append(data_normalizer.normalize_input)
    #
    # if len(input_transformations) > 0:
    #     data_input_transform = transforms.Compose(input_transformations)
    #
    # if train_dataset is None:
    #     # ============================
    #     # Datasets and data loaders
    #     # ============================
    #     print("data_input_transform ", data_input_transform)
    #     dataset = BoneDatasetCT(data_dir=data_dir,
    #                          input_transform=data_input_transform,
    #                           input_channels=config["input_channels"] if "input_channels" in config else None,)
    #     dataset.shuffle(config["seed"])
    #
    #     #train_set, validation_set, test_set = _split_dataset(dataset, config, n_train_samples)
    #
    #     # train_loader_mean_std = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size_train"], shuffle=False)
    #     # input_mean, input_std = get_mean_std(train_loader_mean_std)
    #     # print("DATASET input mean: {}, std:{}".format(input_mean, input_std))
    #
    # else:
    #     train_dataset.init_transform = data_init_transform
    #     train_dataset.input_transform = data_input_transform
    #
    # if "input_transform" in config or "output_transform" in config:
    #     input_transformations = features_transform(config, data_dir, input_transformations,
    #                                                input_channels=config["input_channels"] if "input_channels" in config else None)
    #     if len(input_transformations) > 0:
    #         data_input_transform = transforms.Compose(input_transformations)
    #
    #     dataset = BoneDatasetCT(data_dir=data_dir,
    #                          input_transform=data_input_transform,
    #                           input_channels=config["input_channels"] if "input_channels" in config else None)
    #     dataset.shuffle(config["seed"])
    #
    #     #train_set, validation_set, test_set = _split_dataset(dataset, config, n_train_samples)
    #     #
    #     # train_loader_mean_std = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"],
    #     #                                                     shuffle=False)
    #     # input_mean, input_std, output_mean, output_std, _ = get_mean_std(train_loader_mean_std)
    #     # print("TRAIN SET input mean: {}, std:{}, output mean: {}, std: {}".format(input_mean, input_std, output_mean, output_std))
    #     #
    #     # train_loader_mean_std = torch.utils.data.DataLoader(validation_set, batch_size=config["batch_size_train"],
    #     #                                                     shuffle=False)
    #     # input_mean, input_std, output_mean, output_std, _ = get_mean_std(train_loader_mean_std)
    #     # print("VAL SET input mean: {}, std:{}, output mean: {}, std: {}".format(input_mean, input_std, output_mean,
    #     #                                                                           output_std))
    #     #
    #     # train_loader_mean_std = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size_train"],
    #     #                                                     shuffle=False)
    #     # input_mean, input_std, output_mean, output_std, _ = get_mean_std(train_loader_mean_std)
    #     # print("TEST SET input mean: {}, std:{}, output mean: {}, std: {}".format(input_mean, input_std, output_mean,
    #     #                                                                           output_std))
    #     #
    #     # exit()
    #
    # if study is not None:
    #     if train_dataset is None:
    #         study.set_user_attr("n_samples", len(dataset))
    #         #study.set_user_attr("n_val_samples", len(validation_set))
    #         #study.set_user_attr("n_test_samples", len(test_set))
    #
    #     if "normalize_input_indices" in config:
    #         study.set_user_attr("normalize_input_indices", config["normalize_input_indices"])
    #
    #     if "normalize_output_indices" in config:
    #         study.set_user_attr("normalize_output_indices", config["normalize_output_indices"])
    #
    #     if "log_all_input_channels" in config:
    #         study.set_user_attr("log_all_input_channels", config["log_all_input_channels"])
    #
    #     if "output_transform" in config:
    #         study.set_user_attr("output_transform", config["output_transform"])
    #
    #     if "input_transform" in config:
    #         study.set_user_attr("input_transform", config["input_transform"])
    #
    #     if "init_norm_use_all_features" in config:
    #         study.set_user_attr("init_norm_use_all_features", config["init_norm_use_all_features"])
    #
    #     study.set_user_attr("init_norm", config["init_norm"])
    #     study.set_user_attr("normalize_input", config["normalize_input"])
    #     study.set_user_attr("normalize_output", config["normalize_output"])
    #
    #     study.set_user_attr("input_log", config["log_input"])
    #     study.set_user_attr("input_mean", input_mean)
    #     study.set_user_attr("input_std", input_std)
    #
    #     study.set_user_attr("data_dir", dataset.data_dir)
    #
    # if train_dataset is not None:
    #     return data_init_transform, data_input_transform

    # if serialize_path is not None:
    #    joblib.dump(dataset, os.path.join(serialize_path, "dataset.pkl"))

    return dataset


