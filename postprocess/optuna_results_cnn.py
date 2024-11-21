import os
import sys
import argparse
import joblib
import torch
import numpy as np
import torchvision.transforms as transforms
#from datasets.bone_dataset import BoneDataset
from datasets.cnn_diffusion.bone_dataset_CT import BoneDatasetCT
#from torch.utils.tensorboard import SummaryWriter
from visualization.visualize_data import plot_train_valid_loss
# from metamodel.cnn.models.auxiliary_functions import exp_data, get_eigendecomp, get_mse_nrmse_r2, get_mean_std, log_data, exp_data,\
#     quantile_transform_fit, QuantileTRF, NormalizeData, log_all_data, init_norm, get_mse_nrmse_r2_eigh, log10_data, log10_all_data, power_10_all_data, power_10_data
#from models.graph_diffusion.gnn_models import GNN
from models.cnn_diffusion.UNet import UNet, SimpleUNet, UNet3DWithTimestep
from models.schedulers import NoiseScheduler
#import datasets.dataset_preprocessing
import matplotlib.pyplot as plt
import scipy as sc
import pyvista as pv

#os.environ["CUDA_VISIBLE_DEVICES"]=""


def preprocess_dataset(config, user_attrs, data_dir, results_dir=None):
    print("user attrs ", user_attrs)
    output_file_name = "output_tensor.npy"
    if "vel_avg" in config and config["vel_avg"]:
        output_file_name = "output_vel_avg.npy"

    # ===================================
    # Get mean and std for each channel
    # ===================================
    input_mean, output_mean, input_std, output_std = 0, 0, 1, 1
    data_normalizer = NormalizeData()

    n_train_samples = None
    if "n_train_samples" in config and config["n_train_samples"] is not None:
        n_train_samples = config["n_train_samples"]

    input_transformations = []
    output_transformations = []

    data_input_transform, data_output_transform = None, None
    # Standardize input
    if config["log_input"]:
        if "log_all_input_channels" in config and config["log_all_input_channels"]:
            input_transformations.append(transforms.Lambda(log_all_data))
        else:
            input_transformations.append(transforms.Lambda(log_data))
    if config["normalize_input"]:
        if "normalize_input_indices" in config:
            data_normalizer.input_indices = config["normalize_input_indices"]
        data_normalizer.input_mean = user_attrs["input_mean"]
        data_normalizer.input_std = user_attrs["input_std"]
        input_transformations.append(data_normalizer.normalize_input)

    if len(input_transformations) > 0:
        data_input_transform = transforms.Compose(input_transformations)

    #print("input transforms ", input_transformations)

    print("data dir ", data_dir)
    print("results dir ", results_dir)

    # Standardize output
    if config["log_output"]:
        output_transformations.append(transforms.Lambda(log_data))
    elif config["log10_output"]:
        output_transformations.append(transforms.Lambda(log10_data))
    elif config["log10_all_output"]:
        output_transformations.append(transforms.Lambda(log10_all_data))
    elif config["log_all_output"]:
        output_transformations.append(transforms.Lambda(log_all_data))
    if config["normalize_output"]:
        if "normalize_output_indices" in config:
            data_normalizer.input_indices = config["normalize_output_indices"]
        data_normalizer.output_mean = user_attrs["output_mean"]
        data_normalizer.output_std = user_attrs["output_std"]
        output_transformations.append(data_normalizer.normalize_output)

    if os.path.exists(os.path.join(results_dir, "output_transform.pkl")):
        out_trans_obj = joblib.load(os.path.join(results_dir, "output_transform.pkl"))
        quantile_trf_obj = QuantileTRF()
        quantile_trf_obj.quantile_trfs_out = out_trans_obj
        output_transformations.append(quantile_trf_obj.quantile_transform_out)
    else:
        if "output_transform" in config and config["output_transform"]:
            raise Exception("{} not exists".format(os.path.join(results_dir, "output_transform.pkl")))


    if len(output_transformations) > 0:
        data_output_transform = transforms.Compose(output_transformations)

    init_transform = []
    data_init_transform = None
    if "init_norm" in study.user_attrs and study.user_attrs["init_norm"]:
        init_transform.append(transforms.Lambda(init_norm))

    if len(init_transform) > 0:
        data_init_transform = transforms.Compose(init_transform)

    print("preprocess data input transform ", data_input_transform)
    print("preprocess data output transform ", data_output_transform)


    # ============================
    # Datasets and data loaders
    # ============================
    dataset = DFMDataset(data_dir=data_dir,
                         output_file_name=output_file_name,
                         init_transform=data_init_transform,
                         input_transform=data_input_transform,
                         output_transform=data_output_transform,
                         two_dim=True,
                         input_channels=config["input_channels"] if "input_channels" in config else None,
                         output_channels=config["output_channels"] if "output_channels" in config else None,
                         fractures_sep=config["fractures_sep"] if "fractures_sep" in config else False,
                         cross_section=config["cross_section"] if "cross_section" in config else False,
                         init_norm_use_all_features=config["init_norm_use_all_features"] if "init_norm_use_all_features" in config else False
                         )
    dataset.shuffle(config["seed"])
    print("len dataset ", len(dataset))

    if n_train_samples is None:
        n_train_samples = int(len(dataset) * config["train_samples_ratio"])

    n_train_samples = np.min([n_train_samples, int(len(dataset) * config["train_samples_ratio"])])

    train_val_set = dataset[:n_train_samples]
    train_set = train_val_set[:-int(n_train_samples * config["val_samples_ratio"])]
    validation_set = train_val_set[-int(n_train_samples * config["val_samples_ratio"]):]

    if "n_test_samples" in config and config["n_test_samples"] is not None:
        n_test_samples = config["n_test_samples"]
        test_set = dataset[-n_test_samples:]
    else:
        test_set = dataset[n_train_samples:]

    return train_set[:5000], validation_set, test_set

    # print("len(trainset): {}, len(valset): {}, len(testset): {}".format(len(train_set), len(validation_set),
    #                                                                     len(test_set)))


def get_calibrator():
    return IsotonicRegression(out_of_bounds="clip")
    #return LogisticRegression()

def calibrator_fit(calibrator, predictions, targets):
    #predictions = np.array(predictions).reshape(-1, 1)
    calibrator.fit(predictions, targets)

def calibrator_predict(calibrator, data):
    return calibrator.predict(data)
    #return calibrator(data)[:, 1]

def calibrate_data(data_list, calibs):
    data_array = np.array(data_list)
    #print("data_array.shape ", data_array.shape)
    all_data = []
    for idx, calib in enumerate(calibs):
        cal_data = calib.predict(data_array[:, idx])
        all_data.append(cal_data)

    # pred_0, pred_1, pred_2 = pred_array[:, 0], pred_array[:, 1], pred_array[:, 2]
    #
    # cal_pred_0 = calibrator_0.predict(pred_0)
    # cal_pred_1 = calibrator_1.predict(pred_1)
    # cal_pred_2 = calibrator_2.predict(pred_2)

    return list(np.stack(all_data, axis=1))


def get_saved_model_path(results_dir, best_trial):
    #print(best_trial.user_attrs["model_name"])
    diff_model_path = 'trial_{}_losses_model_{}'.format(best_trial.number, best_trial.user_attrs["diff_model_name"])

    #model_path = "trial_2_losses_model_cnn_net"

    #@TODO: remove ASAP
    #return "/home/martin/Documents/MLMC-DFM/optuna_runs/karolina/cond_conv/exp_2/seed_12345/trial_1_losses_model_cnn_net"
    #return "/home/martin/Documents/MLMC-DFM/optuna_runs/karolina/pooling/new_experiments/exp_13_4_s_36974/seed_12345/trial_0_losses_model_cnn_net"

    # for key, value in best_trial.params.items():
    #     model_path += "_{}_{}".format(key, value)

    #@TODO: remove ASAP
    #model_path = "trial_26_train_samples_15000"
    #model_path = "/home/martin/Documents/MLMC-DFM/optuna_runs/karolina/cnn_3_3/exp_2/seed_12345/trial_6_losses_model_cnn_net_max_channel_72_n_conv_layers_1_kernel_size_3_stride_1_pool_None_pool_size_0_pool_stride_0_lr_0.005_use_batch_norm_True_max_hidden_neurons_48_n_hidden_layers_1"
    return os.path.join(results_dir, diff_model_path)


def load_dataset(results_dir, study):
    data_dir = study.user_attrs["data_dir"]
    dataset = BoneDatasetCT(data_dir=data_dir)

    return dataset

    # n_train_samples = study.user_attrs["n_train_samples"]
    # n_val_samples = study.user_attrs["n_val_samples"]
    # train_val_set = dataset[:(n_train_samples + n_val_samples)]
    #
    # if n_val_samples == 0:
    #     train_set = train_val_set
    #     validation_set = []
    # else:
    #     train_set = train_val_set[:-n_val_samples]
    #     validation_set = train_val_set[-n_val_samples:]
    # n_test_samples = study.user_attrs["n_test_samples"]
    # test_set = dataset[-n_test_samples:]
    #
    # return train_set, validation_set, test_set



def get_rotation_angles(model, loader, inverse_transform):
    targets_list, predictions_list = [], []
    inv_targets_list, inv_predictions_list = [], []

    for i, test_sample in enumerate(loader):
        inputs, targets = test_sample
        inputs = inputs.float()

        if args.cuda and torch.cuda.is_available():
            inputs = inputs.cuda()
        predictions = model(inputs)

        if len(targets.size()) > 1 and np.sum(targets.size()) / len(targets.size()) != 1:
            targets = torch.squeeze(targets.float())
        targets_np = targets.numpy()
        targets_list.append(targets_np)

        if len(predictions.size()) > 1 and np.sum(predictions.size()) / len(predictions.size()) != 1:
            predictions = torch.squeeze(predictions.float())
        predictions = predictions.cpu()
        predictions_np = predictions.cpu().numpy()


        predictions_list.append(predictions_np)

        inv_targets = targets
        inv_predictions = predictions
        if inverse_transform is not None:
            inv_targets = inverse_transform(torch.reshape(targets, (*targets.shape, 1, 1)))
            inv_predictions = inverse_transform(torch.reshape(predictions, (*predictions.shape, 1, 1)))

            inv_targets = np.reshape(inv_targets, targets.shape)
            inv_predictions = np.reshape(inv_predictions, predictions.shape)

        inv_targets_list.append(inv_targets.numpy())
        inv_predictions_list.append(inv_predictions.numpy())

    all_targets = np.array(targets_list)
    all_predictions = np.array(predictions_list)

    if len(all_targets.shape) == 1:
        all_targets = np.reshape(all_targets, (*all_targets.shape, 1))
        all_predictions = np.reshape(all_predictions, (*all_predictions.shape, 1))

    n_channels = 1 if len(all_targets.shape) == 1 else all_targets.shape[1]

    angles = []
    for i in range(n_channels):
        k_target, k_predict = all_targets[:, i], all_predictions[:, i]
        angle, slope, intercept = get_channel_angles(k_target, k_predict)
        angles.append((angle, slope, intercept))
    return angles


# def find_predictions_transform(k_target, k_predict):
#     print("k target ", k_target)
#     print("k predict ", k_predict)
#     pass

def get_channel_angles(targets, predictions):
    slope, intercept = np.polyfit(targets, predictions, 1)
    angle = np.pi/4 - np.arctan(slope)
    return angle, slope, intercept


def rotate_by_angle(targets, predictions, angles):
    n_channels = targets.shape[0]
    rotated_predictions = []

    for i in range(n_channels):
        angle, shift, intercept = angles[i]
        #print("orig predictions ", predictions[i])
        predictions[i] -= shift
        #print("shifted predictions ", predictions[i])
        #yr = (predictions[i] * np.cos(angle)) + shift
        #print("yr ", yr)
        yr = (targets[i] * np.sin(angle)) + (predictions[i] * np.cos(angle)) + shift
        #print("yr tr pr", yr)
        rotated_predictions.append(yr)
    return torch.from_numpy(np.array(rotated_predictions))


def get_inverse_transform_input(study):
    inverse_transform = None
    print("study.user_attrs", study.user_attrs)

    if "normalize_input" in study.user_attrs and study.user_attrs["normalize_input"]:
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

        inverse_transform = transforms.Compose(transforms_list)

    if "init_norm" in study.user_attrs and study.user_attrs["init_norm"]:
        print("init norm ")

    return inverse_transform


def get_transform(study, results_dir=None):
    input_transformations = []
    output_transformations = []
    init_transform = []

    data_normalizer = NormalizeData()

    ###########################
    ## Initial normalization ##
    ###########################
    data_init_transform = None
    if "init_norm" in study.user_attrs and study.user_attrs["init_norm"]:
        init_transform.append(transforms.Lambda(init_norm))

    if len(init_transform) > 0:
        data_init_transform = transforms.Compose(init_transform)

    # if "input_transform" in study.user_attrs or "output_transform" in study.user_attrs:
    #     input_transformations, output_transformations = features_transform(config, data_dir, output_file_name,
    #                                                                        input_transformations,
    #                                                                        output_transformations, train_set)

    data_input_transform, data_output_transform = None, None
    # Standardize input
    if "input_log" in study.user_attrs and study.user_attrs["input_log"]:
        if "log_all_input_channels" in study.user_attrs and study.user_attrs["log_all_input_channels"]:
            input_transformations.append(transforms.Lambda(log_all_data))
        else:
            input_transformations.append(transforms.Lambda(log_data))
    if "normalize_input" in study.user_attrs and study.user_attrs["normalize_input"]:
        if "normalize_input_indices" in study.user_attrs:
            data_normalizer.input_indices = study.user_attrs["normalize_input_indices"]
        data_normalizer.input_mean = study.user_attrs["input_mean"]
        data_normalizer.input_std = study.user_attrs["input_std"]
        input_transformations.append(data_normalizer.normalize_input)

    if len(input_transformations) > 0:
        data_input_transform = transforms.Compose(input_transformations)

    # Standardize output
    if "output_log" in study.user_attrs and study.user_attrs["output_log"]:
        output_transformations.append(transforms.Lambda(log_data))
    if "normalize_output" in study.user_attrs and study.user_attrs["normalize_output"]:
        if "normalize_output_indices" in study.user_attrs:
            data_normalizer.output_indices = study.user_attrs["normalize_output_indices"]
        data_normalizer.output_mean = study.user_attrs["output_mean"]
        data_normalizer.output_std = study.user_attrs["output_std"]
        if "output_quantiles" in study.user_attrs:
            data_normalizer.output_quantiles = study.user_attrs["output_quantiles"]
        output_transformations.append(data_normalizer.normalize_output)

    transforms_list = []
    if ("output_transform" in study.user_attrs and len(study.user_attrs["output_transform"]) > 0) \
            or os.path.exists(os.path.join(results_dir, "output_transform.pkl")):
        output_transform = joblib.load(os.path.join(results_dir, "output_transform.pkl"))
        quantile_trf_obj = QuantileTRF()
        quantile_trf_obj.quantile_trfs_out = output_transform
        transforms_list.append(quantile_trf_obj.quantile_inv_transform_out)

    if len(output_transformations) > 0:
        data_output_transform = transforms.Compose(output_transformations)

    return data_init_transform, data_input_transform, data_output_transform


def get_inverse_transform(study, results_dir=None):
    inverse_transform = None
    print("study.user_attrs", study.user_attrs)

    transforms_list = []
    # if ("output_transform" in study.user_attrs and len(study.user_attrs["output_transform"]) > 0) \
    #         or os.path.exists(os.path.join(results_dir, "output_transform.pkl")):
    #     output_transform = joblib.load(os.path.join(results_dir, "output_transform.pkl"))
    #     quantile_trf_obj = QuantileTRF()
    #     quantile_trf_obj.quantile_trfs_out = output_transform
    #     transforms_list.append(quantile_trf_obj.quantile_inv_transform_out)

    if "normalize_output" in study.user_attrs and study.user_attrs["normalize_output"]:
        std = 1/study.user_attrs["output_std"]
        zeros_mean = np.zeros(len(study.user_attrs["output_mean"]))

        # print("output_mean ", study.user_attrs["output_mean"])
        # print("output_std ",  study.user_attrs["output_std"])

        ones_std = np.ones(len(zeros_mean))
        mean = -study.user_attrs["output_mean"]

        transforms_list.extend([transforms.Normalize(mean=zeros_mean, std=std),
                            transforms.Normalize(mean=mean, std=ones_std)])

        if "output_log" in study.user_attrs and study.user_attrs["output_log"]:
            print("output log to transform list")
            transforms_list.append(transforms.Lambda(exp_data))
        elif "log10_output" in study.user_attrs and study.user_attrs["log10_output"]:
            print("log10_output to transform list")
            transforms_list.append(transforms.Lambda(power_10_data()))
        elif "log10_all_output" in study.user_attrs and study.user_attrs["log10_all_output"]:
            print("log10_all_output to transform list")
            #transforms_list.append(transforms.Lambda(exp_data))
            transforms_list.append(transforms.Lambda(power_10_all_data))
        elif "log_all_output" in study.user_attrs and study.user_attrs["log_all_output"]:
            print("log_all_output to transform list")
            transforms_list.append(transforms.Lambda(exp_data))

        inverse_transform = transforms.Compose(transforms_list)

    print("inverse transform ", inverse_transform)

    return inverse_transform


def renormalize_data(dataset, study, input=False, output=False):
    print("dataset.input_transform ", dataset.input_transform)

    import copy

    new_dataset = copy.deepcopy(dataset)

    if input:
        transforms_list = []
        if "input_log" in study.user_attrs and study.user_attrs["input_log"]:
            transforms_list.append(transforms.Lambda(log_data))

        input_transform = transforms.Compose(transforms_list)

        new_dataset.input_transform = input_transform #None
        #new_dataset.input_transform = None

        loader = torch.utils.data.DataLoader(new_dataset, shuffle=False)

        input_mean, input_std, output_mean, output_std, _ = get_mean_std(loader)
        print("Test loader, INPUT mean: {}, std: {}, OUTPUT mean: {}, std: {}".format(input_mean, input_std, output_mean, output_std))

        if "normalize_input" in study.user_attrs and study.user_attrs["normalize_input"]:
            transforms_list.append(transforms.Normalize(mean=input_mean, std=input_std))

        input_transform = transforms.Compose(transforms_list)

        new_dataset.input_transform = input_transform
        # new_dataset.input_transform = None

        #loader = torch.utils.data.DataLoader(new_dataset, shuffle=False)

    if output:
        # transforms_list = []
        # if "normalize_output" in study.user_attrs and study.user_attrs["normalize_output"]:
        #     std = 1 / study.user_attrs["output_std"]
        #     zeros_mean = np.zeros(len(study.user_attrs["output_mean"]))
        #     print("output_mean ", study.user_attrs["output_mean"])
        #     print("output_std ", study.user_attrs["output_std"])
        #     ones_std = np.ones(len(zeros_mean))
        #     mean = -study.user_attrs["output_mean"]
        #     transforms_list = [transforms.Normalize(mean=zeros_mean, std=std),
        #                        transforms.Normalize(mean=mean, std=ones_std)]
        #
        # if "output_log" in study.user_attrs and study.user_attrs["output_log"]:
        #     print("output log to transform list")
        #     transforms_list.append(transforms.Lambda(exp_data))
        #
        # inverse_transform = transforms.Compose(transforms_list)

        # if "output_log" in study.user_attrs and study.user_attrs["output_log"]:
        #     transforms_list.append(transforms.Lambda(exp_data))
        # output_transform = transforms.Compose(transforms_list)
        #
        # new_dataset.output_transform = output_transform
        # # new_dataset.input_transform = None

        transforms_list = []
        new_dataset.output_transform = None

        loader = torch.utils.data.DataLoader(new_dataset, shuffle=False)

        input_mean, input_std, output_mean, output_std, _ = get_mean_std(loader)
        print(
            "Test loader, INPUT mean: {}, std: {}, OUTPUT mean: {}, std: {}".format(input_mean, input_std, output_mean,
                                                                                    output_std))

        if "normalize_output" in study.user_attrs and study.user_attrs["normalize_output"]:
            transforms_list.append(transforms.Normalize(mean=output_mean, std=output_std))

        output_transform = transforms.Compose(transforms_list)

        new_dataset.output_transform = output_transform
        # new_dataset.input_transform = None

    loader = torch.utils.data.DataLoader(new_dataset, shuffle=False)

    input_mean, input_std, output_mean, output_std, _ = get_mean_std(loader)
    print("Renormalized loader, INPUT mean: {}, std: {}, OUTPUT mean: {}, std: {}".format(input_mean, input_std, output_mean,
                                                                                  output_std))

    return loader


def plot_target_data(train_loader, validation_loader, test_loader):
    import matplotlib.pyplot as plt
    train_targets = []
    val_targets = []
    test_targets = []
    test_targets = []

    k_xx_list_input = []
    k_xy_list_input = []
    k_yy_list_input = []
    for i, test_sample in enumerate(train_loader):
        inputs, targets = test_sample
        inputs = np.squeeze(inputs.numpy())
        print("inputs ", inputs)
        if i > 50:
            break
        print("inputs[0].ravel() ", inputs[0].ravel())
        k_xx_list_input.extend(inputs[0].ravel())
        k_xy_list_input.extend(inputs[1].ravel())
        k_yy_list_input.extend(inputs[2].ravel())

    print("kxx shape ", np.array(k_xx_list_input).shape)
    plt.hist(k_xx_list_input, bins=25, color="red", label="input k_xx", density=True)
    #plt.xlim([-0.001, 1])
    plt.legend()
    plt.show()

    plt.hist(k_xy_list_input, bins=60, color="red", label="input k_xy", density=True)
    plt.legend()
    plt.show()

    plt.hist(k_yy_list_input, bins=60, color="red", label="input k_yy", density=True)
    plt.legend()
    plt.show()

    for i, test_sample in enumerate(test_loader): #@TODO: use train_loader again
        inputs, targets = test_sample
        targets_np = np.squeeze(targets.numpy())
        #print("targets_np ", targets_np)
        train_targets.append(targets_np)
        #exit()

    # for i, test_sample in enumerate(validation_loader):
    #     inputs, targets = test_sample
    #     targets_np = np.squeeze(targets.numpy())
    #     val_targets.append(targets_np)
    #
    # for i, test_sample in enumerate(test_loader):
    #     inputs, targets = test_sample
    #     targets_np = np.squeeze(targets.numpy())
    #     test_targets.append(targets_np)

    train_targets = np.array(train_targets)
    #val_targets = np.array(val_targets)
    #test_targets = np.array(test_targets)

    # ########################
    # # Quantile transform  ##
    # ########################
    # print("QUANTILE TRANSFORM")
    # quantile_trf_obj = QuantileTRF()
    # quantile_trfs_out = quantile_transform_fit(train_targets.T,
    #                                            indices=[1],
    #                                            transform_type="QuantileTransform")
    # quantile_trf_obj.quantile_trfs_out = quantile_trfs_out
    #
    # print("train targets shape ", train_targets.shape)
    # trf_train_targets = quantile_trf_obj.quantile_transform_out_real(train_targets.T)
    #
    # print("trf train tragets shape ", trf_train_targets.shape)
    # plt.hist(trf_train_targets[1, :], bins=1000, color="red", label="QuantileTRF train", alpha=0.5, density=True)
    # plt.legend()
    # plt.show()

    # trf_test_targets = quantile_trf_obj.quantile_transform_out_real(test_targets.T)
    #
    # print("trf test tragets shape ", trf_test_targets.shape)
    # print("trf test targets data ", trf_test_targets[1, :100])
    # plt.hist(trf_test_targets[1, :], bins=1000, color="red", label="QuantileTRF test", alpha=0.5, density=True)
    # plt.legend()
    # plt.show()

    n_channels = 3
    for i in range(n_channels):
        print("train targets shape ", train_targets.shape)
        k_train_targets = train_targets[:, i]

        #k_train_targets = k_train_targets[k_train_targets < 0.6]
        #k_val_targets = val_targets[:, i]
        #k_test_targets = test_targets[:, i]
        print("min: {}, max: {}, avg:{} ".format(np.min(k_train_targets),
                                                 np.max(k_train_targets),
                                                 np.mean(k_train_targets)))
        print("k_train_targets[:100] ", k_train_targets[:100])
        print("k_train_targets ", k_train_targets.shape)
        print("k train targets ")

        density = True
        plt.hist(k_train_targets, bins=60, color="red", label="train, ch: {}".format(i), alpha=0.5, density=density)

        #plt.hist(k_val_targets, bins=60, color="blue", label="val", alpha=0.5, density=density)
        #plt.hist(k_test_targets, bins=60, color="green", label="test", alpha=0.5, density=density)
        #plt.xlabel(xlabel)
        # plt.ylabel("Frequency for relative")
        # if i == 1:
        #     plt.xlim([-1, 1])
        plt.legend()
        #plt.savefig("hist_" + title + ".pdf")
        plt.show()
        # if i == 1:
        #     b = 0.1
        #     # normal_sample = k_train_targets / (b * np.sqrt(2))
        #     # plt.hist(normal_sample, bins=1000, color="red", label="train, ch: {}".format(i), alpha=0.5, density=density)
        #     # plt.legend()
        #     # plt.show()
        #
        #     n_quantiles = 10000
        #     transformer = QuantileTransformer(n_quantiles=n_quantiles, random_state=0, output_distribution="normal")
        #     transform_obj = transformer.fit(k_train_targets.reshape(-1, 1))
        #
        #     transformed_data = transform_obj.transform(k_train_targets.reshape(-1, 1))
        #     plt.hist(transformed_data, bins=1000, color="red", label="train, ch: {}".format(i), alpha=0.5, density=density)
        #     plt.legend()
        #     plt.show()
        #
        #     exit()

        # exit()

# def get_distrs(dataset, density_only=False):
#     import matplotlib.pyplot as plt
#
#     dataset.
#
#     # density = graphs_features[..., 3]
#     # displacement = graphs_features[..., :3]
#
#     # original_samples = []
#     # for sample_id in range(10):#graphs_features.shape[0]):
#     #     print("density[sample_id].shape ",density[sample_id].shape)
#     #     print("density[sample_id] ", density[sample_id])
#     #
#     #     #print("displacement[sample_id][0] ", displacement[sample_id][0])
#     #
#     #     if not density_only:
#     #         print("displacement[sample_id][:, 0] ", displacement[sample_id][:, 0])
#     #         fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
#     #         axes.hist(displacement[sample_id][:, 0], bins=100, density=True, label="displacement[0]")
#     #         axes.hist(displacement[sample_id][:, 1], bins=100, density=True, label="displacement[1]")
#     #         axes.hist(displacement[sample_id][:, 2], bins=100, density=True, label="displacement[2]")
#     #         fig.legend()
#     #         plt.show()
#
#     fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
#     axes.hist(density[sample_id], bins=100, density=True, label="density")
#     fig.legend()
#     plt.show()
#
#     original_samples.append(density[sample_id])
#
#     return original_samples

def render_3d_scan(scan, title="3D Scan"):
    from mayavi import mlab
    # Create a Mayavi figure
    mlab.figure(size=(800, 800), bgcolor=(1, 1, 1))

    # Create a 3D volume visualization
    src = mlab.pipeline.scalar_field(scan)
    #mlab.pipeline.volume(src, vmin=scan.min(), vmax=scan.max())
    volume = mlab.pipeline.volume(src)
    volume._volume_property.scalar_opacity_unit_distance = 0.1
    #mlab.title(f'3D Feature Map from Layer: {layer}, Feature: {feature_index}')
    mlab.show()


    # import plotly.graph_objects as go
    # # Use Plotly's Volume rendering for 3D data
    # fig = go.Figure(data=go.Volume(
    #     x=np.arange(scan.shape[0]),
    #     y=np.arange(scan.shape[1]),
    #     z=np.arange(scan.shape[2]),
    #     value=scan.flatten(),
    #     opacity=0.1,  # Adjust for visibility
    #     surface_count=20,  # Number of surfaces to render
    #     colorscale="Gray"  # Set color to grayscale
    # ))
    #
    # # Update layout
    # fig.update_layout(
    #     title=title,
    #     scene=dict(
    #         xaxis_title='X',
    #         yaxis_title='Y',
    #         zaxis_title='Z'
    #     )
    # )
    #
    # fig.show()

    # print("scan shape ", scan.shape)
    #
    # # Convert the 3D numpy array to a PyVista volume object
    # grid = pv.ImageData()
    #
    # # Set the grid dimensions (must match the shape of the NumPy array)
    # grid.dimensions = scan.shape
    #
    # # Optionally, specify spacing (change if your data needs different scale)
    # grid.spacing = (1, 1, 1)  # Adjust spacing as needed
    #
    # # Add the NumPy array data to the grid as scalars
    # grid.point_data["values"] = scan.flatten(order="F")
    #
    # grid.plot(show_edges=True)


    # #volume = pv.wrap(scan)
    # plotter = pv.Plotter()
    # plotter.add_volume(grid, cmap="gray")
    # plotter.add_text(title)
    # plotter.show()


def load_models(args, study):
    density_only = True
    results_dir = args.results_dir
    model_path = get_saved_model_path(results_dir, study.best_trial)
    #model_path = "/home/martin/Documents/MLMC-DFM/optuna_runs/karolina/MLMC-DFM_general_dataset_2/pooling/mase_loss/fr_div_0_1_exp_rho_2_5_3/seed_36974/trial_0_losses_model_cnn_net"
    print("model path ", model_path)
    dataset = load_dataset(results_dir, study)

    #original_samples = get_distrs(dataset)

    data_loader = torch.utils.data.DataLoader(dataset, shuffle=False)

    for sample in data_loader:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        axes.hist(sample.flatten(), bins=100, density=True, label="bone density distr")
        fig.legend()
        plt.show()

        render_3d_scan(np.squeeze(sample.numpy()), title="Original 3D Scan")

        break

    print("len(dataset): {}".format(len(dataset)))

    #inverse_transform = get_inverse_transform(study, results_dir)
    #input_inverse_transform = get_inverse_transform_input(study)

    plot_separate_images = False
    # Disable grad
    with torch.no_grad():
        # Initialize model
        #print("diff model kwargs ", study.best_trial.user_attrs["diff_model_kwargs"])
        #diff_model_kwargs = study.best_trial.user_attrs["diff_model_kwargs"]
        # model_kwargs = {'n_conv_layers': 1, 'max_channel': 72, 'pool': 'None', 'pool_size': 0, 'kernel_size': 3,
        #                 'stride': 1, 'pool_stride': 0, 'use_batch_norm': True, 'n_hidden_layers': 1,
        #                 'max_hidden_neurons': 48, 'input_size': 3}

        # if "min_channel" in model_kwargs:
        #     model_kwargs["input_channel"] = 1 #model_kwargs["min_channel"]
        #     model_kwargs["n_output_neurons"] = 1
        #     del model_kwargs["min_channel"]

        #gnn_model = study.best_trial.user_attrs["gnn_model"]
        # gnn_kwargs = {'num_node_attrs': study.best_trial.user_attrs["num_node_attrs"], 'adj_matrix': dataset.adj_matrix,
        #               'edge_indices': dataset.edge_indices, 'gnn_config': study.best_trial.user_attrs["gnn_config"]}
        cnn_model = SimpleUNet(dim=32)

        noise_scheduler = NoiseScheduler(**study.best_trial.user_attrs["noise_scheduler_kwargs"])

        diff_model = study.best_trial.user_attrs["diff_model_class"](noise_scheduler=noise_scheduler, cnn_model=cnn_model)

        # Initialize optimizer
        non_frozen_parameters = [p for p in diff_model.parameters() if p.requires_grad]
        optimizer = None
        if len(non_frozen_parameters) > 0:
            optimizer = study.best_trial.user_attrs["optimizer_class"](non_frozen_parameters,
                                                                   **study.best_trial.user_attrs["optimizer_kwargs"])

        print("model path ", model_path)

        #model_path = "/home/martin/Documents/MLMC-DFM/optuna_runs/metacentrum/MLMC-DFM_general_dataset_2/pooling/eigh_loss/fr_div_0_1_exp_6/seed_36974/trial_0_losses_model_cnn_net"
        if not torch.cuda.is_available():
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            print("checkpoint with cuda")
            checkpoint = torch.load(model_path)
        train_loss = checkpoint['train_loss']
        #valid_loss = checkpoint['valid_loss']
        #print("checkpoint ", checkpoint)
        #print("convs weight shape ", checkpoint["best_model_state_dict"]['_convs.0.weight'].shape)
        #print("out layer weight shape ", checkpoint["best_model_state_dict"]['_output_layer.weight'].shape)

        #print("valid loss ", valid_loss)

        print("best loss: {}".format(np.min(train_loss)))
        print("best epoch: {}".format(np.argmin(train_loss)))
        #exit()
        valid_loss = np.ones(len(train_loss))* 50
        plot_train_valid_loss(train_loss, valid_loss)

        print("checkpoint ", list(checkpoint['best_model_state_dict'].keys()))

        #print("_output_layer.bias ", checkpoint['best_model_state_dict']['_output_layer.bias'])

        print("Trainable parameters")
        for key in list(checkpoint['best_model_state_dict'].keys()):
            #print("key ", key)
            print("{}: shape: {}".format(key, checkpoint['best_model_state_dict'][key].shape))
            #checkpoint['best_model_state_dict'][key] = checkpoint['best_model_state_dict'][key].to(device='cuda')
        #
        # print("checkpoint['best_model_state_dict']['model.gnn_tower.mlp_in_t.1.weight'].device ", checkpoint['best_model_state_dict']['model.gnn_tower.mlp_in_t.1.weight'].device)
        # print("checkpoint['best_model_state_dict']['model.gnn_tower.mlp_in_t.1.weight'] ",
        #       checkpoint['best_model_state_dict']['model.gnn_tower.mlp_in_t.1.weight'])

        diff_model.load_state_dict(checkpoint['best_model_state_dict'])
        diff_model.to("cuda")
        diff_model.eval()

        #diff_model.model.gnn_tower.mlp_in_t[1].weight = diff_model.model.gnn_tower.mlp_in_t[1].weight.to(device="cuda")
        #
        #print("diff_model.model.gnn_tower.mlp_in_t[1].weight.device ", diff_model.model.gnn_tower.mlp_in_t[1].weight.device)
        # print("diff_model.model.gnn_tower.mlp_in_t[1].weight ", diff_model.model.gnn_tower.mlp_in_t[1].weight)
        # exit()

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['best_optimizer_state_dict'])
        epoch = checkpoint['best_epoch']

        print("Best epoch ", epoch)

        print("train loss ", train_loss)
        #print("valid loss ", valid_loss)
        print("model training time ", checkpoint["training_time"])

        plot_train_valid_loss(train_loss, valid_loss)

        running_loss, inv_running_loss = 0, 0
        targets_list, predictions_list = [], []
        inv_targets_list, inv_predictions_list = [], []
        batch_size_sample = 1

        #inverse_transform = datasets.dataset_preprocessing.get_inverse_transform(study, results_dir)

        inverse_transform = None

        n_samples = 50
        generated_samples = []
        generated_orig_samples = []
        for i in range(n_samples):
            inv_samples, orig_samples = diff_model.sample(batch_size=batch_size_sample, inverse_transform=inverse_transform, dataset=dataset)
            #samples = torch.squeeze(inv_samples, dim=0)

            print("inv sampels ", inv_samples)

            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            axes.hist(inv_samples.cpu().flatten(), bins=100, density=True, label="Sampled bone density distr")
            fig.legend()
            plt.show()

            inv_samples_np = np.squeeze(inv_samples.cpu().numpy())
            print("inv samples np shape ", inv_samples_np.shape)

            render_3d_scan(np.squeeze(inv_samples.cpu().numpy()), title="Sampled 3D Scan")

            # threshold = 0.1
            # inv_samples_np[inv_samples_np < threshold] = -1

            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            axes.hist(inv_samples_np.flatten(), bins=100, density=True, label="Sampled bone density distr")
            fig.legend()
            plt.show()

            #render_3d_scan(inv_samples, title="Sampled 3D Scan")

            render_3d_scan(inv_samples_np, title="INV Sampled 3D Scan")
            # #print("samples.shape ", samples.shape)
            #
            # if density_only:
            #     density = samples
            # else:
            #     density = samples[..., 3]
            #     displacement = samples[..., :3]
            #
            #     displacement = np.squeeze(displacement.cpu().numpy())
            #
            #     fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            #     axes.hist(displacement[:, 0], bins=100, density=True, label="displacement[0]")
            #     axes.hist(displacement[:, 1], bins=100, density=True, label="displacement[1]")
            #     axes.hist(displacement[:, 2], bins=100, density=True, label="displacement[2]")
            #
            # density = np.squeeze(density.cpu().numpy())


            # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            # axes.hist(density, bins=100, density=True, color="red", label="density")
            # fig.legend()
            # plt.show()
            #
            # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            # axes.hist(np.squeeze(orig_samples.cpu().numpy()), bins=100, density=True, color="green", label="transformed density")
            # fig.legend()
            # plt.show()

            generated_samples.append(density)
            generated_orig_samples.append(orig_samples)

            #loss = loss_fn(predictions, targets)
            # running_loss += loss
            #
            # inv_targets = targets
            # inv_predictions = predictions
            # if inverse_transform is not None:
            #     inv_targets = inverse_transform(torch.reshape(targets, (*targets.shape, 1, 1)))
            #     inv_predictions = inverse_transform(torch.reshape(predictions, (*predictions.shape, 1, 1)))
            #
            #
            #     if dataset.init_transform is not None and input_inverse_transform is not None:
            #         inv_predictions *= inv_input_avg
            #         inv_targets *= inv_input_avg
            #
            #
            #     inv_targets = np.reshape(inv_targets, targets.shape)
            #     inv_predictions = np.reshape(inv_predictions, predictions.shape)
            #
            #
            #
            #
            # inv_running_loss += loss_fn(inv_predictions, inv_targets)
            #
            # inv_targets_list.append(inv_targets.numpy())
            # inv_predictions_list.append(inv_predictions.numpy())


        #inv_targets_arr = np.array(inv_targets_list)
        #inv_predictions_arr = np.array(inv_predictions_list)

        #print("inv targets arr shape", inv_targets_arr.shape)

        predictions_list = np.array(predictions_list)
        #predictions_list += 0.1
        orig_samples_flatten = np.array(original_samples).flatten()
        generated_samples_flatten = np.array(generated_samples).flatten()

        print("orig samples flatten ", orig_samples_flatten.shape)

        # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        # axes.hist(np.log(orig_samples_flatten), bins=100, density=True, color="red", label="log orig density")
        # axes.hist(np.log(generated_samples_flatten), bins=100, density=True, color="blue", label="log generated density", alpha=0.5)
        # fig.legend()
        # plt.show()

        import matplotlib
        matplotlib.rcParams.update({'font.size': 22})

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        axes.hist(orig_samples_flatten, bins=100, density=True, color="red", label="real samples")
        axes.hist(generated_samples_flatten, bins=100, density=True, color="blue", label="generated samples",
                  alpha=0.5)
        plt.xlabel("bone density")
        fig.legend()
        plt.tight_layout()
        plt.savefig("L4_orig_generated.pdf")
        plt.show()

        #fgd = calculate_fgd(real_features=np.array(original_samples), generated_features=np.array(generated_samples))

        #print("fgd ", fgd)

        # mse, rmse, nrmse, r2 = get_mse_nrmse_r2(targets_list, predictions_list)
        # inv_mse, inv_rmse, inv_nrmse, inv_r2 = get_mse_nrmse_r2(inv_targets_arr, inv_predictions_arr)

        #test_loss = running_loss / (i + 1)
        #inv_test_loss = inv_running_loss / (i + 1)

def calculate_fgd(real_features, generated_features):
    print("real features shape ", real_features.shape)
    print("generated features shape ", generated_features.shape)
    # Compute the mean and covariance of the real and generated features
    mu_real, sigma_real = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = generated_features.mean(0), np.cov(generated_features, rowvar=False)

    # Compute Fréchet distance
    mu_diff = mu_real - mu_gen
    print("mu diff ", mu_diff)
    cov_sqrt, _ = sc.linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    fgd = mu_diff @ mu_diff + np.trace(sigma_real + sigma_gen - 2 * cov_sqrt)
    return fgd


#         print("epochs: {}, train loss: {}, valid loss: {}, test loss: {}, inv test loss: {}".format(epoch,
#                                                                                                     train_loss,
#                                                                                                     valid_loss,
#                                                                                                     test_loss,
#                                                                                                     inv_test_loss))
#         mse_str, inv_mse_str = "MSE", "Original data MSE"
#         r2_str, inv_r2_str = "R2", "Original data R2"
#         rmse_str, inv_rmse_str = "RMSE", "Original data RMSE"
#         nrmse_str, inv_nrmse_str = "NRMSE", "Original data NRMSE"
#         for i in range(len(mse)):
#             mse_str += " k_{}: {}".format(i, mse[i])
#             r2_str += " k_{}: {}".format(i, r2[i])
#             rmse_str += " k_{}: {}".format(i, rmse[i])
#             nrmse_str += " k_{}: {}".format(i, nrmse[i])
#
#             inv_mse_str += " k_{}: {}".format(i, inv_mse[i])
#             inv_r2_str += " k_{}: {}".format(i, inv_r2[i])
#             inv_rmse_str += " k_{}: {}".format(i, inv_rmse[i])
#             inv_nrmse_str += " k_{}: {}".format(i, inv_nrmse[i])
#
#             # print("MSE k_xx: {}, k_xy: {}, k_yy: {}".format(mse[0], mse[1], mse[2]))
#             # print("R2 k_xx: {}, k_xy: {}, k_yy: {}".format(r2[0], r2[1], r2[2]))
#             # print("RMSE  k_xx: {}, k_xy: {}, k_yy: {}".format(rmse[0], rmse[1], rmse[2]))
#             # print("NRMSE  k_xx: {}, k_xy: {}, k_yy: {}".format(nrmse[0], nrmse[1], nrmse[2]))
#
#             # print("Original data MSE k_xx: {}, k_xy: {}, k_yy: {}".format(inv_mse[0], inv_mse[1], inv_mse[2]))
#             # print("Original data R2 k_xx: {}, k_xy: {}, k_yy: {}".format(inv_r2[0], inv_r2[1], inv_r2[2]))
#             # print("Original data RMSE k_xx: {}, k_xy: {}, k_yy: {}".format(inv_rmse[0], inv_rmse[1], inv_rmse[2]))
#             # print("Original data NRMSE  k_xx: {}, k_xy: {}, k_yy: {}".format(inv_nrmse[0], inv_nrmse[1], inv_nrmse[2]))
#
#         print(mse_str)
#         print(r2_str)
#         print(rmse_str)
#         print(nrmse_str)
#
#         print("mean R2: {}, NRMSE: {}".format(np.mean(r2), np.mean(nrmse)))
#
#         print(inv_mse_str)
#         print(inv_r2_str)
#         print(inv_rmse_str)
#         print(inv_nrmse_str)
#
#         print("mean R2: {}, NRMSE: {}".format(np.mean(inv_r2), np.mean(inv_nrmse)))
#
#         get_mse_nrmse_r2_eigh(targets_list, predictions_list)
#         print("ORIGINAL DATA")
#         get_mse_nrmse_r2_eigh(inv_targets_arr, inv_predictions_arr)
#
#         plot_target_prediction(np.array(targets_list), np.array(predictions_list), "preprocessed_")
#         plot_target_prediction(inv_targets_arr, inv_predictions_arr)
#
#         inv_targets_arr[:, 0] = np.log10(inv_targets_arr[:, 0])
#         inv_targets_arr[:, 2] = np.log10(inv_targets_arr[:, 2])
#
#         inv_predictions_arr[:, 0] = np.log10(inv_predictions_arr[:, 0])
#         inv_predictions_arr[:, 2] = np.log10(inv_predictions_arr[:, 2])
#
#         log_inv_mse, log_inv_rmse, log_inv_nrmse, log_inv_r2 = get_mse_nrmse_r2(inv_targets_arr, inv_predictions_arr)
#
#         print("log_inv_r2 ", log_inv_r2)
#         print("log_inv_nrmse ", log_inv_nrmse)
#
#         print("mean log_inv_r2 ", np.mean(log_inv_r2))
#         print("mean log_inv_nrmse ", np.mean(log_inv_nrmse))
#
#
#         plot_target_prediction(inv_targets_arr, inv_predictions_arr, title_prefix="log_orig_", r2=log_inv_r2, nrmse=log_inv_nrmse,
#                                x_labels=[r'$log(k_{xx})$', r'$k_{xy}$', r'$log(k_{yy})$']
#                                )
#
#         ######
#         ## main peak fr div 0.1
#         ######
#         #print("inv_targets_arr[inv_targets_arr[:, 0] > -3.8]", inv_targets_arr[inv_targets_arr[:, 0] > -3.8])
#         #print(" inv_predictions_arr[inv_targets_arr[:, 0] > -3.8]",  inv_predictions_arr[inv_targets_arr[:, 0] > -3.8])
#
#
#         log_inv_mse, log_inv_rmse, log_inv_nrmse, log_inv_r2 = get_mse_nrmse_r2(inv_targets_arr[inv_targets_arr[:, 0] > -3.8],
#                                                                                 inv_predictions_arr[inv_targets_arr[:, 0] > -3.8]
# )
#
#         print("log_inv_r2 main peak", log_inv_r2)
#         print("log_inv_nrmse main peak", log_inv_nrmse)



def load_study(results_dir):
    study = joblib.load(os.path.join(results_dir, "study.pkl"))
    print("Best trial until now:")
    print(" Value: ", study.best_trial.value)
    print(" Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    return study


def compare_trials(study):
    df = study.trials_dataframe()
    print("df ", df)

    df.to_csv("trials.csv")
    # Import pandas package

    df_duration = df.sort_values("duration")

    print("df_duration ", df_duration)
    # iterating the columns
    for col in df.columns:
        print(col)
    exit()
    fastest = None
    for trial in study.trials:
        print("datetime start ", trial["datetime_start"])
        print("trial ", trial)

    exit()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', help='results directory')
    parser.add_argument("-c", "--cuda", default=False, action='store_true', help="use cuda")
    args = parser.parse_args(sys.argv[1:])

    print("torch.cuda.is_available() ", torch.cuda.is_available())
    #print(torch.zeros(1).cuda())

    study = load_study(args.results_dir)

    #@TODO: RM ASAP
    print("study attrs ", study.user_attrs)
    #study.user_attrs["output_log"] = True
    #study.set_user_attr("output_log", True)
    print("study attrs ", study.user_attrs)
    #compare_trials(study)

    load_models(args, study)


