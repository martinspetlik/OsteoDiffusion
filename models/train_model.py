import os
import sys
import argparse
import logging
import joblib
import copy
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler, BruteForceSampler
import time
import yaml
import shutil
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
# #from torch.utils.tensorboard import SummaryWriter
# from datetime import datetime
from models.auxiliary_functions import get_mean_std, log_data, exp_data, quantile_transform_fit, QuantileTRF, NormalizeData, log_all_data, init_norm, log10_data, log10_all_data, get_loss_fn
# from metamodel.cnn.visualization.visualize_data import plot_samples, plot_dataset
from datasets.dataset_preprocessing import prepare_dataset


def validate(model, validation_loader, loss_fn=nn.MSELoss(), acc_fn=nn.MSELoss(), use_cuda=False):
    """
    Validate model
    :param model:
    :param loss_fn:
    :return:
    """
    running_vloss = 0.0
    running_vacc = 0
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vtargets = vdata

            if torch.cuda.is_available() and use_cuda:
                vinputs = vinputs.cuda()
                vtargets = vtargets.cuda()

            vinputs = vinputs.float()
            vtargets = vtargets.float()

            voutputs = torch.squeeze(model(vinputs))
            #print("voutputs.shape ", voutputs.shape)
            # print("vtargets.shape ", vtargets.shape)
            vloss = loss_fn(voutputs, vtargets)
            running_vloss += vloss.item()

            #print("validate running loss ", running_vloss)

            vacc = acc_fn(voutputs, vtargets)
            running_vacc += vacc.item()

        avg_vloss = running_vloss / (i + 1)
        avg_vacc = running_vacc / (i + 1)

    return avg_vloss, avg_vacc

def train_one_epoch(model, optimizer, train_loader, config, loss_fn=nn.MSELoss(), use_cuda=True):
    """
    Train NN
    :param model:
    :param optimizer:
    :param loss_fn:
    :return:
    """
    running_loss = 0.
    for i, data in enumerate(train_loader):
        inputs, targets = data
        if torch.cuda.is_available() and use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs = inputs.float()
        targets = targets.float()
        optimizer.zero_grad()

        outputs = torch.squeeze(model(inputs))

        loss = loss_fn(outputs, targets)
        loss.backward()

        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

    train_loss = running_loss / (i + 1)
    return train_loss


def objective(trial, trials_config, train_loader, validation_loader):
    best_vloss = 1_000_000.

    max_channel = trial.suggest_categorical("max_channel", trials_config["max_channel"])
    n_conv_layers = trial.suggest_categorical("n_conv_layers", trials_config["n_conv_layers"])
    kernel_size = trial.suggest_categorical("kernel_size", trials_config["kernel_size"])
    stride = trial.suggest_categorical("stride", trials_config["stride"])
    pool = trial.suggest_categorical("pool", trials_config["pool"])
    pool_size = trial.suggest_categorical("pool_size", trials_config["pool_size"])
    pool_stride = trial.suggest_categorical("pool_stride", trials_config["pool_stride"])
    lr = trial.suggest_categorical("lr", trials_config["lr"])
    use_batch_norm = trial.suggest_categorical("use_batch_norm", trials_config["use_batch_norm"])
    max_hidden_neurons = trial.suggest_categorical("max_hidden_neurons", trials_config["max_hidden_neurons"])
    n_hidden_layers = trial.suggest_categorical("n_hidden_layers", trials_config["n_hidden_layers"])

    batch_size_train = trial.suggest_categorical("batch_size_train", trials_config["batch_size_train"])
    config["batch_size_train"] = batch_size_train

    loss_function = ["MSE", []]
    if "loss_function" in trials_config:
        loss_function = trial.suggest_categorical("loss_function", trials_config["loss_function"])

    loss_fn = get_loss_fn(loss_function)

    pool_indices = None
    if "pool_indices" in trials_config:
        pool_indices = trial.suggest_categorical("pool_indices", trials_config["pool_indices"])

    if "use_cnn_dropout" in trials_config:
        use_cnn_dropout = trial.suggest_categorical("use_cnn_dropout", trials_config["use_cnn_dropout"])

    if "use_fc_dropout" in trials_config:
        use_fc_dropout = trial.suggest_categorical("use_fc_dropout", trials_config["use_fc_dropout"])

    if "cnn_dropout_indices" in trials_config:
        cnn_dropout_indices = trial.suggest_categorical("cnn_dropout_indices", trials_config["cnn_dropout_indices"])

    if "fc_dropout_indices" in trials_config:
        fc_dropout_indices = trial.suggest_categorical("fc_dropout_indices", trials_config["fc_dropout_indices"])

    if "bias_reduction_layer_indices" in trials_config:
        bias_reduction_layer_indices = trial.suggest_categorical("bias_reduction_layer_indices",
                                                                 trials_config["bias_reduction_layer_indices"])

    if "cnn_dropout_ratios" in trials_config:
        cnn_dropout_ratios = trial.suggest_categorical("cnn_dropout_ratios", trials_config["cnn_dropout_ratios"])

    if "fc_dropout_ratios" in trials_config:
        fc_dropout_ratios = trial.suggest_categorical("fc_dropout_ratios", trials_config["fc_dropout_ratios"])

    if "n_train_samples" in trials_config and trials_config["n_train_samples"] is not None:
        n_train_samples = trial.suggest_categorical("n_train_samples", trials_config["n_train_samples"])
        config["n_train_samples"] = n_train_samples

        if "n_test_samples" in trials_config and trials_config["n_test_samples"] is not None:
            config["n_test_samples"] = trials_config["n_test_samples"]

        if "sub_datasets" in config and len(config["sub_datasets"]) > 0:
            train_set, validation_set, test_set = prepare_sub_datasets(study, config, data_dir=data_dir,
                                                                       serialize_path=output_dir)
        else:
            train_set, validation_set, test_set = prepare_dataset(study, config, data_dir=data_dir,
                                                                  serialize_path=output_dir)

        print("len(trainset): {}, len(valset): {}, len(testset): {}".format(len(train_set), len(validation_set),
                                                                            len(test_set)))

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=config["batch_size_train"],
                                                        shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size_test"], shuffle=False)

    # plot_samples(train_loader, n_samples=25)
    # exit()

    optimizer_name = "Adam"
    if "optimizer_name" in trials_config:
        optimizer_name = trial.suggest_categorical("optimizer_name", trials_config["optimizer_name"])

    L2_penalty = 0
    if "L2_penalty" in trials_config:
        L2_penalty = trial.suggest_categorical("L2_penalty", trials_config["L2_penalty"])

    cnn_activation_name = "relu"
    if "cnn_activation_name" in trials_config:
        cnn_activation_name = trial.suggest_categorical("cnn_activation_name",
                                                           trials_config["cnn_activation_name"])
    cnn_activation = getattr(F, cnn_activation_name)

    hidden_activation_name = "relu"
    if "hidden_activation_name" in trials_config:
        hidden_activation_name = trial.suggest_categorical("hidden_activation_name",
                                                           trials_config["hidden_activation_name"])
    hidden_activation = getattr(F, hidden_activation_name)

    # flag, input_size = check_shapes(n_conv_layers, kernel_size, stride, pool_size, pool_stride, pool_indices,
    #                                 input_size=trials_config["input_size"])

    if "global_pool" in trials_config:
        global_pool = trial.suggest_categorical("global_pool", trials_config["global_pool"])

    if "vit_params" in trials_config:
        vit_params = trial.suggest_categorical("vit_params", trials_config["vit_params"])

    # print("max channels ", max_channel)
    # print("flag: {} flatten x: {}".format(flag, input_size * input_size * max_channel))

    # if flag == -1:
    #     print("flag: {}".format(flag))
    #     return best_vloss

    # Initilize model
    model_kwargs = {"n_conv_layers": n_conv_layers,
                    "max_channel": max_channel,
                    "pool": pool,
                    "pool_size": pool_size,
                    "kernel_size": kernel_size,
                    "stride": stride,
                    "pool_stride": pool_stride,
                    "use_batch_norm": use_batch_norm,
                    "n_hidden_layers": n_hidden_layers,
                    "max_hidden_neurons": max_hidden_neurons,
                    "hidden_activation": hidden_activation,
                    "cnn_activation": cnn_activation,
                    "global_pool": global_pool if "global_pool" in trials_config else None,
                    "input_size": trials_config["input_size"],
                    "output_bias": trials_config["output_bias"] if "output_bias" in trials_config else False,
                    "pool_indices": pool_indices if "pool_indices" in trials_config else {},
                    "use_cnn_dropout": use_cnn_dropout if "use_cnn_dropout" in trials_config else False,
                    "use_fc_dropout": use_fc_dropout if "use_fc_dropout" in trials_config else False,
                    "cnn_dropout_indices": cnn_dropout_indices if "cnn_dropout_indices" in trials_config else [],
                    "fc_dropout_indices": fc_dropout_indices if "fc_dropout_indices" in trials_config else [],
                    "cnn_dropout_ratios": cnn_dropout_ratios if "cnn_dropout_ratios" in trials_config else [],
                    "fc_dropout_ratios": fc_dropout_ratios if "fc_dropout_ratios" in trials_config else [],
                    }

    if "bias_reduction_layer_indices" in trials_config:
        model_kwargs["bias_reduction_layer_indices"] = bias_reduction_layer_indices
    if "vit_params" in trials_config:
        model_kwargs["vit_params"] = vit_params
    if "input_channels" in trials_config:
        model_kwargs["input_channel"] = len(trials_config["input_channels"])
    if "output_channels" in trials_config:
        model_kwargs["n_output_neurons"] = len(trials_config["output_channels"])
    model_class_name = "Net"
    if "model_class_name" in trials_config:
        model_class_name = trials_config["model_class_name"]

    if model_class_name == "Net":
        model_class = Net
    elif model_class_name == "CondNet":
        model_class = CondNet
    elif model_class_name == "ViTRegressor":
        model_class = ViTRegressor
    else:
        raise NotImplementedError("model class name {} not supported ".format(model_class_name))

    n_pretrained_layers = 0
    if "trained_layers_dir" in trials_config:
        model_kwargs, n_pretrained_layers = get_trained_layers(trials_config, model_kwargs)
        if "save_output_image" in trials_config:
            model_kwargs["save_output_image"] = trials_config["save_output_image"]

    model = model_class(**model_kwargs).to(device)

    if n_pretrained_layers > 0:
        for index, (cvs, lin) in enumerate(zip(model._convs, model._fcls)):
            if index > n_pretrained_layers - 1:
                break
            for param in cvs.parameters():
                param.requires_grad = False
            for param in lin.parameters():
                param.requires_grad = False

    #model = Net(trial, **model_kwargs).to(device)
    # print("model._convs ", model._convs)
    # print("moodel._hidden_layers ", model._hidden_layers)
    # print("moodel._output_layer ", model._output_layer)

    # Initialize optimizer
    optimizer_kwargs = {"lr": lr, "weight_decay": L2_penalty}
    non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = None
    #print("optimizer kwargs ", optimizer_kwargs)

    #print("non frozen parameters ", non_frozen_parameters)
    if len(non_frozen_parameters) > 0:
        optimizer = getattr(optim, optimizer_name)(params=non_frozen_parameters, **optimizer_kwargs)

    trial.set_user_attr("model_class", model.__class__)
    trial.set_user_attr("optimizer_class", optimizer.__class__)
    trial.set_user_attr("model_name", model._name)
    trial.set_user_attr("model_kwargs", model_kwargs)
    trial.set_user_attr("optimizer_kwargs", optimizer_kwargs)
    trial.set_user_attr("loss_fn", loss_fn)
    trial.set_user_attr("trials_config", trials_config)

    # Training of the model
    start_time = time.time()
    avg_loss_list = []
    avg_vloss_list = []
    avg_vloss, avg_loss = best_vloss, best_vloss
    best_epoch = 0
    model_state_dict = {}
    optimizer_state_dict = {}

    model_path = 'trial_{}_losses_model_{}'.format(trial.number, model._name)
    # print("model path ", model_path)
    # if os.path.exists(model_path):
    #     print("model pat hexists")
    #     return avg_vloss

    scheduler = None
    train = trials_config["train"] if "train" in trials_config else True

    if "scheduler" in trials_config and optimizer is not None:
        trial_scheduler = trial.suggest_categorical("scheduler", trials_config["scheduler"])
        if "class" in trial_scheduler:
            if trial_scheduler["class"] == "ReduceLROnPlateau":
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                           patience=trial_scheduler["patience"],
                                                           factor=trial_scheduler["factor"])
            else:
                scheduler = lr_scheduler.StepLR(optimizer, step_size=trial_scheduler["step_size"],
                                            gamma=trial_scheduler["gamma"])

    if "save_output_image" in trials_config and trials_config["save_output_image"]:
        model.train(False)
        sample_id = 0
        output_data_dir = "/home/martin/Documents/MLMC-DFM_data/layer_outputs/3_3_from_9_9"
        sample_id = save_output_dataset(model, train_loader, study, output_data_dir=output_data_dir, sample_id=sample_id)
        sample_id = save_output_dataset(model, validation_loader, study, output_data_dir=output_data_dir, sample_id=sample_id)
        save_output_dataset(model, test_loader, study, output_data_dir=output_data_dir,sample_id=sample_id)
        exit()

    # input_mean, input_std, output_mean, output_std = get_mean_std(validation_loader)
    # print("Validation loader, input mean: {}, std: {}".format(input_mean, input_std))

    for epoch in range(config["num_epochs"]):
        try:
            if train:
                model.train(True)
                avg_loss = train_one_epoch(model, optimizer, train_loader, config, loss_fn=loss_fn, use_cuda=use_cuda)  # Train the model

            model.train(False)
            if len(validation_set) == 0:
                avg_vloss = avg_loss
                avg_vacc = 0
            else:
                avg_vloss, avg_vacc = validate(model, validation_loader, loss_fn=loss_fn, use_cuda=use_cuda)   # Evaluate the model

            if scheduler is not None:
                scheduler.step(avg_vloss)
                print("scheduler lr: {}".format(scheduler._last_lr))

            avg_loss_list.append(avg_loss)
            avg_vloss_list.append(avg_vloss)

            print("epoch: {}, LOSS train: {}, val: {}, ACC val: {}".format(epoch, avg_loss, avg_vloss, avg_vacc))

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                best_epoch = epoch

                model_state_dict = model.state_dict()
                if train:
                    optimizer_state_dict = optimizer.state_dict()

            # For pruning (stops trial early if not promising)
            trial.report(avg_vloss, epoch)
            # Handle pruning based on the intermediate value.
            # if trial.should_prune():
            #     raise optuna.exceptions.TrialPruned()
        except Exception as e:
           print(str(e))
           return avg_vloss

    #for key, value in trial.params.items():
    #    model_path += "_{}_{}".format(key, value)

    model_path = os.path.join(output_dir, model_path)

    torch.save({
        'best_epoch': best_epoch,
        'best_model_state_dict': model_state_dict,
        'best_optimizer_state_dict': optimizer_state_dict,
        'train_loss': avg_loss_list,
        'valid_loss': avg_vloss_list,
        'training_time': time.time() - start_time,
    }, model_path)

    return best_vloss

def load_trials_config(path_to_config):
    with open(path_to_config, "r") as f:
        trials_config = yaml.load(f, Loader=yaml.FullLoader)
    return trials_config



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trials_config_path', help='Path tp trials config')
    parser.add_argument('data_dir', help='Data directory')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument("-c", "--cuda", default=False, action='store_true', help="use cuda")
    parser.add_argument("-a", "--append", default=False, action='store_true', help="append models")

    args = parser.parse_args(sys.argv[1:])

    data_dir = args.data_dir
    output_dir = args.output_dir
    trials_config = load_trials_config(args.trials_config_path)
    use_cuda = args.cuda

    config = {"num_epochs": trials_config["num_epochs"],
              "batch_size_train": trials_config["batch_size_train"],
              "batch_size_test": trials_config["batch_size_test"] if "batch_size_test" in trials_config else 250,
              "n_train_samples": trials_config["n_train_samples"] if "n_train_samples" in trials_config else None,
              "n_test_samples": trials_config["n_test_samples"] if "n_test_samples" in trials_config else None,
              "train_samples_ratio": trials_config["train_samples_ratio"] if "train_samples_ratio" in trials_config else 0.8,
              "val_samples_ratio": trials_config["val_samples_ratio"] if "val_samples_ratio" in trials_config else 0.2,
              "print_batches": 10,
              "init_norm": trials_config["init_norm"] if "init_norm" in trials_config else False,
              "init_norm_use_all_features": trials_config["init_norm_use_all_features"] if "init_norm_use_all_features" in trials_config else False,
              "log_all_input_channels": trials_config["log_all_input_channels"] if "log_all_input_channels" in trials_config else False,
              "log_input": trials_config["log_input"] if "log_input" in trials_config else True,
              "normalize_input": trials_config["normalize_input"] if "normalize_input" in trials_config else True,
              "log_output": trials_config["log_output"] if "log_output" in trials_config else False,
              "log10_output": trials_config["log10_output"] if "log10_output" in trials_config else False,
              "log_all_output": trials_config["log_all_output"] if "log_all_output" in trials_config else False,
              "log10_all_output": trials_config["log10_all_output"] if "log10_all_output" in trials_config else False,
              "normalize_output": trials_config["normalize_output"] if "normalize_output" in trials_config else True,
              "input_channels": trials_config["input_channels"] if "input_channels" in trials_config else None,
              "output_channels": trials_config["output_channels"] if "output_channels" in trials_config else None,
              "fractures_sep": trials_config["fractures_sep"] if "fractures_sep" in trials_config else False,
              "cross_section": trials_config["cross_section"] if "cross_section" in trials_config else False,
              "seed": trials_config["random_seed"] if "random_seed" in trials_config else 12345,
              "output_dir": output_dir,
              "sub_datasets": trials_config["sub_datasets"] if "sub_datasets" in trials_config else {}
              }

    if "input_transform" in trials_config:
        config["input_transform"] = trials_config["input_transform"]
    if "output_transform" in trials_config:
        config["output_transform"] = trials_config["output_transform"]
    if "output_iqr_scale" in trials_config:
        config["output_iqr_scale"] = trials_config["output_iqr_scale"]
    if "normalize_input_indices" in trials_config:
        config["normalize_input_indices"] = trials_config["normalize_input_indices"]
    if "normalize_output_indices" in trials_config:
        config["normalize_output_indices"] = trials_config["normalize_output_indices"]

    # Optuna params
    num_trials = trials_config["num_trials"]

    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    print("device ", device)
    print("config seed ", config["seed"])

    # Make runs repeatable
    random_seed = trials_config["random_seed"]
    torch.backends.cudnn.enabled = False  # Disable cuDNN use of nondeterministic algorithms
    torch.manual_seed(random_seed)
    output_dir = os.path.join(output_dir, "seed_{}".format(random_seed))
    if os.path.exists(output_dir) and not args.append:
        shutil.rmtree(output_dir)
        #raise IsADirectoryError("Results output dir {} already exists".format(output_dir))
    if not args.append:
        os.mkdir(output_dir)
    elif not os.path.exists(output_dir):
        raise NotADirectoryError("output dir {} not exists".format(output_dir))

    sampler = TPESampler(seed=random_seed)
    if "sampler_class" in trials_config:
        if trials_config["sampler_class"] == "BruteForceSampler":
            sampler = BruteForceSampler(seed=random_seed)

    study = optuna.create_study(sampler=sampler, direction="minimize")

    # ================================
    # Datasets and data loaders
    # ================================
    train_loader, validation_loader = None, None
    # if "n_train_samples" not in config or not isinstance(config["n_train_samples"], (list, np.ndarray)):
    #     train_set, validation_set, test_set = prepare_dataset(study, config, data_dir=data_dir, serialize_path=output_dir)
    #     train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True)
    #     validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=config["batch_size_train"], shuffle=False)
    #     test_loader = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size_test"], shuffle=False)

    def obj_func(trial):
        return objective(trial, trials_config, train_loader, validation_loader)

    study.optimize(obj_func, n_trials=num_trials)

    # ================================
    # Results
    # ================================
    # Find number of pruned and completed trials
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Display the study statistics
    print("\nStudy statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    trial = study.best_trial
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save results to csv file
    df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete', 'duration'], axis=1)  # Exclude columns
    df = df.loc[df['state'] == 'COMPLETE']        # Keep only results that did not prune
    df = df.drop('state', axis=1)                 # Exclude state column
    df = df.sort_values('value')                  # Sort based on accuracy
    df.to_csv(os.path.join(output_dir, 'optuna_results.csv'), index=False)  # Save to csv file

    # Display results in a dataframe
    print("\nOverall Results (ordered by accuracy):\n {}".format(df))

    # Find the most important hyperparameters
    try:
        most_important_parameters = optuna.importance.get_param_importances(study, target=None)

        # Display the most important hyperparameters
        print('\nMost important hyperparameters:')
        for key, value in most_important_parameters.items():
            print('  {}:{}{:.2f}%'.format(key, (15-len(key))*' ', value*100))
    except Exception as e:
        print(str(e))

    # serialize optuna study object
    joblib.dump(study, os.path.join(output_dir, "study.pkl"))

