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
from torch.optim import lr_scheduler
# #from torch.utils.tensorboard import SummaryWriter
# from datetime import datetime
from models.auxiliary_functions import get_mean_std, log_data, exp_data, quantile_transform_fit, QuantileTRF, NormalizeData, log_all_data, init_norm, log10_data, log10_all_data, get_loss_fn
# from metamodel.cnn.visualization.visualize_data import plot_samples, plot_dataset
from datasets.dataset_preprocessing import prepare_dataset
from models.diffusion_model import DiffusionModel
from gnn_models import GNN
from models.schedulers import NoiseScheduler


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
    for i, graphs in enumerate(train_loader):

        if torch.cuda.is_available() and use_cuda:
            graphs = graphs.cuda()

        # inputs = inputs.float()
        # targets = targets.float()
        optimizer.zero_grad()

        noise, predicted_noise = torch.squeeze(model(graphs))
        loss = loss_fn(noise, predicted_noise)

        loss.backward()

        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

    train_loss = running_loss / (i + 1)
    train_loss = running_loss / (i + 1)
    return train_loss


def objective(trial, trials_config, train_loader, validation_loader):
    best_vloss = 1_000_000.
    lr = trial.suggest_categorical("lr", trials_config["lr"])
    batch_size_train = trial.suggest_categorical("batch_size_train", trials_config["batch_size_train"])
    config["batch_size_train"] = batch_size_train

    ###
    # GNN config
    ###
    hidden_t_embd = trials_config["hidden_t_embd"]
    hidden_v_embd = trials_config["hidden_v_embd"]
    num_gnn_layers = trials_config["num_gnn_layers"]
    dropout = trials_config["dropout"]

    ####
    # Noise scheduler config
    ####
    beta_scheduler_type = "linear"
    scheduler_kwargs = {}
    if "beta_scheduler_type" in trials_config:
        beta_scheduler_type = trials_config["beta_scheduler_type"]
    if "scheduler_kwargs" in trials_config:
        scheduler_kwargs = trials_config["scheduler_kwargs"]


    loss_function = ["MSE", []]
    if "loss_function" in trials_config:
        loss_function = trial.suggest_categorical("loss_function", trials_config["loss_function"])

    loss_fn = get_loss_fn(loss_function)

    print("loss fn ", loss_fn)

    if "n_train_samples" in trials_config and trials_config["n_train_samples"] is not None:
        n_train_samples = trial.suggest_categorical("n_train_samples", trials_config["n_train_samples"])
        config["n_train_samples"] = n_train_samples

        if "n_test_samples" in trials_config and trials_config["n_test_samples"] is not None:
            config["n_test_samples"] = trials_config["n_test_samples"]

        # if "sub_datasets" in config and len(config["sub_datasets"]) > 0:
        #     train_set, validation_set, test_set = prepare_sub_datasets(study, config, data_dir=data_dir,
        #                                                                serialize_path=output_dir)
        # else:
        train_set, validation_set, test_set = prepare_dataset(study, config, data_dir=data_dir,
                                                              serialize_path=output_dir)

        print("len(trainset): {}, len(valset): {}, len(testset): {}".format(len(train_set), len(validation_set),
                                                                            len(test_set)))

        print("config batch size train", config["batch_size_train"])

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=config["batch_size_train"],
                                                        shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size_test"], shuffle=False)

    # plot_samples(train_loader, n_samples=25)
    # exit()

    optimizer_name = "AdamW"
    if "optimizer_name" in trials_config:
        optimizer_name = trial.suggest_categorical("optimizer_name", trials_config["optimizer_name"])



    #####################
    #####################
    #  Initilize model #
    ####################
    ####################

    ##################################
    ### Graph neural network model ###
    ##################################
    gnn_config = {"hidden_t_embd": hidden_t_embd,
                  "hidden_v_embd": hidden_v_embd,
                  "num_gnn_layers": num_gnn_layers,
                  "dropout": dropout}
    num_node_attrs = 4
    gnn_model = GNN(num_node_attrs=num_node_attrs, adj_matrix=train_set.adj_matrix, gnn_config=gnn_config)

    #######################
    ### Noise scheduler ###
    #######################
    noise_scheduler = NoiseScheduler(beta_scheduler_type=beta_scheduler_type, num_timesteps=trials_config["num_timesteps"], scheduler_kwargs=scheduler_kwargs)

    #######################
    ### Diffusion model ###
    #######################
    diff_model = DiffusionModel(gnn_model, noise_scheduler).to(device)


    #########################
    #########################
    #########################

    # Initialize optimizer
    optimizer_kwargs = {"lr": lr, "weight_decay": 0}
    non_frozen_parameters = [p for p in diff_model.parameters() if p.requires_grad]
    optimizer = None
    #print("optimizer kwargs ", optimizer_kwargs)

    #print("non frozen parameters ", non_frozen_parameters)
    if len(non_frozen_parameters) > 0:
        optimizer = getattr(optim, optimizer_name)(params=non_frozen_parameters, **optimizer_kwargs)


    print("optimizer ", optimizer)

    trial.set_user_attr("model_class", diff_model.__class__)
    trial.set_user_attr("optimizer_class", optimizer.__class__)
    trial.set_user_attr("model_name", diff_model._name)
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

    model_path = 'trial_{}_losses_model_{}'.format(trial.number, diff_model._name)
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



    for epoch in range(config["num_epochs"]):
        #try:
        if train:
            diff_model.train(True)
            avg_loss = train_one_epoch(diff_model, optimizer, train_loader, config, loss_fn=loss_fn, use_cuda=use_cuda)  # Train the model

        diff_model.train(False)
        if len(validation_set) == 0:
            avg_vloss = avg_loss
            avg_vacc = 0
        else:
            avg_vloss, avg_vacc = validate(diff_model, validation_loader, loss_fn=loss_fn, use_cuda=use_cuda)   # Evaluate the model

        if scheduler is not None:
            scheduler.step(avg_vloss)
            print("scheduler lr: {}".format(scheduler._last_lr))

        avg_loss_list.append(avg_loss)
        avg_vloss_list.append(avg_vloss)

        print("epoch: {}, LOSS train: {}, val: {}, ACC val: {}".format(epoch, avg_loss, avg_vloss, avg_vacc))

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_epoch = epoch

            model_state_dict = diff_model.state_dict()
            if train:
                optimizer_state_dict = optimizer.state_dict()

        # For pruning (stops trial early if not promising)
        trial.report(avg_vloss, epoch)
        # Handle pruning based on the intermediate value.
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()
        # except Exception as e:
        #    print(str(e))
        #    return avg_vloss

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

    config = {
        "num_epochs": trials_config["num_epochs"],
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

