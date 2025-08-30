import os
import sys
import argparse
import joblib
import torch
import torch.nn as nn
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler, BruteForceSampler
import time
import yaml
import shutil
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from models.auxiliary_functions import get_loss_fn
from dataset.cnn_diffusion.dataset_preprocessing import prepare_dataset
from torch.utils.data import DataLoader
from models.vqgan.vqgan_model import VQGAN
from models.cnn_diffusion.diffusion_model import DiffusionModel
from models.schedulers import NoiseScheduler
from models.cnn_diffusion.medicaldiffusion_unet3D import MedicalDiffusionUNet3D
from models.cnn_diffusion.medicaldiffusion_unet3D_own import MedicalDiffusionUNet3DOwn


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"]=""


def get_used_codebook_indices(dataloader, vqgan):
    # Get only used codebook indices
    used_indices = set()
    for batch in dataloader:
        volumes, cond = batch
        input = volumes.to(device)
        h = vqgan.encoder(input)
        h = vqgan.quant_conv(h)
        print("latent space shape ", h.shape)
        # vqgan.quantize.sane_index_shape = True  # get in spatial format
        z_q, loss, (_, _, codebook_indices) = vqgan.quantize(h)
        used_indices.update(codebook_indices.cpu().numpy().flatten())

    used_codebook_indices_file_path = os.path.join("used_indices.npy")
    np.save(used_codebook_indices_file_path, np.array(sorted(used_indices)))
    vqgan.quantize.set_used_indices(used_codebook_indices_file_path)
    return vqgan, used_indices


def validate(model, vqgan, validation_loader, config, loss_fn=nn.CrossEntropyLoss(), acc_fn=nn.CrossEntropyLoss(), use_cuda=False):
    """
    Validate model
    :param model:
    :param loss_fn:
    :return:
    """
    running_vloss = 0.0
    running_vacc = 0
    with torch.no_grad():
        for i, samples in enumerate(validation_loader):
            # if torch.cuda.is_available() and use_cuda:
            #     samples = samples.cuda()
            volumes, cond = samples
            input = volumes.to(device)

            if "train_on_codebooks" in config and config["train_on_codebooks"]:
                ####
                # ALDM approach
                ####
                h = vqgan.encoder(input)
                h = vqgan.quant_conv(h)
                vqgan.quantize.sane_index_shape = True  # get in spatial format
                z_q, loss, (_, _, codebook_indices) = vqgan.quantize(h)
                # In ALDM implementation - diffusion model is trained on !!z_q!! - quantized representations scaled by its std
                ## z_q / z_q.flatten().std()
                model_input = z_q / z_q.flatten().std()
                #######
                #######
            else:
                ####
                # MedicalDiffusion approach
                ####
                h = vqgan.encoder(input)
                h = vqgan.quant_conv(h)
                print("vqgan.quantize.embedding ", vqgan.quantize.embedding)
                # model_input = ((h - vqgan.quantize.embedding.min()) /
                #      (vqgan.quantize.embedding.max() -
                #       vqgan.quantize.embedding.min())) * 2.0 - 1.0
                model_input = h
                # In MedicalDiffusion diffusion model is trained on inputs to quantization which is transformd to [-1, 1]
                #######
                #######

            noise, predicted_noise = model(model_input)
            vloss = loss_fn(noise, predicted_noise)

            running_vloss += vloss.item()

            vacc = acc_fn(noise, predicted_noise)
            running_vacc += vacc.item()

        avg_vloss = running_vloss / (i + 1)
        avg_vacc = running_vacc / (i + 1)

    return avg_vloss, avg_vacc


def train_one_epoch(model, vqgan, optimizer, train_loader, config, loss_fn=nn.CrossEntropyLoss(), use_cuda=True):
    """
    Train NN
    :param model:
    :param optimizer:
    :param loss_fn:
    :return:
    """
    # Get only used codebook indices
    running_loss = 0.
    for i, samples in enumerate(train_loader):
        volumes, conds = samples
        input = volumes.to(device)
        print("input shape ", input.shape)

        if "train_on_codebooks" in config and config["train_on_codebooks"]:
            ####
            # ALDM approach
            ####
            h = vqgan.encoder(input)
            h = vqgan.quant_conv(h)
            vqgan.quantize.sane_index_shape = True  # get in spatial format
            z_q, loss, (_, _, codebook_indices) = vqgan.quantize(h)
            # In ALDM implementation - diffusion model is trained on !!z_q!! - quantized representations scaled by its std
            ## z_q / z_q.flatten().std()
            model_input = z_q / z_q.flatten().std()
            #######
            #######
        else:
            ####
            # MedicalDiffusion approach
            ####
            h = vqgan.encoder(input)
            h = vqgan.quant_conv(h)
            model_input = ((h - vqgan.quantize.embedding.weight.min()) /
                 (vqgan.quantize.embedding.weight.max()-
                  vqgan.quantize.embedding.weight.min())) * 2.0 - 1.0
            # In MedicalDiffusion diffusion model is trained on inputs to quantization which is transformd to [-1, 1]
            #######
            #######

        print("model input min: {}, max: {}".format(model_input.min(), model_input.max()))
        print("model input shape ", model_input.shape)

        # if torch.cuda.is_available() and use_cuda:
        #     samples = samples.cuda()
        optimizer.zero_grad()


        noise, predicted_noise = model(model_input)

        # print("noise.shape ", noise.shape)
        # print("predicted noise shape ", predicted_noise.shape)
        # print("noise ", noise)
        # print("predicted noise ", predicted_noise)
        loss = loss_fn(noise, predicted_noise)

        loss.backward()
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

    train_loss = running_loss / (i + 1)
    return train_loss


def objective(trial, trials_config, train_loader, validation_loader):
    best_vloss = 1_000_000.
    best_epoch = 0
    save_model_best_epoch = 0
    lr = trial.suggest_categorical("lr", trials_config["lr"])
    batch_size_train = trial.suggest_categorical("batch_size_train", trials_config["batch_size_train"])
    batch_size_sample = trial.suggest_categorical("batch_size_sample", trials_config["batch_size_sample"])
    config["batch_size_train"] = batch_size_train
    config["batch_size_sample"] = batch_size_sample

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

    if "n_train_samples" in trials_config and trials_config["n_train_samples"] is not None:
        n_train_samples = trial.suggest_categorical("n_train_samples", trials_config["n_train_samples"])
        config["n_train_samples"] = n_train_samples

        if "n_test_samples" in trials_config and trials_config["n_test_samples"] is not None:
            config["n_test_samples"] = trials_config["n_test_samples"]

        train_set, validation_set, test_set = prepare_dataset(study, config, data_dir=data_dir, serialize_path=output_dir)

        print("len(trainset): {}, len(valset): {}, len(testset): {}".format(len(train_set), len(validation_set), len(test_set)))

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=config["batch_size_train"], shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size_test"], shuffle=False)

    optimizer_name = "AdamW"
    if "optimizer_name" in trials_config:
        optimizer_name = trial.suggest_categorical("optimizer_name", trials_config["optimizer_name"])

    if "mask_loss" in trials_config:
        config["mask_loss"] = trials_config["mask_loss"]
    if "weighted_mask_loss" in trials_config:
        config["weighted_mask_loss"] = trials_config["weighted_mask_loss"]
    if "train_on_codebooks" in trials_config:
        config["train_on_codebooks"] = trials_config["train_on_codebooks"]


    #####################
    #####################
    #  Initilize model #
    ####################
    ####################

    cnn_config = {}
    if "cnn_config" in trials_config:
        cnn_config = trial.suggest_categorical("cnn_config", trials_config["cnn_config"])


    ################
    ## Load VQGAN ##
    ################
    vqgan_study = joblib.load(os.path.join(trials_config["vqgan_results_dir"], "study.pkl"))
    vqgan_model_path = os.path.join(trials_config["vqgan_results_dir"], "logger/version_0/checkpoints/val/last.ckpt")

    vqgan_model_checkpoint = VQGAN.load_from_checkpoint(vqgan_model_path,
                                                        **vqgan_study.best_trial.params["model_config"])

    vqgan_model_checkpoint.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqgan_model_checkpoint.to(device)
    #vqgan_model_checkpoint, used_codebook_indices = get_used_codebook_indices(train_loader, vqgan_model_checkpoint)

    #num_classes = len(used_codebook_indices)

    #cnn_config["num_classes"] = num_classes

    model_class_name = trials_config["model_class_name"]
    if model_class_name == "MedicalDiffusionUNet3D":
        model_class = MedicalDiffusionUNet3D
    elif model_class_name == "MedicalDiffusionUNet3DOwn":
        model_class = MedicalDiffusionUNet3DOwn

    unet_diffusion_model = model_class(**cnn_config)

    #######################
    ### Noise scheduler ###
    #######################
    noise_scheduler_kwargs = {}
    if "noise_scheduler_config" in trials_config:
        noise_scheduler_kwargs = trials_config["noise_scheduler_config"]

    #######################
    ### Noise scheduler ###
    #######################
    noise_scheduler_kwargs = {'beta_scheduler_type': beta_scheduler_type,
                              'num_timesteps': trials_config["num_timesteps"],
                              'scheduler_kwargs': scheduler_kwargs}
    noise_scheduler = NoiseScheduler(**noise_scheduler_kwargs)

    #######################
    ### Diffusion model ###
    #######################
    diff_model = DiffusionModel(unet_diffusion_model, trials_config["image_size"], noise_scheduler).to(device)

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

    trial.set_user_attr("diff_model_class", diff_model.__class__)
    trial.set_user_attr("diff_model_name", diff_model._name)
    trial.set_user_attr("unet_diffusion_model_name", unet_diffusion_model._name)
    trial.set_user_attr("cnn_config", cnn_config)
    trial.set_user_attr("noise_scheduler_kwargs", noise_scheduler_kwargs)
    trial.set_user_attr("optimizer_class", optimizer.__class__)
    trial.set_user_attr("optimizer_kwargs", optimizer_kwargs)
    trial.set_user_attr("loss_fn", loss_fn)
    trial.set_user_attr("trials_config", trials_config)

    # Training of the model
    start_time = time.time()
    avg_loss_list = []
    avg_vloss_list = []
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
        print("scheduler patience: {}, factor: {}".format(trial_scheduler["patience"], trial_scheduler["factor"]))
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
            avg_loss = train_one_epoch(diff_model, vqgan_model_checkpoint, optimizer, train_loader, config, loss_fn=loss_fn, use_cuda=use_cuda)  # Train the model

        diff_model.train(False)
        if len(validation_set) == 0:
            avg_vloss = avg_loss
            avg_vacc = 0
        else:
            avg_vloss, avg_vacc = validate(diff_model, vqgan_model_checkpoint, validation_loader, config, loss_fn=loss_fn,
                                           use_cuda=use_cuda)  # Evaluate the model

        if scheduler is not None:
            scheduler.step(avg_loss)
            scheduler_state_dict = scheduler.state_dict()
            print("scheduler lr: {}".format(scheduler._last_lr))

        avg_loss_list.append(avg_loss)
        avg_vloss_list.append(avg_vloss)

        print("epoch: {}, LOSS train: {}, val: {}, ACC val: {}".format(epoch, avg_loss, avg_vloss, avg_vacc))

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_epoch = epoch
            print("best epoch ", best_epoch)

            model_state_dict = diff_model.state_dict()
            if train:
                optimizer_state_dict = optimizer.state_dict()

            model_path_epoch = os.path.join(output_dir, model_path + "_best_{}".format(epoch))

            scheduler_state_dict = scheduler.state_dict()

            torch.save({
                'best_epoch': best_epoch,
                'best_model_state_dict': model_state_dict,
                'best_optimizer_state_dict': optimizer_state_dict,
                'best_scheduler_state_dict': scheduler_state_dict,
                'train_loss': avg_loss_list,
                'valid_loss': avg_vloss_list,
                'training_time': time.time() - start_time,
            }, model_path_epoch)

        # For pruning (stops trial early if not promising)
        trial.report(avg_vloss, epoch)
        # Handle pruning based on the intermediate value.
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()
        # except Exception as e:
        #    print(str(e))
        #    return avg_vloss
    #
    # inv_samples, orig_samples = diff_model.sample(batch_size=batch_size_sample, inverse_transform=None)
    #
    # exit()

    #for key, value in trial.params.items():
    #    model_path += "_{}_{}".format(key, value)


    #gnn_model.adj_matrix = None

    model_path = os.path.join(output_dir, model_path)

    torch.save({
        'best_epoch': best_epoch,
        'best_model_state_dict': model_state_dict,
        'best_optimizer_state_dict': optimizer_state_dict,
        'best_scheduler_state_dict': scheduler_state_dict,
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
              "train_samples_ratio": trials_config["train_samples_ratio"] if "train_samples_ratio" in trials_config else 0.9,
              "val_samples_ratio": trials_config["val_samples_ratio"] if "val_samples_ratio" in trials_config else 0.2,
              "seed": trials_config["random_seed"] if "random_seed" in trials_config else 12345,
              "output_dir": output_dir,
              #"sub_datasets": trials_config["sub_datasets"] if "sub_datasets" in trials_config else {}
              }

    if "input_transform" in trials_config:
        config["input_transform"] = trials_config["input_transform"]
    if "output_transform" in trials_config:
        config["output_transform"] = trials_config["output_transform"]

    if "data_file_name" in trials_config:
        config["data_file_name"] = trials_config["data_file_name"]

    # Optuna params
    num_trials = trials_config["num_trials"]

    print("use cuda ", use_cuda)
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    print("device ", device)
    print("config seed ", config["seed"])

    # Make runs repeatable
    random_seed = trials_config["random_seed"]
    torch.backends.cudnn.enabled = False  # Disable cuDNN use of nondeterministic algorithms
    torch.manual_seed(random_seed)
    output_dir = os.path.join(output_dir, "seed_{}".format(random_seed))
    if os.path.exists(output_dir) and not args.append:
        #shutil.rmtree(output_dir)
        raise IsADirectoryError("Results output dir {} already exists".format(output_dir))
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
    if "n_train_samples" not in config or not isinstance(config["n_train_samples"], (list, np.ndarray)):
        dataset = prepare_dataset(study, config, data_dir=data_dir, serialize_path=output_dir)
        data_loader = DataLoader(dataset, batch_size=config["batch_size_train"], shuffle=True)

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

