import os
import sys
import copy
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
from tqdm import tqdm
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from models.auxiliary_functions import get_loss_fn
from dataset.cnn_diffusion.dataset_preprocessing import prepare_dataset
from dataset.denoising_diffusion_latents.latents_dataset import LatentsDataset
from torch.utils.data import DataLoader
from models.vqgan.vqgan_model import VQGAN
from models.cnn_diffusion.diffusion_model import DiffusionModel
from models.denoising_diffusion_latents.conditional_denoising_diffusion import ConditionalDiffusion
from models.schedulers import NoiseScheduler
from models.denoising_diffusion_latents.Unet3D import UNet3D
from models.denoising_diffusion_latents.Unet3D import EMA


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"]=""


def validate(
    model,
    validation_loader,
    loss_fn=nn.MSELoss(),     # MSE is the standard loss for diffusion
    metric_fn=nn.L1Loss(),    # optional: MAE as an additional metric
):
    """
    Validate model on a validation set.

    :param model: model to evaluate (often the EMA model).
    :param validation_loader: dataloader yielding (volumes, conds).
    :param loss_fn: loss function (default = MSELoss).
    :param metric_fn: evaluation metric function (default = L1Loss = MAE).
    :return: (avg_vloss, avg_metric) average validation loss and metric.
    """
    running_vloss = 0.0
    running_metric = 0.0
    device = next(model.parameters()).device

    model.eval()  # set evaluation mode
    with torch.no_grad():
        for i, samples in enumerate(validation_loader):
            # unpack batch
            volumes, conds = samples
            volumes = volumes.to(device)
            conds = conds.to(device)

            # forward pass
            noise, predicted_noise = model(volumes, conds)

            # compute loss
            vloss = loss_fn(noise, predicted_noise)
            running_vloss += vloss.item()

            # compute metric (e.g., MAE)
            metric = metric_fn(noise, predicted_noise)
            running_metric += metric.item()

        avg_vloss = running_vloss / (i + 1)
        avg_metric = running_metric / (i + 1)

    return avg_vloss, avg_metric


def train_one_epoch(
    model,
    optimizer,
    train_loader,
    loss_fn=nn.MSELoss(),
    ema=None,
    ema_model=None):
    """
    Train one epoch for denoising diffusion with optional EMA updates.
    :param model: main training model (diff_model).
    :param optimizer: optimizer for model parameters.
    :param train_loader: dataloader yielding (volumes, conds).
    :param loss_fn: loss function (default = MSELoss, standard for diffusion).
    :param ema: EMA helper instance (optional).
    :param ema_model: EMA model copy of `model` (optional, must match ema).
    :return: average training loss over the epoch.
    """
    running_loss = 0.0
    device = next(model.parameters()).device  # auto-detect model device

    model.train(True)  # set training mode
    for i, samples in enumerate(train_loader):
        # unpack batch
        volumes, conds = samples
        volumes = volumes.to(device)
        conds = conds.to(device)

        # zero gradients
        optimizer.zero_grad()

        # forward pass: model predicts noise given (volumes, conds)
        noise, predicted_noise = model(volumes, conds)

        # diffusion objective: minimize MSE between true and predicted noise
        loss = loss_fn(noise, predicted_noise)

        # backpropagation
        loss.backward()
        optimizer.step()

        # EMA update: keep a smoothed version of the model
        if ema is not None and ema_model is not None:
            ema.update_model_average(ema_model, model)

        running_loss += loss.item()

    # average loss over the epoch
    train_loss = running_loss / (i + 1)
    return train_loss


def build_latents_dataset(vqgan, dataset, save_dir, dset_type="train", batch_size=1, device="cuda"):
    """
    Build a latent-space dataset using a trained VQGAN encoder.
    The dataset is saved as .npz files for efficient re-use.

    :param vqgan: Pre-trained VQGAN model (with encoder + quantizer).
    :param dataset: Original dataset of 3D CT volumes + conditions.
    :param save_dir: Directory where latent datasets will be stored.
    :param dset_type: Dataset split type ("train", "validation", "test").
    :param batch_size: Mini-batch size for encoding data.
    :param device: Device to run computations on ("cuda" or "cpu").
    :return: LatentsDataset object containing compressed (z, cond) pairs.
    """

    os.makedirs(save_dir, exist_ok=True)

    # Wrap dataset into a DataLoader (no shuffle = preserve order)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_latents = []  # stores VQGAN latent representations
    all_conds = []    # stores conditioning variables

    # Put VQGAN into eval mode to disable dropout & batchnorm updates
    vqgan.eval()
    vqgan.to(device)

    with torch.no_grad():  # no gradient tracking needed during encoding
        for i, (volumes, conds) in enumerate(tqdm(loader)):
            volumes = volumes.to(device).float()

            if "train_on_codebooks" in config and config["train_on_codebooks"]:
                # --- ALDM approach ---
                # Train diffusion model on discrete quantized VQGAN latents
                h = vqgan.encoder(volumes)   # encode image into feature map
                h = vqgan.quant_conv(h)      # linear projection before quantization
                vqgan.quantize.sane_index_shape = True
                z_q, loss, (_, _, codebook_indices) = vqgan.quantize(h)

                # Scale quantized latents by their std dev for normalization
                model_input = z_q / z_q.flatten().std()

            else:
                # --- MedicalDiffusion approach ---
                # Train diffusion model on continuous *pre-quantized* embeddings
                h = vqgan.encoder(volumes)
                h = vqgan.quant_conv(h)

                # Normalize h to [-1, 1] range using quantizer embeddings
                model_input = ((h - vqgan.quantize.embedding.weight.min()) /
                               (vqgan.quantize.embedding.weight.max() -
                                vqgan.quantize.embedding.weight.min())) * 2.0 - 1.0

                # Save min/max of embeddings for reproducibility
                if not os.path.exists(os.path.join(save_dir, "quantize_embedings_min_max")):
                    np.save(
                        os.path.join(save_dir, "quantize_embedings_min_max"),
                        (
                            vqgan.quantize.embedding.weight.min().cpu().numpy(),
                            vqgan.quantize.embedding.weight.max().cpu().numpy()
                        )
                    )

            # Save latent representation (to CPU → numpy)
            all_latents.append(model_input.cpu().numpy())

            # Convert conditions to numpy array
            # If conds is list/tuple → stack into a single tensor
            conds_np = torch.stack(conds, dim=1).cpu().numpy() if isinstance(conds, (list, tuple)) else conds.cpu().numpy()
            all_conds.append(conds_np)

    # Concatenate all batches along batch dimension
    all_latents = np.concatenate(all_latents, axis=0)
    all_conds = np.concatenate(all_conds, axis=0)

    # Save dataset as compressed .npz file
    saved_data_path = os.path.join(save_dir, f"latents_dataset_{dset_type}.npz")
    np.savez_compressed(
        saved_data_path,
        latents=all_latents,
        conds=all_conds
    )

    print(f" Saved latents dataset to {saved_data_path}, "
          f"shape: {all_latents.shape}, conds: {all_conds.shape}")

    return LatentsDataset(saved_data_path)


def objective(trial, trials_config):
    """
    Optuna objective function for training and evaluating a conditional diffusion model.

    Handles:
    - Hyperparameter sampling via Optuna
    - Dataset preparation (train/val/test latents via VQGAN)
    - Model initialization (UNet3D + ConditionalDiffusion)
    - Optimizer, scheduler, and EMA setup
    - Training/validation loop with checkpointing
    - Returns best validation loss for Optuna to minimize

    :param trial: Optuna trial object for hyperparameter optimization.
    :param trials_config: dictionary of configuration options and search spaces.
    :return: best validation loss (float).
    """
    # ------------------------
    # Trial-level config setup
    # ------------------------
    best_vloss = 1_000_000.  # large sentinel value
    batch_size_train = trial.suggest_categorical("batch_size_train", trials_config["batch_size_train"])
    batch_size_sample = trial.suggest_categorical("batch_size_sample", trials_config["batch_size_sample"])
    config["batch_size_train"] = batch_size_train
    config["batch_size_sample"] = batch_size_sample

    # ------------------------
    # Loss function setup
    # ------------------------
    loss_function = trials_config.get("loss_function", ["MSE", []])
    loss_function = trial.suggest_categorical("loss_function", trials_config["loss_function"]) \
        if "loss_function" in trials_config else loss_function
    loss_fn = get_loss_fn(loss_function)

    # ------------------------
    # Optional dataset size sampling
    # ------------------------
    if "n_train_samples" in trials_config and trials_config["n_train_samples"] is not None:
        n_train_samples = trial.suggest_categorical("n_train_samples", trials_config["n_train_samples"])
        config["n_train_samples"] = n_train_samples
        if "n_test_samples" in trials_config and trials_config["n_test_samples"] is not None:
            config["n_test_samples"] = trials_config["n_test_samples"]

    train_set, validation_set, test_set = prepare_dataset(study, config, data_dir=data_dir,
                                                          data_file_name=trials_config["data_file_name"],
                                                          serialize_path=output_dir)

    # ------------------------
    # Optimizer setup
    # ------------------------
    optimizer_name = trial.suggest_categorical("optimizer_name", trials_config["optimizer_name"]) \
        if "optimizer_name" in trials_config else "AdamW"
    optimizer_kwargs = trial.suggest_categorical("optimizer_kwargs", trials_config["optimizer_kwargs"]) \
        if "optimizer_kwargs" in trials_config else {"lr": 0.0001}

    # Add additional training flags
    for k in ["mask_loss", "weighted_mask_loss", "train_on_codebooks"]:
        if k in trials_config:
            config[k] = trials_config[k]

    # ------------------------
    # Load VQGAN for latent space
    # ------------------------
    vqgan_study = joblib.load(os.path.join(trials_config["vqgan_results_dir"], "study.pkl"))
    vqgan_model_path = trials_config.get(
        "vqgan_model_path",
        os.path.join(trials_config["vqgan_results_dir"], "logger/version_0/checkpoints/val/last.ckpt")
    )
    vqgan_model_config = load_trials_config(trials_config["vqgan_trials_config"])["model_config"][0] \
        if "vqgan_trials_config" in trials_config else vqgan_study.best_trial.params["model_config"]

    vqgan_model_checkpoint = VQGAN.load_from_checkpoint(vqgan_model_path, **vqgan_model_config)
    vqgan_model_checkpoint.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqgan_model_checkpoint.to(device)

    # Build latent datasets
    latents_datasets_dir = os.path.join(output_dir, "latents_datasets")
    latents_train_set = build_latents_dataset(vqgan_model_checkpoint, train_set, latents_datasets_dir,
                                              dset_type="train", batch_size=1, device="cuda")
    latents_validation_set = build_latents_dataset(vqgan_model_checkpoint, validation_set, latents_datasets_dir,
                                                   dset_type="validation", batch_size=1, device="cuda")
    latents_test_set = build_latents_dataset(vqgan_model_checkpoint, validation_set, latents_datasets_dir,
                                             dset_type="test", batch_size=1, device="cuda")

    # Replace loaders with latent space versions
    train_loader = torch.utils.data.DataLoader(latents_train_set, batch_size=config["batch_size_train"], shuffle=True)
    validation_loader = torch.utils.data.DataLoader(latents_validation_set, batch_size=config["batch_size_train"], shuffle=False)
    test_loader = torch.utils.data.DataLoader(latents_test_set, batch_size=config["batch_size_test"], shuffle=False)

    # ------------------------
    # UNet and diffusion model
    # ------------------------
    unet_config = trial.suggest_categorical("unet_config", trials_config["unet_config"]) \
        if "unet_config" in trials_config else {}

    if trials_config["model_class_name"] == "Unet3D":
        model_class = UNet3D
    unet_diffusion_model = model_class(**unet_config)

    noise_scheduler = NoiseScheduler(**trials_config["noise_scheduler_params"])
    diff_model = ConditionalDiffusion(unet_diffusion_model, trials_config["image_size"], noise_scheduler).to(device)

    # ------------------------
    # Optimizer + Scheduler
    # ------------------------
    non_frozen_parameters = [p for p in diff_model.parameters() if p.requires_grad]
    optimizer = getattr(optim, optimizer_name)(params=non_frozen_parameters, **optimizer_kwargs) \
        if len(non_frozen_parameters) > 0 else None

    scheduler = None
    if "scheduler" in trials_config and optimizer is not None:
        trial_scheduler = trial.suggest_categorical("scheduler", trials_config["scheduler"])
        if trial_scheduler["class"] == "ReduceLROnPlateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min",
                patience=trial_scheduler["patience"],
                factor=trial_scheduler["factor"]
            )
        else:
            scheduler = lr_scheduler.StepLR(
                optimizer, step_size=trial_scheduler["step_size"],
                gamma=trial_scheduler["gamma"]
            )

    # ------------------------
    # EMA setup
    # ------------------------
    ema_beta = trials_config.get("EMA_beta", 0.999)
    ema = EMA(beta=ema_beta)
    ema_model = copy.deepcopy(diff_model)
    ema_model.eval()

    # ------------------------
    # Training loop
    # ------------------------
    start_time = time.time()
    avg_loss_list, avg_vloss_list = [], []

    for epoch in range(config["num_epochs"]):
        # Train
        if trials_config.get("train", True):
            diff_model.train(True)
            avg_loss = train_one_epoch(
                diff_model, optimizer, train_loader,
                loss_fn=loss_fn,
                ema=ema, ema_model=ema_model
            )
        else:
            avg_loss = 0

        # Validate (using EMA weights)
        diff_model.train(False)
        avg_vloss, avg_vacc = validate(ema_model, validation_loader, loss_fn=loss_fn)

        # Scheduler step
        if scheduler is not None:
            scheduler.step(avg_loss)

        avg_loss_list.append(avg_loss)
        avg_vloss_list.append(avg_vloss)
        print(f"epoch: {epoch}, LOSS train: {avg_loss:.4f}, val: {avg_vloss:.4f}, ACC val: {avg_vacc:.4f}")

        # Save best model
        if avg_vloss < best_vloss:
            best_vloss, best_epoch = avg_vloss, epoch
            print("best vloss", best_vloss)
            model_state_dict = ema_model.state_dict()
            optimizer_state_dict = optimizer.state_dict() if optimizer else {}

            model_path_epoch = os.path.join(output_dir, f"trial_{trial.number}_model_best_{epoch}.pt")
            torch.save({
                'best_epoch': best_epoch,
                'best_model_state_dict': model_state_dict,
                'best_optimizer_state_dict': optimizer_state_dict,
                'best_scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': avg_loss_list,
                'valid_loss': avg_vloss_list,
                'training_time': time.time() - start_time,
            }, model_path_epoch)

        # Optuna pruning
        trial.report(avg_vloss, epoch)

    return best_vloss


def load_trials_config(path_to_config):
    """
    Load trial configuration YAML file.
    :param path_to_config: Path to YAML config file with trial setup.
    :return: loaded_config dictionary with all experimental settings.
    """
    with open(path_to_config, "r") as f:
        loaded_config = yaml.load(f, Loader=yaml.FullLoader)
    return loaded_config


if __name__ == '__main__':
    """
    Main entry point for Optuna hyperparameter search.
    - Loads dataset & configuration
    - Prepares output directory
    - Sets random seeds for reproducibility
    - Runs Optuna optimization with chosen sampler
    - Saves best results and study statistics
    """

    # --- Parse CLI arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument('trials_config_path', help='Path to trials config (YAML)')
    parser.add_argument('data_dir', help='Path to dataset directory')
    parser.add_argument('output_dir', help='Path to store Optuna results')
    parser.add_argument("-c", "--cuda", default=False, action='store_true', help="Use CUDA if available")
    parser.add_argument("-a", "--append", default=False, action='store_true', help="Append to existing results")
    args = parser.parse_args(sys.argv[1:])

    # --- Setup paths & device ---
    data_dir = args.data_dir
    output_dir = args.output_dir
    trials_config = load_trials_config(args.trials_config_path)
    use_cuda = args.cuda

    # --- Build base config dict ---
    config = {
        "num_epochs": trials_config["num_epochs"],
        "batch_size_train": trials_config["batch_size_train"],
        "batch_size_test": trials_config["batch_size_test"] if "batch_size_test" in trials_config else 250,
        "n_train_samples": trials_config.get("n_train_samples", None),
        "n_test_samples": trials_config.get("n_test_samples", None),
        "train_samples_ratio": trials_config.get("train_samples_ratio", 0.9),
        "val_samples_ratio": trials_config.get("val_samples_ratio", 0.2),
        "seed": trials_config.get("random_seed", 12345),
        "output_dir": output_dir,
    }

    # --- Optuna trial count ---
    num_trials = trials_config["num_trials"]

    # --- Device selection ---
    print("use cuda ", use_cuda)
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    print("device ", device)
    print("config seed ", config["seed"])

    # --- Reproducibility setup ---
    random_seed = trials_config["random_seed"]
    torch.backends.cudnn.enabled = False  # disable nondeterministic CuDNN algos
    torch.manual_seed(random_seed)

    # --- Output directory handling ---
    output_dir = os.path.join(output_dir, f"seed_{random_seed}")
    if os.path.exists(output_dir) and not args.append:
        shutil.rmtree(output_dir)  # overwrite previous results
    if not args.append:
        os.mkdir(output_dir)
    elif not os.path.exists(output_dir):
        raise NotADirectoryError(f"Output dir {output_dir} does not exist")

    # --- Optuna sampler setup ---
    sampler = TPESampler(seed=random_seed)  # TPE = Tree-structured Parzen Estimator
    if "sampler_class" in trials_config:
        if trials_config["sampler_class"] == "BruteForceSampler":
            sampler = BruteForceSampler(seed=random_seed)

    # --- Create Optuna study ---
    study = optuna.create_study(sampler=sampler, direction="minimize")

    # --- Objective function wrapper ---
    def obj_func(trial):
        return objective(trial, trials_config)

    # --- Run Optuna optimization ---
    study.optimize(obj_func, n_trials=num_trials)

    # ================================
    # Results & statistics
    # ================================
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\nStudy statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    # --- Best trial info ---
    trial = study.best_trial
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # --- Save trial results to CSV ---
    df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete', 'duration'], axis=1)
    df = df.loc[df['state'] == 'COMPLETE']   # keep only completed runs
    df = df.drop('state', axis=1)
    df = df.sort_values('value')             # sort by loss value
    df.to_csv(os.path.join(output_dir, 'optuna_results.csv'), index=False)
    print("\nOverall Results (ordered by objective value):\n {}".format(df))

    # --- Hyperparameter importance analysis ---
    try:
        most_important_parameters = optuna.importance.get_param_importances(study, target=None)
        print('\nMost important hyperparameters:')
        for key, value in most_important_parameters.items():
            print(f'  {key}: {(15-len(key))*" "}{value*100:.2f}%')
    except Exception as e:
        print(str(e))

    # --- Save study object for later reload ---
    joblib.dump(study, os.path.join(output_dir, "study.pkl"))
