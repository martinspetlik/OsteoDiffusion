import sys
import argparse
import joblib
import torch
import numpy as np
from dataset.cnn_diffusion.bone_dataset_CT import BoneDatasetCT
from visualization.visualize_data import plot_train_valid_loss, render_3d_scan, render_two_3d_scans
import scipy as sc
from models.vqgan.vqgan_model import VQGAN
import pandas as pd
import nibabel as nib
import os
import glob
import yaml


def load_dataset(dataset_dir, data_file_name):
    """
    Load the bone CT dataset from a specified directory.

    :param dataset_dir: Path to dataset directory containing samples.
    :param data_file_name: Name of the file within each sample folder (e.g. 'data.npz').
    :return: Initialized BoneDatasetCT object.
    """
    dataset = BoneDatasetCT(data_dir=dataset_dir, data_file_name=data_file_name)
    return dataset


def plot_log_images(images_dir, epoch):
    """
    Render saved .nii.gz images from a specified epoch and save them as 3D visualizations.

    :param images_dir: Directory containing logged images (usually Lightning logger outputs).
    :param epoch: Epoch number to visualize.
    :return: None
    """
    epoch_str = f"e-{epoch:06}"
    nii_files = sorted(glob.glob(os.path.join(images_dir, f"*_{epoch_str}_*.nii.gz")))
    print("nii_files ", nii_files)
    for file in nii_files:
        base = os.path.basename(file)
        label = base.split("_")[0]  # e.g., 'input', 'recon', 'target'
        print("label ", label)
        img = np.squeeze(nib.load(file).get_fdata())
        print("img ", img.shape)
        if label in ["source", "recon"]:
            render_3d_scan(img, title=label, fig_name=f"{label}_prediction.png")


def get_vqgan_latents(vqgan, dataloader, device, working_dir):
    """
    Compute and store used VQGAN latent code indices for a dataset.

    :param vqgan: Trained VQGAN model.
    :param dataloader: PyTorch DataLoader providing dataset samples.
    :param device: Device ('cuda' or 'cpu') for computation.
    :param working_dir: Directory to save used code indices file.
    :return: None
    """
    # Track used codebook indices
    used_indices = set()
    for batch in dataloader:
        input, cond = batch
        input = input.to(device)
        h = vqgan.encoder(input)
        h = vqgan.quant_conv(h)
        z_q, loss, (_, _, codebook_indices) = vqgan.quantize(h)
        used_indices.update(codebook_indices.cpu().numpy().flatten())

    # Save used indices to file
    used_codebook_indices_file_path = os.path.join(working_dir, "used_indices.npy")
    np.save(used_codebook_indices_file_path, np.array(sorted(used_indices)))
    vqgan.quantize.set_used_indices(used_codebook_indices_file_path)

    # Compute and store latent codes for later use
    vqgan.eval()
    latent_codes = []
    with torch.no_grad():
        for batch in dataloader:
            input, cond = batch
            input = input.to(device)
            h = vqgan.encoder(input)
            h = vqgan.quant_conv(h)
            vqgan.quantize.sane_index_shape = True  # maintain spatial index format
            z_q, loss, (_, _, codebook_indices) = vqgan.quantize(h)
            latent_codes.append(codebook_indices)


def get_mean_val_loss(df, column_name):
    """
    Compute mean validation loss per epoch for a specific metric column.

    :param df: Pandas DataFrame containing training/validation logs.
    :param column_name: Name of validation loss column (e.g. 'val/reconstruction_loss').
    :return: Pandas Series of mean loss per epoch or None if column missing.
    """
    if column_name not in df:
        return None

    col_data = df[["epoch", column_name]].dropna()
    grouped = col_data.groupby("epoch").mean().reset_index()
    return grouped[column_name]


def load_trials_config(path_to_config):
    """
    Load experiment configuration from a YAML file.

    :param path_to_config: Path to YAML configuration file.
    :return: Parsed dictionary with configuration.
    """
    with open(path_to_config, "r") as f:
        trials_config = yaml.load(f, Loader=yaml.FullLoader)
    return trials_config


def load_models(results_dir, model_checkpoint_path, vqgan_train_config, dataset):
    """
    Load a trained VQGAN model, aggregate training logs, and visualize training progress.

    :param results_dir: Directory containing model outputs (logs, metrics, datasets).
    :param model_checkpoint_path: Path to VQGAN checkpoint file (.ckpt).
    :param vqgan_train_config: Path to YAML config used for model training.
    :param dataset: Dataset object for generating latents or reconstructions.
    :return: None
    """
    # Find all metrics.csv files from multiple Lightning runs
    csv_files = glob.glob(os.path.join(results_dir, "logger/version_*/metrics.csv"))

    dfs = []
    epoch_offset = 0

    # Helper function to sort by version number
    import re
    def extract_version_number(path):
        match = re.search(r"version_(\d+)", path)
        return int(match.group(1)) if match else -1

    csv_files_sorted = sorted(csv_files, key=extract_version_number)

    # Combine all metrics.csv files into one DataFrame with continuous epoch indexing
    for f in csv_files_sorted:
        df = pd.read_csv(f)
        version = extract_version_number(f)
        df["version"] = version
        df["epoch"] = df["epoch"] + epoch_offset
        epoch_offset = df["epoch"].max() + 1
        dfs.append(df)

    all_metrics = pd.concat(dfs, ignore_index=True)

    # Load saved validation subset
    val_dataset = joblib.load(os.path.join(results_dir, "val_dataset.pkl"))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    val_data_loader = None
    if len(val_dataset) > 0:
        BoneDatasetCT.remap_subset_paths(val_dataset)
        val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    plot_losses = True

    # Disable gradients for evaluation
    with torch.no_grad():
        model_config = load_trials_config(vqgan_train_config)["model_config"][0]
        model_checkpoint = VQGAN.load_from_checkpoint(model_checkpoint_path, **model_config)

        df = all_metrics

        # Compute grouped loss statistics
        if "train/reconstruction_loss_epoch" in df:
            recon_train_loss = df[["epoch", "train/reconstruction_loss_epoch"]].dropna()
            recon_train_grouped = recon_train_loss.groupby("epoch").mean().reset_index()
            recon_val_loss = get_mean_val_loss(df, "val/reconstruction_loss")

            g_train_loss = df[["epoch", "train/adversarial_generator_loss_epoch"]].dropna()
            g_train_grouped = g_train_loss.groupby("epoch").mean().reset_index()
            adversarial_generator_val_loss = get_mean_val_loss(df, "val/adversarial_generator_loss")

            p_train_loss = df[["epoch", "train/perceptual_loss_epoch"]].dropna()
            p_train_grouped = p_train_loss.groupby("epoch").mean().reset_index()
            perceptual_val_loss = get_mean_val_loss(df, "val/perceptual_loss")

            entropy_train_loss = df[["epoch", "train/entropy_loss_epoch"]].dropna()
            entropy_train_grouped = entropy_train_loss.groupby("epoch").mean().reset_index()
            entropy_val_loss = get_mean_val_loss(df, "val/entropy_loss")

            quant_train_loss = df[["epoch", "train/codebook_loss_epoch"]].dropna()
            quant_train_grouped = quant_train_loss.groupby("epoch").mean().reset_index()
            quant_val_loss = get_mean_val_loss(df, "val/codebook_loss")

        # Aggregate generator and discriminator losses
        total_train_loss = df[["epoch", "train/generator_total_loss_epoch"]].dropna()
        total_train_grouped = total_train_loss.groupby("epoch").mean().reset_index()
        total_val_loss = get_mean_val_loss(df, "val/generator_total_loss")

        discloss_train_loss = df[["epoch", "train/discriminator_total_loss_epoch"]].dropna()
        discloss_train_grouped = discloss_train_loss.groupby("epoch").mean().reset_index()
        discriminator_val_loss = get_mean_val_loss(df, "val/discriminator_total_loss")

        # Additional training metrics
        used_codebook_percent = df[["epoch", "train/used_codebook_percent"]].dropna().groupby("epoch").mean().reset_index()
        used_codebook_count = df[["epoch", "train/used_codebook_count"]].dropna().groupby("epoch").mean().reset_index()
        codebook_entropy = df[["epoch", "train/codebook_entropy"]].dropna().groupby("epoch").mean().reset_index()
        codebook_entropy_norm = df[["epoch", "train/codebook_entropy_norm"]].dropna().groupby("epoch").mean().reset_index()

        # Report lowest training loss
        min_train_row = total_train_loss.loc[total_train_loss["train/generator_total_loss_epoch"].idxmin()]
        print(f"Minimum train loss: {min_train_row['train/generator_total_loss_epoch']:.4f} at epoch {int(min_train_row.epoch)}")

        # Plot loss trends if enabled
        if plot_losses:
            plot_train_valid_loss(total_train_grouped["train/generator_total_loss_epoch"],
                                  valid_loss=total_val_loss,
                                  y_label="generator_total_loss", log=True)

            if "train/reconstruction_loss_epoch" in df:
                plot_train_valid_loss(quant_train_grouped["train/codebook_loss_epoch"], valid_loss=quant_val_loss,
                                      y_label="codebook_loss", log=True)
                plot_train_valid_loss(g_train_grouped["train/adversarial_generator_loss_epoch"], valid_loss=adversarial_generator_val_loss,
                                      y_label="adversarial_generator_loss", log=True)
                plot_train_valid_loss(p_train_grouped["train/perceptual_loss_epoch"], valid_loss=perceptual_val_loss,
                                      y_label="perceptual_loss", log=True)
                plot_train_valid_loss(entropy_train_grouped["train/entropy_loss_epoch"], valid_loss=entropy_val_loss,
                                      y_label="entropy_loss", log=True)
                plot_train_valid_loss(recon_train_grouped["train/reconstruction_loss_epoch"], valid_loss=recon_val_loss,
                                      y_label="reconstruction_loss", log=True)

            plot_train_valid_loss(discloss_train_grouped["train/discriminator_total_loss_epoch"],
                                  valid_loss=discriminator_val_loss,
                                  y_label="discriminator_total_loss_epoch", line_value=0.85)

            plot_train_valid_loss(used_codebook_percent["train/used_codebook_percent"],
                                  y_label="used codebook percent")
            plot_train_valid_loss(used_codebook_count["train/used_codebook_count"],
                                  y_label="used codebook count")
            plot_train_valid_loss(codebook_entropy["train/codebook_entropy"],
                                  y_label="codebook entropy")
            plot_train_valid_loss(codebook_entropy_norm["train/codebook_entropy_norm"],
                                  y_label="codebook entropy norm")

        # Evaluate model and generate 3D reconstructions
        model_checkpoint.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_checkpoint.to(device)

        threshold = -0.99
        get_vqgan_latents(model_checkpoint, data_loader, device, results_dir)

        data_loader = val_data_loader if val_data_loader is not None else data_loader
        for batch in data_loader:
            input, cond = batch
            input = input.to(device)
            if isinstance(input, (list, tuple)):
                input = input[0].to(device)
            else:
                input = input.to(device)

            output, _ = model_checkpoint(input)
            output[output < threshold] = -1

            render_two_3d_scans(np.squeeze(input.cpu().numpy()),
                                np.squeeze(output.cpu().numpy()),
                                title="OUTPUT Sampled 3D Scan",
                                fig_name="sampled_3D_scan_with_background_output.png")


def calculate_fgd(real_features, generated_features):
    """
    Calculate Fréchet Geometry Distance (FGD) between two feature sets.

    :param real_features: Numpy array of real sample features.
    :param generated_features: Numpy array of generated sample features.
    :return: Scalar FGD score.
    """
    mu_real, sigma_real = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = generated_features.mean(0), np.cov(generated_features, rowvar=False)

    mu_diff = mu_real - mu_gen
    cov_sqrt, _ = sc.linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    fgd = mu_diff @ mu_diff + np.trace(sigma_real + sigma_gen - 2 * cov_sqrt)
    return fgd


if __name__ == "__main__":
    """
    Main entry point for postprocessing VQGAN experiments.

    Loads dataset and trained model, computes loss curves, renders reconstructions,
    and optionally evaluates Fréchet Geometry Distance.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help='Path to VQGAN postprocess YAML config')
    args = parser.parse_args(sys.argv[1:])

    with open(args.config_path, "r") as f:
        postprocess_config = yaml.load(f, Loader=yaml.FullLoader)

    dataset = load_dataset(postprocess_config["dataset_dir"],
                           postprocess_config["dataset_data_file_name"])
    load_models(postprocess_config["results_dir"],
                postprocess_config["model_path"],
                postprocess_config["vqgan_train_config"],
                dataset)
