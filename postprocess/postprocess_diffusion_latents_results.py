import os
import sys
import argparse
import joblib
import torch
import yaml
import numpy as np
import torch.nn.functional as F
from dataset.cnn_diffusion.bone_dataset_CT import BoneDatasetCT
from visualization.visualize_data import plot_train_valid_loss, render_3d_scan, plot_hist
from models.denoising_diffusion_latents.Unet3D import UNet3D
from models.denoising_diffusion_latents.conditional_denoising_diffusion import ConditionalDiffusion
from dataset.denoising_diffusion_latents.latents_dataset import LatentsDataset
from models.schedulers import NoiseScheduler
import matplotlib.pyplot as plt
import scipy as sc

# Select CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trials_config(path_to_config):
    """
    Load the diffusion training configuration from a YAML file.

    :param path_to_config: Path to YAML configuration file.
    :return: Parsed configuration dictionary.
    """
    with open(path_to_config, "r") as f:
        trials_config = yaml.load(f, Loader=yaml.FullLoader)
    return trials_config


def load_model(model_checkpoint_path, diffusion_train_config):
    """
    Load a trained ConditionalDiffusion model and its checkpoint.

    :param model_checkpoint_path: Path to model checkpoint (.pth or .pt file).
    :param diffusion_train_config: Path to YAML configuration file used for training.
    :return: Tuple (diff_model, checkpoint) containing the loaded model and checkpoint data.
    """
    trials_config = load_trials_config(diffusion_train_config)

    # Select model architecture
    if trials_config["unet_class_name"] == "Unet3D":
        model_class = UNet3D
    else:
        raise NotImplementedError("Only UNet3D model is supported")

    # Initialize model and noise scheduler
    unet_model = model_class(**trials_config["unet_config"][0])
    noise_scheduler = NoiseScheduler(**trials_config["noise_scheduler_params"])

    # Wrap into conditional diffusion model
    diff_model = ConditionalDiffusion(
        cnn_model=unet_model,
        image_size=trials_config["image_size"],
        noise_scheduler=noise_scheduler
    )

    # Load checkpoint weights
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    diff_model.load_state_dict(checkpoint['best_model_state_dict'], strict=False)
    diff_model.to(device)
    diff_model.eval()

    return diff_model, checkpoint, trials_config


def load_dataset(dataset_dir, data_file_name):
    """
    Load the bone CT dataset from a specified directory.

    :param dataset_dir: Path to dataset directory containing samples.
    :param data_file_name: Name of the file within each sample folder (e.g. 'data.npz').
    :return: Initialized BoneDatasetCT object.
    """
    dataset = BoneDatasetCT(data_dir=dataset_dir, data_file_name=data_file_name)
    return dataset


def generate_latents_test(results_dir, diff_model, checkpoint, batch_size=1, cond_scale=1.0):
    """
    Generate samples from latent-space diffusion model and compare them to ground truth.

    :param results_dir: Directory containing latent datasets and output folder for results.
    :param diff_model: Trained ConditionalDiffusion model.
    :param checkpoint: Loaded training checkpoint dictionary.
    :param batch_size: Number of samples per batch. Default is 1.
    :param cond_scale: Guidance scale for conditional sampling. Default is 1.0.
    """
    with torch.no_grad():
        # Print training info
        print("Best epoch:", checkpoint['best_epoch'])
        print("Training time (s):", checkpoint["training_time"])
        plot_train_valid_loss(checkpoint['train_loss'], checkpoint['valid_loss'])

        generated_samples = []

        # Locate latent dataset directories
        latents_datasets_dir = os.path.join(results_dir, "latents_datasets")
        train_dataset_path = os.path.join(latents_datasets_dir, "latents_dataset_train")
        valid_dataset_path = os.path.join(latents_datasets_dir, "latents_dataset_validation")
        test_dataset_path = os.path.join(latents_datasets_dir, "latents_dataset_test")

        # Initialize datasets and loaders
        train_dataset = LatentsDataset(train_dataset_path)
        valid_dataset = LatentsDataset(valid_dataset_path)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
        validation_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)

        reconstructions = []

        # ------------------------------------
        # Iterate through training latent data
        # ------------------------------------
        for i, (volumes, conds) in enumerate(train_loader):
            print("volumes min: {}, max: {}".format(volumes.min(), volumes.max()))
            volumes = volumes.to(device).float()
            conds = conds.to(device).float()

            # Forward pass returns (true_noise, predicted_noise)
            true_noise, pred_noise = diff_model(volumes, conds)
            mse = F.mse_loss(pred_noise, true_noise, reduction="mean").item()
            print("MSE ", mse)

            # --- Ground truth ---
            gt = volumes.cpu().numpy()
            print(f"[{i}] Ground truth shape: {gt.shape}")

            # --- Conditional sampling ---
            torch.manual_seed(42)
            samples = diff_model.sample(
                batch_size=volumes.shape[0],
                image_size=volumes.shape[1:],
                cond=conds,
                cond_scale=cond_scale,
            )

            # --- Unconditional sampling ---
            torch.manual_seed(42)
            samples_no_cond = diff_model.sample(
                batch_size=volumes.shape[0],
                image_size=volumes.shape[1:],
            )

            # Compute difference between conditional and unconditional generations
            mse_diff = torch.mean((samples - samples_no_cond) ** 2).item()
            assert mse_diff > 0
            print(f"MSE between cond and uncond (same noise): {mse_diff:.6f}")

            # Clamp to valid intensity range [-1, 1]
            samples = torch.clamp(samples, -1, 1)

            generated_samples.append(samples)
            reconstructions.append(gt)

            print("samples min: {}, max: {}".format(samples.min(), samples.max()))

            # --- Convert to numpy for visualization ---
            samples = np.squeeze(samples.cpu().numpy())
            volumes = np.squeeze(volumes.cpu().numpy())

            # Plot histograms and 3D renderings
            plot_hist(volumes)
            plot_hist(samples)

            for channel in range(samples.shape[1]):
                render_3d_scan(volumes[channel], title=f"Training sample, channel {channel}")
                render_3d_scan(samples[channel], title=f"Generated sample, channel {channel}")


def sample(diff_model, checkpoint, trials_config, n_samples=10, batch_size=1, cond=None, cond_scale=1.0):
    """
    Generate new samples from a trained ConditionalDiffusion model.

    :param diff_model: Trained ConditionalDiffusion model.
    :param checkpoint: Loaded training checkpoint dictionary.
    :param trials_config: Loaded diffusion training configuration (YAML dict).
    :param n_samples: Number of samples to generate. Default is 10.
    :param batch_size: Number of samples per batch. Default is 1.
    :param cond: Conditioning tensor, or None for unconditional sampling.
    :param cond_scale: Classifier-free guidance scale. Default is 1.0.
    :return: List of generated torch.Tensor samples.
    """
    with torch.no_grad():
        print("Best epoch:", checkpoint['best_epoch'])
        print("Training time (s):", checkpoint["training_time"])
        plot_train_valid_loss(checkpoint['train_loss'], checkpoint['valid_loss'])

        print("trials config ", trials_config)

        generated_samples = []
        for i in range(n_samples):
            # If no conditioning provided, sample unconditionally
            if cond is None:
                cond = torch.zeros(
                    (batch_size, trials_config["cond_dim"]),
                    device=next(diff_model.parameters()).device
                )
            else:
                # Create conditioning vector for males (sex=0) with random ages
                cond = torch.zeros((batch_size, 2), device=device)
                cond[:, 1] = torch.rand(batch_size, device=device)

            samples = diff_model.sample(
                batch_size=batch_size,
                image_size=trials_config["image_size"],
                cond=cond,
                cond_scale=cond_scale,
            )

            samples = samples.cpu()
            print("samples min: {}, max: {}".format(np.min(samples.numpy()), np.max(samples.numpy())))

            generated_samples.append(samples)

            # --- Visualization ---
            np_sample = np.squeeze(samples.numpy())
            render_3d_scan(np_sample[0], title=f"Sample_ch_0", fig_name=f"sample_ch_0.png")

        return generated_samples


if __name__ == "__main__":
    """
    Main entry point for diffusion post-processing script.
    Loads configuration, model, dataset, and performs sampling and reconstruction tests.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help='Path to Denoising diffusion postprocess YAML config')
    args = parser.parse_args(sys.argv[1:])

    # Load post-processing configuration
    with open(args.config_path, "r") as f:
        postprocess_config = yaml.load(f, Loader=yaml.FullLoader)

    # Load dataset and model
    dataset = load_dataset(
        postprocess_config["dataset_dir"],
        postprocess_config["dataset_data_file_name"]
    )

    diffusion_model, checkpoint, diffusion_trials_config = load_model(
        postprocess_config["model_path"],
        postprocess_config["diffusion_train_config"]
    )

    # Evaluate model on latent dataset and visualize generated samples
    generate_latents_test(postprocess_config["results_dir"], diffusion_model, checkpoint)

    # Generate standalone diffusion samples
    sample(diffusion_model, checkpoint, diffusion_trials_config)
