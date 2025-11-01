import os
import sys
import argparse
import torch
import shutil
import numpy as np
import yaml
from dataset.cnn_diffusion.bone_dataset_CT import BoneDatasetCT
from models.denoising_diffusion_latents.Unet3D import UNet3D
from models.denoising_diffusion_latents.conditional_denoising_diffusion import ConditionalDiffusion
from models.vqgan.vqgan_model import VQGAN
from models.schedulers import NoiseScheduler
from models.auxiliary_functions import inverse_latent_transform
from visualization.visualize_data import plot_hist, render_3d_scan

# Select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trials_config(path_to_config):
    """
    Method that loads experiment configuration from a YAML file.

    :param path_to_config: Path to the YAML configuration file.
    :type path_to_config: str
    :return: Dictionary containing configuration parameters.
    :rtype: dict
    """
    with open(path_to_config, "r") as f:
        trials_config = yaml.load(f, Loader=yaml.FullLoader)
    return trials_config


def load_diffusion_model(model_checkpoint_path, diffusion_train_config):
    """
    Method that loads a trained Conditional Diffusion model and its configuration.

    :param model_checkpoint_path: Path to the trained diffusion model checkpoint.
    :type model_checkpoint_path: str
    :param diffusion_train_config: Path to the diffusion training configuration YAML file.
    :type diffusion_train_config: str
    :return: Tuple containing (diffusion_model, checkpoint_dict, trials_config_dict).
    :rtype: Tuple[torch.nn.Module, dict, dict]
    """
    trials_config = load_trials_config(diffusion_train_config)

    # Select model architecture
    if trials_config["unet_class_name"] == "Unet3D":
        model_class = UNet3D
    else:
        raise NotImplementedError("Only UNet3D model is supported.")

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


def load_vqgan_model(vqgan_model_path, diff_model_trials_config):
    """
    Method that loads a pretrained VQ-GAN model based on diffusion model configuration.

    :param vqgan_model_path: Path to the VQ-GAN checkpoint.
    :type vqgan_model_path: str
    :param diff_model_trials_config: Dictionary containing the diffusion model's configuration.
    :type diff_model_trials_config: dict
    :return: Loaded VQ-GAN model set to evaluation mode.
    :rtype: torch.nn.Module
    """
    vqgan_model_config = load_trials_config(diff_model_trials_config["vqgan_trials_config"])["model_config"][0]
    vqgan_model_checkpoint = VQGAN.load_from_checkpoint(vqgan_model_path, **vqgan_model_config)
    vqgan_model_checkpoint.eval()
    vqgan_model_checkpoint.to(device)
    return vqgan_model_checkpoint


def load_dataset(data_dir, data_file_name):
    """
    Method that loads the CT bone dataset.

    :param data_dir: Directory containing sample data.
    :type data_dir: str
    :param data_file_name: Name of the .npz file with CT data.
    :type data_file_name: str
    :return: Instance of BoneDatasetCT.
    :rtype: BoneDatasetCT
    """
    dataset = BoneDatasetCT(data_dir=data_dir, data_file_name=data_file_name)
    return dataset


def generate_samples(latent_diffusion_model, vqgan_model, trials_config, cond=None, cond_scale=1.0,
                     results_dir=None, clamp_diffusion_samples=False):
    """
    Method that generates 3D samples using a pretrained latent diffusion model and VQ-GAN decoder.

    This function performs the following steps:
      1. Samples latent tensors from the diffusion model (optionally conditioned).
      2. Optionally clamps generated samples to [-1, 1].
      3. Converts latent samples back to spatial domain via inverse VQ-GAN transformation.
      4. Decodes quantized features using the VQ-GAN decoder.
      5. Saves and visualizes results in both raw and inverse-transformed form.

    :param latent_diffusion_model: Trained latent diffusion model.
    :type latent_diffusion_model: torch.nn.Module
    :param vqgan_model: Trained VQ-GAN model for decoding.
    :type vqgan_model: torch.nn.Module
    :param trials_config: Diffusion model training configuration.
    :type trials_config: dict
    :param cond: Conditioning tensor for guided generation (optional).
    :type cond: torch.Tensor, optional
    :param cond_scale: Scale factor for conditional guidance.
    :type cond_scale: float
    :param results_dir: Directory to store generated samples and visualizations.
    :type results_dir: str, optional
    :param clamp_diffusion_samples: Whether to clamp latent samples to [-1, 1].
    :type clamp_diffusion_samples: bool
    :return: None
    """
    batch_size_sample = 1
    n_samples = 50
    generated_samples = []

    torch.manual_seed(333)

    with torch.no_grad():
        for i in range(n_samples):
            # Prepare result directory for this sample
            if results_dir is not None:
                sample_dir = os.path.join(results_dir, f"sample_{i}")
                if os.path.exists(sample_dir):
                    shutil.rmtree(sample_dir)
                os.mkdir(sample_dir)
                os.chdir(sample_dir)

            # If no conditioning provided, use zeros (unconditional generation)
            if cond is None:
                cond = torch.zeros(
                    (batch_size_sample, trials_config["cond_dim"]),
                    device=next(latent_diffusion_model.parameters()).device
                )

            # Generate latent samples
            samples = latent_diffusion_model.sample(
                batch_size=batch_size_sample,
                image_size=trials_config["image_size"],
                cond=cond,
                cond_scale=cond_scale,
            )

            if clamp_diffusion_samples:
                samples = torch.clamp(samples, -1, 1)

            print(f"Sample {i}: min={samples.min():.4f}, max={samples.max():.4f}")
            plot_hist(samples.cpu().numpy(), title="Generated Latents")

            generated_samples.append(samples.cpu())

            # Map latent samples back through VQ-GAN inverse transform
            samples_to_vqgan = inverse_latent_transform(samples, vqgan_model).to(dtype=torch.float32)
            print(f"Sample {i} (post-inverse): min={samples_to_vqgan.min():.4f}, max={samples_to_vqgan.max():.4f}")

            # Decode samples
            if not ("train_on_codebooks" in trials_config and trials_config["train_on_codebooks"]):
                quant, emb_loss, info = vqgan_model.quantize(samples_to_vqgan)
                decoded_samples = vqgan_model.decode(quant)

                print(f"Decoded sample {i}: min={decoded_samples.min():.4f}, max={decoded_samples.max():.4f}")

                # Save and visualize
                np.save(os.path.join(sample_dir, "decoded_samples"), np.squeeze(decoded_samples.cpu().numpy()))
                render_3d_scan(np.squeeze(decoded_samples.cpu().numpy()),
                               title="Generated sample",
                               fig_name=os.path.join(sample_dir, "gen_sample.png"))

                # Apply threshold and convert to HU scale
                threshold = -0.97
                decoded_samples = np.squeeze(decoded_samples.cpu().numpy())
                decoded_samples[decoded_samples < threshold] = -1

                min_value, max_value = -1024, 1650
                inv_decoded_samples = (decoded_samples + 1) / 2
                inv_decoded_samples = inv_decoded_samples * (max_value - min_value) + min_value

                render_3d_scan(np.squeeze(inv_decoded_samples),
                               title="Generated sample (inverse transform)",
                               fig_name=os.path.join(sample_dir, "gen_sample_inv.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Bone Sample Generation Pipeline")
    parser.add_argument('sampling_config_path', help='Path to YAML configuration file for sampling.')
    parser.add_argument('results_dir', help='Directory to store generated samples.')
    args = parser.parse_args(sys.argv[1:])

    # Load sampling configuration
    with open(args.sampling_config_path, "r") as f:
        sampling_config = yaml.load(f, Loader=yaml.FullLoader)

    # Load diffusion model and configuration
    latent_diffusion_model, checkpoint, diffusion_trials_config = load_diffusion_model(
        sampling_config["denoising_diffusion_model_path"],
        sampling_config["denoising_diffusion_train_config"]
    )

    # Load VQ-GAN decoder
    vqgan_model_path = sampling_config.get("vqgan_model_path", None)
    vqgan_model = load_vqgan_model(vqgan_model_path=vqgan_model_path,
                                   diff_model_trials_config=diffusion_trials_config)

    # Generate samples
    generate_samples(
        latent_diffusion_model,
        vqgan_model,
        diffusion_trials_config,
        results_dir=args.results_dir,
        clamp_diffusion_samples=sampling_config["clamp_diffusion_samples"]
    )
