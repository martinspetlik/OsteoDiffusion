import os
import torch
import numpy as np
from torch.utils.data import Subset
import torchvision.transforms as transforms
from tqdm import tqdm
from dataset.denoising_diffusion_latents.latents_dataset import LatentsDataset
from torch.utils.data import DataLoader


class MinMaxNormalize:
    def __init__(self, min_vals, max_vals, target_range=(-1, 1)):
        self.min_vals = min_vals  # Shape: [128]
        self.max_vals = max_vals
        self.target_min, self.target_max = target_range

    def __call__(self, tensor):
        # tensor: [128, 40, 24, 40]
        min_vals = self.min_vals.view(-1, 1, 1, 1).to(tensor.device)
        max_vals = self.max_vals.view(-1, 1, 1, 1).to(tensor.device)
        # Normalize to [target_min, target_max]
        tensor_norm = (tensor - min_vals) / (max_vals - min_vals + 1e-8)
        tensor_norm = tensor_norm * (self.target_max - self.target_min) + self.target_min
        return tensor_norm


def compute_data_min_max(train_loader, num_channels=128):
    # Initialize with infinities
    min_vals = torch.ones(num_channels, 1, 1, 1) * float('inf')
    max_vals = torch.ones(num_channels, 1, 1, 1) * -float('inf')
    for batch in train_loader:
        latents, conds = batch
        latents = np.squeeze(latents)
        batch_min = latents.amin(dim=(1, 2, 3), keepdim=True)  # shape: (channels,)
        batch_max = latents.amax(dim=(1, 2, 3), keepdim=True)  # shape: (channels,)

        # Update global min/max
        min_vals = torch.minimum(min_vals, batch_min)
        max_vals = torch.maximum(max_vals, batch_max)

    return min_vals, max_vals  # Shape: [128]


def latents_prepare_dataset(study, config, vqgan_train_val_test_datasets, vqgan_model_checkpoint, latents_datasets_dir):
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
    train_set, validation_set, test_set = vqgan_train_val_test_datasets

    latents_train_set = build_latents_dataset(vqgan_model_checkpoint, config, train_set, latents_datasets_dir,
                                              dset_type="train", device="cuda")
    latents_val_set = build_latents_dataset(vqgan_model_checkpoint, config, validation_set, latents_datasets_dir,
                                              dset_type="validation", device="cuda")
    latents_test_set = build_latents_dataset(vqgan_model_checkpoint, config, test_set, latents_datasets_dir,
                                              dset_type="test", device="cuda")

    latents_train_loader = torch.utils.data.DataLoader(latents_train_set, batch_size=1, shuffle=False)
    min_values, max_values = compute_data_min_max(latents_train_loader, num_channels=latents_train_set[0][0].shape[0])

    # Define transform pipeline
    transform = transforms.Compose([
        MinMaxNormalize(min_values, max_values, target_range=(-1, 1)),
    ])

    latents_train_set.transform = transform
    latents_val_set.transform = transform
    latents_test_set.transform = transform

    # Store dataset attributes in study object
    study.set_user_attr("global_min_value", min_values)
    study.set_user_attr("global_max_value", max_values)

    return latents_train_set, latents_val_set, latents_test_set


def build_latents_dataset(vqgan, config, dataset, save_dir, dset_type="train", device="cuda"):
    """
    Build a latent-space dataset using a trained VQGAN encoder.
    Each sample is saved into a separate .npz file for efficient streaming.

    :param vqgan: Pre-trained VQGAN model (with encoder + quantizer).
    :param config: trial configuration
    :param dataset: Original dataset of 3D CT volumes + conditions.
    :param save_dir: Directory where latent datasets will be stored.
    :param dset_type: Dataset split type ("train", "validation", "test").
    :param device: Device to run computations on ("cuda" or "cpu").
    """
    os.makedirs(save_dir, exist_ok=True)

    # Directory for this dataset split
    split_dir = os.path.join(save_dir, f"latents_dataset_{dset_type}")

    # Create LatentsDataset from existing dataset directory
    if os.path.exists(split_dir):
        LatentsDataset(split_dir)

    os.makedirs(split_dir, exist_ok=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    vqgan.eval()
    vqgan.to(device)

    counter = 0
    with torch.no_grad():
        for i, (volume, conds) in enumerate(tqdm(loader)):
            volume = volume.to(device).float()

            if "train_on_codebooks" in config and config["train_on_codebooks"]:
                # --- ALDM approach ---
                h = vqgan.encoder(volume)
                h = vqgan.quant_conv(h)
                vqgan.quantize.sane_index_shape = True
                z_q, loss, (_, _, codebook_indices) = vqgan.quantize(h)
                #model_input = z_q / z_q.flatten().std()
                model_input = z_q

            else:
                # --- MedicalDiffusion approach ---
                h = vqgan.encoder(volume)
                h = vqgan.quant_conv(h)

                model_input = h

                # model_input = forward_latent_transform(h, vqgan)
                #
                # if not os.path.exists(os.path.join(save_dir, "quantize_embedings_min_max.npy")):
                #     np.save(
                #         os.path.join(save_dir, "quantize_embedings_min_max.npy"),
                #         (
                #             vqgan.quantize.embedding.weight.min().cpu().numpy(),
                #             vqgan.quantize.embedding.weight.max().cpu().numpy()
                #         )
                #     )

            model_input_np = np.squeeze(model_input.cpu().numpy())
            conds_np = torch.stack(conds, dim=1).cpu().numpy() if isinstance(conds, (list, tuple)) else conds.cpu().numpy()
            conds_np = np.squeeze(conds_np)

            # Save one sample per file
            sample_path = os.path.join(split_dir, f"sample_{counter:04d}.npz")
            np.savez_compressed(sample_path, latent=model_input_np, cond=conds_np)
            counter += 1

    print(f" Saved {counter} samples into {split_dir}")

    return LatentsDataset(split_dir)
