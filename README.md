
# OsteoDiffusion

A VQGAN + Latent Diffusion framework for generating 3D pelvic bone structures.
---


## 🔧 Features

This repository consists of three components:

1. **VQGAN training**  
   - Trains a Vector-Quantized GAN (VQGAN) to learn compact latent representations of pelvic bone CT data.
   - The encoder compresses 3D volumes into a latent space, and the decoder reconstructs them back with minimal loss.
   - The trained VQGAN is later used as the backbone for diffusion training on latents.

2. **Denoising diffusion training**  
   - Trains a Denoising Diffusion Probabilistic Model (DDPM) on the learned latent space from VQGAN.
   - By operating in latent space instead of voxel space, training becomes computationally efficient while preserving structural detail.
   - The model learns to generate realistic latent codes that correspond to plausible pelvic bone structures.
   
3. **Sampling**  
   - Uses the trained diffusion model to generate synthetic latent codes.
   - These latents are decoded through the VQGAN decoder to reconstruct 3D pelvic bone volumes.

Each part can be run independently, using the provided data.

---



## 🛠 Installation & Requirements

- Developed and tested using **Python 3.10** with libraries listed in dependency file: [`requirements.txt`](requirements.txt)   

#### Set up Python environment:

```bash
cd OsteoDiffusion
export PYTHONPATH=.
```

---

## 📦 Dataset Generation

[BoneDat](https://www.nature.com/articles/s41597-025-05161-y) dataset of pelvic bones CT scans is adopted. The complete dataset can be found at [https://zenodo.org/records/15189761](https://zenodo.org/records/15189761)

In particular "masked.nii.gz" files in its /derived/segmentation directories are preprocessed to form the dataset used for VQGAN training.

Before training VQGAN, raw CT scans need to be clipped, normalized, and resampled into a consistent voxel grid.
We provide a preprocessing script to generate a training-ready dataset.

```bash
python dataset/form_dataset.py databse dataset_dir
```
- `database`: Path to the BoneDat dataset (must contain `derived/segmentation/*/masked.nii.gz` and `raw/*/metadata.xlsx`)
- `dataset_dir`: Path where the processed dataset will be stored.

For test purposes there are few samples in `data/bones_dataset_subset`


## 🧠 VQGAN Training

To train vqgan run:

```bash
python models/vqgan/train_model_vqgan.py configuration dataset_dir results_dir -c --mlflow
```

- `configuration` (e.g. [`configs/vqgan/test_vqgan_config.yaml`](configs/vqgan/test_vqgan_config.yaml))
- `data_dir`: Path to the dataset (e.g., `data/bones_dataset_subset` - small dataset (38 samples))  
- `results_dir`: Where results and logs will be saved
- `-c`: Use GPU (CUDA or AMD ROCm) if available
- `--mlflow`: Use MLFlow monitor


To postprocess trained vqgan (loss plots, reconstructions, etc.) run:

```bash
python postprocess/postprocess_vqgan_results.py configuration
```

- `configuration` (e.g. [`configs/vqgan/vqgan_postprocess_config.yaml`](configs/vqgan/vqgan_postprocess_config.yaml))



## Denoising Diffusion Training

To train denoising diffusion model on latents run:

```bash
python models/denoising_diffusion_latents/train_denoising_diffusion.py configuration data_dir results_dir -c --mlfloww
```
- `configuration` (e.g. [`configs/denoising_diffusion_latents/test_diffusion_config.yaml`](configs/denoising_diffusion_latents/test_diffusion_config.yaml))
- `data_dir`: Path to the dataset (e.g., `data/bones_dataset_subset` - small dataset (38 samples))
- `results_dir`: Where results and logs will be saved
- `-c`: Use GPU (CUDA or AMD ROCm) if available
- `--mlflow`: Use MLFlow monitor

To postprocess trained denoising diffusion model (loss plots, reconstructions, etc.) run:

```bash
python postprocess/postprocess_diffusion_latents_results.py configuration
```

- `configuration` (e.g. [`configs/denoising_diffusion_latents/diffusion_postprocess_config.yaml`](configs/denoising_diffusion_latents/diffusion_postprocess_config.yaml))

## Sampling
