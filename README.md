# OsteoDiffusion

A **VQGAN + Latent Diffusion** framework for generating realistic 3D pelvic bone structures.

This project is partly inspired by the following repositories and prior work:
- [medicaldiffusion](https://github.com/FirasGit/medicaldiffusion)
- [ALDM](https://github.com/jongdory/ALDM)
- Project structure and code organization are inspired by [MLMC-DFM](https://github.com/martinspetlik/MLMC-DFM)

---

## 🔧 Features

This repository is composed of three main components:

1. **🧠 VQGAN Training**  
   - Trains a **Vector-Quantized GAN (VQGAN)** to learn compact latent representations of pelvic bone CT data.  
   - The encoder compresses 3D volumes into a latent space (codebooks), while the decoder reconstructs them with minimal information loss.  
   - The trained VQGAN serves as the backbone for diffusion training on latent representations.

2. **🌫️ Denoising Diffusion Training**  
   - Trains a **Denoising Diffusion Probabilistic Model (DDPM)** in the learned latent space.  
   - Operating in latent space (instead of voxel space) reduces computational cost while preserving anatomical structure.  
   - The diffusion model learns to generate realistic latent codes.

3. **🎨 Generating Sampling**  
   - Uses the trained diffusion model to sample new latent codes.  
   - The VQGAN decoder then reconstructs them into full 3D pelvic bone volumes.

Each module can be executed independently using the provided datasets and configuration files.

---

## 🛠 Installation & Requirements

- Developed and tested with **Python 3.12**
- Dependencies are listed in [`requirements.txt`](requirements.txt)
- For full **3D visualization** during postprocessing, install **[Mayavi](https://docs.enthought.com/mayavi/mayavi/)**:
  ```bash
  pip install mayavi
  ```

### Set up the Python environment

```bash
cd OsteoDiffusion
export PYTHONPATH=.
```
---

## 📦 Dataset Generation

[BoneDat](https://www.nature.com/articles/s41597-025-05161-y) dataset of pelvic bones CT scans is adopted. The complete dataset can be found at [https://zenodo.org/records/15189761](https://zenodo.org/records/15189761)

Specifically, the "masked.nii.gz" files from the /derived/segmentation directories are preprocessed to form the dataset for VQGAN training.

Before training, raw CT scans are clipped, normalized, and resampled into a consistent voxel grid.
We provide a preprocessing script to generate a training-ready dataset:

```bash
python dataset/form_dataset.py database dataset_dir
```
**Arguments:**
- `database`: Path to the BoneDat dataset (must contain `derived/segmentation/*/masked.nii.gz` and `raw/*/metadata.xlsx`)
- `dataset_dir`: Directory where the processed dataset will be saved.

> For quick testing, a small subset is available in: `data/bones_dataset_subset`


## 🧠 VQGAN Training

Train the VQGAN model using:

```bash
python models/vqgan/train_model_vqgan.py configuration dataset_dir results_dir -c --mlflow
```
**Arguments:**
- `configuration` (e.g. [`configs/vqgan/test_vqgan_config.yaml`](configs/vqgan/test_vqgan_config.yaml))
- `data_dir`: Path to the dataset (e.g., `data/bones_dataset_subset` - small dataset (38 samples))  
- `results_dir`: Directory for saving training results and logs  
- `-c`: Use GPU (CUDA or AMD ROCm) if available
- `--mlflow`: Use MLFlow monitor. **Do not include this flag if you want to run without MLflow.**

### Postprocessing

To visualize training curves, reconstructions, and metrics:

```bash
python postprocess/postprocess_vqgan_results.py configuration
```
- `configuration` (e.g. [`configs/vqgan/vqgan_postprocess_config.yaml`](configs/vqgan/vqgan_postprocess_config.yaml))



## 🌫️ Denoising Diffusion Training

Train the latent-space diffusion model:

```bash
python models/denoising_diffusion_latents/train_denoising_diffusion.py configuration data_dir results_dir -c --mlflow
```
**Arguments:**
- `configuration` (e.g. [`configs/denoising_diffusion_latents/test_diffusion_config.yaml`](configs/denoising_diffusion_latents/test_diffusion_config.yaml))
- `data_dir`: Path to the dataset (e.g., `data/bones_dataset_subset` - small dataset (38 samples))
- `results_dir`: Where results and logs will be saved
- `-c`: Use GPU (CUDA or AMD ROCm) if available
- `--mlflow`: Use MLFlow monitor. **Do not include this flag if you want to run without MLflow.**

### Postprocessing
To analyze and visualize diffusion model results:

```bash
python postprocess/postprocess_diffusion_latents_results.py configuration
```
- `configuration` (e.g. [`configs/denoising_diffusion_latents/diffusion_postprocess_config.yaml`](configs/denoising_diffusion_latents/diffusion_postprocess_config.yaml))


## 🎨 Generating Samples

To generate new 3D pelvic bone structures using the trained VQGAN and diffusion model:

```bash
python postprocess/generate_samples.py configuration results_dir
```
**Arguments:**
- `configuration`: e.g. [`configs/test_sampling_config.yaml`](configs/test_sampling_config.yaml)
- `results_dir`: Directory where generated samples will be saved as `.npy` files


✅ **Tip:**  
For reproducible results, make sure that:
- The same configuration files and random seeds are used during training and sampling.
- Model checkpoints (`.pt` or `.ckpt`) are correctly referenced in your configuration files.

