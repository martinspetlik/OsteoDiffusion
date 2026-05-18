import os
import sys
import argparse
import torch
import shutil
import numpy as np
import yaml
import joblib
from dataset.cnn_diffusion.bone_dataset_CT import BoneDatasetCT
from models.denoising_diffusion_latents.Unet3D import UNet3D
from models.denoising_diffusion_latents.conditional_denoising_diffusion import ConditionalDiffusion
from models.vqgan.vqgan_model import VQGAN
from models.schedulers import NoiseScheduler
from models.auxiliary_functions import inverse_latent_transform
from visualization.visualize_data import plot_hist, render_3d_scan
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import torch.nn.functional as F


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


def load_study(results_dir):
    # Load the Optuna study object from a pickle file in the results directory
    study = joblib.load(os.path.join(results_dir, "study.pkl"))

    # Print summary info about the best trial found so far
    print("Best trial until now:")
    print(" Value: ", study.best_trial.value)
    print(" Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    # Return the loaded study object
    return study


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


# -----------------------------------------------------------------------
# Step 1: Extract latents + labels from the full dataset
# -----------------------------------------------------------------------

def extract_latents(vqgan_model, data_dir, data_file_name, batch_size=1):
    """
    Encode all CT scans through the VQGAN encoder and collect:
      - mean-pooled latent vectors  (B, latent_dim)
      - age values                  (B,)
      - sex values                  (B,)

    Mean-pooling over spatial dims (D, H, W) gives one vector per scan,
    which is what the linear probe operates on.
    """
    dataset = BoneDatasetCT(data_dir=data_dir, data_file_name=data_file_name)
    print("data_dir ", data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_latents_pooled, all_latents_pooled_4, all_latents_pooled_8, all_ages, all_sexes = [], [], [], [], []

    print(f"Encoding {len(dataset)} scans through VQGAN...")

    with torch.no_grad():
        for batch_idx, (x, cond) in enumerate(loader):
            x = x.to(device).float()
            print('x.shape ', x.shape)
            sex = cond[0].cpu().numpy()  # (B,)
            age = cond[1].cpu().numpy()  # (B,)  normalized age

            # Encode: returns (quant, emb_loss, info)
            #quant, _, _ = vqgan_model.encode(x)  # (B, C, D, H, W)

            h = vqgan_model.encoder(x)
            h = vqgan_model.quant_conv(h)

            z_pooled = h.mean(dim=[2, 3, 4])
            #z_flat = h.flatten(start_dim=1)

            z_pool_4 = F.adaptive_avg_pool3d(h, (4, 4, 4))
            z_pool_8 = F.adaptive_avg_pool3d(h, (8, 8, 8))

            z_pool_4 = z_pool_4.flatten(start_dim=1)
            z_pool_8 = z_pool_8.flatten(start_dim=1)

            # # Mean-pool spatial dims → one vector per scan
            # z_pooled = quant.mean(dim=[2, 3, 4])  # (B, latent_dim)

            all_latents_pooled.append(z_pooled.cpu().numpy())
            all_latents_pooled_4.append(z_pool_4.cpu().numpy())
            all_latents_pooled_8.append(z_pool_8.cpu().numpy())
            all_ages.append(age)
            all_sexes.append(sex)

            print(f"  Batch {batch_idx + 1}/{len(loader)} — "
                  f"latent shape: {z_pooled.shape}, "
                  f"ages: {np.round(age * 100).astype(int).tolist()}")

    latents_pooled = np.concatenate(all_latents_pooled, axis=0)  # (N, latent_dim)
    latents_pooled_4 = np.concatenate(all_latents_pooled_4, axis=0)
    latents_pooled_8 = np.concatenate(all_latents_pooled_8, axis=0)
    #latents = np.concatenate(all_latents, axis=0)  # (N, latent_dim)
    ages = np.concatenate(all_ages, axis=0)  # (N,)
    sexes = np.concatenate(all_sexes, axis=0)  # (N,)

    print(f"\nExtracted latents: {h.shape}")
    print(f"Age  — min: {ages.min() * 100:.1f}, max: {ages.max() * 100:.1f}, "
          f"mean: {ages.mean() * 100:.1f}, std: {ages.std() * 100:.1f}")
    print(f"Sex  — female: {(sexes == 1.0).sum()}, male: {(sexes == 0.0).sum()}")

    return latents_pooled, latents_pooled_4, latents_pooled_8, ages, sexes


# -----------------------------------------------------------------------
# Step 2: Run the linear probe
# -----------------------------------------------------------------------

def run_linear_probe(latents, ages, sexes, n_splits=5):
    """
    Probe whether age and sex are linearly decodable from latent vectors.

    Age  → Ridge regression   → R² score
    Sex  → Logistic regression → accuracy

    Both use cross-validation with StandardScaler inside the pipeline
    so the probe itself doesn't overfit to scale.

    Interpretation:
      Age R²  > 0.70  → age well preserved in latents
      Age R²  < 0.30  → age largely destroyed → must fix VQGAN first
      Sex Acc > 0.75  → sex preserved
      Sex Acc ~ 0.50  → sex destroyed (random chance baseline)
    """
    n = len(latents)
    print(f"\n{'=' * 60}")
    print(f"LINEAR PROBE  ({n} samples, {n_splits}-fold CV)")
    print(f"{'=' * 60}")

    # --- Age probe (regression) ---
    age_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])

    # With small N, use KFold and shuffle
    kf = KFold(n_splits=min(n_splits, n // 5), shuffle=True, random_state=42)
    age_r2_scores = cross_val_score(age_pipeline, latents, ages, cv=kf, scoring="r2")

    print(f"\nAge (regression — Ridge):")
    print(f"  R² per fold : {np.round(age_r2_scores, 3).tolist()}")
    print(f"  Mean R²     : {age_r2_scores.mean():.3f}  ±  {age_r2_scores.std():.3f}")

    # --- Sex probe (classification) ---
    sex_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, C=1.0))
    ])

    # Stratify by sex to keep class balance across folds
    skf = StratifiedKFold(n_splits=min(n_splits, n // 5), shuffle=True, random_state=42)
    sex_acc_scores = cross_val_score(sex_pipeline, latents, sexes, cv=skf, scoring="accuracy")

    print(f"\nSex (classification — Logistic Regression):")
    print(f"  Accuracy per fold : {np.round(sex_acc_scores, 3).tolist()}")
    print(f"  Mean accuracy     : {sex_acc_scores.mean():.3f}  ±  {sex_acc_scores.std():.3f}")

    # --- Interpretation ---
    age_r2 = age_r2_scores.mean()
    sex_acc = sex_acc_scores.mean()

    print(f"\n{'=' * 60}")
    print("INTERPRETATION")
    print(f"{'=' * 60}")

    # Age
    if age_r2 > 0.70:
        age_verdict = "PRESERVED  ✓  Fix diffusion conditioning only"
    elif age_r2 > 0.30:
        age_verdict = "PARTIAL    ~  Fix diffusion first, re-evaluate"
    else:
        age_verdict = "DESTROYED  ✗  Must fix VQGAN before diffusion"
    print(f"  Age R² = {age_r2:.3f}  →  {age_verdict}")

    # Sex
    if sex_acc > 0.75:
        sex_verdict = "PRESERVED  ✓"
    elif sex_acc > 0.60:
        sex_verdict = "PARTIAL    ~"
    else:
        sex_verdict = "DESTROYED  ✗  (or class imbalance — check counts above)"
    print(f"  Sex Acc = {sex_acc:.3f}  →  {sex_verdict}")

    return age_r2, sex_acc


# -----------------------------------------------------------------------
# Step 3: HU preservation check (bonus — uses your existing diagnostic)
# -----------------------------------------------------------------------

# def run_hu_preservation_check(vqgan_model, data_dir, data_file_name, batch_size=1):
#     """
#     Checks whether the VQGAN roundtrip (encode → decode) preserves
#     mean HU values in bone regions.
#
#     Correlation > 0.85 → VQGAN is not the bottleneck for density.
#     """
#     dataset = BoneDatasetCT(data_dir=data_dir, data_file_name=data_file_name)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
#
#     hu_real_list, hu_recon_list, age_list = [], [], []
#
#     print(f"\n{'=' * 60}")
#     print("HU PRESERVATION CHECK  (VQGAN encode → decode roundtrip)")
#     print(f"{'=' * 60}")
#
#     with torch.no_grad():
#         for x, cond in loader:
#             x = x.to(device).float()
#             print("cond ", cond)
#             age = cond[1].cpu().numpy()
#
#             # Full roundtrip
#             quant, _, _ = vqgan_model.encode(x)
#             recon = vqgan_model.decode(quant)
#
#             # Bone mask: normalized HU > 0.3 corresponds roughly to HU > -300
#             bone_mask = x > 0.3  # (B, 1, D, H, W)
#
#             for i in range(x.shape[0]):
#                 mask = bone_mask[i, 0]
#                 if mask.sum() < 100:  # skip if almost no bone voxels
#                     continue
#                 hu_real_list.append(x[i, 0][mask].mean().item())
#                 hu_recon_list.append(recon[i, 0][mask].mean().item())
#                 age_list.append(age[i] * 100)
#
#     if len(hu_real_list) < 5:
#         print("  Not enough bone voxels found — check your HU threshold.")
#         return None
#
#     hu_real = np.array(hu_real_list)
#     hu_recon = np.array(hu_recon_list)
#     ages = np.array(age_list)
#
#     corr = np.corrcoef(hu_real, hu_recon)[0, 1]
#     mae = np.abs(hu_real - hu_recon).mean()
#
#     print(f"  Samples with bone voxels : {len(hu_real)}")
#     print(f"  Real  HU range : [{hu_real.min():.3f}, {hu_real.max():.3f}]")
#     print(f"  Recon HU range : [{hu_recon.min():.3f}, {hu_recon.max():.3f}]")
#     print(f"  Correlation    : {corr:.4f}  (> 0.85 = VQGAN is fine)")
#     print(f"  MAE            : {mae:.4f}")
#
#     # Also check if real HU correlates with age (sanity check on your data)
#     age_hu_corr = np.corrcoef(ages, hu_real)[0, 1]
#     print(f"\n  Age vs real HU correlation : {age_hu_corr:.4f}")
#     if abs(age_hu_corr) < 0.2:
#         print("  WARNING: Age barely correlates with HU in your raw scans.")
#         print("           This may indicate your dataset lacks age-related density variation")
#         print("           independent of the model — the problem may be in the data itself.")
#     else:
#         print("  Age-HU correlation looks reasonable — density signal exists in raw data.")
#
#     if corr > 0.85:
#         print("\n  → VQGAN preserves HU well. Problem is in diffusion conditioning.")
#     else:
#         print("\n  → VQGAN is losing HU information. Fix VQGAN before diffusion.")
#
#     print(f"{'=' * 60}\n")
#     return corr


def compute_bone_statistics(vox):

    stats = {}

    stats["mean"] = vox.mean().item()
    stats["std"] = vox.std().item()

    stats["median"] = torch.quantile(vox, 0.50).item()

    stats["p10"] = torch.quantile(vox, 0.10).item()
    stats["p25"] = torch.quantile(vox, 0.25).item()
    stats["p75"] = torch.quantile(vox, 0.75).item()
    stats["p90"] = torch.quantile(vox, 0.90).item()
    stats["p95"] = torch.quantile(vox, 0.95).item()

    stats["high_frac_040"] = (vox > 0.40).float().mean().item()
    stats["high_frac_045"] = (vox > 0.45).float().mean().item()
    stats["high_frac_050"] = (vox > 0.50).float().mean().item()

    return stats


def correlation_safe(a, b):

    a = np.asarray(a)
    b = np.asarray(b)

    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return np.nan

    return np.corrcoef(a, b)[0,1]


def run_hu_preservation_check(
    vqgan_model,
    data_dir,
    data_file_name,
    batch_size=1
):

    dataset = BoneDatasetCT(
        data_dir=data_dir,
        data_file_name=data_file_name
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    metric_names = [
        "mean",
        "std",
        "median",
        "p10",
        "p25",
        "p75",
        "p90",
        "p95",
        "high_frac_040",
        "high_frac_045",
        "high_frac_050"
    ]

    real_metrics = {
        k: [] for k in metric_names
    }

    recon_metrics = {
        k: [] for k in metric_names
    }

    ages = []
    sexes = []

    print("\n" + "=" * 80)
    print("IMPROVED HU PRESERVATION CHECK")
    print("=" * 80)

    with torch.no_grad():

        for batch_idx, (x, cond) in enumerate(loader):

            x = x.to(device).float()

            sex = cond[0].cpu().numpy()
            age = cond[1].cpu().numpy() * 100.0

            # -----------------------------------------
            # VQGAN roundtrip
            # -----------------------------------------

            quant, _, _ = vqgan_model.encode(x)
            recon = vqgan_model.decode(quant)

            # -----------------------------------------
            # Multiple bone thresholds
            # -----------------------------------------

            for threshold in [0.10, 0.20, 0.30]:

                bone_mask = x > threshold

                for i in range(x.shape[0]):

                    mask = bone_mask[i,0]

                    if mask.sum() < 100:
                        continue

                    vox_real = x[i,0][mask]
                    vox_recon = recon[i,0][mask]

                    real_stats = compute_bone_statistics(vox_real)
                    recon_stats = compute_bone_statistics(vox_recon)

                    for k in metric_names:
                        real_metrics[k].append(real_stats[k])
                        recon_metrics[k].append(recon_stats[k])

                    ages.append(age[i])
                    sexes.append(sex[i])

    ages = np.array(ages)
    sexes = np.array(sexes)

    print("\n")
    print("=" * 80)
    print("REAL vs RECON CORRELATIONS")
    print("=" * 80)

    for k in metric_names:

        real_vals = np.array(real_metrics[k])
        recon_vals = np.array(recon_metrics[k])

        corr = correlation_safe(real_vals, recon_vals)

        mae = np.mean(np.abs(real_vals - recon_vals))

        print(
            f"{k:20s} | "
            f"corr = {corr:7.4f} | "
            f"mae = {mae:7.4f}"
        )

    print("\n")
    print("=" * 80)
    print("AGE CORRELATIONS (REAL)")
    print("=" * 80)

    for k in metric_names:

        real_vals = np.array(real_metrics[k])

        corr = correlation_safe(ages, real_vals)

        print(
            f"{k:20s} | "
            f"age corr = {corr:7.4f}"
        )

    print("\n")
    print("=" * 80)
    print("AGE CORRELATIONS (RECON)")
    print("=" * 80)

    for k in metric_names:

        recon_vals = np.array(recon_metrics[k])

        corr = correlation_safe(ages, recon_vals)

        print(
            f"{k:20s} | "
            f"age corr = {corr:7.4f}"
        )

    print("\n")
    print("=" * 80)
    print("SEX DIFFERENCES")
    print("=" * 80)

    female_mask = sexes == 1.0
    male_mask = sexes == 0.0

    for k in metric_names:

        real_vals = np.array(real_metrics[k])

        female_mean = real_vals[female_mask].mean()
        male_mean = real_vals[male_mask].mean()

        diff = male_mean - female_mean

        print(
            f"{k:20s} | "
            f"male-female diff = {diff:7.4f}"
        )

    print("\n")
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    print("""
Good signs:
- real/recon corr > 0.8
- recon preserves age correlations
- recon preserves sex differences
- p90/high-density fractions preserved

Bad signs:
- compressed std/p90
- collapsed high-density fractions
- weak age correlation after reconstruction
- dynamic-range shrinkage
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Bone Sample Generation Pipeline")
    parser.add_argument('sampling_config_path', help='Path to YAML configuration file for sampling.')
    parser.add_argument('results_dir', help='Directory to store generated samples.')
    args = parser.parse_args(sys.argv[1:])

    # Load sampling configuration
    with open(args.sampling_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Load diffusion model and configuration
    latent_diffusion_model, checkpoint, diffusion_trials_config = load_diffusion_model(
        config["denoising_diffusion_model_path"],
        config["denoising_diffusion_train_config"]
    )
    latent_diffusion_study = load_study(config["denoising_diffusion_results_dir"])

    # Load VQ-GAN decoder
    vqgan_model_path = config.get("vqgan_model_path", None)
    vqgan_model = load_vqgan_model(vqgan_model_path=vqgan_model_path,
                                   diff_model_trials_config=diffusion_trials_config)

    # gen_params = {}
    # if "gen_params" in sampling_config:
    #     gen_params = sampling_config["gen_params"]

    # # Generate samples
    # generate_samples(
    #     latent_diffusion_model,
    #     vqgan_model,
    #     diffusion_trials_config,
    #     latent_diffusion_study,
    #     results_dir=args.results_dir,
    #     clamp_diffusion_samples=sampling_config["clamp_diffusion_samples"],
    #     **gen_params
    # )

    # Run HU preservation check first (quick sanity check)

    run_hu_preservation_check(
            vqgan_model,
            config["dataset_dir"],
            config["dataset_data_file_name"],
            #batch_size=args.batch_size
        )

    # Extract latents
    latents_pooled, latents_pooled_4, latents_pooled_8, ages, sexes = extract_latents(
        vqgan_model,
        config["dataset_dir"],
        config["dataset_data_file_name"],
        #batch_size=args.batch_size
    )

    print("#### Latents pooled 4 ####")

    # Run linear probe
    age_r2, sex_acc = run_linear_probe(
        latents_pooled_4, ages, sexes,
        #n_splits=args.cv_folds
    )

    print("#### Latents pooled 8 ####")

    # Run linear probe
    age_r2, sex_acc = run_linear_probe(
        latents_pooled_8, ages, sexes,
        # n_splits=args.cv_folds
    )


