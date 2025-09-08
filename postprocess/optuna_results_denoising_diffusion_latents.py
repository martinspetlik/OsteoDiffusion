import os
import sys
import argparse
import joblib
import torch
import yaml
import numpy as np
import torch.nn.functional as F
from dataset.cnn_diffusion.bone_dataset_CT import BoneDatasetCT
from visualization.visualize_data import plot_train_valid_loss


from models.denoising_diffusion_latents.Unet3D import UNet3D
from models.denoising_diffusion_latents.conditional_denoising_diffusion import ConditionalDiffusion
from dataset.denoising_diffusion_latents.latents_dataset import LatentsDataset
from models.schedulers import NoiseScheduler
import matplotlib.pyplot as plt
import scipy as sc
import pyvista as pv

#os.environ["CUDA_VISIBLE_DEVICES"]=""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trials_config(path_to_config):
    with open(path_to_config, "r") as f:
        trials_config = yaml.load(f, Loader=yaml.FullLoader)
    return trials_config


def load_diffusion_model(results_dir, model_path=None, device="cuda"):
    optuna_study = load_study(results_dir)
    trials_config = load_trials_config(os.path.join(results_dir, "trials_config.yaml"))

    print("optuna_study.best_trial.user_attrs ", optuna_study.best_trial.user_attrs)

    # Use explicit model_path if provided, otherwise construct from study
    if model_path is None:
        model_path = os.path.join(
            results_dir,
            f"trial_{optuna_study.best_trial.number}_model_best.pt"
        )
    print("Loading model checkpoint from:", model_path)

    # Pick model class
    if trials_config["unet_class_name"] == "Unet3D":
        model_class = UNet3D
    else:
        raise NotImplementedError("Only UNet3D model is supported")

    # Init model + noise scheduler
    print("unet config ", trials_config["unet_config"][0])
    unet_model = model_class(**trials_config["unet_config"][0])
    noise_scheduler = NoiseScheduler(**trials_config["noise_scheduler_params"])
    noise_scheduler.num_gen_timesteps = 50

    diff_model = ConditionalDiffusion(
        cnn_model=unet_model,
        image_size=trials_config["image_size"],
        noise_scheduler=noise_scheduler
    )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    diff_model.load_state_dict(checkpoint['best_model_state_dict'], strict=False)
    diff_model.to(device)
    diff_model.eval()

    return diff_model, checkpoint, trials_config



def load_dataset(results_dir, study):
    data_dir = study.user_attrs["data_dir"]
    dataset = BoneDatasetCT(data_dir=data_dir, data_file_name="lumbopelvic_masked_normed_local_resampled_32_32_32.npz")
    return dataset



def render_3d_scan(scan, title="3D Scan", fig_name=""):
    from mayavi import mlab
    # Create a Mayavi figure
    mlab.figure(size=(800, 800), bgcolor=(1, 1, 1))

    # Create a 3D volume visualization
    src = mlab.pipeline.scalar_field(scan)
    #mlab.pipeline.volume(src, vmin=scan.min(), vmax=scan.max())
    volume = mlab.pipeline.volume(src)
    volume._volume_property.scalar_opacity_unit_distance = 0.1
    #mlab.title(f'3D Feature Map from Layer: {layer}, Feature: {feature_index}')
    colorbar = mlab.colorbar(title="Intensity", orientation="vertical")
    colorbar.label_text_property.font_size = 12  # Adjust font size of numbers
    mlab.savefig(fig_name)
    mlab.show()


def plot_hist(sample, title="Histogram", bins=100):
    """
    Plots histogram of values inside a PyTorch tensor.
    """
    values = sample.flatten()
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=bins, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def train_valid_results(results_dir, diff_model, checkpoint, trials_config, n_samples=10, batch_size=1, cond=None, cond_scale=1.0):
    with torch.no_grad():
        print("Best epoch:", checkpoint['best_epoch'])
        print("Training time (s):", checkpoint["training_time"])
        plot_train_valid_loss(checkpoint['train_loss'], checkpoint['valid_loss'])

        generated_samples = []

        latents_datasets_dir = os.path.join(results_dir, "latents_datasets")

        train_dataset_path = os.path.join(latents_datasets_dir, "latents_dataset_train")
        valid_dataset_path = os.path.join(latents_datasets_dir, "latents_dataset_validation")
        test_dataset_path = os.path.join(latents_datasets_dir, "latents_dataset_test")

        train_dataset = LatentsDataset(train_dataset_path)
        valid_dataset = LatentsDataset(valid_dataset_path)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
        validation_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)

        reconstructions = []
        with torch.no_grad():
            for i, (volumes, conds) in enumerate(train_loader):
                print("volumes min: {}, max: {}".format(volumes.min(), volumes.max()))
                volumes = volumes.to(device).float()
                conds = conds.to(device).float()

                # Forward pass returns (true_noise, predicted_noise)
                true_noise, pred_noise = diff_model(volumes, conds)
                mse = F.mse_loss(pred_noise, true_noise, reduction="mean").item()
                print("MSE ", mse)

                print("conds ", conds)

                # ----------------------------
                # 1) Show ground-truth volumes
                # ----------------------------
                gt = volumes.cpu().numpy()
                print(f"[{i}] Ground truth shape: {gt.shape}")

                # ----------------------------
                # 2) Generate samples
                # ----------------------------
                samples = diff_model.sample(
                    batch_size=volumes.shape[0],
                    cond=conds,
                    cond_scale=cond_scale,
                ).cpu().numpy()

                # Clamp to training range
                samples = torch.clamp(samples, -1, 1)

                # print("betas range:", diff_model.noise_scheduler.betas.min().item(), diff_model.noise_scheduler.betas.max().item())
                # print("sqrt_recip_alphas range:", diff_model.noise_scheduler.sqrt_recip_alphas.min().item(),
                #       diff_model.noise_scheduler.sqrt_recip_alphas.max().item())
                # print("sqrt_one_minus_alphas_cumprod range:",
                #       diff_model.noise_scheduler.sqrt_one_minus_alphas_cumprod.min().item(),
                #       diff_model.noise_scheduler.sqrt_one_minus_alphas_cumprod.max().item())

                generated_samples.append(samples)
                reconstructions.append(gt)

                print("samples min: {}, max: {}".format(samples.min(), samples.max()))

                print("samples.shape ", samples.shape)
                plot_hist(volumes.cpu().numpy())
                plot_hist(samples)

                # ----------------------------
                # Optional: visualize one example
                # ----------------------------
                if i < n_samples:
                    render_3d_scan(np.squeeze(gt)[0,...], title=f"Ground Truth {i}", fig_name=f"gt_{i}.png")
                    render_3d_scan(np.squeeze(samples)[0, ...], title=f"Generated {i}", fig_name=f"gen_{i}.png")

                if i >= n_samples - 1:
                    break

        # diff_model.eval()  # set evaluation mode
        # with torch.no_grad():
        #     for i, samples in enumerate(train_loader):
        #         # unpack batch
        #         volumes, conds = samples
        #         volumes = volumes.to(device)
        #         conds = conds.to(device)
        #
        #         # forward pass
        #         noise, predicted_noise = diff_model(volumes, conds)


def sample(diff_model, checkpoint, trials_config, n_samples=10, batch_size=1, cond=None, cond_scale=1.0):
    """
    Generate samples from a trained ConditionalDiffusion model.
    """

    with torch.no_grad():
        print("Best epoch:", checkpoint['best_epoch'])
        print("Training time (s):", checkpoint["training_time"])
        plot_train_valid_loss(checkpoint['train_loss'], checkpoint['valid_loss'])

        generated_samples = []

        for i in range(n_samples):
            # If no conditioning provided, use zeros (unconditional sampling)
            if cond is None:
                cond = torch.zeros((batch_size, trials_config["unet_config"][0]["cond_dim"]), device=next(diff_model.parameters()).device)

            samples = diff_model.sample(
                batch_size=batch_size,
                cond=cond,
                cond_scale=cond_scale,
            )

            samples = samples.cpu()

            print("samples min: {}, max: {}".format(np.min(samples.numpy()) , np.max(samples.numpy())))

            generated_samples.append(samples)

            # Render and histogram (optional)
            np_sample = np.squeeze(samples.numpy())

            render_3d_scan(np_sample, title=f"Sample {i}", fig_name=f"sample_{i}.png")

            plt.figure(figsize=(8, 6))
            plt.hist(np_sample.flatten(), bins=100, density=True)
            plt.title(f"Bone density distribution - sample {i}")
            plt.savefig(f"bone_density_distr_{i}.pdf")
            plt.close()

        return generated_samples
        #
        #     # #print("samples.shape ", samples.shape)
        #     #
        #     # if density_only:
        #     #     density = samples
        #     # else:
        #     #     density = samples[..., 3]
        #     #     displacement = samples[..., :3]
        #     #
        #     #     displacement = np.squeeze(displacement.cpu().numpy())
        #     #
        #     #     fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        #     #     axes.hist(displacement[:, 0], bins=100, density=True, label="displacement[0]")
        #     #     axes.hist(displacement[:, 1], bins=100, density=True, label="displacement[1]")
        #     #     axes.hist(displacement[:, 2], bins=100, density=True, label="displacement[2]")
        #     #
        #     # density = np.squeeze(density.cpu().numpy())
        #
        #
        #     # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        #     # axes.hist(density, bins=100, density=True, color="red", label="density")
        #     # fig.legend()
        #     # plt.show()
        #     #
        #     # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        #     # axes.hist(np.squeeze(orig_samples.cpu().numpy()), bins=100, density=True, color="green", label="transformed density")
        #     # fig.legend()
        #     # plt.show()
        #
        #     generated_samples.append(density)
        #     generated_orig_samples.append(orig_samples)
        #
        #     #loss = loss_fn(predictions, targets)
        #     # running_loss += loss
        #     #
        #     # inv_targets = targets
        #     # inv_predictions = predictions
        #     # if inverse_transform is not None:
        #     #     inv_targets = inverse_transform(torch.reshape(targets, (*targets.shape, 1, 1)))
        #     #     inv_predictions = inverse_transform(torch.reshape(predictions, (*predictions.shape, 1, 1)))
        #     #
        #     #
        #     #     if dataset.init_transform is not None and input_inverse_transform is not None:
        #     #         inv_predictions *= inv_input_avg
        #     #         inv_targets *= inv_input_avg
        #     #
        #     #
        #     #     inv_targets = np.reshape(inv_targets, targets.shape)
        #     #     inv_predictions = np.reshape(inv_predictions, predictions.shape)
        #     #
        #     #
        #     #
        #     #
        #     # inv_running_loss += loss_fn(inv_predictions, inv_targets)
        #     #
        #     # inv_targets_list.append(inv_targets.numpy())
        #     # inv_predictions_list.append(inv_predictions.numpy())
        #
        #
        # #inv_targets_arr = np.array(inv_targets_list)
        # #inv_predictions_arr = np.array(inv_predictions_list)
        #
        # #print("inv targets arr shape", inv_targets_arr.shape)
        #
        # predictions_list = np.array(predictions_list)
        # #predictions_list += 0.1
        # orig_samples_flatten = np.array(original_samples).flatten()
        # generated_samples_flatten = np.array(generated_samples).flatten()
        #
        # print("orig samples flatten ", orig_samples_flatten.shape)
        #
        # # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        # # axes.hist(np.log(orig_samples_flatten), bins=100, density=True, color="red", label="log orig density")
        # # axes.hist(np.log(generated_samples_flatten), bins=100, density=True, color="blue", label="log generated density", alpha=0.5)
        # # fig.legend()
        # # plt.show()
        #
        # import matplotlib
        # matplotlib.rcParams.update({'font.size': 22})
        #
        # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        # axes.hist(orig_samples_flatten, bins=100, density=True, color="red", label="real samples")
        # axes.hist(generated_samples_flatten, bins=100, density=True, color="blue", label="generated samples",
        #           alpha=0.5)
        # plt.xlabel("bone density")
        # fig.legend()
        # plt.tight_layout()
        # plt.savefig("L4_orig_generated.pdf")
        # plt.show()
        #
        # #fgd = calculate_fgd(real_features=np.array(original_samples), generated_features=np.array(generated_samples))
        #
        # #print("fgd ", fgd)
        #
        # # mse, rmse, nrmse, r2 = get_mse_nrmse_r2(targets_list, predictions_list)
        # # inv_mse, inv_rmse, inv_nrmse, inv_r2 = get_mse_nrmse_r2(inv_targets_arr, inv_predictions_arr)
        #
        # #test_loss = running_loss / (i + 1)
        # #inv_test_loss = inv_running_loss / (i + 1)

def calculate_fgd(real_features, generated_features):
    print("real features shape ", real_features.shape)
    print("generated features shape ", generated_features.shape)
    # Compute the mean and covariance of the real and generated features
    mu_real, sigma_real = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = generated_features.mean(0), np.cov(generated_features, rowvar=False)

    # Compute Fréchet distance
    mu_diff = mu_real - mu_gen
    print("mu diff ", mu_diff)
    cov_sqrt, _ = sc.linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    fgd = mu_diff @ mu_diff + np.trace(sigma_real + sigma_gen - 2 * cov_sqrt)
    return fgd


def load_study(results_dir):
    study = joblib.load(os.path.join(results_dir, "study.pkl"))
    print("Best trial until now:")
    print(" Value: ", study.best_trial.value)
    print(" Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', help='results directory')
    parser.add_argument("-c", "--cuda", default=False, action='store_true', help="use cuda")
    args = parser.parse_args(sys.argv[1:])

    diffusion_model, checkpoint, trials_config = load_diffusion_model(args.results_dir)

    #train_valid_results(args.results_dir, diffusion_model, checkpoint, trials_config)

    sample(diffusion_model, checkpoint, trials_config)


