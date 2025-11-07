import os
import sys
import argparse
import joblib
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_study(results_dir):
    study = joblib.load(os.path.join(results_dir, "study.pkl"))
    print("Best trial until now:")
    print(" Value: ", study.best_trial.value)
    print(" Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    return study


def load_trials_config(path_to_config):
    with open(path_to_config, "r") as f:
        trials_config = yaml.load(f, Loader=yaml.FullLoader)
    return trials_config


def load_diffusion_model(results_dir, model_path):
    #optuna_study = load_study(results_dir)
    trials_config = load_trials_config(os.path.join(results_dir, "trials_config.yaml"))

    #print("optuna_study.best_trial.user_attrs ", optuna_study.best_trial.user_attrs)

    # # Use explicit model_path if provided, otherwise construct from study
    # if model_path is None:
    #     model_path = os.path.join(
    #         results_dir,
    #         f"trial_{optuna_study.best_trial.number}_model_best.pt"
    #     )
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
    #noise_scheduler.num_gen_timesteps = 50

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

    return diff_model, trials_config


def load_vqgan_model(results_dir, vqgan_model_path, diff_model_trials_config):
    #optuna_study = load_study(results_dir)

    #trials_config = load_trials_config(os.path.join(results_dir, "trials_config.yaml"))

    # if vqgan_model_path is None:
    #     vqgan_model_path = os.path.join(results_dir, "logger/version_0/checkpoints/val/last.ckpt")
    print("VQGAN model path ", vqgan_model_path)

    print("diff model trials config vqga trials config ", diff_model_trials_config["vqgan_trials_config"])

    vqgan_model_config = load_trials_config(diff_model_trials_config["vqgan_trials_config"])["model_config"][0]

    print("vqgan_model_config ", vqgan_model_config)

    vqgan_model_checkpoint = VQGAN.load_from_checkpoint(vqgan_model_path,
                                                        **vqgan_model_config)

    vqgan_model_checkpoint.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqgan_model_checkpoint.to(device)

    return vqgan_model_checkpoint


def load_dataset(data_dir, data_file_name):
    dataset = BoneDatasetCT(data_dir=data_dir, data_file_name=data_file_name)
    return dataset


def generate_samples(latent_diffusion_model, vqgan_model, trials_config, cond=None, cond_scale=1.0, results_dir=None, clamp_diffusion_samples=False):
    batch_size_sample = 1
    n_samples = 50
    generated_samples = []
    torch.manual_seed(333)
    with torch.no_grad():
        for i in range(n_samples):
            if results_dir is not None:
                sample_dir = os.path.join(results_dir, "sample_{}".format(i))
                if os.path.exists(sample_dir):
                    shutil.rmtree(sample_dir)
                os.mkdir(sample_dir)
                os.chdir(sample_dir)

            # If no conditioning provided, use zeros (unconditional sampling)
            if cond is None:
                cond = torch.zeros((batch_size_sample, trials_config["unet_config"][0]["cond_dim"]),
                                   device=next(latent_diffusion_model.parameters()).device)

            # samples = latent_diffusion_model.sample(
            #     batch_size=batch_size_sample,
            #     cond=cond,
            #     cond_scale=cond_scale,
            # )

            samples = latent_diffusion_model.sample(
                batch_size=batch_size_sample,
                image_size=trials_config["image_size"], clip_denoised=True
            )

            # if clamp_diffusion_samples:
            #     samples = torch.clamp(samples, -1, 1)

            print("samples min: {}, max: {}".format(samples.min(), samples.max()))

            plot_hist(samples.cpu().numpy(), title="Generated")

            print("samples.shape ", samples.shape)
            #plot_hist(samples.cpu().numpy())

            generated_samples.append(samples.cpu())

            samples_to_vqgan = inverse_latent_transform(samples, vqgan_model).to(dtype=torch.float32)

            print("samples_to_vqgan.shape ", samples_to_vqgan.shape)

            print("samples_to_vqgan.dtype ", samples_to_vqgan.dtype)
            print("samples_to_vqgan min: {}, max: {}".format(samples_to_vqgan.min(), samples_to_vqgan.max()))

            if "train_on_codebooks" in trials_config and trials_config["train_on_codebooks"]:
                pass
            else:
                quant, emb_loss, info = vqgan_model.quantize(samples_to_vqgan)
                decoded_samples = vqgan_model.decode(quant)

            print("decoded samples shape ", decoded_samples.shape)

            print("decoded_samples min: {}, max: {}".format(decoded_samples.min(), decoded_samples.max()))

            np.save(os.path.join(sample_dir, "decoded_samples"), np.squeeze(decoded_samples.cpu().numpy()))

            #render_3d_scan(np.squeeze(decoded_samples.cpu().numpy()), title="Generated sample", fig_name=os.path.join(sample_dir, "gen_sample.png"), show=False)

            #render_3d_scan(np.load(os.path.join(sample_dir, "decoded_samples.npy")), fig_name=None)

            print("np.squeeze(decoded_samples.cpu().numpy()) ", np.squeeze(decoded_samples.cpu().numpy()).shape)

            threshold = -0.97

            decoded_samples = np.squeeze(decoded_samples.cpu().numpy())

            decoded_samples[decoded_samples < threshold] = -1

            #for slice_idx in range(0, trials_config["image_size"][0], 2):
            # render_3d_scan(np.squeeze(samples_to_vqgan.cpu().numpy())[0, ...], title=f"samples to vqgan {i}",
            #                fig_name=f"samples_to_vqgan_{i}.png")
            #render_3d_scan(decoded_samples, title=f"Decoded samples - VQGAN output {i}",
            #               fig_name=f"gen_vqgan_decoded_{i}.png")

            #@TODO inverse transform


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('sampling_config_path', help='Sampling config path')
    parser.add_argument('results_dir', help='Directory to save generated samples')
    args = parser.parse_args(sys.argv[1:])

    with open(args.sampling_config_path, "r") as f:
        sampling_config = yaml.load(f, Loader=yaml.FullLoader)

    print("sampling config ", sampling_config)

    latent_diffusion_model, diff_model_trials_config = load_diffusion_model(sampling_config["denoising_diffusion_results_dir"], model_path=sampling_config["denoising_diffusion_model_path"])

    vqgan_model_path = None
    if "vqgan_model_path" in sampling_config:
        vqgan_model_path = sampling_config["vqgan_model_path"]

    vqgan_model = load_vqgan_model(sampling_config["vqgan_results_dir"], vqgan_model_path=vqgan_model_path, diff_model_trials_config=diff_model_trials_config)
    #dataset = load_dataset(sampling_config["dataset_dir"], sampling_config["dataset_data_file_name"])

    generate_samples(latent_diffusion_model, vqgan_model, diff_model_trials_config, results_dir=args.results_dir,
                     clamp_diffusion_samples=sampling_config["clamp_diffusion_samples"])
