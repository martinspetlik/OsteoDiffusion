import os
import sys
import argparse
import joblib
import torch
import shutil
import numpy as np
import yaml
from dataset.cnn_diffusion.bone_dataset_CT import BoneDatasetCT
from models.vqgan.vqgan_model import VQGAN

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



def get_vqgan_samples(vqgan_results_dir, vqgan_model_path, vqgan_trials_config, output_results_dir, dataset):
    val_dataset = joblib.load(os.path.join(vqgan_results_dir, "val_dataset.pkl"))

    print("val dataset ", val_dataset)
    print(len(val_dataset))

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    val_data_loader = None
    if len(val_dataset) > 0:
        BoneDatasetCT.remap_subset_paths(val_dataset)
        val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)


    vqgan_model_config = load_trials_config(vqgan_trials_config)["model_config"][0]

    print("vqgan_model_config ", vqgan_model_config)

    vqgan_model_checkpoint = VQGAN.load_from_checkpoint(vqgan_model_path,
                                                        **vqgan_model_config)

    vqgan_model_checkpoint.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqgan_model_checkpoint.to(device)

    vqgan_model_checkpoint.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device ", device)
    vqgan_model_checkpoint.to(device)

    print("model checkpoint ", vqgan_model_checkpoint)

    # get_vqgan_latents(model_checkpoint, data_loader, device, results_dir)

    threshold = -0.99  # -0.85

    data_loader = val_data_loader if val_data_loader is not None else data_loader
    targets_predictions = []
    i = 0
    for batch in data_loader:
        if output_results_dir is not None:
            sample_dir = os.path.join(output_results_dir, "sample_{}".format(i))
            if os.path.exists(sample_dir):
                shutil.rmtree(sample_dir)
            os.mkdir(sample_dir)
            os.chdir(sample_dir)
            i += 1

        input, cond = batch
        print("input ", input)
        input = input.to(device)
        if isinstance(input, (list, tuple)):
            input = input[0].to(device)  # Assuming (input, target) format
        else:
            input = input.to(device)  # If batch is input only

        print('input.dtype ', input.dtype)

        output, _ = vqgan_model_checkpoint(input)

        print("output.dtype ", output.dtype)

        output[output < threshold] = -1

        print("output  threshold", output)

        np.save(os.path.join(sample_dir, "vqgan_input"), np.squeeze(input.cpu().numpy()))
        np.save(os.path.join(sample_dir, "vqgan_output"), np.squeeze(output.cpu().numpy()))


def load_dataset(data_dir, data_file_name):
    dataset = BoneDatasetCT(data_dir=data_dir, data_file_name=data_file_name)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('sampling_config_path', help='Sampling config path')
    parser.add_argument('results_dir', help='Directory to save generated samples')
    args = parser.parse_args(sys.argv[1:])

    with open(args.sampling_config_path, "r") as f:
        sampling_config = yaml.load(f, Loader=yaml.FullLoader)

    print("sampling config ", sampling_config)


    vqgan_model_path = sampling_config["vqgan_model_path"]

    dataset = load_dataset(sampling_config["dataset_dir"], sampling_config["dataset_data_file_name"])

    get_vqgan_samples(sampling_config["vqgan_results_dir"], vqgan_model_path=sampling_config["vqgan_model_path"],
                      vqgan_trials_config=sampling_config["vqgan_trials_config"], output_results_dir=args.results_dir, dataset=dataset)

