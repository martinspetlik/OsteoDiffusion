import os
import sys
import argparse
import joblib
import torch
import numpy as np
import torchvision.transforms as transforms
from dataset.cnn_diffusion.bone_dataset_CT import BoneDatasetCT
from visualization.visualize_data import plot_train_valid_loss
import matplotlib.pyplot as plt
import scipy as sc
from models.vqgan.vqgan_model import VQGAN
import pandas as pd
import pyvista as pv
import nibabel as nib
import os
import glob

#os.environ["CUDA_VISIBLE_DEVICES"]=""


def get_saved_model_path(results_dir, best_trial):
    #path_to_model = "lightning_logs/version_0/checkpoints/latest_checkpoint.ckpt"
    path_to_model = "logger/version_0/checkpoints/latest_checkpoint.ckpt"

    vqgan_model_path = os.path.join(results_dir, path_to_model)

    print(vqgan_model_path)

    return vqgan_model_path


def load_dataset(results_dir, study):
    data_dir = study.user_attrs["data_dir"]
    print("data_dir ", data_dir)
    dataset = BoneDatasetCT(data_dir=data_dir, data_file_name="lumbopelvic_masked_normed_global_clip_resampled_320_192_320.npz")
    dataset = BoneDatasetCT(data_dir=data_dir, data_file_name="lumbopelvic_masked_normed_global_clip_resampled_128_128_128.npz")
    return dataset


def plot_target_data(train_loader, validation_loader, test_loader):
    import matplotlib.pyplot as plt
    train_targets = []
    val_targets = []
    test_targets = []
    test_targets = []

    k_xx_list_input = []
    k_xy_list_input = []
    k_yy_list_input = []
    for i, test_sample in enumerate(train_loader):
        inputs, targets = test_sample
        inputs = np.squeeze(inputs.numpy())
        print("inputs ", inputs)
        if i > 50:
            break
        print("inputs[0].ravel() ", inputs[0].ravel())
        k_xx_list_input.extend(inputs[0].ravel())
        k_xy_list_input.extend(inputs[1].ravel())
        k_yy_list_input.extend(inputs[2].ravel())

    print("kxx shape ", np.array(k_xx_list_input).shape)
    plt.hist(k_xx_list_input, bins=25, color="red", label="input k_xx", density=True)
    #plt.xlim([-0.001, 1])
    plt.legend()
    plt.show()

    plt.hist(k_xy_list_input, bins=60, color="red", label="input k_xy", density=True)
    plt.legend()
    plt.show()

    plt.hist(k_yy_list_input, bins=60, color="red", label="input k_yy", density=True)
    plt.legend()
    plt.show()

    for i, test_sample in enumerate(test_loader): #@TODO: use train_loader again
        inputs, targets = test_sample
        targets_np = np.squeeze(targets.numpy())
        #print("targets_np ", targets_np)
        train_targets.append(targets_np)
        #exit()

    # for i, test_sample in enumerate(validation_loader):
    #     inputs, targets = test_sample
    #     targets_np = np.squeeze(targets.numpy())
    #     val_targets.append(targets_np)
    #
    # for i, test_sample in enumerate(test_loader):
    #     inputs, targets = test_sample
    #     targets_np = np.squeeze(targets.numpy())
    #     test_targets.append(targets_np)

    train_targets = np.array(train_targets)
    #val_targets = np.array(val_targets)
    #test_targets = np.array(test_targets)


    n_channels = 3
    for i in range(n_channels):
        print("train targets shape ", train_targets.shape)
        k_train_targets = train_targets[:, i]

        #k_train_targets = k_train_targets[k_train_targets < 0.6]
        #k_val_targets = val_targets[:, i]
        #k_test_targets = test_targets[:, i]
        print("min: {}, max: {}, avg:{} ".format(np.min(k_train_targets),
                                                 np.max(k_train_targets),
                                                 np.mean(k_train_targets)))
        print("k_train_targets[:100] ", k_train_targets[:100])
        print("k_train_targets ", k_train_targets.shape)
        print("k train targets ")

        density = True
        plt.hist(k_train_targets, bins=60, color="red", label="train, ch: {}".format(i), alpha=0.5, density=density)

        #plt.hist(k_val_targets, bins=60, color="blue", label="val", alpha=0.5, density=density)
        #plt.hist(k_test_targets, bins=60, color="green", label="test", alpha=0.5, density=density)
        #plt.xlabel(xlabel)
        # plt.ylabel("Frequency for relative")
        # if i == 1:
        #     plt.xlim([-1, 1])
        plt.legend()
        #plt.savefig("hist_" + title + ".pdf")
        plt.show()

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


def plot_log_images(images_dir, epoch):
    epoch_str = f"e-{epoch:06}"
    nii_files = sorted(glob.glob(os.path.join(images_dir, f"*_{epoch_str}_*.nii.gz")))
    print("nii_files ", nii_files)
    for file in nii_files:
        base = os.path.basename(file)
        label = base.split("_")[0]  # e.g., 'input', 'pred', 'target'
        print("label ", label)
        img = np.squeeze(nib.load(file).get_fdata())

        print("img ", img.shape)
        if label in ["source", "recon"]:
            render_3d_scan(img, title=label, fig_name="{}_prediction.png".format(label))


def get_vqgan_latents(vqgan, dataloader, device, working_dir):
    # Get only used codebook indices
    used_indices = set()
    print("len dataloader ", len(dataloader))
    for batch in dataloader:
        input, cond = batch
        print("input.shape ", input.shape)
        input = input.to(device)
        h = vqgan.encoder(input)
        h = vqgan.quant_conv(h)
        print("latent space shape ", h.shape)
        #vqgan.quantize.sane_index_shape = True  # get in spatial format
        z_q, loss, (_, _, codebook_indices) = vqgan.quantize(h)
        used_indices.update(codebook_indices.cpu().numpy().flatten())

    print("len used indices ", len(used_indices))
    exit()
    used_codebook_indices_file_path = os.path.join(working_dir, "used_indices.npy")
    np.save(used_codebook_indices_file_path, np.array(sorted(used_indices)))
    vqgan.quantize.set_used_indices(used_codebook_indices_file_path)

    vqgan.eval()
    latent_codes = []
    with torch.no_grad():
        for batch in dataloader:
            input, cond = batch
            input = input.to(device)
            h = vqgan.encoder(input)
            print("encoder output shape ", h.shape)
            h = vqgan.quant_conv(h)
            print("quant conv shape ", h.shape)

            vqgan.quantize.sane_index_shape = True  # get in spatial format
            z_q, loss, (_, _, codebook_indices) = vqgan.quantize(h)
            print("code indices ", codebook_indices)
            latent_codes.append(codebook_indices)


def load_models(args, study):
    results_dir = args.results_dir
    model_checkpoint_path = get_saved_model_path(results_dir, study.best_trial)
    # metrics_csv_path = os.path.join(results_dir, "lightning_logs/version_0/metrics.csv")
    #
    # train_image_log = os.path.join(results_dir, "lightning_logs/version_0/images/train")
    # val_image_log = os.path.join(results_dir, "lightning_logs/version_0/images/val")

    metrics_csv_path = os.path.join(results_dir, "logger/version_0/metrics.csv")

    train_image_log = os.path.join(results_dir, "logger/version_0/images/train")
    val_image_log = os.path.join(results_dir, "logger/version_0/images/val")

    dataset = load_dataset(results_dir, study)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    print("study.best_trial.params.keys() ", list(study.best_trial.params.keys()))

    #threshold = -0.5  # @TODO: determine the most suitable threshold value

    # i = 0
    # for sample in data_loader:
    #     # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    #     # axes.hist(sample.flatten(), bins=100, density=True, label="bone density distr")
    #     # fig.legend()
    #     # plt.show()
    #
    #     sample_to_rander = np.squeeze(sample.numpy())
    #     #sample_to_rander[sample_to_rander > ]
    #
    #     #sample_to_rander[np.abs(sample_to_rander) < 1e-4] = -1
    #     sample_to_rander = sample_to_rander.flatten()
    #
    #     fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    #     axes.hist(sample_to_rander, bins=100, density=True,
    #               label="Dataset bone density distr with background")
    #     fig.legend()
    #     plt.show()
    #
    #
    #     fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    #     axes.hist(sample_to_rander[sample_to_rander > -0.5], bins=100, density=True,
    #               label="Dataset bone density distr without background")
    #     fig.legend()
    #     plt.show()
    #
    #     i+=1
    #
    #     if i > 1:
    #         break
    #
    #     #render_3d_scan(sample_to_rander, title="Original 3D Scan")
    #
    #

    #print("len(dataset): {}".format(len(dataset)))

    plot_separate_images = False
    # Disable grad
    with torch.no_grad():
        model_checkpoint = VQGAN.load_from_checkpoint(model_checkpoint_path, **study.best_trial.params["model_config"])
        # Load CSV
        df = pd.read_csv(metrics_csv_path)
        # Inspect columns
        print(df.columns)
        # Group and average the training reconstruction loss per epoch
        recon_train_loss = df[["epoch", "train/recon_loss_epoch"]].dropna()
        recon_train_grouped = recon_train_loss.groupby("epoch").mean().reset_index()

        # Group and average the validation reconstruction loss per epoch
        # recon_valid_loss = df[["epoch", "val/recon_loss_epoch"]].dropna()
        # recon_valid_grouped = recon_valid_loss.groupby("epoch").mean().reset_index()

        # Group and average the training AE loss per epoch
        aeloss_train_loss = df[["epoch", "train/aeloss_epoch"]].dropna()
        aeloss_train_grouped = aeloss_train_loss.groupby("epoch").mean().reset_index()

        # # Group and average the validation AE loss per epoch
        # aeloss_valid_loss = df[["epoch", "val/aeloss_epoch"]].dropna()
        # aeloss_valid_grouped = aeloss_valid_loss.groupby("epoch").mean().reset_index()

        # Group and average the training AE loss per epoch
        discloss_train_loss = df[["epoch", "train/discloss_epoch"]].dropna()
        discloss_train_grouped = discloss_train_loss.groupby("epoch").mean().reset_index()

        # # Group and average the validation AE loss per epoch
        # discloss_valid_loss = df[["epoch", "val/discloss_epoch"]].dropna()
        # discloss_valid_grouped = discloss_valid_loss.groupby("epoch").mean().reset_index()

        # used codebook percent
        used_codebook_percent = df[["epoch", "train/used_codebook_percent"]].dropna()
        print("used_codebook_percent ", used_codebook_percent)
        used_codebook_percent = used_codebook_percent.groupby("epoch").mean().reset_index()

        # used codebook percent
        used_codebook_count = df[["epoch", "train/used_codebook_count"]].dropna()
        print("used_codebook_count ", used_codebook_count)
        used_codebook_count = used_codebook_count.groupby("epoch").mean().reset_index()



        print("recon_train_loss ", recon_train_loss)
        #print("recon valid loss ", recon_valid_loss)

        # Find epoch with minimum train loss
        min_train_row = recon_train_loss.loc[recon_train_loss["train/recon_loss_epoch"].idxmin()]
        print(f"Minimum train loss: {min_train_row['train/recon_loss_epoch']:.4f} at epoch {int(min_train_row.epoch)}")

        # min_valid_row = recon_valid_loss.loc[recon_valid_loss["val/recon_loss_epoch"].idxmin()]
        # print(f"Minimum validation loss: {min_valid_row['val/recon_loss_epoch']:.4f} at epoch {int(min_valid_row.epoch)}")

        best_train_epoch = int(min_train_row.epoch)
        #best_valid_epoch = int(min_valid_row.epoch)

        # plot_train_valid_loss(recon_train_grouped["train/recon_loss_epoch"], recon_valid_grouped["val/recon_loss_epoch"], y_label="recon_loss")
        # plot_train_valid_loss(aeloss_train_grouped["train/aeloss_epoch"], aeloss_valid_grouped["val/aeloss_epoch"],
        #                       y_label="aeloss")
        #
        # plot_train_valid_loss(discloss_train_grouped["train/discloss_epoch"], discloss_valid_grouped["val/discloss_epoch"],
        #                       y_label="discloss")
        #
        # plot_train_valid_loss(used_codebook_percent["train/used_codebook_percent"],
        #                       used_codebook_percent["train/used_codebook_percent"],
        #                       y_label="used codebook percent")

        plot_train_valid_loss(recon_train_grouped["train/recon_loss_epoch"], valid_loss=None, y_label="recon_loss")
        plot_train_valid_loss(aeloss_train_grouped["train/aeloss_epoch"], valid_loss=None, y_label="aeloss")

        plot_train_valid_loss(discloss_train_grouped["train/discloss_epoch"], valid_loss=None, y_label="discloss")

        plot_train_valid_loss(used_codebook_percent["train/used_codebook_percent"], valid_loss=None,
                              y_label="used codebook percent")

        plot_train_valid_loss(used_codebook_count["train/used_codebook_count"], valid_loss=None,
                              y_label="used codebook count")

        #plot_log_images(train_image_log, best_valid_epoch)
        #plot_log_images(val_image_log, best_valid_epoch)

        model_checkpoint.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device ", device)
        model_checkpoint.to(device)

        get_vqgan_latents(model_checkpoint, data_loader, device, results_dir)

        targets_predictions = []
        for batch in data_loader:
            input, cond = batch
            input = input.to(device)
            if isinstance(input, (list, tuple)):
                input = input[0].to(device)  # Assuming (input, target) format
            else:
                input = input.to(device)  # If batch is input only

            output, _ = model_checkpoint(input)

            targets_predictions.extend(zip(input.cpu(), output.cpu()))

            render_3d_scan(np.squeeze(input.cpu().numpy()), title="INPUT Sampled 3D Scan",
                           fig_name="sampled_3D_scan_with_background_input.png")
            render_3d_scan(np.squeeze(output.cpu().numpy()), title="OUTPUT Sampled 3D Scan",
                           fig_name="sampled_3D_scan_with_background_output.png")



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

    print("torch.cuda.is_available() ", torch.cuda.is_available())
    #print(torch.zeros(1).cuda())

    study = load_study(args.results_dir)

    #@TODO: RM ASAP
    print("study attrs ", study.user_attrs)
    #study.user_attrs["output_log"] = True
    #study.set_user_attr("output_log", True)
    print("study attrs ", study.user_attrs)
    #compare_trials(study)

    load_models(args, study)


