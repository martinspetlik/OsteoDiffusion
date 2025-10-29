import copy
import os
import sys
import argparse
import torch
import glob
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from visualization.visualize_data import plot_train_valid_loss, render_3d_scan, render_two_3d_scans



def plot_samples(results_dir):
    # Find all subdirectories that start with "sample_"
    sample_dirs = sorted(glob.glob(os.path.join(results_dir, "sample_*")))

    decoded_data = []
    for sdir in sample_dirs:
        input_file_path = os.path.join(sdir, "vqgan_input.npy")
        output_file_path = os.path.join(sdir, "vqgan_output.npy")
        if os.path.exists(input_file_path):
            input_sample = np.load(input_file_path)
            output_sample = np.load(output_file_path)

            threshold = -0.99

            output_sample = np.squeeze(output_sample)

            output_sample[output_sample < threshold] = -1

            render_two_3d_scans(input_sample, output_sample,
                                title="OUTPUT Sampled 3D Scan",
                                fig_name="sampled_3D_scan_with_background_output.png")

            # render_3d_scan(generated_sample, fig_name="")
            #
            # min_value = -1024
            # max_value = 1650
            # orig_generated_sample = (orig_generated_sample + 1) / 2
            # orig_generated_sample = orig_generated_sample * (max_value - min_value) + min_value
            # render_3d_scan(orig_generated_sample, fig_name="")

            #decoded_data.append(generated_sample)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', help='Directory to save generated samples')
    args = parser.parse_args(sys.argv[1:])

    plot_samples(args.results_dir)