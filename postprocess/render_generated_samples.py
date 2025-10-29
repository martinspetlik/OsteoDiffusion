import copy
import os
import sys
import argparse
import torch
import glob
import numpy as np
from visualization.visualize_data import plot_hist, render_3d_scan

# Automatically select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_samples(results_dir: str):
    """
    Method that loads and visualizes decoded 3D samples stored in subdirectories of a results directory.
    It searches for folders matching 'sample_*', loads the corresponding 'decoded_samples.npy' files,
    applies thresholding and normalization, and renders the resulting 3D volumes.

    :param results_dir: Path to the directory containing 'sample_*' subdirectories with decoded .npy files.
    :type results_dir: str
    :raises FileNotFoundError: If no sample directories are found in the provided results_dir.
    :return: None
    """

    # Locate subdirectories that match the expected naming pattern
    sample_dirs = sorted(glob.glob(os.path.join(results_dir, "sample_*")))

    if not sample_dirs:
        raise FileNotFoundError(f"No sample directories found in: {results_dir}")

    decoded_data = []

    for sdir in sample_dirs:
        file_path = os.path.join(sdir, "decoded_samples.npy")

        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found")
            continue

        # Load the .npy file as a NumPy array
        generated_sample = np.load(file_path)
        orig_generated_sample = copy.deepcopy(generated_sample)

        # Apply thresholding to highlight structures
        threshold = -0.99
        generated_sample = np.squeeze(generated_sample)
        generated_sample[generated_sample < threshold] = -1

        # Render the raw thresholded 3D volume
        render_3d_scan(generated_sample, fig_name="")

        # Normalize from [-1, 1] to approximate Hounsfield Unit (HU) range
        min_value, max_value = -1024, 1650
        generated_sample = (generated_sample + 1) / 2
        generated_sample = generated_sample * (max_value - min_value) + min_value

        # Render the normalized 3D volume
        render_3d_scan(generated_sample, fig_name="")

        # decoded_data.append(generated_sample)  # Optional: collect results
        print(f"Loaded {file_path}, shape = {generated_sample.shape}")


def main():
    """
    Method that parses command-line arguments and visualizes decoded 3D samples.
    This function is the entry point when running the script directly from the command line.

    :return: None
    """

    parser = argparse.ArgumentParser(
        description="Visualize 3D decoded samples stored in subdirectories."
    )
    parser.add_argument(
        'results_dir',
        help='Path to the directory containing "sample_*" folders with decoded .npy files.'
    )
    args = parser.parse_args(sys.argv[1:])

    plot_samples(args.results_dir)


if __name__ == "__main__":
    main()
