import os
import sys
import glob
import ants
import argparse
import shutil
import numpy as np


def form_dataset(database, dataset_dir):
    # Path to segmentation-derived data inside the BoneDat database
    data_dir_path = os.path.join(database, "derived/segmentation")
    # Path to metadata (sex, age) information
    metadata_dir_path = os.path.join(database, "raw")
    data_file_name = "masked.nii.gz"  # CT scan masked with pelvic segmentation
    metadata_file_name = "metadata.xlsx"  # Contains patient ID, year of birth, sex, and CT date

    # Global intensity clipping range (based on dataset analysis)
    # -1024 is typical air HU in CT, 1650 chosen as a reasonable upper bound for bone
    min_value_global = -1024.0
    max_value_global = 1650

    # Target voxel dimensions for resampling all scans
    # Ensures uniform size for VQGAN training
    new_shape = (320, 192, 320)

    # File name template for the processed samples
    saved_data_file = "lumbopelvic_masked_normed_global_clip_resampled_{}_{}_{}".format(*new_shape)

    # Loop over all patient subdirectories in segmentation data
    for data_dir in glob.glob(data_dir_path + '/*'):
        path_parts = os.path.normpath(data_dir).split(os.sep)
        sample_dir_name = path_parts[-1]  # Unique patient/sample identifier

        # Load the masked CT image
        image_file = os.path.join(data_dir, data_file_name)
        image = ants.image_read(image_file)
        original_shape = image.shape
        original_spacing = image.spacing

        # Convert ANTs image to numpy array
        image_data = image.numpy()

        # Clip intensities to global min/max range (remove outliers and standardize range)
        image_data = np.clip(image_data, min_value_global, max_value_global)

        # Normalize to [0, 1], then scale to [-1, 1] for VQGAN training
        image_data_normalized_global = (image_data - min_value_global) / (max_value_global - min_value_global)  # [0, 1]
        image_data_normalized_global = 2 * image_data_normalized_global - 1  # [-1, 1]

        # Convert back to ANTs image for resampling
        image_normalized_global = ants.from_numpy(image_data_normalized_global, spacing=image.spacing)

        # Resample to fixed voxel grid (new_shape), nearest-neighbor (interp_type=0) to preserve labels/masks
        resampled_img_normalized_global = ants.resample_image(image_normalized_global, new_shape, use_voxels=True, interp_type=0)

        # Compute new voxel spacing to preserve physical dimensions of the scan
        new_spacing = [
            original_spacing[i] * original_shape[i] / new_shape[i] for i in range(3)
        ]
        resampled_img_normalized_global.set_spacing(new_spacing)

        # Convert to numpy for saving
        resampled_img_normalized_global_data = resampled_img_normalized_global.numpy()

        # Debug print: value range after normalization and resampling
        print("resampled_img_normalized_global_data values min: {}, max: {}".format(
            np.min(resampled_img_normalized_global_data),
            np.max(resampled_img_normalized_global_data))
        )

        dataset_sample_dir = os.path.join(dataset_dir, sample_dir_name)
        # Create patient/sample directory inside dataset output folder
        if not os.path.exists(dataset_sample_dir):
            os.mkdir(dataset_sample_dir)

        # Save processed scan as compressed NumPy archive
        np.savez_compressed(
            os.path.join(dataset_sample_dir, saved_data_file),
            data=resampled_img_normalized_global_data
        )

        # Path to metadata file
        metadata_file = os.path.join(os.path.join(metadata_dir_path, sample_dir_name), metadata_file_name)

        # Copy metadata file into dataset_sample_dir
        if os.path.exists(metadata_file):  # make sure file exists
            shutil.copy(metadata_file, dataset_sample_dir)
        else:
            print(f"Warning: Metadata file not found for {sample_dir_name}")


# =========================================================
# Main entrypoint
# =========================================================
if __name__ == '__main__':
    # --- CLI arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument('database', help='Path to BoneDat')
    parser.add_argument('dataset_dir', help='Path to dataset directory')
    args = parser.parse_args(sys.argv[1:])

    # Process dataset and save normalized/resampled scans
    form_dataset(database=args.database, dataset_dir=args.dataset_dir)
