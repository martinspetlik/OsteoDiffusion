import matplotlib.pyplot as plt


def plot_hist_orig_generated(orig_sample, generated_sample, title="Histogram", bins=100):
    """
    Method that plots overlapping histograms of the original and generated samples.

    :param orig_sample: Original sample tensor or array.
    :param generated_sample: Generated sample tensor or array.
    :param title: Title of the plot.
    :param bins: Number of bins for the histogram.
    :return: None
    """
    plt.figure(figsize=(6, 4))
    plt.hist(orig_sample.flatten(), bins=bins, color='blue', density=True, alpha=0.99, label='Original')
    plt.hist(generated_sample.flatten(), bins=bins, color='red', density=True, alpha=0.5, label='Generated')
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_hist(sample, title="Histogram", bins=100):
    """
    Method that plots a histogram of values inside a tensor or NumPy array.

    :param sample: Sample tensor or array to plot.
    :param title: Title of the histogram plot.
    :param bins: Number of bins in the histogram.
    :return: None
    """
    values = sample.flatten()
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=bins, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_train_valid_loss(train_loss, valid_loss=None, y_label="", log=False, line_value=None):
    """
    Method that plots training and validation loss curves over epochs.

    :param train_loss: List or array of training loss values.
    :param valid_loss: Optional list or array of validation loss values.
    :param y_label: Label for the Y-axis.
    :param log: Whether to use logarithmic scaling on the Y-axis.
    :param line_value: Optional horizontal reference line (e.g., target loss).
    :return: None
    """
    plt.plot(train_loss, label="Train loss")
    if valid_loss is not None:
        plt.plot(valid_loss, label="Validation loss")
    if log:
        plt.yscale("log")
    if line_value is not None:
        plt.axhline(y=line_value, color='r', linestyle='-')
    plt.xlabel("Epochs")
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig("train_val_loss.pdf")
    plt.show()


def render_3d_scan(scan, title="3D Scan", fig_name=None):
    """
    Method that renders a 3D volumetric scan using Mayavi.

    :param scan: 3D array representing the volumetric data.
    :param title: Title displayed above the rendered volume.
    :param fig_name: Optional filename to save the rendered figure.
    :return: None
    """
    try:
        from mayavi import mlab
    except ImportError as e:
        raise ImportError(
            "Mayavi is required for 3D visualization. Install it with 'pip install mayavi'."
        ) from e

    mlab.figure(size=(800, 800), bgcolor=(1, 1, 1))
    src = mlab.pipeline.scalar_field(scan)
    volume = mlab.pipeline.volume(src)
    volume._volume_property.scalar_opacity_unit_distance = 0.1

    cb = mlab.colorbar(object=volume, orientation='vertical')

    lut_manager = volume.module_manager.scalar_lut_manager
    scalar_bar = lut_manager.scalar_bar
    scalar_bar.title = "Intensity"
    scalar_bar.component_title = ""
    scalar_bar.title_text_property.font_size = 14
    scalar_bar.label_text_property.font_size = 10
    scalar_bar.label_text_property.color = (0, 0, 0)
    scalar_bar.title_text_property.color = (0, 0, 0)

    mlab.title(title, size=0.5, height=0.95)

    if fig_name is not None:
        mlab.savefig(fig_name)

    mlab.show()


def render_two_3d_scans(scan1, scan2, label1="Target", label2="Reconstruction",
                        title="3D Comparison", fig_name=""):
    """
    Method that renders two 3D volumetric scans side by side for visual comparison.

    The scans are placed with a small gap between them. Color scales are normalized
    to the first scan to allow direct intensity comparison.

    :param scan1: First volumetric scan (e.g., target or ground truth).
    :param scan2: Second volumetric scan (e.g., generated or reconstructed).
    :param label1: Label text displayed above the first scan.
    :param label2: Label text displayed above the second scan.
    :param title: Title of the 3D scene.
    :param fig_name: Optional filename to save the rendered figure.
    :return: None
    """
    try:
        from mayavi import mlab
    except ImportError as e:
        raise ImportError(
            "Mayavi is required for 3D visualization. Install it with 'pip install mayavi'."
        ) from e
    import numpy as np

    assert scan1.shape == scan2.shape, "Scans must have the same shape"
    sx, sy, sz = scan1.shape
    gap = 20

    vmin, vmax = np.min(scan1), np.max(scan1)
    scan2_clamped = np.clip(scan2, vmin, vmax)

    combined = np.zeros((sx * 2 + gap, sy, sz), dtype=scan1.dtype)
    combined[:sx, :, :] = scan1
    combined[sx + gap:, :, :] = scan2_clamped

    mlab.figure(size=(1000, 800), bgcolor=(1, 1, 1))
    src = mlab.pipeline.scalar_field(combined)
    vol = mlab.pipeline.volume(src)
    vol._volume_property.scalar_opacity_unit_distance = 0.1
    vol.module_manager.scalar_lut_manager.lut_mode = 'viridis'
    vol.module_manager.scalar_lut_manager.data_range = (vmin, vmax)

    cb = mlab.colorbar(title='Intensity', orientation='vertical')
    cb.label_text_property.font_size = 14
    cb.label_text_property.color = (0, 0, 0)
    cb.title_text_property.color = (0, 0, 0)

    label_z = sz + 10
    mlab.text3d(x=sx // 2, y=sy // 2, z=label_z,
                text=label1, scale=(10, 10, 10), color=(0, 0, 0))
    x_offset = sx + gap + (sx // 2)
    mlab.text3d(x=x_offset, y=sy // 2, z=label_z,
                text=label2, scale=(10, 10, 10), color=(0, 0, 0))

    mlab.title(title, size=0.5, height=0.95)
    mlab.view(azimuth=180, elevation=80, distance='auto')

    if fig_name:
        mlab.savefig(fig_name)

    mlab.show()
