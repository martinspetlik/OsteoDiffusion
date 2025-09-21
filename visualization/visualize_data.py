import numpy as np
import matplotlib.pyplot as plt

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


def plot_train_valid_loss(train_loss, valid_loss=None, y_label="", log=False):
    plt.plot(train_loss, label="train loss")
    if valid_loss is not None:
        plt.plot(valid_loss, label="valid loss")
    if log:
        plt.yscale("log")
    #plt.ylim([0, np.min([10000, np.max(train_loss), np.max(valid_loss)])])
    plt.xlabel("epochs")
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig("train_val_loss.pdf")
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

    # Add a title (overlay in scene)
    mlab.title(title, size=0.5, height=0.95)

    mlab.savefig(fig_name)
    mlab.show()


def render_two_3d_scans(scan1, scan2, label1="Target", label2="Reconstruction",
                        title="3D Comparison", fig_name=""):
    from mayavi import mlab
    import numpy as np

    # Ensure scans are the same shape
    assert scan1.shape == scan2.shape, "Scans must have the same shape"
    sx, sy, sz = scan1.shape
    gap = 20

    # Define scale based on scan1
    vmin, vmax = np.min(scan1), np.max(scan1)

    # Clamp scan2 to scan1's range
    scan2_clamped = np.clip(scan2, vmin, vmax)

    # Combine both scans into a padded array
    combined = np.zeros((sx * 2 + gap, sy, sz), dtype=scan1.dtype)
    combined[:sx, :, :] = scan1
    combined[sx + gap:, :, :] = scan2_clamped

    # Create Mayavi figure
    mlab.figure(size=(1000, 800), bgcolor=(1, 1, 1))

    # Render volume
    src = mlab.pipeline.scalar_field(combined)
    vol = mlab.pipeline.volume(src)
    vol._volume_property.scalar_opacity_unit_distance = 0.1
    vol.module_manager.scalar_lut_manager.lut_mode = 'viridis'

    # Lock LUT to scan1’s scale
    vol.module_manager.scalar_lut_manager.data_range = (vmin, vmax)

    # Add vertical colorbar with black text
    cb = mlab.colorbar(title='Intensity', orientation='vertical')
    cb.label_text_property.font_size = 14
    cb.label_text_property.color = (0, 0, 0)
    cb.title_text_property.color = (0, 0, 0)

    # Add 3D labels
    label_z = sz + 10
    mlab.text3d(x=sx // 2, y=sy // 2, z=label_z,
                text=label1, scale=(10, 10, 10), color=(0, 0, 0))
    x_offset = sx + gap + (sx // 2)
    mlab.text3d(x=x_offset, y=sy // 2, z=label_z,
                text=label2, scale=(10, 10, 10), color=(0, 0, 0))

    # View settings
    mlab.view(azimuth=180, elevation=80, distance='auto')

    if fig_name:
        mlab.savefig(fig_name)

    mlab.show()

    # import napari
    # viewer = napari.Viewer()
    # viewer.add_image(scan1, name="Scan 1")
    # viewer.add_image(scan2, name="Scan 2")
    # napari.run()
    #
    # viewer = napari.Viewer(ndisplay=2)  # still works with 3D volumes
    #
    # # Add Scan 1 on the left
    # viewer.add_image(scan1, name="Scan 1", colormap="gray")
    #
    # # Add Scan 2 on the right (shifted in x for display)
    # import numpy as np
    # shift = scan1.shape[0] + 10  # gap between volumes
    # scan2_shifted = np.zeros((scan1.shape[0] * 2 + 10, scan1.shape[1], scan1.shape[2]))
    # scan2_shifted[:scan1.shape[0], :, :] = scan1
    # scan2_shifted[scan1.shape[0] + 10:, :, :] = scan2
    #
    # diff = scan1 - scan2
    # viewer.add_image(diff, name="Difference", colormap="bwr")
    #
    # napari.run()
