import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from utils import get_sample_descriptions


def create_3d_volumetric_visualization(diam1, intr1, diam2, intr2, diam3, intr3, save_path="3D_volumetric_pore_distribution.png"):
    """
    Create a 3D volumetric visualization of pore distribution in thermal insulating boards.
    Uses voxels to represent pore density throughout the volume of the boards.
    """
    # Create both vertical and horizontal orientation visualizations
    board_dimensions_vertical = (40, 40, 160)  # (width, depth, height)
    board_dimensions_horizontal = (160, 40, 40)  # (width, depth, height)

    # Generate vertical orientation
    fig_vertical = _create_volumetric_visualization(
        diam1, intr1, diam2, intr2, diam3, intr3,
        board_dimensions=board_dimensions_vertical,
        save_path=save_path
    )

    # Generate horizontal orientation with pores lying down
    horizontal_save_path = save_path.replace(
        ".png", "_horizontal.png") if save_path else None
    fig_horizontal = _create_volumetric_visualization(
        diam1, intr1, diam2, intr2, diam3, intr3,
        board_dimensions=board_dimensions_horizontal,
        save_path=horizontal_save_path,
        is_horizontal=True
    )

    return fig_vertical, fig_horizontal


def _create_volumetric_visualization(diam1, intr1, diam2, intr2, diam3, intr3,
                                     board_dimensions=(40, 40, 160),
                                     save_path=None,
                                     is_horizontal=False):
    """
    Helper function to create a 3D volumetric visualization with specific board dimensions.
    """
    board_width, board_depth, board_height = board_dimensions

    # Adjust figure size based on orientation
    figsize = (26, 10) if not is_horizontal else (28, 9)

    # Create figure with 3 subplots (side by side)
    fig = plt.figure(figsize=figsize)
    orientation_text = "Horizontal" if is_horizontal else "Vertical"
    fig.suptitle(
        f"3D Volumetric Visualization of Pore Distribution in Thermal Insulating Boards ({orientation_text})",
        fontsize=16)

    # Create subplot axes
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    # Set up the axes with board dimensions
    axes = [ax1, ax2, ax3]
    sample_names = ["T1", "T2", "T3"]
    descriptions = get_sample_descriptions()
    diameters_list = [diam1, diam2, diam3]
    intrusion_list = [intr1, intr2, intr3]
    cmaps = [plt.cm.Reds, plt.cm.Blues, plt.cm.Oranges]

    # Adjust resolution based on orientation to keep pores proportional
    if is_horizontal:
        # For horizontal orientation, elongate along x-axis
        res_x, res_y, res_z = 32, 16, 16
    else:
        # For vertical orientation, elongate along z-axis
        res_x, res_y, res_z = 16, 16, 32

    # Initialize voxel data array to store pore density values
    max_density = 3000  # Maximum pore density value

    # Process each sample
    for i, (ax, name, desc, diameters, intrusion, cmap) in enumerate(zip(
            axes, sample_names, descriptions, diameters_list, intrusion_list, cmaps)):

        print(
            f"\nGenerating volumetric data for {name} ({orientation_text})...")

        # Initialize voxel grid for this sample
        voxel_data = np.zeros((res_x, res_y, res_z))

        # Normalize log diameters and intrusion values
        log_diam = np.log10(diameters)
        min_log, max_log = np.min(log_diam), np.max(log_diam)

        # Generate pore density data based on actual distribution
        if is_horizontal:
            # For horizontal layout, distribute pores along the x-axis (width)
            for x in tqdm(range(res_x), desc=f"{name} voxel generation"):
                # Calculate position along board length (normalized 0-1)
                x_pos = x / res_x

                # Use different pore sizes along the length
                board_pos = x_pos
                idx = min(int(board_pos * len(diameters)), len(diameters)-1)

                base_density = intrusion[idx] / \
                    np.max(intrusion) * max_density * 1.25

                for y in range(res_y):
                    for z in range(res_z):
                        # Add variation based on distance from center of cross-section
                        dist_from_center = np.sqrt(
                            ((y/res_y) - 0.5)**2 + ((z/res_z) - 0.5)**2
                        ) * 2

                        # More porosity in center, less at edges
                        center_factor = 1.0 - 0.5 * dist_from_center

                        # Add some random variation
                        random_factor = 0.9 + 0.2 * np.random.random()

                        density = base_density * center_factor * random_factor
                        voxel_data[x, y, z] = max(0, density)
        else:
            # Original vertical orientation code
            for x in tqdm(range(res_x), desc=f"{name} voxel generation"):
                for y in range(res_y):
                    # Calculate distance from center
                    dist_from_center = np.sqrt(
                        ((x/res_x) - 0.5)**2 + ((y/res_y) - 0.5)**2
                    ) * 2

                    idx = min(int(dist_from_center * len(diameters)),
                              len(diameters)-1)
                    base_density = intrusion[idx] / \
                        np.max(intrusion) * max_density * 1.25

                    for z in range(res_z):
                        depth_factor = 1.0 - 0.4 * abs(2 * (z / res_z) - 1.0)
                        random_factor = 0.9 + 0.2 * np.random.random()
                        density = base_density * depth_factor * random_factor
                        voxel_data[x, y, z] = max(0, density)

        # Create voxel mask and colors
        density_threshold = max_density * 0.05
        voxel_mask = voxel_data > density_threshold

        norm_density = np.clip((voxel_data - density_threshold) /
                               (max_density - density_threshold), 0, 1)

        colors_3d = np.zeros(voxel_mask.shape + (4,))
        for ix in range(res_x):
            for iy in range(res_y):
                for iz in range(res_z):
                    if voxel_mask[ix, iy, iz]:
                        color = cmap(norm_density[ix, iy, iz])
                        colors_3d[ix, iy, iz, :3] = color[:3]
                        colors_3d[ix, iy, iz, 3] = min(1.0, color[3] * 1.2)

        # Plot the voxels with fixed colors
        ax.voxels(voxel_mask, facecolors=colors_3d, edgecolor=None)

        # Set labels and title with increased size
        ax.set_xlabel('Width (mm)', labelpad=15, fontsize=12)
        ax.set_ylabel('Depth (mm)', labelpad=15, fontsize=12)
        ax.set_zlabel('Height (mm)', labelpad=15, fontsize=12)
        ax.set_title(f"{name}: {desc}", fontsize=12)

        # Improve coordinate display with distinct tick marks and clearer labels
        # Create more space between tick labels
        if is_horizontal:
            # For horizontal orientation
            x_ticks = [0, 40, 80, 120, 160]
            y_ticks = [0, 10, 20, 30, 40]
            z_ticks = [0, 10, 20, 30, 40]
        else:
            # For vertical orientation
            x_ticks = [0, 10, 20, 30, 40]
            y_ticks = [0, 10, 20, 30, 40]
            z_ticks = [0, 40, 80, 120, 160]

        # Normalize ticks to 0-1 range for plotting
        ax.set_xticks(np.linspace(0, 1, len(x_ticks)))
        ax.set_xticklabels([str(x) for x in x_ticks],
                           fontsize=11, fontweight='bold')

        ax.set_yticks(np.linspace(0, 1, len(y_ticks)))
        ax.set_yticklabels([str(y) for y in y_ticks],
                           fontsize=11, fontweight='bold')

        ax.set_zticks(np.linspace(0, 1, len(z_ticks)))
        ax.set_zticklabels([str(z) for z in z_ticks],
                           fontsize=11, fontweight='bold')

        # Make tick marks more prominent
        ax.tick_params(axis='x', pad=7, labelsize=11,
                       width=2, length=5, colors='black')
        ax.tick_params(axis='y', pad=7, labelsize=11,
                       width=2, length=5, colors='black')
        ax.tick_params(axis='z', pad=7, labelsize=11,
                       width=2, length=5, colors='black')

        # Add a coordinate grid for better visibility
        ax.grid(True, linestyle='-', alpha=0.3, color='gray')

        # Set consistent viewing angle based on orientation
        if is_horizontal:
            # For horizontal view, show length along x-axis
            ax.view_init(elev=25, azim=30)
        else:
            # For vertical view, show height along z-axis
            ax.view_init(elev=30, azim=45)

        # Scale the axes for proper dimensions
        if is_horizontal:
            ax.set_box_aspect(
                [board_width/board_height, board_depth/board_height, 1])
        else:
            ax.set_box_aspect([1, 1, board_height/board_width])

    # Add a shared colorbar
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis, norm=plt.Normalize(0, max_density))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Pore Density', fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    # Use only subplots_adjust with carefully tuned values
    fig.subplots_adjust(left=0.05, bottom=0.12, right=0.88,
                        top=0.88, wspace=0.25, hspace=0.3)

    # Add dimensional information to the figure
    dimension_text = f"Board dimensions: {board_width}×{board_depth}×{board_height} mm"
    fig.text(0.5, 0.03, dimension_text, ha='center',
             fontsize=12, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)

    return fig
