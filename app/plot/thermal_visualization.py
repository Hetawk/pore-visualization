import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from utils import get_sample_descriptions


def visualize_thermal_board(ax, diameters, intrusion, cumulative, sample_name,
                            board_dimensions, description, cmap_name):
    """
    Create a visualization of a thermal board showing heat distribution

    Parameters:
    -----------
    ax : matplotlib 3D axis
        Axis to plot on
    diameters : array
        Pore diameters (nm)
    intrusion : array
        Intrusion values
    cumulative : array
        Cumulative intrusion values
    sample_name : str
        Sample name (e.g., "T1")
    board_dimensions : tuple
        Board dimensions in mm (width, depth, height)
    description : str
        Sample description
    cmap_name : str
        Colormap name

    Returns:
    --------
    surf : matplotlib surface
        Surface plot object
    """
    # Type hint to help IDE recognize 3D axes methods
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    assert isinstance(ax, Axes3D), "Expected a 3D axes"

    width, depth, height = board_dimensions
    cmap = plt.get_cmap(cmap_name)

    # Create board surfaces
    x = np.linspace(0, width, 50)
    y = np.linspace(0, depth, 50)
    z = np.linspace(0, height, 20)
    X, Y = np.meshgrid(x, y)

    # Create top surface with thermal pattern using both intrusion and cumulative data
    # Normalize our data for visualization
    norm_cum = cumulative / \
        np.max(cumulative) if np.max(cumulative) > 0 else cumulative
    norm_intr = intrusion / \
        np.max(intrusion) if np.max(intrusion) > 0 else intrusion

    # Create a thermal conductivity pattern on the top surface
    Z_top = np.zeros_like(X)

    # Calculate log pore size distribution to use for thermal mapping
    log_diam = np.log10(diameters)

    # Create a thermal pattern based on pore distribution patterns
    print(f"Generating thermal pattern for {sample_name}...")
    for i in tqdm(range(len(x)), desc="Processing x-axis"):
        for j in range(len(y)):
            # Create radial distance from center
            dist_from_center = np.sqrt(
                (X[j, i] - width/2)**2 + (Y[j, i] - depth/2)**2)
            dist_normalized = dist_from_center / \
                (np.sqrt(width**2 + depth**2)/2)

            # Map the distance to an index in our data
            idx = min(int(dist_normalized * len(diameters)), len(diameters)-1)

            # Thermal pattern is influenced by both intrusion and cumulative data
            # Smaller pores (center) create more insulation (less heat transfer)
            thermal_factor = 0.7 * norm_cum[idx] + 0.3 * norm_intr[idx]

            # Create a pattern where "warmer" regions represent higher thermal conductivity
            Z_top[j, i] = height + thermal_factor * height * 0.2 * (
                1 + 0.2 *
                np.sin(5 * np.arctan2(Y[j, i] - depth/2, X[j, i] - width/2))
            )

    # Plot the thermal surface with colormap indicating thermal conductivity
    surf = ax.plot_surface(X, Y, Z_top, cmap=cmap, edgecolor='none', alpha=0.8)

    # Add side faces of the board with thermal gradients
    # Front face (y=0)
    X_front, Z_front = np.meshgrid(x, z)
    Y_front = np.zeros_like(X_front)

    # Create gradient on front face
    thermal_front = np.zeros_like(X_front)
    print(f"Creating front face thermal gradient for {sample_name}...")
    for i in tqdm(range(len(x)), desc="Processing front face"):
        for k in range(len(z)):
            # Map x-position to diameter index
            x_normalized = x[i] / width
            idx = min(int(x_normalized * len(diameters)), len(diameters)-1)
            thermal_front[k, i] = norm_cum[idx] * \
                (1 + 0.3 * Z_front[k, i]/height)

    # Plot front face with thermal gradient
    front_surf = ax.plot_surface(X_front, Y_front, Z_front,
                                 facecolors=cmap(thermal_front), alpha=0.6)

    # Side face (x=0)
    Y_side, Z_side = np.meshgrid(y, z)
    X_side = np.zeros_like(Y_side)

    # Create gradient on side face
    thermal_side = np.zeros_like(Y_side)
    print(f"Creating side face thermal gradient for {sample_name}...")
    for j in tqdm(range(len(y)), desc="Processing side face"):
        for k in range(len(z)):
            # Map y-position to diameter index
            y_normalized = y[j] / depth
            idx = min(int(y_normalized * len(diameters)), len(diameters)-1)
            thermal_side[k, j] = norm_cum[idx] * \
                (1 + 0.3 * Z_side[k, j]/height)

    # Plot side face with thermal gradient
    side_surf = ax.plot_surface(X_side, Y_side, Z_side,
                                facecolors=cmap(thermal_side), alpha=0.6)

    # Add thermal indicator arrows showing heat flow
    # Heat flows more through regions with higher thermal conductivity
    print(f"Adding thermal flow indicators for {sample_name}...")
    arrow_count = 0
    total_arrows = 25  # 5x5 grid
    with tqdm(total=total_arrows, desc="Adding thermal arrows") as pbar:
        for i in range(5):
            for j in range(5):
                x_pos = width * (0.1 + 0.2 * i)
                y_pos = depth * (0.1 + 0.2 * j)

                # Map position to data index
                dist = np.sqrt((x_pos - width/2)**2 + (y_pos - depth/2)**2)
                dist_norm = dist / (np.sqrt(width**2 + depth**2)/2)
                idx = min(int(dist_norm * len(diameters)), len(diameters)-1)

                # Arrow length based on thermal conductivity (from cumulative data)
                arrow_len = 0.5 + 3.0 * norm_cum[idx]

                # Draw arrow from bottom to top (z-direction)
                ax.quiver(x_pos, y_pos, 0, 0, 0, arrow_len,
                          color=cmap(0.3 + 0.7 * norm_cum[idx]), alpha=0.8,
                          arrow_length_ratio=0.3)
                arrow_count += 1
                pbar.update(1)

    # Add colorbar
    cbar = plt.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label('Relative Thermal Conductivity')

    # Set labels and title
    ax.set_xlabel('Width (mm)')
    ax.set_ylabel('Depth (mm)')
    ax.set_zlabel('Height (mm)')
    ax.set_title(f"{sample_name}", fontsize=12)

    # Add sample description
    ax.text2D(0.05, 0.05, description, transform=ax.transAxes, fontsize=9,
              bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

    # Set limits
    ax.set_xlim(0, width)
    ax.set_ylim(0, depth)
    ax.set_zlim(0, height * 1.3)

    # Set viewing angle
    ax.view_init(elev=30, azim=-60)

    return surf


def plot_3d_thermal_boards(diam1, intr1, cum1, diam2, intr2, cum2, diam3, intr3, cum3, save_path="3D_thermal_boards.png"):
    """
    Create a 3D visualization specifically for thermal boards showing thermal conductivity properties

    Parameters:
    -----------
    diam1, intr1, cum1, diam2, intr2, cum2, diam3, intr3, cum3 : arrays
        Diameter, intrusion, and cumulative data for each sample
    save_path : str
        Path to save the figure

    Returns:
    --------
    fig : matplotlib figure
        Figure object
    """
    # Create figure with 3 panels - one for each sample (T1, T2, T3)
    fig = plt.figure(figsize=(18, 10))

    # Create subplots for the three samples
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    # Board dimensions
    board_dim = (40, 40, 10)  # width, depth, height in mm

    # Get sample descriptions
    descriptions = get_sample_descriptions()

    # Create thermal conductivity visualization for each board
    visualize_thermal_board(ax1, diam1, intr1, cum1,
                            "T1", board_dim, descriptions[0], 'Reds')
    visualize_thermal_board(ax2, diam2, intr2, cum2,
                            "T2", board_dim, descriptions[1], 'Blues')
    visualize_thermal_board(ax3, diam3, intr3, cum3,
                            "T3", board_dim, descriptions[2], 'Oranges')

    # Title and layout adjustments
    fig.suptitle(
        "Thermal Conductivity Visualization of Board Samples", fontsize=16)
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
