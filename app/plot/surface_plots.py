import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from utils import get_pore_cutoff_values


def create_3d_surface_model(diameters, values, resolution=50, board_dim=(40, 40),
                            title="Pore Distribution", cmap='coolwarm', zlabel=""):
    """
    Create a 3D surface model visualization of pore distribution.

    Parameters:
    -----------
    diameters : array
        Pore diameters (nm)
    values : array
        Values corresponding to diameters (intrusion or cumulative)
    resolution : int
        Resolution of the grid
    board_dim : tuple
        Board dimensions in mm (width, height)
    title : str
        Title for the plot
    cmap : str
        Colormap to use
    zlabel : str
        Label for z-axis

    Returns:
    --------
    fig, ax : matplotlib figure and axis
    """
    # Create a regular grid covering the board dimensions
    x = np.linspace(0, board_dim[0], resolution)
    y = np.linspace(0, board_dim[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Create Z values based on pore distribution
    center_x, center_y = board_dim[0]/2, board_dim[1]/2
    dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2) / \
        np.sqrt(center_x**2 + center_y**2)

    # First normalize the log of diameters to 0-1 range
    log_diam = np.log10(diameters)
    min_log = np.min(log_diam)
    max_log = np.max(log_diam)
    norm_log_diam = (log_diam - min_log) / (max_log - min_log)

    # Create a Z value that changes based on the distribution pattern
    Z = np.zeros_like(X)

    # Generate the surface
    for i in range(resolution):
        for j in range(resolution):
            r = dist[i, j]
            theta = np.arctan2(Y[i, j] - center_y, X[i, j] - center_x)
            idx = int(r * (len(diameters) - 1))
            Z[i, j] = values[idx] * (0.8 + 0.2 * np.sin(5 * theta))

            # Add some fine details based on the local region
            region = int(3 * r)
            if region < 3:
                # Center region - emphasize smaller pores
                Z[i, j] *= 1.0 + 0.2 * np.cos(10 * norm_log_diam[idx])
            else:
                # Outer region - emphasize larger pores
                Z[i, j] *= 1.0 + 0.2 * np.sin(5 * norm_log_diam[idx])

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=0.8)

    # Add color bar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Relative Porosity')

    # Add contour plot projection on the bottom
    offset = np.min(Z) - 0.1 * (np.max(Z) - np.min(Z))
    cset = ax.contourf(X, Y, Z, zdir='z', offset=offset, cmap=cmap, alpha=0.5)

    # Add labels and title
    ax.set_xlabel('Width (mm)')
    ax.set_ylabel('Height (mm)')
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    # Set limits
    ax.set_xlim(0, board_dim[0])
    ax.set_ylim(0, board_dim[1])

    # Add annotations for mesopores, macropores
    ax.text(board_dim[0]*0.8, board_dim[1]*0.2, np.max(Z),
            "Mesopores\n(<100 nm)", color='black', fontsize=9)
    ax.text(board_dim[0]*0.2, board_dim[1]*0.8, np.max(Z),
            "Macropores\n(>2000 nm)", color='black', fontsize=9)

    return fig, ax


def create_differential_surface(ax, diameters, intrusion, title, cmap):
    """
    Helper function to create a differential intrusion surface on an existing axis

    Parameters:
    -----------
    ax : matplotlib 3D axis
        Axis to plot on
    diameters : array
        Pore diameters (nm)
    intrusion : array
        Intrusion values
    title : str
        Plot title
    cmap : str
        Colormap to use

    Returns:
    --------
    surf : matplotlib surface
        Surface plot object
    """
    # Create a regular grid
    resolution = 50
    x = np.linspace(0, 40, resolution)
    y = np.linspace(0, 40, resolution)
    X, Y = np.meshgrid(x, y)

    # Create Z values based on the intrusion distribution
    Z = np.zeros_like(X)

    # Normalize log diameters
    log_diam = np.log10(diameters)
    min_log, max_log = np.min(log_diam), np.max(log_diam)
    norm_log_diam = (log_diam - min_log) / (max_log - min_log)

    # Create distance matrix from center
    center_x, center_y = 20, 20
    dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2) / \
        28.28  # normalized 0-1

    # Map the 1D intrusion data to a 2D surface
    print(f"Creating {title} surface...")
    for i in tqdm(range(resolution), desc="Processing rows"):
        for j in range(resolution):
            # Create a pattern where height varies with distance and angle
            r = dist[i, j]
            theta = np.arctan2(Y[i, j] - center_y, X[i, j] - center_x)

            # Find the index in the pore data to use based on normalized distance
            idx = int(r * (len(diameters) - 1))

            # Modulate Z using the intrusion value and some trigonometric functions
            Z[i, j] = intrusion[idx] * (0.8 + 0.2 * np.sin(5 * theta))

            # Add some fine details based on the local region
            region = int(3 * r)
            if region < 3:
                # Center region - emphasize smaller pores
                Z[i, j] *= 1.0 + 0.2 * np.cos(10 * norm_log_diam[idx])
            else:
                # Outer region - emphasize larger pores
                Z[i, j] *= 1.0 + 0.2 * np.sin(5 * norm_log_diam[idx])

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=0.8)

    # Add contour plot at the bottom for better visibility
    offset = np.min(Z) - 0.1 * (np.max(Z) - np.min(Z))
    ax.contourf(X, Y, Z, zdir='z', offset=offset, cmap=cmap, alpha=0.5)

    # Add color bar
    plt.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1,
                 label='Log Differential Intrusion (mL/g)')

    # Add pore size boundary indicators
    cutoffs = get_pore_cutoff_values()
    theta = np.linspace(0, 2*np.pi, 100)

    # For mesopores (<100 nm) boundary
    r_meso = np.interp(np.log10(cutoffs['mesopores']), [
                       min_log, max_log], [0.1, 0.9]) * 20
    h_min, h_max = offset, np.max(Z) + 0.1

    x_circle = center_x + r_meso * np.cos(theta)
    y_circle = center_y + r_meso * np.sin(theta)

    # Plot circles at top and bottom
    ax.plot(x_circle, y_circle, [h_min] *
            len(theta), 'k--', alpha=0.5, linewidth=1)
    ax.plot(x_circle, y_circle, [h_max] *
            len(theta), 'k--', alpha=0.5, linewidth=1)

    # For macropores (>2000 nm) boundary
    r_macro = np.interp(np.log10(cutoffs['macropores']), [
                        min_log, max_log], [0.1, 0.9]) * 20

    x_circle = center_x + r_macro * np.cos(theta)
    y_circle = center_y + r_macro * np.sin(theta)

    # Plot circles at top and bottom
    ax.plot(x_circle, y_circle, [h_min] *
            len(theta), 'k--', alpha=0.5, linewidth=1)
    ax.plot(x_circle, y_circle, [h_max] *
            len(theta), 'k--', alpha=0.5, linewidth=1)

    # Add text annotations for pore regions
    ax.text(center_x, center_y, h_max + 0.1, "Mesopores\n(<100 nm)",
            ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

    mid_r = (r_meso + r_macro) / 2
    ax.text(center_x + mid_r*0.7, center_y, h_max + 0.1,
            "Larger mesopores/\nsmaller macropores",
            ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

    ax.text(center_x + r_macro*1.2, center_y, h_max + 0.1, "Macropores\n(>2000 nm)",
            ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

    # Set labels and title
    ax.set_xlabel('Width (mm)')
    ax.set_ylabel('Height (mm)')
    ax.set_zlabel('Log Differential Intrusion (mL/g)')
    ax.set_title(title)

    # Set limits
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)

    # Adjust view angle
    ax.view_init(elev=30, azim=45)

    return surf


def create_comparative_surface(ax, title, diam1, intr1, diam2, intr2, diam3, intr3):
    """
    Create a comparative surface showing all three samples together

    Parameters:
    -----------
    ax : matplotlib 3D axis
        Axis to plot on
    title : str
        Plot title
    diam1, intr1, diam2, intr2, diam3, intr3 : arrays
        Diameter and intrusion data for each sample

    Returns:
    --------
    tuple : (surf1, surf2, surf3)
        Surface plot objects
    """
    resolution = 50
    x = np.linspace(0, 40, resolution)
    y = np.linspace(0, 40, resolution)
    X, Y = np.meshgrid(x, y)

    # Create three Z-surfaces, one for each board
    Z1 = np.zeros_like(X)
    Z2 = np.zeros_like(X)
    Z3 = np.zeros_like(X)

    # Normalize log diameters
    log_diam1 = np.log10(diam1)
    log_diam2 = np.log10(diam2)
    log_diam3 = np.log10(diam3)

    min_log = min(np.min(log_diam1), np.min(log_diam2), np.min(log_diam3))
    max_log = max(np.max(log_diam1), np.max(log_diam2), np.max(log_diam3))

    # Create distance matrix from center
    center_x, center_y = 20, 20
    dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2) / 28.28

    # Process each board
    print(f"Creating comparative surface...")
    for i in tqdm(range(resolution), desc="Processing rows"):
        for j in range(resolution):
            r = dist[i, j]

            # Convert distance to a log diameter between min_log and max_log
            # Small r (center) = small pores, large r (edge) = large pores
            target_log_diam = min_log + r * (max_log - min_log)

            # Find closest diameter index for each board
            idx1 = np.argmin(np.abs(log_diam1 - target_log_diam))
            idx2 = np.argmin(np.abs(log_diam2 - target_log_diam))
            idx3 = np.argmin(np.abs(log_diam3 - target_log_diam))

            # Set Z values with some spatial variation
            theta = np.arctan2(Y[i, j] - center_y, X[i, j] - center_x)
            var = 0.8 + 0.2 * np.sin(5 * theta + r * np.pi)

            Z1[i, j] = intr1[idx1] * var
            Z2[i, j] = intr2[idx2] * var
            Z3[i, j] = intr3[idx3] * var

    # Calculate a common z-offset for stacking
    h_offset = 0.2
    max_z = max(np.max(Z1), np.max(Z2), np.max(Z3))

    # Plot surfaces with a vertical offset to see all three
    surf1 = ax.plot_surface(X, Y, Z1, cmap='Reds', alpha=0.7, label='T1')
    surf2 = ax.plot_surface(X, Y, Z2 + max_z*h_offset,
                            cmap='Blues', alpha=0.7, label='T2')
    surf3 = ax.plot_surface(X, Y, Z3 + 2*max_z*h_offset,
                            cmap='Oranges', alpha=0.7, label='T3')

    # Add labels for each surface
    ax.text(40, 20, np.max(Z1), "T1", color='red',
            fontsize=12, fontweight='bold')
    ax.text(40, 20, np.max(Z2) + max_z*h_offset, "T2",
            color='blue', fontsize=12, fontweight='bold')
    ax.text(40, 20, np.max(Z3) + 2*max_z*h_offset, "T3",
            color='orange', fontsize=12, fontweight='bold')

    # Set labels and title
    ax.set_xlabel('Width (mm)')
    ax.set_ylabel('Height (mm)')
    ax.set_zlabel('Log Differential Intrusion (mL/g)')
    ax.set_title(title)

    # Set limits
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)

    # Adjust view angle
    ax.view_init(elev=30, azim=45)

    return surf1, surf2, surf3


def plot_3d_differential_surfaces(diam1, intr1, diam2, intr2, diam3, intr3, save_path="3D_differential_intrusion.png"):
    """
    Create 3D surface plots for Log Differential Intrusion for all three samples

    Parameters:
    -----------
    diam1, intr1, diam2, intr2, diam3, intr3 : arrays
        Diameter and intrusion data for each sample
    save_path : str
        Path to save the figure

    Returns:
    --------
    fig : matplotlib figure
        Figure object
    """
    # Create 2x2 subplot layout
    fig = plt.figure(figsize=(18, 16))

    # Plot T1 (top left)
    ax1 = fig.add_subplot(221, projection='3d')
    create_differential_surface(
        ax1, diam1, intr1, "T1: Log Differential Intrusion", 'Reds')

    # Plot T2 (top right)
    ax2 = fig.add_subplot(222, projection='3d')
    create_differential_surface(
        ax2, diam2, intr2, "T2: Log Differential Intrusion", 'Blues')

    # Plot T3 (bottom left)
    ax3 = fig.add_subplot(223, projection='3d')
    create_differential_surface(
        ax3, diam3, intr3, "T3: Log Differential Intrusion", 'Oranges')

    # Create merged view (bottom right)
    ax4 = fig.add_subplot(224, projection='3d')
    create_comparative_surface(ax4, "Comparative Log Differential Intrusion",
                               diam1, intr1, diam2, intr2, diam3, intr3)

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.suptitle(
        "3D Visualization of Pore Size Distribution in Thermal Boards", fontsize=16)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
