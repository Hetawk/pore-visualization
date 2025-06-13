import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from utils import get_sample_descriptions


def create_scientific_visualization(ax, diameters, intrusion_values, volume_size, sample_name, description, fixed_view=False):
    """
    Create a scientific visualization of board material with internal pores,
    similar to reference images from scientific papers, with a clean modern appearance.

    Parameters:
    -----------
    ax : matplotlib 3D axis
        Axis to plot on
    diameters : array
        Pore diameters (nm)
    intrusion_values : array
        Intrusion values
    volume_size : tuple
        Volume size in mm (width, depth, height)
    sample_name : str
        Sample name (e.g., "T1")
    description : str
        Sample description
    fixed_view : bool
        Whether to use a fixed viewing angle

    Returns:
    --------
    ax : matplotlib 3D axis
        Modified axis
    """
    width, depth, height = volume_size

    # Adjust cutaway to better show the board
    cutaway_x = width * 0.75
    cutaway_y = depth * 0.75
    cutaway_z = height * 0.85  # Increase height cutaway to make more of the top visible

    # Step 1: Create boundary outlines for the board shape
    wireframe_points = np.array([
        # Bottom face
        [0, 0, 0], [width, 0, 0], [width, depth, 0], [0, depth, 0], [0, 0, 0],
        # Connect to top
        [0, 0, height],
        # Top face
        [width, 0, height], [width, depth, height], [
            0, depth, height], [0, 0, height],
        # Remaining edges
        [width, 0, height], [width, 0, 0],
        [width, depth, 0], [width, depth, height],
        [0, depth, 0], [0, depth, height]
    ])

    # Plot wireframe with thin, light gray lines
    ax.plot(wireframe_points[:, 0], wireframe_points[:, 1], wireframe_points[:, 2],
            color='lightgray', linewidth=0.5, alpha=0.3)

    # Create a diagonal cutaway plane
    diagonal_plane_xs = np.linspace(0, cutaway_x, 20)
    diagonal_plane_ys = np.linspace(0, cutaway_y, 20)

    # Create a diagonal grid for the cutaway surface
    diag_X, diag_Y = np.meshgrid(diagonal_plane_xs, diagonal_plane_ys)
    # Create diagonal plane cutting through the volume
    diag_Z = cutaway_z * (1 - 0.7*diag_X/cutaway_x) * \
        (1 - 0.7*diag_Y/cutaway_y)
    # Clip to maximum height
    diag_Z = np.minimum(diag_Z, cutaway_z)

    # Add the diagonal cutaway plane
    ax.plot_surface(diag_X, diag_Y, diag_Z, color='whitesmoke', alpha=0.05)

    # Generate pores to fill the volume
    n_pores = 1200

    # Calculate percentage of each pore size from intrusion data
    norm_intrusion = intrusion_values / np.sum(intrusion_values)

    # Choose pore diameters based on experimental distribution
    indices = np.random.choice(len(diameters), size=n_pores, p=norm_intrusion)
    selected_diameters = diameters[indices]

    # Convert diameters to μm for visualization
    selected_diameters_um = selected_diameters / 1000.0  # nm to μm
    selected_radii = selected_diameters_um / 2.0

    # Scale radii for visualization (8-15 μm)
    min_radius, max_radius = 8.0, 15.0
    if np.max(selected_radii) != np.min(selected_radii):
        scaled_radii = min_radius + (max_radius - min_radius) * (
            selected_radii - np.min(selected_radii)) / (np.max(selected_radii) - np.min(selected_radii))
    else:
        scaled_radii = np.ones_like(
            selected_radii) * ((min_radius + max_radius) / 2)

    # Distribute pores
    pore_positions = []

    # Helper function to add a pore with small jitter
    def add_pore_at(x, y, z, jitter=3.0):
        jx = np.random.normal(0, jitter)
        jy = np.random.normal(0, jitter)
        jz = np.random.normal(0, jitter)
        # Ensure pore stays within bounds
        x = max(min(x + jx, width-2), 2)
        y = max(min(y + jy, depth-2), 2)
        z = max(min(z + jz, height-2), 2)
        return [x, y, z]

    # 1. Add pores in the upper region
    top_pores = int(n_pores * 0.4)
    print(f"Generating {top_pores} top pores for {sample_name}...")
    for _ in tqdm(range(top_pores), desc="Top pores"):
        # Use exponential distribution to concentrate toward top
        z_val = cutaway_z * (1 - np.random.exponential(0.3))
        # Ensure z_val is in the upper half
        z_val = min(cutaway_z, max(cutaway_z * 0.5, z_val))

        # Distribute across the x-y plane
        x = np.random.uniform(0, cutaway_x)
        y = np.random.uniform(0, cutaway_y)

        pore_positions.append([x, y, z_val])

    # 2. Add pores in a diagonal pattern
    diag_pores = int(n_pores * 0.25)
    print(f"Generating {diag_pores} diagonal pores for {sample_name}...")
    for i in tqdm(range(diag_pores), desc="Diagonal pores"):
        # Parametric position along diagonal (weighted toward top)
        t = np.random.beta(1.5, 1.0)

        # Create points along the diagonal
        x = cutaway_x * (1 - t * 0.9)
        y = cutaway_y * (1 - t * 0.9)
        z = cutaway_z * (1 - t * 0.5)

        pore_positions.append(add_pore_at(x, y, z, jitter=2.0))

    # 3. Add pores along the edges
    edge_pores = int(n_pores * 0.15)
    print(f"Generating {edge_pores} edge pores for {sample_name}...")
    for _ in tqdm(range(edge_pores), desc="Edge pores"):
        edge = np.random.choice(['top-x', 'top-y', 'corner'])

        if edge == 'top-x':
            x = np.random.uniform(0, cutaway_x)
            y = cutaway_y * (0.9 + 0.1 * np.random.random())
            z = cutaway_z * (0.8 + 0.2 * np.random.random())
        elif edge == 'top-y':
            x = cutaway_x * (0.9 + 0.1 * np.random.random())
            y = np.random.uniform(0, cutaway_y)
            z = cutaway_z * (0.8 + 0.2 * np.random.random())
        else:  # corner
            x = cutaway_x * (0.85 + 0.15 * np.random.random())
            y = cutaway_y * (0.85 + 0.15 * np.random.random())
            z = cutaway_z * (0.85 + 0.15 * np.random.random())

        pore_positions.append(add_pore_at(x, y, z, jitter=1.5))

    # 4. Add remaining pores throughout the volume
    remaining = n_pores - len(pore_positions)
    print(f"Generating {remaining} remaining pores for {sample_name}...")
    for _ in tqdm(range(remaining), desc="Remaining pores"):
        x = np.random.uniform(0, cutaway_x)
        y = np.random.uniform(0, cutaway_y)
        z = np.random.uniform(0, cutaway_z)
        pore_positions.append([x, y, z])

    pore_positions = np.array(pore_positions)

    # Set up camera position for better viewing
    camera_pos = np.array([width*1.5, width*1.5, height*1.5])

    # Define colormap and norm
    colormap = plt.get_cmap('jet')
    norm = colors.Normalize(vmin=8.0, vmax=15.0)

    # Sort pores by distance from camera with z-height bonus
    distances = np.linalg.norm(
        pore_positions - camera_pos.reshape(1, 3), axis=1)
    # Add a small modifier that keeps high-z pores more visible
    z_bonus = pore_positions[:, 2] / height * 0.2 * np.max(distances)
    adjusted_distances = distances - z_bonus

    # Sort indices based on adjusted distances
    sort_indices = np.argsort(-adjusted_distances)
    pore_positions = pore_positions[sort_indices]
    scaled_radii = scaled_radii[sort_indices]

    # Plot pores
    print(f"Rendering {len(pore_positions)} pores for {sample_name}...")
    for i in tqdm(range(len(pore_positions)), desc="Rendering pores"):
        radius = scaled_radii[i]
        color = colormap(norm(radius))

        # Create a sphere for each pore
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 15)
        x = pore_positions[i, 0] + radius * np.outer(np.cos(u), np.sin(v))
        y = pore_positions[i, 1] + radius * np.outer(np.sin(u), np.sin(v))
        z = pore_positions[i, 2] + radius * \
            np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(x, y, z, color=color, shade=True, alpha=1.0,
                        rstride=1, cstride=1, linewidth=0)

    # Set viewing angle
    if fixed_view:
        ax.view_init(elev=25, azim=-45)
    else:
        ax.view_init(elev=20, azim=-35)

    # Set aspect ratio to match board dimensions
    ax.set_box_aspect([1, 1, height/width])

    # Remove axis elements for clean visualization
    ax.set_axis_off()

    # Add invisible "bounding box" for proper scaling
    ax.plot([0, width], [0, 0], [0, 0], color='none')
    ax.plot([0, 0], [0, depth], [0, 0], color='none')
    ax.plot([0, 0], [0, 0], [0, height], color='none')

    # Add sample information
    ax.text2D(0.05, 0.02, sample_name, transform=ax.transAxes, fontsize=12,
              fontweight='bold', bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.3'))

    # Set background to white
    ax.set_facecolor('white')

    # Set limits with small margins
    ax.set_xlim(-5, width+5)
    ax.set_ylim(-5, depth+5)
    ax.set_zlim(-5, height+5)

    # Add dimension text
    ax.text2D(0.05, 0.95, f"{width}×{depth}×{height} mm", transform=ax.transAxes,
              fontsize=10, color='dimgray', alpha=0.7)

    return ax


def create_3d_pore_visualization(diam1, intr1, diam3, intr3, save_path="3D_pore_structure_scientific.png"):
    """
    Create 3D visualization of pore distribution in rectangular prism thermal boards.
    Shows realistic cross-sections of the boards with colored pores visible inside.

    Parameters:
    -----------
    diam1, intr1 : arrays
        Diameter and intrusion data for sample T1
    diam3, intr3 : arrays
        Diameter and intrusion data for sample T3
    save_path : str
        Path to save the figure

    Returns:
    --------
    fig : matplotlib figure
        Figure object
    """
    # Create figure with two subplots (side by side)
    fig = plt.figure(figsize=(16, 9))

    # Create subplots for T1 and T3 (the most different samples)
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Use actual board dimensions (40×40×160 mm)
    volume_size = (40, 40, 160)

    # Get sample descriptions
    descriptions = get_sample_descriptions()

    # Create material samples
    create_scientific_visualization(
        ax1, diam1, intr1, volume_size, "T1", descriptions[0], fixed_view=True)
    create_scientific_visualization(
        ax2, diam3, intr3, volume_size, "T3", descriptions[2], fixed_view=True)

    # Add a colorbar for both plots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    normalize = colors.Normalize(vmin=8.0, vmax=15.0)
    cmap = plt.get_cmap('jet')
    cbar = fig.colorbar(cm.ScalarMappable(
        norm=normalize, cmap=cmap), cax=cbar_ax)
    cbar.set_label('Pore Radius (μm)', rotation=270, labelpad=15)

    # Add subplot labels
    ax1.set_title("(a)", fontsize=14, loc='left')
    ax2.set_title("(b)", fontsize=14, loc='left')

    # Add main title
    fig.suptitle(
        "3D Visualization of Pore Structure in Thermal Insulating Boards", fontsize=14, y=0.98)

    # Adjust layout
    fig.subplots_adjust(left=0.02, right=0.90,
                        bottom=0.05, top=0.90, wspace=0.05)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
