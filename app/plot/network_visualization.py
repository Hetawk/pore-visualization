#!/usr/bin/env python3
"""
Module for creating 3D network-like visualizations of pore distributions.
This represents pores as atoms with chemical-bond-like connections in a 3D space,
mimicking molecular structures.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.cm as cm
from matplotlib import colors


def create_3d_pore_network(diam1, intr1, diam2, intr2, diam3, intr3, output_file):
    """
    Create a 3D atom-bond-like visualization of pore distributions for three samples.

    Parameters:
    -----------
    diam1, diam2, diam3 : arrays
        Pore diameter arrays for the three samples
    intr1, intr2, intr3 : arrays
        Intrusion values for the three samples
    output_file : str
        File path to save the visualization

    Returns:
    --------
    None
        The visualization is saved to output_file
    """
    # Define more professional atom-like colors similar to the reference image
    board_colors = {
        'T1': '#1F77B4',  # Blue
        'T2': '#2C3E50',  # Dark blue
        'T3': '#3498DB'   # Light blue
    }

    # Define a more professional color scheme for atom types
    atom_colors = {
        'small': '#1F77B4',    # Standard blue (most common)
        'medium': '#FF7F0E',   # Orange (highlight)
        'large': '#2CA02C',    # Green (highlight)
        'xlarge': '#D62728'    # Red (rare/important)
    }

    # Create output file path derivatives for individual plots
    base_name = output_file.rsplit('.', 1)[0]  # Remove extension
    ext = output_file.rsplit('.', 1)[1] if '.' in output_file else 'png'
    t1_output = f"{base_name}_T1.{ext}"
    t2_output = f"{base_name}_T2.{ext}"
    t3_output = f"{base_name}_T3.{ext}"
    combined_output = f"{base_name}_combined.{ext}"

    # Modified function to draw prism frame with 40×40×160 mm proportions - horizontal orientation
    def plot_box_frame(ax, color='#FF8C00', linewidth=1.5, alpha=0.8):
        # Define the 8 corners of the prism with 40×40×160 mm proportions (4:1 length:width ratio)
        # Longest dimension (160mm) is now along the x-axis (horizontal)
        corners = np.array([
            [-4, -1, -1],  # Back face
            [-4, 1, -1],
            [-4, 1, 1],
            [-4, -1, 1],
            [4, -1, -1],   # Front face
            [4, 1, -1],
            [4, 1, 1],
            [4, -1, 1]
        ])

        # Define the 12 edges of the prism
        edges = [(0, 1), (1, 2), (2, 3), (3, 0),  # Back face
                 (4, 5), (5, 6), (6, 7), (7, 4),  # Front face
                 (0, 4), (1, 5), (2, 6), (3, 7)]  # Connecting edges

        # Plot the edges with increased line width for better visibility
        for edge in edges:
            ax.plot([corners[edge[0], 0], corners[edge[1], 0]],
                    [corners[edge[0], 1], corners[edge[1], 1]],
                    [corners[edge[0], 2], corners[edge[1], 2]],
                    color=color, linewidth=linewidth, alpha=alpha)

        # Set axis limits to ensure the horizontal prism is fully visible with some margin
        ax.set_xlim(-4.5, 4.5)  # Wider x-axis for the longest dimension
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-1.2, 1.2)

    # Modified function to convert pore data to network nodes in a horizontal prism shape with increased density
    # Increased from 60 to 100 for higher density
    def create_molecular_network(diam, intr, n_samples=100):
        # Sample the data if there are too many points
        if len(diam) > n_samples:
            indices = np.linspace(0, len(diam)-1, n_samples, dtype=int)
            diam_sample = diam[indices]
            intr_sample = intr[indices]
        else:
            diam_sample = diam
            intr_sample = intr

        # Create a graph
        G = nx.Graph()

        # Create node positions based on pore characteristics
        # Using log scale for diameter for better visualization
        log_diam = np.log10(diam_sample)

        # Normalize values for positioning
        norm_log_diam = (log_diam - np.min(log_diam)) / \
            (np.max(log_diam) - np.min(log_diam))
        norm_intr = intr_sample / np.max(intr_sample)

        # Categorize pores by size to assign "atom types"
        atom_types = []
        for d in diam_sample:
            if d < 100:  # Mesopores
                atom_types.append('small')
            elif d < 2000:  # Larger mesopores/smaller macropores
                atom_types.append('medium')
            elif d < 50000:  # Medium macropores
                atom_types.append('large')
            else:  # Large macropores
                atom_types.append('xlarge')

        # Create positions optimized for horizontal prism shape
        positions = []
        existing_positions = []  # Track positions to avoid overlap

        # Generate more seed positions distributed throughout the horizontal prism volume
        # Increased from 15 to 25 for better distribution
        num_seeds = min(25, len(diam_sample)//4)
        seed_positions = []

        # Create seed positions throughout the horizontal prism volume with tighter bounds
        # to ensure nothing extends outside the orange box
        for _ in range(num_seeds):
            # Use tighter bounds to account for node size, ensuring nodes don't extend outside
            x = np.random.uniform(-3.6, 3.6)  # Tighter bound from -3.8
            y = np.random.uniform(-0.8, 0.8)  # Tighter bound from -0.9
            z = np.random.uniform(-0.8, 0.8)  # Tighter bound from -0.9
            seed_positions.append(np.array([x, y, z]))

        # Reduced minimum distance to allow nodes to be closer together
        min_distance = 0.2  # Reduced from 0.3 to 0.2 for higher density

        # Function to check if a position is too close to existing positions
        def is_too_close(pos, existing_pos, min_dist):
            if not existing_pos:
                return False
            for ep in existing_pos:
                if np.linalg.norm(pos - ep) < min_dist:
                    return True
            return False

        # Now create all positions
        for i in range(len(diam_sample)):
            if i < num_seeds:
                # Use seed positions for first few nodes
                positions.append(seed_positions[i])
                existing_positions.append(seed_positions[i])
            else:
                # Position based on nearest seed position but with variation
                attempts = 0
                pos_found = False

                while not pos_found and attempts < 50:
                    # Choose a random seed as starting point
                    closest_seed_idx = np.random.randint(
                        0, len(seed_positions))
                    seed_pos = seed_positions[closest_seed_idx]

                    # Use normalized diameter to influence positioning
                    diam_factor = norm_log_diam[i]

                    # Random position within an ellipsoid around the seed (stretched in x-direction)
                    # Reduced radii for tighter packing
                    radius_yz = 0.2 + 0.5 * diam_factor  # Reduced slightly for higher density
                    radius_x = 0.7 + 1.5 * diam_factor  # Reduced slightly for higher density

                    theta = np.random.uniform(0, 2*np.pi)
                    phi = np.random.uniform(0, np.pi)

                    # Adjust for horizontal prism proportions - longest dimension along x-axis
                    # Stretched in x-direction
                    x = seed_pos[0] + radius_x * np.sin(phi) * np.cos(theta)
                    y = seed_pos[1] + radius_yz * np.sin(phi) * np.sin(theta)
                    z = seed_pos[2] + radius_yz * np.cos(phi)

                    # Use tighter bounds to ensure nodes (including their visual size)
                    # stay within the orange box
                    x = max(-3.6, min(3.6, x))  # Tighter bounds for x-axis
                    y = max(-0.8, min(0.8, y))  # Tighter bounds for y-axis
                    z = max(-0.8, min(0.8, z))  # Tighter bounds for z-axis

                    new_pos = np.array([x, y, z])

                    # Check if this position is too close to existing ones
                    if not is_too_close(new_pos, existing_positions, min_distance):
                        positions.append(new_pos)
                        existing_positions.append(new_pos)
                        pos_found = True

                    attempts += 1

                # If we couldn't find a non-overlapping position, use the last attempted one
                # but adjust its minimum distance
                if not pos_found:
                    positions.append(new_pos)
                    existing_positions.append(new_pos)

        # Add additional boundary check to reposition any nodes that might still be too close to the edge
        max_node_size_factor = 0  # Track the maximum node size factor

        # First pass: determine the maximum node size factor
        for i in range(len(diam_sample)):
            size_factor = 30 + 70 * norm_intr[i]  # Larger contrast in size
            size_factor *= 0.7  # Reduced from 0.8 to 0.7 for higher density
            max_node_size_factor = max(max_node_size_factor, size_factor)

        # Calculate a safety margin based on node size (in data units)
        # Convert from scatter point size to approximate data units
        safety_margin = np.sqrt(max_node_size_factor) * 0.02

        # Second pass: check and reposition nodes that are too close to the boundary
        for i in range(len(positions)):
            pos = positions[i]

            # Check if too close to x boundaries
            if pos[0] < -4 + safety_margin:
                positions[i][0] = -4 + safety_margin
            elif pos[0] > 4 - safety_margin:
                positions[i][0] = 4 - safety_margin

            # Check if too close to y boundaries
            if pos[1] < -1 + safety_margin:
                positions[i][1] = -1 + safety_margin
            elif pos[1] > 1 - safety_margin:
                positions[i][1] = 1 - safety_margin

            # Check if too close to z boundaries
            if pos[2] < -1 + safety_margin:
                positions[i][2] = -1 + safety_margin
            elif pos[2] > 1 - safety_margin:
                positions[i][2] = 1 - safety_margin

        # Add nodes with atom-like attributes
        for i in range(len(diam_sample)):
            size_factor = 30 + 70 * norm_intr[i]  # Larger contrast in size

            # Scale down node size to prevent visual overlap with increased density
            size_factor *= 0.7  # Reduced from 0.8 to 0.7 for higher density

            G.add_node(i,
                       pos=positions[i],
                       diameter=diam_sample[i],
                       intrusion=intr_sample[i],
                       size=size_factor,
                       atom_type=atom_types[i])

        # Connect nodes to form a molecular structure
        # First, ensure a connected graph by connecting sequential nodes
        for i in range(1, len(diam_sample)):
            G.add_edge(i-1, i, bond_type='single')

        # Add additional "bonds" based on spatial proximity and pore characteristics
        # but limit to reduce visual clutter
        max_bonds_per_node = 7  # Increased from 5 to 7 for more connections
        # Start with 1 for sequential bonds
        bond_counts = {i: 1 for i in range(1, len(diam_sample))}
        bond_counts[0] = 1

        for i in range(len(diam_sample)):
            if bond_counts[i] >= max_bonds_per_node:
                continue

            pos_i = positions[i]

            # Find nearby nodes to create bond-like connections
            for j in range(i+1, len(diam_sample)):
                if bond_counts[j] >= max_bonds_per_node:
                    continue

                pos_j = positions[j]

                # Calculate distance
                dist = np.linalg.norm(pos_i - pos_j)

                # Create different bond types based on distance and pore similarity
                if dist < 0.8:  # Close proximity = potential bond
                    # Calculate similarity in pore characteristics
                    diam_similarity = abs(norm_log_diam[i] - norm_log_diam[j])
                    intr_similarity = abs(norm_intr[i] - norm_intr[j])

                    # Decide bond type based on similarities
                    if diam_similarity < 0.1 and intr_similarity < 0.1:
                        # Very similar pores get triple bonds
                        G.add_edge(i, j, bond_type='triple', weight=3)
                        bond_counts[i] += 1
                        bond_counts[j] += 1
                    elif diam_similarity < 0.2 and intr_similarity < 0.2:
                        # Somewhat similar pores get double bonds
                        G.add_edge(i, j, bond_type='double', weight=2)
                        bond_counts[i] += 1
                        bond_counts[j] += 1
                    elif dist < 0.6 and bond_counts[i] < max_bonds_per_node and bond_counts[j] < max_bonds_per_node:
                        # Very close nodes get at least a single bond
                        G.add_edge(i, j, bond_type='single', weight=1)
                        bond_counts[i] += 1
                        bond_counts[j] += 1

                # Exit early if this node has enough bonds
                if bond_counts[i] >= max_bonds_per_node:
                    break

        return G, diam_sample, intr_sample, atom_types

    # Function to draw molecular-like network with professional appearance
    def draw_molecular_network(ax, diam, intr, color_base, title):
        # Increase sample size to show more nodes
        # Increased from 150 to 250 for higher density
        n_samples = min(250, len(diam))
        G, diam_sample, intr_sample, atom_types = create_molecular_network(
            diam, intr, n_samples)

        # Remove all default axes elements (background, grid, etc.)
        ax.set_axis_off()  # Turn off the default matplotlib axis

        # Make background transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # Remove grid
        ax.grid(False)

        # Draw box frame similar to reference image (keep only this orange frame)
        box_color = '#FF8C00'  # Orange box outline like in reference
        plot_box_frame(ax, box_color)

        # Set up color mapping based on intrusion values
        norm = Normalize(vmin=np.min(intr_sample), vmax=np.max(intr_sample))

        # Draw bonds/connections with reduced density
        for u, v, data in G.edges(data=True):
            x1, y1, z1 = G.nodes[u]['pos']
            x2, y2, z2 = G.nodes[v]['pos']

            # Use darker and more subtle color for connections
            line_color = '#666666'  # Dark gray for all connections
            line_alpha = 0.25        # More transparent
            line_width = 0.25        # Thinner lines

            # Draw simple, thin connections
            ax.plot([x1, x2], [y1, y2], [z1, z2],
                    color=line_color, alpha=line_alpha, linewidth=line_width)

        # Draw atoms (nodes) with adjusted sizes for higher density
        for node in G.nodes():
            x, y, z = G.nodes[node]['pos']
            atom_type = G.nodes[node]['atom_type']

            # Make most nodes blue like the reference, with occasional highlights
            if atom_type == 'small':  # Most common type
                node_color = atom_colors['small']  # Blue
                # Further reduce size to ensure they stay within bounds
                node_size = 15 + 25 * \
                    (G.nodes[node]['intrusion'] / np.max(intr_sample))
            else:
                # Highlighted nodes (less common)
                node_color = atom_colors[atom_type]
                # Further reduce size to ensure they stay within bounds
                node_size = 30 + 45 * \
                    (G.nodes[node]['intrusion'] / np.max(intr_sample))

            # Remove the black edge by setting edgecolor to 'none'
            ax.scatter(x, y, z, color=node_color, s=node_size,
                       edgecolor='none', alpha=0.85)

        # Set title and labels - cleaner, more professional font
        ax.set_title(title, fontsize=12, color='#333333', weight='bold')
        ax.set_xlabel('X', fontsize=9, color='#333333')
        ax.set_ylabel('Y', fontsize=9, color='#333333')
        ax.set_zlabel('Z', fontsize=9, color='#333333')

        # Remove ticks for cleaner visualization
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Set equal aspect ratio for height and depth, but allow x-axis to stretch
        # This maintains the 160×40×40 proportions (horizontal)
        ax.set_box_aspect([4, 1, 1])  # 4× wider than tall

        # Set viewpoint to better show the horizontal prism shape
        ax.view_init(elev=25, azim=30)

        return G

    # Modified function for individual visualizations with proper horizontal prism shape
    def create_individual_visualization(diam, intr, color_base, title, output_file):
        """Create an individual visualization for a single thermal board"""
        # Use figure size that maintains the 160×40×40 proportions (horizontal)
        fig = plt.figure(figsize=(12, 8))  # Wider figure for horizontal layout
        ax = fig.add_subplot(111, projection='3d')

        # Draw the molecular network on this axis
        draw_molecular_network(ax, diam, intr, color_base, title)

        # Add information about this sample with horizontal dimensions
        info_text = f"Visualization of pore networks in {title} thermal insulating board (160×40×40 mm).\nNode size and color represent pore characteristics."
        plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Individual visualization saved to {output_file}")
        plt.close()

    # Modified function for combined visualization with increased density
    def create_combined_visualization(diam1, intr1, diam2, intr2, diam3, intr3, output_file):
        """Create a visualization with all samples in one orange box"""
        # Use figure size appropriate for the 160×40×40 proportions (horizontal)
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Turn off axis and make background transparent
        ax.set_axis_off()
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(False)

        # Draw box frame
        box_color = '#FF8C00'
        plot_box_frame(ax, box_color)

        # Create networks for each sample but keep nodes smaller
        # so they don't overwhelm the visualization
        # Use more nodes for each sample in combined view
        # Increased from 80 to 120 for higher density
        n_samples = min(120, len(diam1))

        # Redefine positioning to avoid overlap in the prism shape
        # Smaller scale for better separation
        def get_positioned_graph(diam, intr, offset, scale=0.65):
            G, diam_sample, intr_sample, atom_types = create_molecular_network(
                diam, intr, n_samples)

            # Apply offset to each node position to separate the samples
            # Apply different scaling to z-axis to maintain prism proportions
            for node in G.nodes():
                pos = G.nodes[node]['pos']
                # Scale x,y by standard scale but scale z separately to maintain proportions
                scaled_pos = np.array(
                    [pos[0] * scale, pos[1] * scale, pos[2] * scale])
                G.nodes[node]['pos'] = scaled_pos + np.array(offset)

            return G, diam_sample, intr_sample, atom_types

        # Position each sample in different regions with good separation in the horizontal prism
        # Position samples along the x-axis (the longest dimension)
        G1, diam1_sample, intr1_sample, atom_types1 = get_positioned_graph(
            diam1, intr1, [-2.5, -0.6, -0.6])
        G2, diam2_sample, intr2_sample, atom_types2 = get_positioned_graph(
            diam2, intr2, [0, -0.6, 0.6])
        G3, diam3_sample, intr3_sample, atom_types3 = get_positioned_graph(
            diam3, intr3, [2.5, 0.6, 0])

        # Draw each network with distinct colors
        # Draw connections
        def draw_connections(G, color, alpha=0.2, width=0.2):
            for u, v in G.edges():
                x1, y1, z1 = G.nodes[u]['pos']
                x2, y2, z2 = G.nodes[v]['pos']
                ax.plot([x1, x2], [y1, y2], [z1, z2],
                        color=color, alpha=alpha, linewidth=width)

        draw_connections(G1, board_colors['T1'])
        draw_connections(G2, board_colors['T2'])
        draw_connections(G3, board_colors['T3'])

        # Draw nodes
        def draw_nodes(G, atom_types, atom_colors, color_label):
            for node in G.nodes():
                x, y, z = G.nodes[node]['pos']
                atom_type = atom_types[node]
                intrusion = G.nodes[node]['intrusion']
                max_intr = np.max([G.nodes[n]['intrusion'] for n in G.nodes()])

                # Use smaller sizes in combined view
                if atom_type == 'small':
                    node_color = atom_colors['small']
                    node_size = 15 + 25 * (intrusion / max_intr)
                else:
                    node_color = atom_colors[atom_type]
                    node_size = 25 + 40 * (intrusion / max_intr)

                ax.scatter(x, y, z, color=node_color, s=node_size,
                           edgecolor='none', alpha=0.85)

        draw_nodes(G1, atom_types1, atom_colors, 'T1')
        draw_nodes(G2, atom_types2, atom_colors, 'T2')
        draw_nodes(G3, atom_types3, atom_colors, 'T3')

        # Set title and create legend
        ax.set_title('Combined 3D Pore Network Visualization',
                     fontsize=14, color='#333333', weight='bold')

        # Set viewpoint and aspect ratio for horizontal prism
        ax.set_box_aspect([4, 1, 1])  # 4× wider than tall
        ax.view_init(elev=25, azim=30)

        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=board_colors['T1'],
                       label='T$_1$: Vermiculite + CSA Cement', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=board_colors['T2'],
                       label='T$_2$: Vermiculite + Rice Husk + Bamboo', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=board_colors['T3'],
                       label='T$_3$: High Rice Husk + High Bamboo', markersize=10)
        ]

        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

        # Add information text including horizontal dimensions
        info_text = ("Combined visualization of pore networks across all three thermal insulating boards (160×40×40 mm).\n"
                     "Node size represents intrusion value, node color represents the thermal board type.")
        plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Combined visualization saved to {output_file}")
        plt.close()

    # 1. Original visualization with 3 subplots - modified for horizontal prism shape
    # Set up the figure with 3 subplots for each sample
    fig = plt.figure(figsize=(18, 8))  # Wider figure for horizontal prisms

    # Create a subplot for each thermal board
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    # Draw each molecular network
    G1 = draw_molecular_network(ax1, diam1, intr1, board_colors['T1'], 'T1')
    G2 = draw_molecular_network(ax2, diam2, intr2, board_colors['T2'], 'T2')
    G3 = draw_molecular_network(ax3, diam3, intr3, board_colors['T3'], 'T3')

    # Add an overall title - more professional
    plt.suptitle('3D Pore Network Visualization (160×40×40 mm Horizontal Prisms)',
                 fontsize=14, fontweight='bold', color='#333333', y=0.98)

    # Simplify and clean up legends
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=atom_colors['small'],
                   label='Mesopores (<100nm)', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=atom_colors['medium'],
                   label='Mesopores/Macropores (100-2000nm)', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=atom_colors['large'],
                   label='Macropores (2000-50000nm)', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=atom_colors['xlarge'],
                   label='Large Macropores (>50000nm)', markersize=8)
    ]

    # Add single legend at the bottom of the figure
    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, 0.01), ncol=4, fontsize=9)

    # Add information about the samples with horizontal dimensions
    info_text = (
        "Visualization of pore networks in thermal insulating boards (160×40×40 mm).\n"
        "Node size and color represent pore characteristics, with connections showing relationships between pores."
    )
    plt.figtext(0.5, 0.08, info_text, ha='center', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

    # More professional layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.2)

    # Save the original 3-subplot visualization
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"3D network visualization saved to {output_file}")
    plt.close()

    # 2-4. Create individual visualizations for each thermal board
    print("Creating individual visualizations...")
    create_individual_visualization(
        diam1, intr1, board_colors['T1'], 'T1', t1_output)
    create_individual_visualization(
        diam2, intr2, board_colors['T2'], 'T2', t2_output)
    create_individual_visualization(
        diam3, intr3, board_colors['T3'], 'T3', t3_output)

    # 5. Create combined visualization with all samples in one orange box
    print("Creating combined visualization...")
    create_combined_visualization(
        diam1, intr1, diam2, intr2, diam3, intr3, combined_output)

    print("All visualizations completed successfully!")
