#!/usr/bin/env python3
"""
3D Volumetric Pore Distribution Visualization
Specialized module for creating volumetric representations of pore distribution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from matplotlib import cm
from scipy.stats import lognorm
import random
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class VolumetricPoreVisualizer:
    """Class for creating 3D volumetric visualizations of pore distribution."""

    def __init__(self, show_progress=True):
        # Standard board dimensions in mm
        self.board_dimensions = (40, 40, 160)
        self.max_density = 3000
        self.show_progress = show_progress

    def _progress_wrapper(self, iterable, desc="Processing"):
        """Wrapper for tqdm that can be disabled in GUI threads."""
        if self.show_progress:
            return tqdm(iterable, desc=desc)
        else:
            return iterable

    def load_and_process_data(self, file_path):
        """Load and process CSV data for visualization."""
        try:
            # Read every line of the file
            with open(file_path, 'r') as f:
                all_lines = f.readlines()

            # Keep only those lines whose first field can be parsed as a float
            clean_lines = []
            for line in all_lines:
                stripped = line.strip()
                if stripped == "":
                    continue
                first_token = stripped.split(',')[0]
                try:
                    _ = float(first_token)
                    clean_lines.append(line)
                except ValueError:
                    continue

            # Join lines and parse with pandas
            csv_data = "".join(clean_lines)
            # Rename columns for clarity - handle different column counts
            df = pd.read_csv(StringIO(csv_data), header=None)
            if len(df.columns) == 9:
                df.columns = [
                    "diam_T1", "int_T1", "cond_T1",
                    "diam_T2", "int_T2", "cond_T2",
                    "diam_T3", "int_T3", "cond_T3"
                ]
            elif len(df.columns) == 6:
                df.columns = [
                    "diam_T1", "int_T1",
                    "diam_T2", "int_T2",
                    "diam_T3", "int_T3"
                ]
            else:
                logger.error(
                    f"Unexpected number of columns: {len(df.columns)}")
                raise ValueError(
                    f"Expected 6 or 9 columns, got {len(df.columns)}")

            # Force everything to numeric and drop any leftover NaNs
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Only check for required columns (diameter and intrusion)
            required_cols = ["diam_T1", "int_T1",
                             "diam_T2", "int_T2", "diam_T3", "int_T3"]
            df = df.dropna(subset=required_cols)

            # Extract and sort each curve
            samples = {}
            for i, sample in enumerate(['T1', 'T2', 'T3'], 1):
                diam_col = f"diam_T{i}"
                int_col = f"int_T{i}"

                diameters = df[diam_col].values
                intrusion = df[int_col].values

                # Sort by diameter
                idx = np.argsort(diameters)
                samples[sample] = {
                    'diameters': diameters[idx],
                    'intrusion': intrusion[idx]
                }

            logger.info(f"Successfully processed data from {file_path}")
            return samples

        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            raise

    def load_processed_dataframe(self, df):
        """Process an already loaded pandas DataFrame for visualization."""
        try:
            # Handle different column counts
            if len(df.columns) == 9:
                df.columns = [
                    "diam_T1", "int_T1", "cond_T1",
                    "diam_T2", "int_T2", "cond_T2",
                    "diam_T3", "int_T3", "cond_T3"
                ]
            elif len(df.columns) == 6:
                df.columns = [
                    "diam_T1", "int_T1",
                    "diam_T2", "int_T2",
                    "diam_T3", "int_T3"
                ]
            else:
                logger.error(
                    f"Unexpected number of columns: {len(df.columns)}")
                raise ValueError(
                    f"Expected 6 or 9 columns, got {len(df.columns)}")

            # Force everything to numeric and drop any leftover NaNs
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Only check for required columns (diameter and intrusion)
            required_cols = ["diam_T1", "int_T1",
                             "diam_T2", "int_T2", "diam_T3", "int_T3"]
            df = df.dropna(subset=required_cols)

            # Extract and sort each curve
            samples = {}
            for i, sample in enumerate(['T1', 'T2', 'T3'], 1):
                diam_col = f"diam_T{i}"
                int_col = f"int_T{i}"

                diameters = df[diam_col].values
                intrusion = df[int_col].values

                # Sort by diameter
                idx = np.argsort(diameters)
                samples[sample] = {
                    'diameters': diameters[idx],
                    'intrusion': intrusion[idx]
                }

            logger.info(
                "Successfully processed DataFrame for volumetric visualization")
            return samples

        except Exception as e:
            logger.error(f"Failed to process DataFrame: {e}")
            raise

    def create_3d_volumetric_visualization(self, samples_data):
        """
        Create a 3D volumetric visualization of pore distribution in thermal insulating boards.
        Uses voxels to represent pore density throughout the volume of the boards.
        """        # Create figure with 3 subplots (side by side) - adjusted for GUI compatibility
        fig = plt.figure(figsize=(15, 6))
        fig.suptitle(
            "3D Volumetric Visualization of Pore Distribution in Thermal Insulating Boards",
            fontsize=12
        )

        # Define board dimensions (40×40×160 mm)
        board_width, board_depth, board_height = self.board_dimensions

        # Create subplot axes
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')

        # Set up the axes with board dimensions
        axes = [ax1, ax2, ax3]
        sample_names = ["T1", "T2", "T3"]
        descriptions = [
            "CSA cement with expanded vermiculite",
            "CSA cement with expanded vermiculite and rice husk ash",
            "CSA cement with vermiculite, rice husk ash and bamboo fiber"
        ]
        cmaps = [plt.cm.Reds, plt.cm.Blues, plt.cm.Oranges]

        # Set resolution for the voxel grid (lower for performance)
        res_x, res_y, res_z = 20, 20, 40

        # Process each sample
        for i, (ax, name, desc, cmap) in enumerate(zip(
                axes, sample_names, descriptions, cmaps)):

            logger.info(f"Generating volumetric data for {name}...")

            # Get sample data
            diameters = samples_data[name]['diameters']
            intrusion = samples_data[name]['intrusion']

            # Initialize voxel grid for this sample
            voxel_data = np.zeros((res_x, res_y, res_z))

            # Normalize log diameters and intrusion values
            log_diam = np.log10(diameters)
            min_log, max_log = np.min(log_diam), np.max(log_diam)

            # Generate pore density data based on actual distribution
            for x in self._progress_wrapper(range(res_x), desc=f"{name} voxel generation"):
                for y in range(res_y):
                    # Calculate distance from center
                    dist_from_center = np.sqrt(
                        ((x/res_x) - 0.5)**2 + ((y/res_y) - 0.5)**2
                    ) * 2  # Normalize to 0-1 range

                    # Map the distance to an index in our data
                    # Closer to center = smaller pores, edge = larger pores
                    idx = min(int(dist_from_center * len(diameters)),
                              len(diameters)-1)

                    # Use intrusion value to determine density profile along z-axis
                    # Higher intrusion = more pores
                    base_density = intrusion[idx] / \
                        np.max(intrusion) * self.max_density

                    # Create depth profile with some natural variation
                    for z in range(res_z):
                        # Add depth variation - more pores toward the edges
                        depth_factor = 1.0 - 0.5 * abs(2 * (z / res_z) - 1.0)

                        # Add some random variation for realistic appearance
                        random_factor = 0.8 + 0.4 * np.random.random()

                        # Calculate final density value
                        density = base_density * depth_factor * random_factor

                        # Ensure positive values
                        voxel_data[x, y, z] = max(0, density)

            # Create boolean mask for filled voxels with density threshold
            voxel_mask = voxel_data > 0  # All non-zero values get a voxel

            # Normalize density values for color mapping (0-1 range)
            norm_density = voxel_data / self.max_density

            # Create a 3D array of RGBA colors
            colors_3d = np.zeros(voxel_mask.shape + (4,))
            for ix in range(res_x):
                for iy in range(res_y):
                    for iz in range(res_z):
                        if voxel_mask[ix, iy, iz]:
                            # Apply the right colormap for each sample
                            colors_3d[ix, iy, iz] = cmap(
                                norm_density[ix, iy, iz])

            # Plot the voxels with fixed colors
            ax.voxels(voxel_mask, facecolors=colors_3d, edgecolor=None)

            # Set labels and title
            ax.set_xlabel('Width (mm)', labelpad=15)
            ax.set_ylabel('Depth (mm)', labelpad=15)
            ax.set_zlabel('Height (mm)', labelpad=10)
            ax.set_title(f"{name}: {desc}", fontsize=11)

            # Set explicit ticks for better spacing
            ax.set_xticks(np.linspace(0, board_width, 5))
            ax.set_yticks(np.linspace(0, board_depth, 5))
            ax.zaxis.set_major_locator(plt.MaxNLocator(nbins=6))

            # Adjust tick label padding
            ax.tick_params(axis='x', pad=7)
            ax.tick_params(axis='y', pad=7)
            ax.tick_params(axis='z', pad=5)

            # Set consistent viewing angle
            ax.view_init(elev=30, azim=45)

            # Remove grid for cleaner appearance
            ax.grid(False)

            # Scale the axes for proper dimensions
            ax.set_box_aspect([1, 1, board_height/board_width])

        # Add a shared colorbar
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis, norm=plt.Normalize(0, self.max_density))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        # Adjust layout for better GUI spacing and compatibility
        cbar.set_label('Pore Density')
        try:
            plt.tight_layout()
        except UserWarning:
            # If tight_layout fails, continue with manual spacing
            pass
        fig.subplots_adjust(left=0.08, bottom=0.12, right=0.92,
                            top=0.88, wspace=0.20, hspace=0.25)

        logger.info("3D volumetric visualization created successfully")
        return fig

    def create_single_sample_volumetric_view(self, sample_data, sample_name, description):
        """Create a volumetric visualization for a single sample."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Get data
        diameters = sample_data['diameters']
        intrusion = sample_data['intrusion']

        # Set resolution
        res_x, res_y, res_z = 25, 25, 50
        board_width, board_depth, board_height = self.board_dimensions

        # Initialize voxel grid
        voxel_data = np.zeros((res_x, res_y, res_z))

        # Generate density data
        log_diam = np.log10(diameters)
        min_log, max_log = np.min(log_diam), np.max(log_diam)

        logger.info(
            f"Generating detailed volumetric data for {sample_name}...")

        for x in self._progress_wrapper(range(res_x), desc=f"{sample_name} detailed generation"):
            for y in range(res_y):
                # Distance from center
                dist_from_center = np.sqrt(
                    ((x/res_x) - 0.5)**2 + ((y/res_y) - 0.5)**2
                ) * 2

                # Map to data index
                idx = min(int(dist_from_center * len(diameters)),
                          len(diameters)-1)
                base_density = intrusion[idx] / \
                    np.max(intrusion) * self.max_density

                for z in range(res_z):
                    depth_factor = 1.0 - 0.5 * abs(2 * (z / res_z) - 1.0)
                    random_factor = 0.8 + 0.4 * np.random.random()
                    density = base_density * depth_factor * random_factor
                    voxel_data[x, y, z] = max(0, density)

        # Create visualization
        voxel_mask = voxel_data > 0
        norm_density = voxel_data / self.max_density

        # Use viridis colormap for single sample
        cmap = plt.cm.viridis
        colors_3d = np.zeros(voxel_mask.shape + (4,))

        for ix in range(res_x):
            for iy in range(res_y):
                for iz in range(res_z):
                    if voxel_mask[ix, iy, iz]:
                        colors_3d[ix, iy, iz] = cmap(norm_density[ix, iy, iz])

        # Plot voxels
        ax.voxels(voxel_mask, facecolors=colors_3d, edgecolor=None)

        # Styling
        ax.set_xlabel('Width (mm)', labelpad=15)
        ax.set_ylabel('Depth (mm)', labelpad=15)
        ax.set_zlabel('Height (mm)', labelpad=10)
        ax.set_title(f"{sample_name}: 3D Volumetric Pore Distribution\n{description}",
                     fontsize=14)

        ax.view_init(elev=25, azim=45)
        ax.grid(False)
        ax.set_box_aspect([1, 1, board_height/board_width]
                          )        # Add colorbar
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(0, self.max_density))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
        cbar.set_label('Pore Density')

        # Improve layout for GUI compatibility
        try:
            plt.tight_layout()
        except UserWarning:
            # If tight_layout fails, continue with manual spacing
            pass
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.85, top=0.9)
        return fig


def create_volumetric_pore_visualization(data_path, sample_type='all'):
    """
    Main function to create 3D volumetric pore distribution visualization.

    Args:
        data_path (str): Path to the CSV data file
        sample_type (str): 'all' for all samples, 'T1', 'T2', or 'T3' for specific sample

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Detect if we're running in a QThread to disable progress bars
    show_progress = True
    try:
        from PyQt5.QtCore import QThread
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app and QThread.currentThread() != app.thread():
            show_progress = False
            print(
                "VolumetricVisualizer: Running in background thread, disabling progress bars")
    except (ImportError, AttributeError):
        pass  # Not in PyQt environment, keep progress bars enabled

    visualizer = VolumetricPoreVisualizer(show_progress=show_progress)

    try:
        # Load and process data
        samples_data = visualizer.load_and_process_data(data_path)

        if sample_type == 'all':
            # Create visualization for all samples
            return visualizer.create_3d_volumetric_visualization(samples_data)
        elif sample_type in ['T1', 'T2', 'T3']:
            # Create single sample visualization
            descriptions = {
                'T1': "CSA cement with expanded vermiculite",
                'T2': "CSA cement with expanded vermiculite and rice husk ash",
                'T3': "CSA cement with vermiculite, rice husk ash and bamboo fiber"
            }
            return visualizer.create_single_sample_volumetric_view(
                samples_data[sample_type], sample_type, descriptions[sample_type])
        else:
            raise ValueError(
                f"Invalid sample_type: {sample_type}. Use 'all', 'T1', 'T2', or 'T3'")

    except Exception as e:
        logger.error(f"Failed to create volumetric visualization: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    data_path = "../dataset/pore_data.csv"
    fig = create_volumetric_pore_visualization(data_path, 'all')
    plt.show()
