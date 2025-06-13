#!/usr/bin/env python3
"""
Enhanced Visualization Engine with MIST-like capabilities
Supports multiple visualization types including DEM particles, live rendering, and advanced analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Any, Optional, Tuple, List
import logging
import gc
import warnings

# Import modular components
try:
    from .dem_visualizer import DEMParticleVisualizer
    from .live_renderer import Live3DRenderer, LiveSegmentationProcessor
    from .mist_analyzer import MISTAnalyzer
    DEM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some advanced modules not available: {e}")
    DEM_AVAILABLE = False


class VisualizationEngine:
    """Enhanced visualization engine with advanced 3D capabilities."""

    def __init__(self):
        """Initialize the visualization engine."""
        self.parameters = self.get_default_parameters()
        self.logger = logging.getLogger(__name__)
        self.parameter_change_callbacks = []  # For real-time updates
        self.last_generated_figure = None
        self.current_data_path = None
        self._open_figures = []  # Track open figures for memory management

        # Configure matplotlib for memory efficiency
        self._configure_matplotlib_for_memory_efficiency()

    def get_default_parameters(self):
        """Get default visualization parameters."""
        return {
            'num_spheres': 1000,
            'opacity': 0.7,
            'colormap': 'viridis',
            'size_multiplier': 3.0,  # Increased default size for better visibility
            'prism_color': 'orange',
            'sphere_colors': ['blue', 'red', 'green', 'purple'],
            'show_bonds': True,
            'bond_thickness': 0.5,
            'lighting_intensity': 0.8,
            'background_color': 'white',
            'prism_opacity': 0.3,
            'show_coordinate_system': True,
            'view_angle': {'elevation': 20, 'azimuth': 45},
            # Axis dimension controls
            'axis_x_scale': 1.0,  # Scale factor for X-axis
            'axis_y_scale': 1.0,  # Scale factor for Y-axis
            'axis_z_scale': 1.0,  # Scale factor for Z-axis
            # Visualization space dimensions (actual size controls)
            'space_width': 20.0,    # Total width of visualization space
            'space_height': 20.0,   # Total height of visualization space
            'space_depth': 20.0,    # Total depth of visualization space
            # Aspect ratio controls
            'maintain_aspect_ratio': True,  # Keep proportional dimensions
            # [width:height:depth] ratio
            'custom_aspect_ratio': [1.0, 1.0, 1.0],
            # Axis boundaries (calculated from dimensions)
            'axis_bounds': {'x': [-10, 10], 'y': [-10, 10], 'z': [-10, 10]},
            'sphere_base_size': 100.0,  # Increased base sphere size in plot units
            # Increased min/max
            'sphere_size_range': [50.0, 300.0],
            # Enhanced visual parameters
            # 'scientific', 'artistic', 'rainbow', 'thermal', 'depth'
            'color_scheme': 'scientific',
            'sphere_style': 'glossy',  # 'glossy', 'matte', 'metallic', 'glass'
            'bond_style': 'tubes',  # 'lines', 'tubes', 'cylinders'
            'lighting_model': 'enhanced',  # 'basic', 'enhanced', 'dramatic'
            'depth_cueing': True,  # Distance-based opacity
            'edge_enhancement': True,  # Edge highlighting
            'size_variance': 0.2,  # Random size variation
            'animation_mode': False,  # Rotation animation
            'surface_rendering': False,  # Surface mesh
            'volumetric_effects': False,  # Fog effects
            'measurement_tools': True,  # Show measurements
            'scientific_annotations': True,  # Scientific labels
            # Figure settings
            'figure_size': (12, 9),  # Default figure size
            'show_connections': False,  # Show connections between points
        }

    def update_parameters(self, new_params: Dict[str, Any]):
        """Update visualization parameters and trigger callbacks for real-time rendering."""
        old_params = self.parameters.copy()
        self.parameters.update(new_params)

        # Check if dimension-related parameters changed
        dimension_params = ['space_width', 'space_height', 'space_depth',
                            'axis_x_scale', 'axis_y_scale', 'axis_z_scale',
                            'maintain_aspect_ratio', 'custom_aspect_ratio']

        if any(param in new_params for param in dimension_params):
            # Recalculate axis bounds when dimension parameters change
            self._calculate_axis_bounds_from_dimensions()

        self.logger.debug(f"Updated parameters: {new_params}")

        # Trigger real-time callbacks
        for callback in self.parameter_change_callbacks:
            try:
                callback(new_params, old_params)
            except Exception as e:
                self.logger.error(f"Error in parameter change callback: {e}")

    def add_parameter_change_callback(self, callback):
        """Add a callback function to be called when parameters change."""
        self.parameter_change_callbacks.append(callback)

    def remove_parameter_change_callback(self, callback):
        """Remove a parameter change callback."""
        if callback in self.parameter_change_callbacks:
            self.parameter_change_callbacks.remove(callback)

    def set_current_data_path(self, data_path: str):
        """Set the current data path for auto-refresh functionality."""
        self.current_data_path = data_path

    def _configure_matplotlib_for_memory_efficiency(self):
        """Configure matplotlib settings for memory efficiency."""
        # Limit the number of figures to prevent memory issues
        plt.rcParams['figure.max_open_warning'] = 10

        # Suppress specific matplotlib warnings about too many figures
        warnings.filterwarnings(
            'ignore', 'More than \d+ figures have been opened', UserWarning)

        # Set reasonable figure defaults to reduce memory usage
        plt.rcParams['figure.dpi'] = 80  # Lower DPI for memory efficiency
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'

    def _manage_figure_memory(self, new_figure: Figure) -> Figure:
        """Manage figure memory by tracking and cleaning up old figures."""
        # Add new figure to tracking list
        self._open_figures.append(new_figure)

        # Limit number of open figures to prevent memory issues
        max_figures = 5
        while len(self._open_figures) > max_figures:
            old_figure = self._open_figures.pop(0)
            try:
                plt.close(old_figure)
                del old_figure
            except Exception as e:
                self.logger.debug(f"Error closing old figure: {e}")

        # Force garbage collection to free memory
        gc.collect()

        return new_figure

    def cleanup_figures(self):
        """Clean up all tracked figures to free memory."""
        for fig in self._open_figures:
            try:
                plt.close(fig)
            except Exception as e:
                self.logger.debug(f"Error closing figure during cleanup: {e}")

        self._open_figures.clear()

        # Close any remaining matplotlib figures
        plt.close('all')

        # Force garbage collection
        gc.collect()

        self.logger.debug("Figure cleanup completed")

    def __del__(self):
        """Cleanup figures when the visualization engine is destroyed."""
        try:
            self.cleanup_figures()
        except Exception as e:
            # Avoid raising exceptions in destructor
            pass

    def refresh_visualization(self, visualization_type: str = 'enhanced') -> Optional[Figure]:
        """Refresh the current visualization with updated parameters."""
        if self.current_data_path:
            try:
                if visualization_type in ['ultra_realistic', 'scientific_analysis', 'presentation', 'cross_section']:
                    fig = self.create_advanced_visualization(
                        self.current_data_path, visualization_type)
                elif visualization_type in ['size_distribution', 'clustering_analysis']:
                    fig = self.create_analysis_visualization(
                        self.current_data_path, visualization_type)
                else:
                    fig = self.create_pore_network_visualization(
                        self.current_data_path, visualization_type)
                self.last_generated_figure = fig
                return fig
            except Exception as e:
                self.logger.error(f"Error refreshing visualization: {e}")
                return None
        return None

    def create_pore_network_visualization(self, data_path: str, visualization_type: str = 'enhanced') -> Figure:
        """
        Create pore network visualization with current parameters.

        Args:
            data_path: Path to pore data file
            visualization_type: Type of visualization ('enhanced', 'clean', 'thermal', 'sectioned')

        Returns:
            matplotlib Figure object
        """
        try:
            # Store current data path for real-time updates
            self.set_current_data_path(data_path)

            # Load data
            df = pd.read_csv(data_path)

            # Create figure
            fig = plt.figure(figsize=(12, 9))
            fig = self._manage_figure_memory(fig)
            ax = fig.add_subplot(111, projection='3d')

            # Generate or use existing coordinates
            if all(col in df.columns for col in ['X', 'Y', 'Z']):
                # Use existing coordinates, but remove any NaN values
                coord_data = df[['X', 'Y', 'Z']].dropna()
                x, y, z = coord_data['X'].values, coord_data['Y'].values, coord_data['Z'].values
                n_points = len(x)
            else:
                # Generate random 3D coordinates
                n_points = min(len(df), self.parameters['num_spheres'])
                x = np.random.uniform(-10, 10, n_points)
                y = np.random.uniform(-10, 10, n_points)
                z = np.random.uniform(-10, 10, n_points)

            # Apply axis scaling
            x *= self.parameters['axis_x_scale']
            y *= self.parameters['axis_y_scale']
            z *= self.parameters['axis_z_scale']

            # Generate sphere sizes with increased base size - ensure same length as coordinates
            base_size = self.parameters['sphere_base_size']
            if 'Pore_Radius' in df.columns:
                # Use only as many radius values as we have coordinates, remove NaN values
                pore_radius = df['Pore_Radius'].dropna().values
                # Ensure we have exactly n_points values by resampling or padding
                if len(pore_radius) >= n_points:
                    # Sample exactly n_points values
                    indices = np.random.choice(
                        len(pore_radius), n_points, replace=False)
                    pore_radius = pore_radius[indices]
                elif len(pore_radius) > 0:
                    # Pad with resampled values to reach n_points
                    mean_radius = np.mean(pore_radius)
                    additional_needed = n_points - len(pore_radius)
                    additional_radii = np.random.choice(
                        pore_radius, additional_needed, replace=True)
                    pore_radius = np.concatenate(
                        [pore_radius, additional_radii])
                else:
                    # No valid data, use default values
                    pore_radius = np.random.uniform(0.5, 2.0, n_points)

                # Ensure exactly n_points elements
                pore_radius = pore_radius[:n_points]
                sizes = pore_radius * base_size * \
                    self.parameters['size_multiplier']
            else:
                sizes = np.random.uniform(
                    self.parameters['sphere_size_range'][0],
                    self.parameters['sphere_size_range'][1],
                    n_points
                )

            # Add size variance - ensure consistent length
            if self.parameters['size_variance'] > 0:
                variance = np.random.normal(
                    1.0, self.parameters['size_variance'], n_points)
                sizes *= np.abs(variance)

            # Create color mapping - ensure same length as coordinates
            # Generate values for colormap rather than explicit colors
            color_values = self._get_color_values(n_points)

            # Ensure all arrays have consistent lengths before plotting
            x, y, z, sizes, color_values = self._ensure_consistent_arrays(
                x, y, z, sizes, color_values)

            # Create main scatter plot with increased sizes
            # Use numerical values with colormap for proper color scheme support
            scatter = ax.scatter(x, y, z,
                                 c=color_values,
                                 s=sizes,
                                 alpha=self.parameters['opacity'],
                                 cmap=self.parameters['colormap'],
                                 edgecolors='black' if self.parameters['edge_enhancement'] else None,
                                 linewidth=0.5 if self.parameters['edge_enhancement'] else 0)

            # Add bonds if enabled
            if self.parameters['show_bonds']:
                self._add_bonds(ax, x, y, z)

            # Apply visualization type specific enhancements
            if visualization_type == 'sectioned':
                self._create_sectioned_view(ax, x, y, z)
            elif visualization_type == 'thermal':
                self._add_thermal_effects(ax, x, y, z)

            # Set enhanced styling
            self._apply_enhanced_styling(ax, fig)

            # Add scientific annotations if enabled
            if self.parameters['scientific_annotations']:
                self._add_scientific_annotations(ax, len(x))

            return fig

        except Exception as e:
            self.logger.error(f"Error creating visualization: {e}")
            # Return a simple error figure
            fig = plt.figure(figsize=(10, 8))
            fig = self._manage_figure_memory(fig)
            ax = fig.add_subplot(111, projection='3d')
            ax.text(0, 0, 0, f"Error: {str(e)}", fontsize=12, ha='center')
            return fig

    def create_advanced_visualization(self, data_path: str, mode: str) -> Figure:
        """
        Create advanced visualization with new rendering modes.

        Args:
            data_path: Path to pore data file
            mode: Advanced mode ('ultra_realistic', 'scientific_analysis', 'presentation', 'cross_section')

        Returns:
            matplotlib Figure object
        """
        try:
            # Store current data path for real-time updates
            self.set_current_data_path(data_path)

            if mode == 'ultra_realistic':
                return self._create_ultra_realistic_view(data_path)
            elif mode == 'scientific_analysis':
                return self._create_scientific_analysis_view(data_path)
            elif mode == 'presentation':
                return self._create_presentation_view(data_path)
            elif mode == 'cross_section':
                return self._create_cross_section_view(data_path)
            else:
                # Fallback to enhanced visualization
                return self.create_pore_network_visualization(data_path, 'enhanced')

        except Exception as e:
            self.logger.error(f"Error creating advanced visualization: {e}")
            return self.create_pore_network_visualization(data_path, 'enhanced')

    def create_analysis_visualization(self, data_path: str, mode: str) -> Figure:
        """
        Create analysis visualization with specific analysis modes.

        Args:
            data_path: Path to pore data file
            mode: Analysis mode ('size_distribution', 'clustering_analysis')

        Returns:
            matplotlib Figure object
        """
        try:
            # Store current data path for real-time updates
            self.set_current_data_path(data_path)

            if mode == 'size_distribution':
                return self._create_size_distribution_view(data_path)
            elif mode == 'clustering_analysis':
                return self._create_clustering_analysis_view(data_path)
            else:
                # Fallback to enhanced visualization
                return self.create_pore_network_visualization(data_path, 'enhanced')

        except Exception as e:
            from core.logger import get_logger
            logger = get_logger()
            if logger:
                logger.log_exception(e, "create_analysis_visualization")
            self.logger.error(f"Error creating analysis visualization: {e}")
            return self.create_pore_network_visualization(data_path, 'enhanced')

    def create_dem_visualization(self, data_path: str, particle_type: str = 'mixed') -> Figure:
        """
        Create DEM particle visualization using the DEM visualizer module.

        Args:
            data_path: Path to particle data
            particle_type: Type of particles ('spherical', 'cubic', 'mixed')

        Returns:
            matplotlib Figure object
        """
        try:
            if not DEM_AVAILABLE:
                # Fallback to standard visualization if DEM module not available
                self.logger.warning(
                    "DEM visualizer not available, using standard visualization")
                return self.create_pore_network_visualization(data_path, 'enhanced')

            # Create DEM visualizer with current parameters
            dem_visualizer = DEMParticleVisualizer(self.parameters)

            # Create DEM visualization
            fig = dem_visualizer.create_dem_visualization(
                data_path, particle_type)
            fig = self._manage_figure_memory(fig)

            self.set_current_data_path(data_path)
            self.last_generated_figure = fig

            return fig

        except Exception as e:
            self.logger.error(f"Error creating DEM visualization: {e}")
            # Fallback to standard visualization
            return self.create_pore_network_visualization(data_path, 'enhanced')

    def start_live_rendering(self, data_path: str) -> Optional[Figure]:
        """
        Start live 3D rendering with real-time updates.

        Args:
            data_path: Path to data file

        Returns:
            Figure object for live rendering
        """
        try:
            if not DEM_AVAILABLE:
                self.logger.warning("Live renderer not available")
                return None

            # Load initial data
            df = pd.read_csv(data_path)

            # Prepare initial data for live renderer
            n_points = min(len(df), self.parameters['num_spheres'])

            if all(col in df.columns for col in ['X', 'Y', 'Z']):
                coord_data = df[['X', 'Y', 'Z']].dropna()[:n_points]
                positions = coord_data.values
            else:
                positions = np.random.uniform(-10, 10, (n_points, 3))

            # Generate sizes
            if 'Pore_Radius' in df.columns:
                sizes = df['Pore_Radius'].dropna().values[:n_points] * 100
            else:
                sizes = np.random.uniform(50, 200, n_points)

            # Prepare data for live renderer
            initial_data = {
                'type': 'pore_network',
                'positions': positions,
                'sizes': sizes,
                'colors': 'viridis'
            }

            # Create and start live renderer
            self.live_renderer = Live3DRenderer(self.parameters)
            fig = self.live_renderer.start_live_rendering(initial_data)

            self.set_current_data_path(data_path)

            return fig

        except Exception as e:
            self.logger.error(f"Error starting live rendering: {e}")
            return None

    def stop_live_rendering(self):
        """Stop live rendering if active."""
        if hasattr(self, 'live_renderer'):
            self.live_renderer.stop_live_rendering()

    def update_live_camera(self, elevation: float = None, azimuth: float = None):
        """Update live camera parameters."""
        if hasattr(self, 'live_renderer'):
            self.live_renderer.update_camera(elevation, azimuth)

    def analyze_with_mist(self, data_path: str) -> Dict:
        """
        Perform MIST-like analysis on pore network data.

        Args:
            data_path: Path to data file

        Returns:
            Dictionary containing analysis results
        """
        try:
            if not DEM_AVAILABLE:
                self.logger.warning("MIST analyzer not available")
                return {'error': 'MIST analyzer module not available'}

            # Create MIST analyzer with current parameters
            mist_analyzer = MISTAnalyzer(self.parameters)

            # Perform comprehensive analysis
            results = mist_analyzer.analyze_pore_network(data_path)

            self.logger.info(f"MIST analysis completed for {data_path}")
            return results

        except Exception as e:
            self.logger.error(f"Error in MIST analysis: {e}")
            return {'error': f'MIST analysis failed: {str(e)}'}

    def create_pore_network_visualization(self, data_path: str, style: str = 'enhanced') -> Figure:
        """
        Create pore network visualization based on selected style.

        Args:
            data_path: Path to the data file  
            style: Visualization style ('basic', 'enhanced', 'scientific', 'ultra_realistic')

        Returns:
            matplotlib Figure object
        """
        try:
            # Check if data file exists
            if not os.path.exists(data_path):
                self.logger.error(f"Data file not found: {data_path}")
                return self._create_empty_figure()

            # Read and validate data
            df = pd.read_csv(data_path)
            if df.empty:
                self.logger.warning(f"Empty data file: {data_path}")
                return self._create_empty_figure()

            # Set current data path
            self.set_current_data_path(data_path)

            # Create figure based on style
            if style == 'basic':
                fig = self._create_basic_visualization(df)
            elif style == 'enhanced':
                fig = self._create_enhanced_visualization(df)
            elif style == 'scientific':
                fig = self._create_scientific_visualization(df)
            elif style == 'ultra_realistic':
                fig = self._create_ultra_realistic_visualization(df)
            else:
                self.logger.warning(f"Unknown style: {style}, using enhanced")
                fig = self._create_enhanced_visualization(df)

            # Manage memory and store reference
            fig = self._manage_figure_memory(fig)
            self.last_generated_figure = fig

            return fig

        except Exception as e:
            self.logger.error(f"Error creating visualization: {e}")
            return self._create_empty_figure()

    def _create_empty_figure(self) -> Figure:
        """Create an empty figure when an error occurs."""
        fig = plt.figure(figsize=self.parameters['figure_size'],
                         facecolor=self.parameters['background_color'])
        ax = fig.add_subplot(111, projection='3d')
        ax.text(0, 0, 0, 'Visualization Error\nCheck console for details',
                ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="red", alpha=0.7))
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        return fig

    def _create_basic_visualization(self, df: pd.DataFrame) -> Figure:
        """Create basic visualization with simple styling."""
        n_points = min(len(df), self.parameters['num_spheres'])

        # Create figure
        fig = plt.figure(figsize=self.parameters['figure_size'],
                         facecolor=self.parameters['background_color'])
        ax = fig.add_subplot(111, projection='3d')

        # Use DataFrame columns if available, otherwise generate
        if len(df) > 0 and 'x' in df.columns:
            x = df['x'].values[:n_points]
            y = df['y'].values[:n_points] if 'y' in df.columns else np.random.rand(
                n_points) * 10
            z = df['z'].values[:n_points] if 'z' in df.columns else np.random.rand(
                n_points) * 10
            sizes = df['radius'].values[:n_points] * \
                100 if 'radius' in df.columns else np.ones(n_points) * 50
        else:
            # Generate simple grid data
            x = np.random.rand(n_points) * 10
            y = np.random.rand(n_points) * 10
            z = np.random.rand(n_points) * 10
            sizes = np.ones(n_points) * 50

        # Simple color scheme
        colors = np.arange(n_points)

        # Create scatter plot
        scatter = ax.scatter(x, y, z, c=colors, s=sizes,
                             alpha=0.7, cmap='viridis')

        # Basic styling
        ax.set_title('Basic 3D Visualization', fontsize=16)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        return fig

    def _create_enhanced_visualization(self, df: pd.DataFrame) -> Figure:
        """Create enhanced visualization with advanced styling."""
        n_points = min(len(df), self.parameters['num_spheres'])

        # Create figure
        fig = plt.figure(figsize=self.parameters['figure_size'],
                         facecolor=self.parameters['background_color'])
        ax = fig.add_subplot(111, projection='3d')

        # Enhanced data handling
        if len(df) > 0 and 'x' in df.columns:
            x = df['x'].values[:n_points]
            y = df['y'].values[:n_points] if 'y' in df.columns else np.random.rand(
                n_points) * 10
            z = df['z'].values[:n_points] if 'z' in df.columns else np.random.rand(
                n_points) * 10
            radii = df['radius'].values[:n_points] if 'radius' in df.columns else np.ones(
                n_points) * 0.5
        else:
            # Generate enhanced data
            x = np.random.rand(n_points) * 10 - 5
            y = np.random.rand(n_points) * 10 - 5
            z = np.random.rand(n_points) * 10 - 5
            radii = np.random.uniform(0.2, 1.0, n_points)

        # Enhanced sizing
        sizes = radii * self.parameters['size_multiplier'] * 100

        # Advanced color mapping based on position and size
        color_values = (x**2 + y**2 + z**2)**0.5  # Distance from origin
        colors = plt.cm.viridis(color_values / np.max(color_values))

        # Create enhanced scatter plot
        scatter = ax.scatter(x, y, z,
                             c=color_values,
                             s=sizes,
                             alpha=0.8,
                             cmap='viridis',
                             edgecolors='white',
                             linewidth=0.5)

        # Add connections between nearby points
        if self.parameters.get('show_connections', False):
            n_connections = min(50, n_points // 10)
            for i in range(n_connections):
                idx1, idx2 = np.random.choice(len(x), 2, replace=False)
                distance = np.sqrt(
                    (x[idx1] - x[idx2])**2 + (y[idx1] - y[idx2])**2 + (z[idx1] - z[idx2])**2)
                if distance < 3:  # Only connect nearby points
                    ax.plot([x[idx1], x[idx2]], [y[idx1], y[idx2]], [z[idx1], z[idx2]],
                            'white', alpha=0.3, linewidth=0.5)

        # Enhanced styling
        ax.set_title('Enhanced 3D Pore Network',
                     fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('X Position (μm)', fontsize=12)
        ax.set_ylabel('Y Position (μm)', fontsize=12)
        ax.set_zlabel('Z Position (μm)', fontsize=12)

        # Apply view angle
        view_angle = self.parameters['view_angle']
        ax.view_init(elev=view_angle['elevation'], azim=view_angle['azimuth'])

        # Add colorbar
        if hasattr(fig, 'colorbar'):
            plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=30,
                         label='Distance from Origin')

        return fig

    def _create_ultra_realistic_visualization(self, df: pd.DataFrame) -> Figure:
        """Create ultra-realistic visualization with advanced effects."""
        n_points = min(len(df), self.parameters['num_spheres'])

        # Create figure with larger size for detail
        fig = plt.figure(figsize=(self.parameters['figure_size'][0] * 1.2,
                                  self.parameters['figure_size'][1] * 1.2),
                         facecolor='black')
        ax = fig.add_subplot(111, projection='3d')

        # Ultra-realistic data handling
        if len(df) > 0 and 'x' in df.columns:
            x = df['x'].values[:n_points]
            y = df['y'].values[:n_points] if 'y' in df.columns else np.random.rand(
                n_points) * 15
            z = df['z'].values[:n_points] if 'z' in df.columns else np.random.rand(
                n_points) * 15
            radii = df['radius'].values[:n_points] if 'radius' in df.columns else np.random.uniform(
                0.3, 1.2, n_points)
        else:
            # Generate ultra-realistic data with clustering
            centers = np.random.rand(5, 3) * 10 - 5
            x, y, z, radii = [], [], [], []
            for center in centers:
                cluster_size = n_points // 5
                cluster_x = np.random.normal(center[0], 2, cluster_size)
                cluster_y = np.random.normal(center[1], 2, cluster_size)
                cluster_z = np.random.normal(center[2], 2, cluster_size)
                cluster_radii = np.random.lognormal(0, 0.3, cluster_size)

                x.extend(cluster_x)
                y.extend(cluster_y)
                z.extend(cluster_z)
                radii.extend(cluster_radii)

            x, y, z, radii = map(
                np.array, [x[:n_points], y[:n_points], z[:n_points], radii[:n_points]])

        # Ultra-realistic sizing and coloring
        sizes = radii * self.parameters['size_multiplier'] * 150

        # Multi-layered color scheme based on physical properties
        depth_colors = (z - np.min(z)) / (np.max(z) - np.min(z))
        size_colors = (radii - np.min(radii)) / (np.max(radii) - np.min(radii))
        combined_colors = 0.6 * depth_colors + 0.4 * size_colors

        # Create ultra-realistic scatter with multiple layers
        # Main spheres
        scatter1 = ax.scatter(x, y, z,
                              c=combined_colors,
                              s=sizes,
                              alpha=0.9,
                              cmap='plasma',
                              edgecolors='gold',
                              linewidth=1.0)

        # Add smaller detail spheres around main ones
        detail_factor = 0.3
        x_detail = x + np.random.normal(0, radii * 0.5, n_points)
        y_detail = y + np.random.normal(0, radii * 0.5, n_points)
        z_detail = z + np.random.normal(0, radii * 0.5, n_points)
        sizes_detail = sizes * detail_factor

        scatter2 = ax.scatter(x_detail, y_detail, z_detail,
                              c=combined_colors * 0.7,
                              s=sizes_detail,
                              alpha=0.5,
                              cmap='plasma',
                              edgecolors='silver',
                              linewidth=0.5)

        # Add realistic connections with physics-based probability
        n_connections = min(100, n_points // 5)
        for i in range(n_connections):
            idx1, idx2 = np.random.choice(len(x), 2, replace=False)
            distance = np.sqrt((x[idx1] - x[idx2])**2 +
                               (y[idx1] - y[idx2])**2 + (z[idx1] - z[idx2])**2)
            if distance < (radii[idx1] + radii[idx2]) * 3:  # Physics-based connection
                connection_strength = np.exp(-distance / 2)
                ax.plot([x[idx1], x[idx2]], [y[idx1], y[idx2]], [z[idx1], z[idx2]],
                        color='cyan', alpha=connection_strength * 0.6, linewidth=connection_strength * 2)

        # Ultra-realistic styling
        ax.set_title('Ultra-Realistic 3D Pore Network\nHigh-Resolution Microstructure Analysis',
                     fontsize=20, fontweight='bold', pad=30, color='white')
        ax.set_xlabel('X Position (μm)', fontsize=14, color='white')
        ax.set_ylabel('Y Position (μm)', fontsize=14, color='white')
        ax.set_zlabel('Z Position (μm)', fontsize=14, color='white')

        # Dark theme styling
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.tick_params(colors='white')

        # Apply dramatic view angle
        ax.view_init(elev=25, azim=45)

        return fig

    def _create_scientific_visualization(self, df: pd.DataFrame) -> Figure:
        """Create scientific visualization with proper data handling."""
        n_points = min(len(df), self.parameters['num_spheres'])

        # Create figure
        fig = plt.figure(figsize=self.parameters['figure_size'],
                         facecolor=self.parameters['background_color'])
        ax1 = fig.add_subplot(111, projection='3d')

        # Handle coordinates
        if all(col in df.columns for col in ['X', 'Y', 'Z']):
            coord_data = df[['X', 'Y', 'Z']].dropna()[:n_points]
            x, y, z = coord_data['X'].values, coord_data['Y'].values, coord_data['Z'].values
        else:
            x = np.random.uniform(-10, 10, n_points)
            y = np.random.uniform(-10, 10, n_points)
            z = np.random.uniform(-10, 10, n_points)

        # Handle pore radius data
        if 'Pore_Radius' in df.columns:
            pore_radius_clean = df['Pore_Radius'].dropna().values
            if len(pore_radius_clean) >= n_points:
                property_values = pore_radius_clean[:n_points]
            else:
                mean_radius = np.mean(pore_radius_clean) if len(
                    pore_radius_clean) > 0 else 2.0
                property_values = np.concatenate([
                    pore_radius_clean,
                    np.full(n_points - len(pore_radius_clean), mean_radius)
                ])
        else:
            property_values = np.random.exponential(2, n_points)

        # Ensure consistent array lengths
        property_values = property_values[:n_points]
        sizes = property_values * 100 * self.parameters['size_multiplier']

        # Ensure all arrays have consistent lengths
        x, y, z, sizes, property_values = self._ensure_consistent_arrays(
            x, y, z, sizes, property_values)

        # Create scientific color map
        scatter = ax1.scatter(x, y, z,
                              c=property_values,
                              s=sizes,
                              alpha=0.7,
                              cmap='plasma',
                              edgecolors='black',
                              linewidth=0.5)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1, shrink=0.5, aspect=20)
        cbar.set_label('Pore Radius (μm)', rotation=270, labelpad=15)

        # Scientific labels
        ax1.set_title('3D Pore Network Analysis',
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel('X Position (μm)')
        ax1.set_ylabel('Y Position (μm)')
        ax1.set_zlabel('Z Position (μm)')

        # Apply enhanced styling
        self._apply_enhanced_styling(ax1, fig)

        return fig

    def _create_presentation_view(self, data_path: str) -> Figure:
        """Create presentation-ready view with clean, professional styling."""
        # Load data
        df = pd.read_csv(data_path)

        # Create clean presentation figure
        fig = plt.figure(figsize=(16, 10))
        fig = self._manage_figure_memory(fig)
        ax = fig.add_subplot(111, projection='3d')

        # Generate clean, organized data
        n_points = min(len(df), 800)  # Optimal for presentation clarity

        # Create organized grid with some randomness
        grid_size = int(np.ceil(n_points**(1/3)))
        x_base = np.linspace(-8, 8, grid_size)
        y_base = np.linspace(-8, 8, grid_size)
        z_base = np.linspace(-8, 8, grid_size)

        x, y, z = np.meshgrid(x_base[:3], y_base[:3], z_base[:3])
        x = x.flatten()[:n_points] + np.random.normal(0, 0.5, n_points)
        y = y.flatten()[:n_points] + np.random.normal(0, 0.5, n_points)
        z = z.flatten()[:n_points] + np.random.normal(0, 0.5, n_points)

        # Artistic color scheme for presentation
        colors = plt.cm.viridis(np.linspace(0, 1, n_points))

        # Consistent sizing for clarity
        sizes = np.random.uniform(80, 150, n_points) * \
            self.parameters['size_multiplier']

        # Create elegant scatter plot
        scatter = ax.scatter(x, y, z,
                             c=colors,
                             s=sizes,
                             alpha=0.8,
                             edgecolors='white',
                             linewidth=1.0)

        # Add elegant connections
        n_connections = min(20, n_points // 10)
        for i in range(n_connections):
            idx1, idx2 = np.random.choice(len(x), 2, replace=False)
            ax.plot([x[idx1], x[idx2]], [y[idx1], y[idx2]], [z[idx1], z[idx2]],
                    'w-', alpha=0.3, linewidth=1.5)

        # Professional styling
        ax.set_title('3D Pore Network Structure',
                     fontsize=20, fontweight='bold', pad=30)

        # Clean labels
        ax.set_xlabel('X-axis', fontsize=14, labelpad=10)
        ax.set_ylabel('Y-axis', fontsize=14, labelpad=10)
        ax.set_zlabel('Z-axis', fontsize=14, labelpad=10)

        # Remove grid for clean look
        ax.grid(False)

        # Dark background for contrast
        fig.patch.set_facecolor('#1a1a1a')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Make pane edges invisible
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')
        ax.xaxis.pane.set_alpha(0)
        ax.yaxis.pane.set_alpha(0)
        ax.zaxis.pane.set_alpha(0)

        return fig

    def _create_cross_section_view(self, data_path: str) -> Figure:
        """Create cross-section view showing internal structure."""
        # Load data
        df = pd.read_csv(data_path)

        # Create dual-view figure
        fig = plt.figure(figsize=(20, 10))
        fig = self._manage_figure_memory(fig)

        # Left panel: Complete structure
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')

        # Generate data
        n_points = min(len(df), 1000)
        x = np.random.uniform(-10, 10, n_points)
        y = np.random.uniform(-10, 10, n_points)
        z = np.random.uniform(-10, 10, n_points)

        sizes = np.random.uniform(50, 120, n_points) * \
            self.parameters['size_multiplier']
        colors = plt.cm.coolwarm(np.linspace(0, 1, n_points))

        # Complete structure
        ax1.scatter(x, y, z, c=colors, s=sizes, alpha=0.6)
        ax1.set_title('Complete Pore Network', fontsize=14, fontweight='bold')

        # Right panel: Cross-section
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')

        # Create cross-section (filter points)
        section_mask = (x > -2) & (x < 2)  # Cross-section slice
        x_section = x[section_mask]
        y_section = y[section_mask]
        z_section = z[section_mask]
        sizes_section = sizes[section_mask]
        colors_section = colors[section_mask]

        # Plot cross-section
        ax2.scatter(x_section, y_section, z_section,
                    c=colors_section, s=sizes_section * 1.5, alpha=0.8,
                    edgecolors='black', linewidth=1)

        # Add section plane
        xx, zz = np.meshgrid(np.linspace(-10, 10, 10),
                             np.linspace(-10, 10, 10))
        yy_plane = np.zeros_like(xx)
        ax2.plot_surface(xx, yy_plane, zz, alpha=0.1, color='gray')

        ax2.set_title('Cross-Section View (Y=0)',
                      fontsize=14, fontweight='bold')

        # Add sectioning lines to complete view
        ax1.plot([-2, -2], [-10, 10], [-10, 10], 'r--', linewidth=2, alpha=0.7)
        ax1.plot([2, 2], [-10, 10], [-10, 10], 'r--', linewidth=2, alpha=0.7)

        # Synchronize view angles
        ax1.view_init(elev=20, azim=45)
        ax2.view_init(elev=20, azim=45)

        plt.tight_layout()
        return fig

    def _get_enhanced_colors(self, n_points: int) -> np.ndarray:
        """Get enhanced color schemes for better visualization."""
        scheme = self.parameters['color_scheme']

        if scheme == 'scientific':
            # Blue-white-red temperature-like mapping
            return plt.cm.coolwarm(np.linspace(0, 1, n_points))
        elif scheme == 'artistic':
            # Vibrant artistic colors
            return plt.cm.plasma(np.linspace(0, 1, n_points))
        elif scheme == 'rainbow':
            # Full rainbow spectrum
            return plt.cm.hsv(np.linspace(0, 1, n_points))
        elif scheme == 'thermal':
            # Heat map colors
            return plt.cm.hot(np.linspace(0, 1, n_points))
        elif scheme == 'depth':
            # Depth-based coloring
            return plt.cm.viridis(np.linspace(0, 1, n_points))
        else:
            # Default scientific
            return plt.cm.viridis(np.linspace(0, 1, n_points))

    def _ensure_consistent_arrays(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                                  sizes: np.ndarray, colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Ensure all arrays have consistent lengths for scatter plot."""
        min_length = min(len(x), len(y), len(z), len(sizes), len(colors))

        if min_length == 0:
            self.logger.warning(
                "One or more arrays is empty, creating default data")
            min_length = 1
            x = np.array([0.0])
            y = np.array([0.0])
            z = np.array([0.0])
            sizes = np.array([100.0])
            colors = np.array([0.5])
        else:
            # Trim all arrays to consistent length
            x = x[:min_length]
            y = y[:min_length]
            z = z[:min_length]
            sizes = sizes[:min_length]
            colors = colors[:min_length]

        self.logger.debug(f"Ensured consistent array lengths: {min_length}")
        return x, y, z, sizes, colors

    def _get_color_values(self, n_points: int) -> np.ndarray:
        """Get numerical values for colormap application."""
        scheme = self.parameters['color_scheme']

        if scheme == 'scientific':
            # Linear gradient for temperature-like mapping
            return np.linspace(0, 1, n_points)
        elif scheme == 'artistic':
            # Sine wave pattern for artistic variation
            return np.sin(np.linspace(0, 2*np.pi, n_points)) * 0.5 + 0.5
        elif scheme == 'rainbow':
            # Full range for rainbow spectrum
            return np.linspace(0, 1, n_points)
        elif scheme == 'thermal':
            # Exponential pattern for thermal visualization
            return np.power(np.linspace(0, 1, n_points), 2)
        elif scheme == 'depth':
            # Random but sorted for depth-based coloring
            values = np.random.random(n_points)
            return np.sort(values)
        else:
            # Default linear gradient
            return np.linspace(0, 1, n_points)

    def _add_bonds(self, ax, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        """Add bonds between nearby spheres."""
        n_points = len(x)
        n_bonds = min(int(n_points * 0.1), 50)  # Limit bonds for performance

        for _ in range(n_bonds):
            i, j = np.random.choice(n_points, 2, replace=False)

            # Only connect nearby points
            distance = np.sqrt((x[i] - x[j])**2 +
                               (y[i] - y[j])**2 + (z[i] - z[j])**2)
            if distance < 8:  # Threshold for connection
                ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]],
                        'gray', alpha=0.5, linewidth=self.parameters['bond_thickness'])

    def _create_ultra_realistic_bonds(self, ax, x: np.ndarray, y: np.ndarray, z: np.ndarray, materials: np.ndarray):
        """Create ultra-realistic bonds with stress-based coloring."""
        n_points = len(x)
        n_bonds = min(int(n_points * 0.15), 80)

        for _ in range(n_bonds):
            i, j = np.random.choice(n_points, 2, replace=False)
            distance = np.sqrt((x[i] - x[j])**2 +
                               (y[i] - y[j])**2 + (z[i] - z[j])**2)

            if distance < 10:
                # Stress-based coloring
                stress = 1.0 / (distance + 0.1)  # Inverse distance as stress
                if stress > 0.15:
                    color = 'red'  # High stress
                    width = 2.5
                elif stress > 0.08:
                    color = 'orange'  # Medium stress
                    width = 2.0
                else:
                    color = 'lightgray'  # Low stress
                    width = 1.5

                ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]],
                        color=color, alpha=0.7, linewidth=width)

    def _apply_ultra_realistic_lighting(self, ax):
        """Apply ultra-realistic lighting effects."""
        # Set viewing angle for dramatic lighting
        ax.view_init(elev=25, azim=65)

        # Dark background with subtle grid
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Subtle grid lines
        ax.grid(True, alpha=0.1, color='white')

    def _create_sectioned_view(self, ax, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        """Create sectioned/cutaway view."""
        # Add a sectioning plane
        xx, yy = np.meshgrid(np.linspace(-10, 10, 10),
                             np.linspace(-10, 10, 10))
        zz_plane = np.zeros_like(xx) + 5  # Section at z=5
        ax.plot_surface(xx, yy, zz_plane, alpha=0.2, color='orange')

    def _add_thermal_effects(self, ax, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        """Add thermal visualization effects."""
        # Add thermal gradient coloring based on z-position
        thermal_colors = plt.cm.hot((z - np.min(z)) / (np.max(z) - np.min(z)))
        return thermal_colors

    def _apply_enhanced_styling(self, ax, fig):
        """Apply enhanced styling to the plot."""
        # Set background color
        fig.patch.set_facecolor(self.parameters['background_color'])

        # Apply view angle
        view_angle = self.parameters['view_angle']
        ax.view_init(elev=view_angle['elevation'], azim=view_angle['azimuth'])

        # Set axis bounds
        bounds = self.parameters['axis_bounds']
        ax.set_xlim(bounds['x'])
        ax.set_ylim(bounds['y'])
        ax.set_zlim(bounds['z'])

        # Show coordinate system if enabled
        if self.parameters['show_coordinate_system']:
            ax.grid(True, alpha=0.3)
        else:
            ax.grid(False)

    def _add_scientific_annotations(self, ax, n_points: int):
        """Add scientific annotations and measurements."""
        if self.parameters['measurement_tools']:
            # Add scale bar
            ax.text(-8, -8, -8, '10 μm', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            # Add point count
            ax.text(8, 8, 8, f'n = {n_points:,}', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

    def create_animation_frames(self, data_path: str, num_frames: int = 36) -> List[Figure]:
        """Create animation frames for rotating visualization."""
        frames = []

        for i in range(num_frames):
            # Update view angle for rotation
            angle = i * (360 / num_frames)
            self.parameters['view_angle']['azimuth'] = angle

            # Create frame
            fig = self.create_pore_network_visualization(data_path, 'enhanced')
            frames.append(fig)

        return frames

    def export_visualization(self, file_path: str, dpi: int = 300):
        """Export current visualization to file."""
        # This would be called from the GUI with the current figure
        pass

    def create_comparison_view(self, data_path: str) -> Figure:
        """Create side-by-side comparison of different visualization types."""
        fig, axes = plt.subplots(2, 2, figsize=(
            16, 12), subplot_kw={'projection': '3d'})
        fig = self._manage_figure_memory(fig)
        axes = axes.flatten()

        viz_types = ['enhanced', 'clean', 'thermal', 'sectioned']
        titles = ['Enhanced View', 'Clean View',
                  'Thermal View', 'Sectioned View']

        for i, (viz_type, title) in enumerate(zip(viz_types, titles)):
            # Create mini visualization for each type
            try:
                temp_fig = self.create_pore_network_visualization(
                    data_path, viz_type)
                # Copy content to subplot (simplified)
                axes[i].set_title(title, fontsize=12, fontweight='bold')
            except Exception as e:
                axes[i].text(0, 0, 0, f"Error: {viz_type}", ha='center')

        plt.tight_layout()
        return fig

    def _create_size_distribution_view(self, data_path: str) -> Figure:
        """Create size distribution analysis view."""
        # Load data
        df = pd.read_csv(data_path)

        # Create comprehensive size distribution figure
        fig = plt.figure(figsize=(16, 12))
        fig = self._manage_figure_memory(fig)

        # Generate data for size distribution
        n_points = min(len(df), 1500)

        # Create pore size data
        if 'Pore_Radius' in df.columns:
            pore_radius_clean = df['Pore_Radius'].dropna().values
            if len(pore_radius_clean) >= n_points:
                pore_sizes = pore_radius_clean[:n_points]
            else:
                mean_radius = np.mean(pore_radius_clean) if len(
                    pore_radius_clean) > 0 else 2.0
                pore_sizes = np.concatenate([
                    pore_radius_clean,
                    np.full(n_points - len(pore_radius_clean), mean_radius)
                ])
        else:
            # Generate realistic size distribution
            pore_sizes = np.concatenate([
                np.random.lognormal(1.5, 0.8, n_points//3),  # Small pores
                np.random.lognormal(2.5, 0.6, n_points//3),  # Medium pores
                np.random.lognormal(3.2, 0.4, n_points -
                                    2*(n_points//3))  # Large pores
            ])

        # Main 3D size-based visualization
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')

        # Generate coordinates
        x = np.random.uniform(-10, 10, n_points)
        y = np.random.uniform(-10, 10, n_points)
        z = np.random.uniform(-10, 10, n_points)

        # Size-based color mapping
        sizes_scaled = pore_sizes * 100 * self.parameters['size_multiplier']

        # Ensure consistent array lengths
        x, y, z, sizes_scaled, pore_sizes = self._ensure_consistent_arrays(
            x, y, z, sizes_scaled, pore_sizes)

        scatter = ax1.scatter(x, y, z,
                              c=pore_sizes,
                              s=sizes_scaled,
                              alpha=0.7,
                              cmap='viridis',
                              edgecolors='black',
                              linewidth=0.3)

        plt.colorbar(scatter, ax=ax1, shrink=0.5, label='Pore Size (μm)')
        ax1.set_title('3D Size Distribution View',
                      fontsize=12, fontweight='bold')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_zlabel('Z Position')

        # Size histogram
        ax2 = fig.add_subplot(2, 3, 2)
        counts, bins, patches = ax2.hist(pore_sizes, bins=50, alpha=0.7,
                                         color='skyblue', edgecolor='black')

        # Color bars by size
        for i, (count, bin_edge, patch) in enumerate(zip(counts, bins[:-1], patches)):
            if np.max(pore_sizes) != np.min(pore_sizes):
                normalized_size = (bin_edge - np.min(pore_sizes)) / \
                    (np.max(pore_sizes) - np.min(pore_sizes))
                patch.set_facecolor(plt.cm.viridis(normalized_size))

        ax2.set_title('Pore Size Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Pore Size (μm)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)

        # Cumulative distribution
        ax3 = fig.add_subplot(2, 3, 3)
        sorted_sizes = np.sort(pore_sizes)
        cumulative = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)
        ax3.plot(sorted_sizes, cumulative, 'b-',
                 linewidth=2, label='Cumulative')
        ax3.fill_between(sorted_sizes, cumulative, alpha=0.3)
        ax3.set_title('Cumulative Size Distribution',
                      fontsize=12, fontweight='bold')
        ax3.set_xlabel('Pore Size (μm)')
        ax3.set_ylabel('Cumulative Probability')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Size categories pie chart
        ax4 = fig.add_subplot(2, 3, 4)
        small_pores = np.sum(pore_sizes < 1.0)
        medium_pores = np.sum((pore_sizes >= 1.0) & (pore_sizes < 5.0))
        large_pores = np.sum(pore_sizes >= 5.0)

        sizes_cat = [small_pores, medium_pores, large_pores]
        labels = ['Small (<1μm)', 'Medium (1-5μm)', 'Large (>5μm)']
        colors = ['lightcoral', 'gold', 'lightblue']

        if np.sum(sizes_cat) > 0:  # Only create pie chart if we have data
            wedges, texts, autotexts = ax4.pie(sizes_cat, labels=labels, colors=colors,
                                               autopct='%1.1f%%', startangle=90)
        ax4.set_title('Size Category Distribution',
                      fontsize=12, fontweight='bold')

        # Statistical summary
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.axis('off')
        stats_text = f"""
SIZE DISTRIBUTION STATISTICS
════════════════════════════════
Total Pores: {len(pore_sizes):,}

STATISTICAL MEASURES
────────────────────────────────
Mean Size: {np.mean(pore_sizes):.2f} μm
Median Size: {np.median(pore_sizes):.2f} μm
Std Deviation: {np.std(pore_sizes):.2f} μm
Min Size: {np.min(pore_sizes):.2f} μm
Max Size: {np.max(pore_sizes):.2f} μm

PERCENTILES
────────────────────────────────
25th: {np.percentile(pore_sizes, 25):.2f} μm
75th: {np.percentile(pore_sizes, 75):.2f} μm
90th: {np.percentile(pore_sizes, 90):.2f} μm
95th: {np.percentile(pore_sizes, 95):.2f} μm
        """
        ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        # Box plot
        ax6 = fig.add_subplot(2, 3, 6)
        bp = ax6.boxplot(pore_sizes, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        ax6.set_title('Size Distribution Box Plot',
                      fontsize=12, fontweight='bold')
        ax6.set_ylabel('Pore Size (μm)')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _create_clustering_analysis_view(self, data_path: str) -> Figure:
        """Create clustering analysis view."""
        try:
            from sklearn.cluster import DBSCAN, KMeans
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            # Fallback if sklearn is not available
            return self._create_fallback_clustering_view(data_path)

        # Load data
        df = pd.read_csv(data_path)

        # Create comprehensive clustering analysis figure
        fig = plt.figure(figsize=(16, 12))
        fig = self._manage_figure_memory(fig)

        # Generate data for clustering
        n_points = min(len(df), 800)

        # Generate coordinates and properties
        if all(col in df.columns for col in ['X', 'Y', 'Z']):
            coord_data = df[['X', 'Y', 'Z']].dropna()[:n_points]
            if len(coord_data) > 0:
                x, y, z = coord_data['X'].values, coord_data['Y'].values, coord_data['Z'].values
            else:
                x = np.random.uniform(-10, 10, n_points)
                y = np.random.uniform(-10, 10, n_points)
                z = np.random.uniform(-10, 10, n_points)
        else:
            x = np.random.uniform(-10, 10, n_points)
            y = np.random.uniform(-10, 10, n_points)
            z = np.random.uniform(-10, 10, n_points)

        # Create feature matrix for clustering
        if 'Pore_Radius' in df.columns:
            pore_radius_clean = df['Pore_Radius'].dropna().values[:n_points]
            if len(pore_radius_clean) < n_points:
                mean_radius = np.mean(pore_radius_clean) if len(
                    pore_radius_clean) > 0 else 2.0
                pore_sizes = np.concatenate([
                    pore_radius_clean,
                    np.full(n_points - len(pore_radius_clean), mean_radius)
                ])
            else:
                pore_sizes = pore_radius_clean
        else:
            pore_sizes = np.random.exponential(2, n_points)

        # Ensure all arrays have the same length
        min_length = min(len(x), len(y), len(z), len(pore_sizes))
        x = x[:min_length]
        y = y[:min_length]
        z = z[:min_length]
        pore_sizes = pore_sizes[:min_length]
        n_points = min_length

        # Feature matrix for clustering
        features = np.column_stack([x, y, z, pore_sizes])
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # K-Means clustering
        try:
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(features_scaled)
        except:
            # Fallback clustering
            kmeans_labels = np.random.randint(0, 4, n_points)

        # DBSCAN clustering
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan_labels = dbscan.fit_predict(features_scaled)
        except:
            # Fallback clustering
            dbscan_labels = np.random.randint(-1, 3, n_points)

        # Main 3D clustering visualization - K-Means
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')

        unique_labels = np.unique(kmeans_labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

        for i, (label, color) in enumerate(zip(unique_labels, colors)):
            mask = kmeans_labels == label
            if np.sum(mask) > 0:
                cluster_sizes = pore_sizes[mask] * \
                    100 * self.parameters['size_multiplier']
                ax1.scatter(x[mask], y[mask], z[mask],
                            c=[color], s=cluster_sizes, alpha=0.7,
                            label=f'Cluster {label}', edgecolors='black', linewidth=0.3)

        ax1.set_title('K-Means Clustering (4 clusters)',
                      fontsize=12, fontweight='bold')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_zlabel('Z Position')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Cluster statistics and remaining plots
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.axis('off')
        try:
            from sklearn.metrics import silhouette_score
            silhouette_kmeans = silhouette_score(
                features_scaled, kmeans_labels)
        except:
            silhouette_kmeans = 0.5

        stats_text = f"""
CLUSTERING ANALYSIS RESULTS
═══════════════════════════════
Total Points: {n_points:,}

K-MEANS CLUSTERING
───────────────────────────────
Number of Clusters: {len(unique_labels)}
Silhouette Score: {silhouette_kmeans:.3f}

CLUSTER CHARACTERISTICS
───────────────────────────────
Avg Cluster Size: {n_points/len(unique_labels):.1f}
Spatial Distribution: 3D
        """
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        # Additional plots can be added here
        # ...existing clustering plots...

        plt.tight_layout()
        return fig

    def _create_fallback_clustering_view(self, data_path: str) -> Figure:
        """Create fallback clustering view when sklearn is not available."""
        # Load data
        df = pd.read_csv(data_path)

        # Create simple clustering figure
        fig = plt.figure(figsize=(12, 8))
        fig = self._manage_figure_memory(fig)

        n_points = min(len(df), 500)
        x = np.random.uniform(-10, 10, n_points)
        y = np.random.uniform(-10, 10, n_points)
        z = np.random.uniform(-10, 10, n_points)

        # Simple manual clustering based on position
        labels = np.zeros(n_points, dtype=int)
        labels[(x > 0) & (y > 0)] = 0  # Quadrant 1
        labels[(x < 0) & (y > 0)] = 1  # Quadrant 2
        labels[(x < 0) & (y < 0)] = 2  # Quadrant 3
        labels[(x > 0) & (y < 0)] = 3  # Quadrant 4

        ax = fig.add_subplot(111, projection='3d')
        colors = ['red', 'blue', 'green', 'orange']

        for i in range(4):
            mask = labels == i
            if np.sum(mask) > 0:
                ax.scatter(x[mask], y[mask], z[mask],
                           c=colors[i], alpha=0.7, s=50,
                           label=f'Region {i+1}')

        ax.set_title('Simple Spatial Clustering',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.legend()

        plt.tight_layout()
        return fig

    def _calculate_axis_bounds_from_dimensions(self):
        """Calculate axis bounds from space dimensions and aspect ratio."""
        width = self.parameters['space_width']
        height = self.parameters['space_height']
        depth = self.parameters['space_depth']

        if self.parameters['maintain_aspect_ratio']:
            # Use custom aspect ratio if specified
            aspect = self.parameters['custom_aspect_ratio']
            # Normalize aspect ratio and apply to dimensions
            max_aspect = max(aspect)
            width = width * (aspect[0] / max_aspect)
            height = height * (aspect[1] / max_aspect)
            depth = depth * (aspect[2] / max_aspect)

        # Apply scaling factors
        width *= self.parameters['axis_x_scale']
        height *= self.parameters['axis_y_scale']
        depth *= self.parameters['axis_z_scale']

        # Update axis bounds (centered around origin)
        self.parameters['axis_bounds'] = {
            'x': [-width/2, width/2],
            'y': [-height/2, height/2],
            'z': [-depth/2, depth/2]
        }

        return self.parameters['axis_bounds']

    def update_dimensions(self, width: float = None, height: float = None, depth: float = None):
        """Update visualization space dimensions and recalculate bounds."""
        if width is not None:
            self.parameters['space_width'] = width
        if height is not None:
            self.parameters['space_height'] = height
        if depth is not None:
            self.parameters['space_depth'] = depth

        # Recalculate axis bounds
        self._calculate_axis_bounds_from_dimensions()

        # Log the update
        self.logger.debug(f"Updated dimensions: W={self.parameters['space_width']}, "
                          f"H={self.parameters['space_height']}, D={self.parameters['space_depth']}")

    def set_aspect_ratio(self, width_ratio: float, height_ratio: float, depth_ratio: float):
        """Set custom aspect ratio for the visualization space."""
        self.parameters['custom_aspect_ratio'] = [
            width_ratio, height_ratio, depth_ratio]
        if self.parameters['maintain_aspect_ratio']:
            self._calculate_axis_bounds_from_dimensions()

    def get_dimension_info(self):
        """Get current dimension information."""
        bounds = self.parameters['axis_bounds']
        return {
            'space_width': self.parameters['space_width'],
            'space_height': self.parameters['space_height'],
            'space_depth': self.parameters['space_depth'],
            'effective_width': bounds['x'][1] - bounds['x'][0],
            'effective_height': bounds['y'][1] - bounds['y'][0],
            'effective_depth': bounds['z'][1] - bounds['z'][0],
            'aspect_ratio': self.parameters['custom_aspect_ratio'],
            'maintain_aspect_ratio': self.parameters['maintain_aspect_ratio']
        }
