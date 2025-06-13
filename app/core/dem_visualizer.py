#!/usr/bin/env python3
"""
DEM (Discrete Element Method) Particle Visualizer
Handles visualization of particle assemblies similar to MIST software
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging


class DEMParticleVisualizer:
    """Visualizer for Discrete Element Method particle assemblies."""

    def __init__(self, parameters: Dict[str, Any]):
        """Initialize DEM visualizer with parameters."""
        self.parameters = parameters
        self.logger = logging.getLogger(__name__)

    def create_dem_visualization(self, data_path: str, particle_type: str = 'mixed') -> Figure:
        """
        Create DEM particle visualization with stress visualization.

        Args:
            data_path: Path to particle data
            particle_type: Type of particles ('spherical', 'cubic', 'mixed')

        Returns:
            matplotlib Figure object
        """
        try:
            # Load data
            df = pd.read_csv(data_path)

            # Create figure
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Generate particle assembly
            particles = self._generate_particle_assembly(df, particle_type)

            # Render particles with different colors based on material/stress
            self._render_particles(ax, particles)

            # Add stress visualization (arrows showing stress directions)
            self._add_stress_arrows(ax, particles)

            # Apply DEM-specific styling
            self._apply_dem_styling(ax, fig)

            return fig

        except Exception as e:
            self.logger.error(f"Error creating DEM visualization: {e}")
            raise

    def _generate_particle_assembly(self, df: pd.DataFrame, particle_type: str) -> List[Dict]:
        """Generate particle assembly data."""
        n_particles = min(len(df), self.parameters.get('num_spheres', 1000))

        particles = []

        # Define material colors based on the image
        material_colors = {
            'material_1': '#FF1493',  # Magenta/Pink
            'material_2': '#00FFFF',  # Cyan/Blue
            'material_3': '#32CD32',  # Lime Green
        }

        for i in range(n_particles):
            # Generate position within bounds
            bounds = self.parameters.get(
                'axis_bounds', {'x': [-10, 10], 'y': [-10, 10], 'z': [-10, 10]})

            x = np.random.uniform(bounds['x'][0], bounds['x'][1])
            y = np.random.uniform(bounds['y'][0], bounds['y'][1])
            z = np.random.uniform(bounds['z'][0], bounds['z'][1])

            # Generate particle properties
            material = np.random.choice(
                ['material_1', 'material_2', 'material_3'])

            if particle_type == 'cubic':
                # Cubic particles (like in the image)
                size = np.random.uniform(0.5, 1.5)
                particle = {
                    'type': 'cube',
                    'position': np.array([x, y, z]),
                    'size': size,
                    'material': material,
                    'color': material_colors[material],
                    'stress': np.random.uniform(0, 100),  # Stress level
                    # Random rotation
                    'rotation': np.random.uniform(0, 2*np.pi, 3)
                }
            elif particle_type == 'spherical':
                # Spherical particles
                radius = np.random.uniform(0.3, 1.0)
                particle = {
                    'type': 'sphere',
                    'position': np.array([x, y, z]),
                    'radius': radius,
                    'material': material,
                    'color': material_colors[material],
                    'stress': np.random.uniform(0, 100)
                }
            else:  # mixed
                # Mixed particle types
                ptype = np.random.choice(['cube', 'sphere'])
                if ptype == 'cube':
                    size = np.random.uniform(0.5, 1.5)
                    particle = {
                        'type': 'cube',
                        'position': np.array([x, y, z]),
                        'size': size,
                        'material': material,
                        'color': material_colors[material],
                        'stress': np.random.uniform(0, 100),
                        'rotation': np.random.uniform(0, 2*np.pi, 3)
                    }
                else:
                    radius = np.random.uniform(0.3, 1.0)
                    particle = {
                        'type': 'sphere',
                        'position': np.array([x, y, z]),
                        'radius': radius,
                        'material': material,
                        'color': material_colors[material],
                        'stress': np.random.uniform(0, 100)
                    }

            particles.append(particle)

        return particles

    def _render_particles(self, ax: Axes3D, particles: List[Dict]):
        """Render particles on the 3D axis."""
        for particle in particles:
            if particle['type'] == 'cube':
                self._render_cube(ax, particle)
            else:
                self._render_sphere(ax, particle)

    def _render_cube(self, ax: Axes3D, particle: Dict):
        """Render a cubic particle."""
        pos = particle['position']
        size = particle['size']
        color = particle['color']
        rotation = particle.get('rotation', np.array([0, 0, 0]))

        # Create cube vertices
        vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Bottom face
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # Top face
        ]) * size / 2

        # Apply rotation (simplified - just around Z axis for now)
        cos_z, sin_z = np.cos(rotation[2]), np.sin(rotation[2])
        rotation_matrix = np.array([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ])
        vertices = vertices @ rotation_matrix.T

        # Translate to position
        vertices += pos

        # Define cube faces
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
            [vertices[4], vertices[7], vertices[3], vertices[0]]   # Left
        ]

        # Create 3D polygon collection
        poly3d = Poly3DCollection(
            faces, alpha=0.8, facecolor=color, edgecolor='black', linewidth=0.5)
        ax.add_collection3d(poly3d)

    def _render_sphere(self, ax: Axes3D, particle: Dict):
        """Render a spherical particle."""
        pos = particle['position']
        radius = particle['radius']
        color = particle['color']

        # Create sphere using scatter plot
        ax.scatter(pos[0], pos[1], pos[2],
                   s=radius*500, c=color, alpha=0.8,
                   edgecolors='black', linewidth=0.5)

    def _add_stress_arrows(self, ax: Axes3D, particles: List[Dict]):
        """Add stress direction arrows (like σx, σy, σz in the image)."""
        # Add coordinate system arrows at the corner
        bounds = self.parameters.get(
            'axis_bounds', {'x': [-10, 10], 'y': [-10, 10], 'z': [-10, 10]})

        # Origin for stress arrows
        origin_x = bounds['x'][0] + 2
        origin_y = bounds['y'][0] + 2
        origin_z = bounds['z'][0] + 2

        # Arrow length
        arrow_length = 3.0

        # σx arrow (red)
        ax.quiver(origin_x, origin_y, origin_z,
                  arrow_length, 0, 0,
                  color='red', arrow_length_ratio=0.1, linewidth=3)
        ax.text(origin_x + arrow_length + 0.5, origin_y, origin_z,
                'σₓ(εₓ)', color='red', fontsize=12, fontweight='bold')

        # σy arrow (green)
        ax.quiver(origin_x, origin_y, origin_z,
                  0, arrow_length, 0,
                  color='green', arrow_length_ratio=0.1, linewidth=3)
        ax.text(origin_x, origin_y + arrow_length + 0.5, origin_z,
                'σᵧ(εᵧ)', color='green', fontsize=12, fontweight='bold')

        # σz arrow (blue)
        ax.quiver(origin_x, origin_y, origin_z,
                  0, 0, arrow_length,
                  color='blue', arrow_length_ratio=0.1, linewidth=3)
        ax.text(origin_x, origin_y, origin_z + arrow_length + 0.5,
                'σᵣ(εᵣ)', color='blue', fontsize=12, fontweight='bold')

    def _apply_dem_styling(self, ax: Axes3D, fig: Figure):
        """Apply DEM-specific styling."""
        # Set title
        ax.set_title('DEM Particle Assembly\nParticles with different shapes by different colors',
                     fontsize=14, fontweight='bold', pad=20)

        # Set axis bounds
        bounds = self.parameters.get(
            'axis_bounds', {'x': [-10, 10], 'y': [-10, 10], 'z': [-10, 10]})
        ax.set_xlim(bounds['x'])
        ax.set_ylim(bounds['y'])
        ax.set_zlim(bounds['z'])

        # Remove axis labels for cleaner look (like in the image)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')

        # Hide axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Make panes transparent
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Make grid lines more subtle
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)

        # Set background
        fig.patch.set_facecolor('white')

        # Adjust viewing angle for better perspective
        ax.view_init(elev=20, azim=45)
