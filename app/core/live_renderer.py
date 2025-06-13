#!/usr/bin/env python3
"""
Live 3D Renderer for MIST-like real-time visualization
Handles real-time rendering, camera controls, and live updates
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import threading
import time
from typing import Dict, Any, Optional, Callable, List
import logging
from queue import Queue


class Live3DRenderer:
    """Real-time 3D renderer with live camera controls and updates."""

    def __init__(self, parameters: Dict[str, Any]):
        """Initialize live renderer."""
        self.parameters = parameters
        self.logger = logging.getLogger(__name__)

        # Animation and rendering state
        self.is_rendering = False
        self.animation = None
        self.figure = None
        self.ax = None

        # Camera state
        self.camera_elevation = 20
        self.camera_azimuth = 45
        self.camera_distance = 1.0

        # Data state
        self.current_data = None
        self.update_queue = Queue()

        # Callbacks
        self.update_callbacks = []

    def start_live_rendering(self, initial_data: Dict[str, Any]) -> Figure:
        """Start live rendering with initial data."""
        try:
            # Create figure
            self.figure = plt.figure(figsize=(12, 9))
            self.ax = self.figure.add_subplot(111, projection='3d')

            # Set initial data
            self.current_data = initial_data

            # Configure interactive mode
            plt.ion()

            # Start animation
            self.is_rendering = True
            self.animation = FuncAnimation(
                self.figure,
                self._animation_frame,
                interval=50,  # 20 FPS
                blit=False,
                cache_frame_data=False
            )

            # Initial render
            self._render_frame()

            self.logger.info("Live rendering started")
            return self.figure

        except Exception as e:
            self.logger.error(f"Error starting live rendering: {e}")
            raise

    def stop_live_rendering(self):
        """Stop live rendering."""
        self.is_rendering = False
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None

        plt.ioff()
        self.logger.info("Live rendering stopped")

    def update_camera(self, elevation: float = None, azimuth: float = None, distance: float = None):
        """Update camera parameters in real-time."""
        if elevation is not None:
            self.camera_elevation = elevation
        if azimuth is not None:
            self.camera_azimuth = azimuth
        if distance is not None:
            self.camera_distance = distance

        # Queue camera update
        self.update_queue.put({
            'type': 'camera',
            'elevation': self.camera_elevation,
            'azimuth': self.camera_azimuth,
            'distance': self.camera_distance
        })

    def update_data(self, new_data: Dict[str, Any]):
        """Update visualization data in real-time."""
        # Queue data update
        self.update_queue.put({
            'type': 'data',
            'data': new_data
        })

    def update_parameters(self, new_params: Dict[str, Any]):
        """Update visualization parameters in real-time."""
        self.parameters.update(new_params)

        # Queue parameter update
        self.update_queue.put({
            'type': 'parameters',
            'parameters': new_params
        })

    def add_update_callback(self, callback: Callable):
        """Add callback for update events."""
        self.update_callbacks.append(callback)

    def _animation_frame(self, frame):
        """Animation frame update function."""
        if not self.is_rendering:
            return

        # Process queued updates
        while not self.update_queue.empty():
            try:
                update = self.update_queue.get_nowait()
                self._process_update(update)
            except:
                break

        # Render frame
        self._render_frame()

        # Call update callbacks
        for callback in self.update_callbacks:
            try:
                callback(frame)
            except Exception as e:
                self.logger.debug(f"Error in update callback: {e}")

    def _process_update(self, update: Dict[str, Any]):
        """Process a queued update."""
        if update['type'] == 'camera':
            self.ax.view_init(
                elev=update['elevation'],
                azim=update['azimuth']
            )
            # Note: distance/zoom would require more complex handling

        elif update['type'] == 'data':
            self.current_data = update['data']

        elif update['type'] == 'parameters':
            # Parameters already updated in update_parameters
            pass

    def _render_frame(self):
        """Render current frame."""
        if not self.current_data or not self.ax:
            return

        # Clear previous frame
        self.ax.clear()

        # Render based on data type
        data_type = self.current_data.get('type', 'pore_network')

        if data_type == 'pore_network':
            self._render_pore_network()
        elif data_type == 'dem_particles':
            self._render_dem_particles()
        elif data_type == 'segmentation':
            self._render_segmentation()

        # Apply common styling
        self._apply_live_styling()

    def _render_pore_network(self):
        """Render pore network data."""
        data = self.current_data

        if 'positions' in data and 'sizes' in data:
            positions = data['positions']
            sizes = data['sizes']
            colors = data.get('colors', 'blue')

            # Handle color mapping properly
            if isinstance(colors, str) and colors in ['viridis', 'plasma', 'coolwarm', 'hot']:
                # Use colormap with numerical values
                color_values = np.arange(len(positions))
                self.ax.scatter(
                    positions[:, 0], positions[:, 1], positions[:, 2],
                    s=sizes, c=color_values, cmap=colors, alpha=self.parameters.get(
                        'opacity', 0.7)
                )
            else:
                # Use direct color specification
                self.ax.scatter(
                    positions[:, 0], positions[:, 1], positions[:, 2],
                    s=sizes, c=colors, alpha=self.parameters.get(
                        'opacity', 0.7)
                )

        # Add bonds if available
        if 'bonds' in data and self.parameters.get('show_bonds', True):
            bonds = data['bonds']
            for bond in bonds:
                start, end = bond
                self.ax.plot3D(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    'k-', alpha=0.3, linewidth=0.5
                )

    def _render_dem_particles(self):
        """Render DEM particle data."""
        data = self.current_data

        if 'particles' in data:
            particles = data['particles']

            for particle in particles:
                if particle['type'] == 'sphere':
                    pos = particle['position']
                    radius = particle['radius']
                    color = particle['color']

                    self.ax.scatter(
                        pos[0], pos[1], pos[2],
                        s=radius*500, c=color, alpha=0.8
                    )

                # Note: Cube rendering would require more complex 3D polygons

    def _render_segmentation(self):
        """Render segmentation data."""
        data = self.current_data

        if 'segments' in data:
            segments = data['segments']

            for i, segment in enumerate(segments):
                if 'positions' in segment:
                    positions = segment['positions']
                    color = segment.get('color', f'C{i}')

                    self.ax.scatter(
                        positions[:, 0], positions[:, 1], positions[:, 2],
                        c=color, alpha=0.6, s=50
                    )

    def _apply_live_styling(self):
        """Apply styling for live rendering."""
        # Set axis bounds
        bounds = self.parameters.get(
            'axis_bounds', {'x': [-10, 10], 'y': [-10, 10], 'z': [-10, 10]})
        self.ax.set_xlim(bounds['x'])
        self.ax.set_ylim(bounds['y'])
        self.ax.set_zlim(bounds['z'])

        # Set labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Set background
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False

        # Set grid
        self.ax.grid(True, alpha=0.3)


class LiveSegmentationProcessor:
    """Real-time segmentation processor for live analysis."""

    def __init__(self, parameters: Dict[str, Any]):
        """Initialize segmentation processor."""
        self.parameters = parameters
        self.logger = logging.getLogger(__name__)

        # Segmentation state
        self.current_threshold = 0.5
        self.segmentation_method = 'kmeans'
        self.num_segments = 3

    def process_live_segmentation(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """Process live segmentation of 3D data."""
        try:
            if self.segmentation_method == 'kmeans':
                return self._kmeans_segmentation(data)
            elif self.segmentation_method == 'threshold':
                return self._threshold_segmentation(data)
            elif self.segmentation_method == 'watershed':
                return self._watershed_segmentation(data)
            else:
                return self._default_segmentation(data)

        except Exception as e:
            self.logger.error(f"Error in live segmentation: {e}")
            return []

    def _kmeans_segmentation(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """K-means based segmentation."""
        from sklearn.cluster import KMeans

        # Reshape data for clustering
        if len(data.shape) == 2 and data.shape[1] >= 3:
            positions = data[:, :3]
        else:
            # Generate positions from data indices
            positions = np.array(np.meshgrid(
                np.arange(data.shape[0]),
                np.arange(data.shape[1]),
                np.arange(data.shape[2])
            )).T.reshape(-1, 3)
            positions = positions[data.flatten() > self.current_threshold]

        if len(positions) == 0:
            return []

        # Apply K-means
        kmeans = KMeans(n_clusters=self.num_segments,
                        random_state=42, n_init=10)
        labels = kmeans.fit_predict(positions)

        # Create segments
        segments = []
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']

        for i in range(self.num_segments):
            mask = labels == i
            if np.any(mask):
                segments.append({
                    'positions': positions[mask],
                    'color': colors[i % len(colors)],
                    'label': f'Segment {i+1}',
                    'size': np.sum(mask)
                })

        return segments

    def _threshold_segmentation(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """Threshold-based segmentation."""
        # Simple threshold segmentation
        above_threshold = data > self.current_threshold
        below_threshold = data <= self.current_threshold

        segments = []

        if np.any(above_threshold):
            positions = np.array(np.where(above_threshold)).T
            segments.append({
                'positions': positions,
                'color': 'red',
                'label': f'Above {self.current_threshold}',
                'size': len(positions)
            })

        if np.any(below_threshold):
            positions = np.array(np.where(below_threshold)).T
            segments.append({
                'positions': positions,
                'color': 'blue',
                'label': f'Below {self.current_threshold}',
                'size': len(positions)
            })

        return segments

    def _watershed_segmentation(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """Watershed-based segmentation."""
        try:
            from scipy import ndimage
            from skimage.segmentation import watershed
            from skimage.feature import peak_local_maxima

            # Find local maxima as seeds
            if len(data.shape) == 3:
                # 3D data
                local_maxima = peak_local_maxima(data, min_distance=3)
                markers = np.zeros_like(data, dtype=int)
                for i, peak in enumerate(local_maxima):
                    markers[tuple(peak)] = i + 1
            else:
                # 2D or other data - use default segmentation
                return self._default_segmentation(data)

            # Apply watershed
            labels = watershed(-data, markers)

            # Create segments
            segments = []
            colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']

            for i in range(1, labels.max() + 1):
                mask = labels == i
                if np.any(mask):
                    positions = np.array(np.where(mask)).T
                    segments.append({
                        'positions': positions,
                        'color': colors[(i-1) % len(colors)],
                        'label': f'Region {i}',
                        'size': len(positions)
                    })

            return segments

        except ImportError:
            self.logger.warning("Watershed segmentation requires scikit-image")
            return self._default_segmentation(data)

    def _default_segmentation(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """Default segmentation when others fail."""
        # Simple random segmentation
        if len(data.shape) == 2 and data.shape[1] >= 3:
            positions = data[:, :3]
        else:
            positions = np.random.uniform(-5, 5, (100, 3))

        # Random assignment to segments
        labels = np.random.randint(0, self.num_segments, len(positions))

        segments = []
        colors = ['red', 'green', 'blue']

        for i in range(self.num_segments):
            mask = labels == i
            if np.any(mask):
                segments.append({
                    'positions': positions[mask],
                    'color': colors[i % len(colors)],
                    'label': f'Random Segment {i+1}',
                    'size': np.sum(mask)
                })

        return segments

    def update_segmentation_parameters(self, threshold: float = None, method: str = None, num_segments: int = None):
        """Update segmentation parameters."""
        if threshold is not None:
            self.current_threshold = threshold
        if method is not None:
            self.segmentation_method = method
        if num_segments is not None:
            self.num_segments = num_segments
