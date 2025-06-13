#!/usr/bin/env python3
"""
High-Performance Renderer - Python interface to C++ fast renderer
Provides MIST-like real-time visualization capabilities
"""

import numpy as np
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging

try:
    # Try to import the compiled C++ extension
    sys.path.append(str(Path(__file__).parent / 'cpp_extensions'))

    # Log C++ extension loading attempt
    try:
        from logger import get_logger
        logger = get_logger()
        logger.log_cpp_operation(
            "import_attempt", True, "Attempting to load fast_renderer_cpp")
    except:
        logger = None

    import fast_renderer_cpp
    CPP_AVAILABLE = True
    print("✓ C++ fast renderer loaded successfully")

    if logger:
        logger.log_cpp_operation("load_success", True,
                                 "fast_renderer_cpp loaded successfully")

except ImportError as e:
    if logger:
        logger.log_cpp_operation(
            "primary_import", False, f"Primary import failed: {e}")

    try:
        # Alternative import path
        from .cpp_extensions import fast_renderer_cpp
        CPP_AVAILABLE = True
        print("✓ C++ fast renderer loaded successfully")

        if logger:
            logger.log_cpp_operation(
                "alternative_import", True, "Alternative import successful")

    except ImportError as e2:
        CPP_AVAILABLE = False
        error_msg = f"C++ fast renderer not available: {e2}"
        print(f"Warning: {error_msg}")

        if logger:
            logger.log_cpp_operation("all_imports", False, error_msg)


class HighPerformanceRenderer:
    """
    Python wrapper for the C++ fast renderer.
    Provides MIST-like real-time 3D visualization with interactive controls.
    """

    def __init__(self, use_cpp: bool = True, use_threading: bool = True):
        self.logger = logging.getLogger(__name__)
        self.use_cpp = use_cpp and CPP_AVAILABLE

        if self.use_cpp:
            self.cpp_renderer = fast_renderer_cpp.FastRenderer(use_threading)
            self.logger.info("Initialized C++ fast renderer")
        else:
            self.cpp_renderer = None
            self.logger.warning("Using fallback Python renderer")

        # Fallback Python implementation
        self.spheres = []
        self.bonds = []
        self.view_params = self._default_view_params()

    def _default_view_params(self) -> Dict[str, Any]:
        """Get default view parameters."""
        return {
            'elevation': 20.0,
            'azimuth': 45.0,
            'zoom': 1.0,
            'camera_position': [0, 0, 10],
            'target_position': [0, 0, 0],
            'lighting_intensity': 0.8,
            'background_color': [1.0, 1.0, 1.0]
        }

    def set_spheres_from_data(self,
                              centers: np.ndarray,
                              radii: np.ndarray,
                              colors: Optional[np.ndarray] = None,
                              opacities: Optional[np.ndarray] = None) -> None:
        """
        Set spheres from numpy arrays.

        Args:
            centers: Nx3 array of sphere centers
            radii: N array of sphere radii
            colors: Nx4 array of RGBA colors (optional)
            opacities: N array of opacities (optional)
        """
        n_spheres = len(centers)

        if colors is None:
            colors = np.random.rand(n_spheres, 4)
            colors[:, 3] = 1.0  # Full alpha

        if opacities is None:
            opacities = np.ones(n_spheres)

        if self.use_cpp:
            spheres = []
            for i in range(n_spheres):
                center = fast_renderer_cpp.Point3D(
                    centers[i, 0], centers[i, 1], centers[i, 2])
                color = [colors[i, 0], colors[i, 1],
                         colors[i, 2], colors[i, 3]]
                sphere = fast_renderer_cpp.Sphere(
                    center, radii[i], color, opacities[i])
                spheres.append(sphere)

            self.cpp_renderer.set_spheres(spheres)
        else:
            # Fallback implementation
            self.spheres = []
            for i in range(n_spheres):
                self.spheres.append({
                    'center': centers[i],
                    'radius': radii[i],
                    'color': colors[i],
                    'opacity': opacities[i]
                })

    def set_bonds_from_connections(self,
                                   start_points: np.ndarray,
                                   end_points: np.ndarray,
                                   thicknesses: Optional[np.ndarray] = None,
                                   colors: Optional[np.ndarray] = None) -> None:
        """
        Set bonds from connection arrays.

        Args:
            start_points: Nx3 array of bond start points
            end_points: Nx3 array of bond end points
            thicknesses: N array of bond thicknesses (optional)
            colors: Nx4 array of RGBA colors (optional)
        """
        n_bonds = len(start_points)

        if thicknesses is None:
            thicknesses = np.full(n_bonds, 0.5)

        if colors is None:
            colors = np.full((n_bonds, 4), [0.5, 0.5, 0.5, 0.8])

        if self.use_cpp:
            bonds = []
            for i in range(n_bonds):
                start = fast_renderer_cpp.Point3D(
                    start_points[i, 0], start_points[i, 1], start_points[i, 2])
                end = fast_renderer_cpp.Point3D(
                    end_points[i, 0], end_points[i, 1], end_points[i, 2])
                color = [colors[i, 0], colors[i, 1],
                         colors[i, 2], colors[i, 3]]
                bond = fast_renderer_cpp.Bond(
                    start, end, thicknesses[i], color)
                bonds.append(bond)

            self.cpp_renderer.set_bonds(bonds)
        else:
            # Fallback implementation
            self.bonds = []
            for i in range(n_bonds):
                self.bonds.append({
                    'start': start_points[i],
                    'end': end_points[i],
                    'thickness': thicknesses[i],
                    'color': colors[i]
                })

    def update_view(self, elevation: float, azimuth: float, zoom: float) -> None:
        """Update view parameters for real-time interaction."""
        if self.use_cpp:
            self.cpp_renderer.update_view(elevation, azimuth, zoom)
        else:
            self.view_params.update({
                'elevation': elevation,
                'azimuth': azimuth,
                'zoom': zoom
            })

    def set_camera_position(self, position: List[float], target: List[float]) -> None:
        """Set camera position and target."""
        if self.use_cpp:
            pos = fast_renderer_cpp.Point3D(
                position[0], position[1], position[2])
            tgt = fast_renderer_cpp.Point3D(target[0], target[1], target[2])
            self.cpp_renderer.set_camera_position(pos, tgt)
        else:
            self.view_params.update({
                'camera_position': position,
                'target_position': target
            })

    def set_lighting(self, intensity: float) -> None:
        """Set lighting intensity."""
        if self.use_cpp:
            self.cpp_renderer.set_lighting(intensity)
        else:
            self.view_params['lighting_intensity'] = intensity

    def set_background_color(self, r: float, g: float, b: float) -> None:
        """Set background color."""
        if self.use_cpp:
            self.cpp_renderer.set_background_color(r, g, b)
        else:
            self.view_params['background_color'] = [r, g, b]

    def render_frame(self, width: int, height: int) -> np.ndarray:
        """
        Render a frame to a numpy array.

        Returns:
            RGBA image as numpy array of shape (height, width, 4)
        """
        if self.use_cpp:
            # Get flat buffer from C++
            buffer = self.cpp_renderer.render_frame(width, height)
            # Reshape to image format
            image = np.array(buffer, dtype=np.float32)
            return image.reshape((height, width, 4))
        else:
            # Fallback Python implementation
            return self._python_render_fallback(width, height)

    def _python_render_fallback(self, width: int, height: int) -> np.ndarray:
        """Fallback Python renderer (simplified)."""
        # Create a simple gradient background as fallback
        image = np.zeros((height, width, 4), dtype=np.float32)

        # Background gradient
        bg_color = self.view_params['background_color']
        for y in range(height):
            factor = y / height
            image[y, :, 0] = bg_color[0] * (1.0 - factor * 0.2)
            image[y, :, 1] = bg_color[1] * (1.0 - factor * 0.2)
            image[y, :, 2] = bg_color[2] * (1.0 - factor * 0.2)
            image[y, :, 3] = 1.0

        # Draw simple sphere representations
        center_x, center_y = width // 2, height // 2
        # Limit for performance
        for sphere in self.spheres[:min(100, len(self.spheres))]:
            # Simple projection
            x = int(center_x + sphere['center'][0]
                    * self.view_params['zoom'] * 10)
            y = int(center_y + sphere['center'][1]
                    * self.view_params['zoom'] * 10)
            r = max(1, int(sphere['radius'] * self.view_params['zoom'] * 5))

            if 0 <= x < width and 0 <= y < height:
                # Draw a simple circle
                for dy in range(-r, r+1):
                    for dx in range(-r, r+1):
                        if dx*dx + dy*dy <= r*r:
                            px, py = x + dx, y + dy
                            if 0 <= px < width and 0 <= py < height:
                                alpha = sphere['opacity'] * 0.7
                                image[py, px, :3] = (
                                    1 - alpha) * image[py, px, :3] + alpha * sphere['color'][:3]

        return image

    def needs_update(self) -> bool:
        """Check if renderer needs update."""
        if self.use_cpp:
            return self.cpp_renderer.needs_update()
        return True  # Always update for fallback

    def mark_updated(self) -> None:
        """Mark renderer as updated."""
        if self.use_cpp:
            self.cpp_renderer.mark_updated()

    def get_statistics(self) -> Dict[str, Any]:
        """Get rendering statistics."""
        if self.use_cpp:
            return {
                'sphere_count': self.cpp_renderer.get_sphere_count(),
                'bond_count': self.cpp_renderer.get_bond_count(),
                'using_cpp': True,
                'renderer_type': 'C++ Fast Renderer'
            }
        else:
            return {
                'sphere_count': len(self.spheres),
                'bond_count': len(self.bonds),
                'using_cpp': False,
                'renderer_type': 'Python Fallback Renderer'
            }

    def pick_objects(self, screen_x: float, screen_y: float, width: int, height: int) -> List[int]:
        """Pick objects at screen coordinates."""
        if self.use_cpp:
            return self.cpp_renderer.pick_spheres(screen_x, screen_y, width, height)
        else:
            # Simple fallback picking
            return []

    def update_opacity(self, opacity: float) -> None:
        """Update opacity for all spheres."""
        if self.use_cpp:
            # Update all spheres with new opacity
            if hasattr(self, '_current_opacity'):
                self._current_opacity = opacity
            else:
                self._current_opacity = opacity
            # Force recreation of spheres if needed
            self.cpp_renderer.mark_updated()
        else:
            # Update fallback sphere data
            for sphere in self.spheres:
                sphere['opacity'] = opacity

    def update_size_multiplier(self, multiplier: float) -> None:
        """Update size multiplier for all spheres."""
        if self.use_cpp:
            # Store the multiplier for future sphere updates
            if hasattr(self, '_size_multiplier'):
                self._size_multiplier = multiplier
            else:
                self._size_multiplier = multiplier
            # Force recreation of spheres if needed
            self.cpp_renderer.mark_updated()
        else:
            # Update fallback sphere data
            for sphere in self.spheres:
                sphere['radius'] *= multiplier

    def update_colormap(self, colormap: str) -> None:
        """Update colormap for sphere coloring."""
        if self.use_cpp:
            # Store the colormap for future sphere updates
            if hasattr(self, '_colormap'):
                self._colormap = colormap
            else:
                self._colormap = colormap
            # Force recreation of spheres if needed
            self.cpp_renderer.mark_updated()
        else:
            # Update fallback sphere colors based on colormap
            import matplotlib.pyplot as plt
            try:
                cmap = plt.cm.get_cmap(colormap)
                for i, sphere in enumerate(self.spheres):
                    color_val = i / max(1, len(self.spheres) - 1)
                    rgba = cmap(color_val)
                    sphere['color'] = rgba
            except Exception:
                # Fallback to default colors if colormap fails
                pass

    def update_lighting(self, intensity: float) -> None:
        """Update lighting intensity."""
        self.set_lighting(intensity)

    def update_render_settings(self, settings: Dict[str, Any]) -> None:
        """Update various render settings."""
        if self.use_cpp:
            # Apply settings to C++ renderer
            for key, value in settings.items():
                if key == 'show_bonds':
                    # Toggle bond visibility - would need C++ support
                    pass
                elif key == 'bond_thickness':
                    # Update bond thickness - would need C++ support
                    pass
                elif key == 'prism_opacity':
                    # Update prism opacity if applicable
                    pass
                # Add more settings as needed
            self.cpp_renderer.mark_updated()
        else:
            # Update fallback settings
            for key, value in settings.items():
                if key in self.view_params:
                    self.view_params[key] = value


def create_spheres_from_pore_data(diameters: np.ndarray,
                                  volumes: np.ndarray,
                                  num_spheres: int = 1000,
                                  bounds: Tuple[float, float, float] = (10.0, 10.0, 10.0)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sphere positions and radii from pore data.

    Args:
        diameters: Pore diameters
        volumes: Cumulative intrusion volumes
        num_spheres: Number of spheres to generate
        bounds: (x, y, z) bounds for sphere positions

    Returns:
        Tuple of (centers, radii) arrays
    """
    # Generate random positions within bounds
    centers = np.random.uniform(-np.array(bounds),
                                np.array(bounds), (num_spheres, 3))

    # Map pore diameters to sphere radii with some scaling
    if len(diameters) > 0:
        # Filter out invalid diameters (NaN, inf, zero, negative)
        valid_diameters = diameters[np.isfinite(diameters) & (diameters > 0)]

        if len(valid_diameters) > 0:
            min_diameter, max_diameter = np.min(
                valid_diameters), np.max(valid_diameters)

            # Ensure max_diameter is not zero to avoid division by zero
            if max_diameter > 0:
                # Sample diameters based on the distribution
                indices = np.random.choice(
                    len(valid_diameters), num_spheres, replace=True)
                sampled_diameters = valid_diameters[indices]
                # Convert to radii with scaling, avoiding division by zero
                radii = (sampled_diameters / max_diameter) * \
                    1.5  # Increased scale to reasonable size for better visibility

                # Ensure no NaN or invalid values in final radii
                radii = np.nan_to_num(radii, nan=0.3, posinf=1.0, neginf=0.3)
                # Ensure minimum radius to avoid invisible spheres - increased minimum
                radii = np.maximum(radii, 0.2)
            else:
                # All diameters are zero, use default radii - increased size
                radii = np.random.uniform(0.3, 1.0, num_spheres)
        else:
            # No valid diameters, use default radii - increased size
            radii = np.random.uniform(0.3, 1.0, num_spheres)
    else:
        radii = np.random.uniform(0.1, 0.5, num_spheres)

    return centers, radii


def generate_bonds_from_spheres(centers: np.ndarray,
                                radii: np.ndarray,
                                max_distance: float = 3.0,
                                max_bonds_per_sphere: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate bonds between nearby spheres.

    Args:
        centers: Sphere centers
        radii: Sphere radii
        max_distance: Maximum bond distance
        max_bonds_per_sphere: Maximum bonds per sphere

    Returns:
        Tuple of (start_points, end_points) for bonds
    """
    start_points = []
    end_points = []

    for i, center_i in enumerate(centers):
        bonds_count = 0
        for j, center_j in enumerate(centers):
            if i >= j or bonds_count >= max_bonds_per_sphere:
                continue

            distance = np.linalg.norm(center_i - center_j)
            if distance <= max_distance and distance > (radii[i] + radii[j]):
                start_points.append(center_i)
                end_points.append(center_j)
                bonds_count += 1

    return np.array(start_points), np.array(end_points)
