#!/usr/bin/env python3
"""
Enhanced Visualization Engine Integration
Connects embedded OpenGL renderer with existing pore analysis
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import logging
from pathlib import Path

# Import existing components
try:
    from .embedded_opengl_renderer import EmbeddedPoreVisualizationWidget
    from .visualization_engine import VisualizationEngine
    from .data_manager import DataManager
    from .pore_analyzer import PoreAnalyzer
    ENHANCED_RENDERING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Enhanced rendering not available: {e}")
    ENHANCED_RENDERING_AVAILABLE = False

# Try to import C++ renderer
try:
    from .cpp_extensions import fast_renderer_cpp
    CPP_RENDERER_AVAILABLE = True
except ImportError:
    CPP_RENDERER_AVAILABLE = False


class EnhancedPoreVisualizationEngine:
    """
    Advanced pore visualization engine with embedded OpenGL and C++ acceleration
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_manager = DataManager()
        self.pore_analyzer = PoreAnalyzer()
        self.visualization_engine = VisualizationEngine()

        # Enhanced rendering components
        self.opengl_widget = None
        self.cpp_renderer = None

        # Current data
        self.current_pore_data = None
        self.current_bond_data = None
        self.current_df = None

        self._setup_enhanced_rendering()

    def _setup_enhanced_rendering(self):
        """Setup enhanced rendering components"""
        if ENHANCED_RENDERING_AVAILABLE:
            try:
                self.opengl_widget = EmbeddedPoreVisualizationWidget()
                self.logger.info("✓ Embedded OpenGL renderer initialized")
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize OpenGL renderer: {e}")

        if CPP_RENDERER_AVAILABLE:
            try:
                self.cpp_renderer = fast_renderer_cpp.FastRenderer(
                    use_threading=True)
                self.logger.info("✓ C++ fast renderer initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize C++ renderer: {e}")

    def load_data(self, file_path: str) -> bool:
        """Load pore data from file"""
        try:
            self.current_df = self.data_manager.load_data(file_path)
            if self.current_df is not None:
                self.logger.info(f"Data loaded: {len(self.current_df)} rows")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            return False

    def create_realistic_pore_network(self,
                                      board_type: str = 'T1',
                                      connection_threshold: float = 2.0,
                                      min_pore_size: float = 10.0,
                                      max_pore_size: float = 1000.0) -> Tuple[List[Dict], List[Dict]]:
        """
        Create realistic 3D pore network with atomic-style connections
        """
        if self.current_df is None:
            raise ValueError("No data loaded")

        # Filter data for the specified board
        board_data = self.current_df[self.current_df.columns[self.current_df.columns.str.contains(
            board_type)]]
        if board_data.empty:
            # Use first available columns
            board_data = self.current_df.iloc[:, 1:4]

        # Extract pore size and volume data
        pore_diameters = board_data.iloc[:, 0].dropna()
        pore_volumes = board_data.iloc[:, 1].dropna(
        ) if board_data.shape[1] > 1 else pore_diameters * 0.1

        # Filter by size range
        valid_mask = (pore_diameters >= min_pore_size) & (
            pore_diameters <= max_pore_size)
        pore_diameters = pore_diameters[valid_mask]
        pore_volumes = pore_volumes[valid_mask]

        # Generate 3D positions using realistic spatial distribution
        positions = self._generate_realistic_positions(
            len(pore_diameters), pore_diameters)

        # Create pore objects
        pore_data = []
        for i, (diameter, volume, pos) in enumerate(zip(pore_diameters, pore_volumes, positions)):
            # Convert diameter (nm) to visualization radius
            radius = (diameter / 1000.0) * 0.5  # Convert nm to relative units

            # Color based on size (scientific visualization)
            size_factor = (diameter - min_pore_size) / \
                (max_pore_size - min_pore_size)
            color = self._get_scientific_color(size_factor, volume)

            pore_data.append({
                'id': i,
                'position': pos,
                'radius': max(radius, 0.05),  # Minimum visible size
                'diameter': diameter,
                'volume': volume,
                'color': color
            })

        # Generate bonds using physical connectivity rules
        bond_data = self._generate_physical_bonds(
            pore_data, connection_threshold)

        self.current_pore_data = pore_data
        self.current_bond_data = bond_data

        self.logger.info(
            f"Created realistic network: {len(pore_data)} pores, {len(bond_data)} bonds")
        return pore_data, bond_data

    def _generate_realistic_positions(self, n_pores: int, diameters: pd.Series) -> np.ndarray:
        """Generate realistic 3D positions based on pore size distribution"""
        positions = []

        # Use different distribution strategies based on pore size
        large_pores = diameters > diameters.median()

        for i, (diameter, is_large) in enumerate(zip(diameters, large_pores)):
            if is_large:
                # Large pores tend to be more centrally located
                pos = np.random.normal(0, 2, 3)
            else:
                # Small pores can be more distributed
                pos = np.random.uniform(-4, 4, 3)

            # Avoid overlaps with existing pores
            max_attempts = 50
            attempts = 0
            while attempts < max_attempts and len(positions) > 0:
                too_close = False
                for existing_pos, existing_diameter in zip(positions, diameters[:len(positions)]):
                    distance = np.linalg.norm(pos - existing_pos)
                    # Convert to relative units
                    min_distance = (diameter + existing_diameter) / 2000.0
                    if distance < min_distance:
                        too_close = True
                        break

                if not too_close:
                    break

                # Try new position
                if is_large:
                    pos = np.random.normal(0, 2, 3)
                else:
                    pos = np.random.uniform(-4, 4, 3)
                attempts += 1

            positions.append(pos)

        return np.array(positions)

    def _get_scientific_color(self, size_factor: float, volume: float) -> List[float]:
        """Get scientific visualization color based on pore properties"""
        # Multi-factor coloring: size + volume
        # Normalize volume influence
        volume_factor = min(volume / (volume + 0.1), 1.0)

        # Create color mixing
        if size_factor < 0.3:  # Small pores - blue
            r = 0.2 + size_factor * 0.3
            g = 0.4 + size_factor * 0.4
            b = 0.8 + size_factor * 0.2
        elif size_factor < 0.7:  # Medium pores - green/yellow
            r = 0.2 + size_factor * 0.6
            g = 0.8
            b = 0.2 + (1 - size_factor) * 0.6
        else:  # Large pores - red/orange
            r = 0.9
            g = 0.6 - (size_factor - 0.7) * 0.4
            b = 0.1

        # Apply volume influence
        alpha = 0.6 + volume_factor * 0.3

        return [r, g, b, alpha]

    def _generate_physical_bonds(self, pore_data: List[Dict], threshold: float) -> List[Dict]:
        """Generate bonds based on physical connectivity rules"""
        bonds = []

        for i, pore1 in enumerate(pore_data):
            for j, pore2 in enumerate(pore_data[i+1:], i+1):
                distance = np.linalg.norm(
                    pore1['position'] - pore2['position'])

                # Connection probability based on:
                # 1. Distance
                # 2. Pore sizes
                # 3. Physical feasibility

                max_connection_distance = threshold * \
                    (pore1['radius'] + pore2['radius'])

                if distance <= max_connection_distance:
                    # Calculate bond properties
                    bond_radius = min(pore1['radius'], pore2['radius']) * 0.3

                    # Bond color based on connection strength
                    strength = 1.0 - (distance / max_connection_distance)
                    bond_color = [
                        0.8 * strength + 0.2,
                        0.6 * strength + 0.4,
                        0.2 + 0.3 * strength,
                        0.7 * strength + 0.3
                    ]

                    bonds.append({
                        'start': pore1['position'],
                        'end': pore2['position'],
                        'radius': bond_radius,
                        'color': bond_color,
                        'pore1_id': pore1['id'],
                        'pore2_id': pore2['id'],
                        'strength': strength
                    })

        return bonds

    def create_ultra_realistic_view(self) -> bool:
        """Create ultra-realistic 3D view with all enhancements"""
        if not self.current_pore_data:
            self.logger.warning("No pore data available for realistic view")
            return False

        try:
            # Update OpenGL renderer if available
            if self.opengl_widget:
                self.opengl_widget.update_visualization(
                    self.current_pore_data,
                    self.current_bond_data
                )
                self.logger.info("✓ OpenGL view updated")

            # Update C++ renderer if available
            if self.cpp_renderer:
                self._update_cpp_renderer()
                self.logger.info("✓ C++ renderer updated")

            return True

        except Exception as e:
            self.logger.error(f"Failed to create realistic view: {e}")
            return False

    def _update_cpp_renderer(self):
        """Update C++ renderer with current data"""
        if not self.cpp_renderer or not self.current_pore_data:
            return

        # Convert pore data to C++ format
        spheres = []
        for pore in self.current_pore_data:
            sphere = fast_renderer_cpp.Sphere(
                fast_renderer_cpp.Point3D(*pore['position']),
                pore['radius'],
                pore['color']
            )
            spheres.append(sphere)

        # Convert bond data to C++ format
        bonds = []
        if self.current_bond_data:
            for bond in self.current_bond_data:
                cpp_bond = fast_renderer_cpp.Bond(
                    fast_renderer_cpp.Point3D(*bond['start']),
                    fast_renderer_cpp.Point3D(*bond['end']),
                    bond['radius'],
                    bond['color']
                )
                bonds.append(cpp_bond)

        # Update renderer
        self.cpp_renderer.set_spheres(spheres)
        self.cpp_renderer.set_bonds(bonds)
        self.cpp_renderer.mark_updated()

    def get_embedded_widget(self):
        """Get the embedded OpenGL widget for integration into GUI"""
        return self.opengl_widget

    def export_realistic_visualization(self, output_path: str,
                                       resolution: Tuple[int, int] = (1920, 1080)):
        """Export high-resolution realistic visualization"""
        if not self.cpp_renderer:
            self.logger.warning("C++ renderer not available for export")
            return False

        try:
            # Render to buffer
            width, height = resolution
            buffer = np.zeros((height, width, 4), dtype=np.float32)

            self.cpp_renderer.render_to_buffer(buffer, width, height)

            # Save to file
            import imageio
            # Convert float buffer to uint8
            image_data = (buffer * 255).astype(np.uint8)
            imageio.imwrite(output_path, image_data)

            self.logger.info(
                f"✓ Realistic visualization exported to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export visualization: {e}")
            return False

    def update_visualization_parameters(self, **params):
        """Update visualization parameters in real-time"""
        for key, value in params.items():
            if hasattr(self.visualization_engine, 'parameters'):
                self.visualization_engine.parameters[key] = value

        # Trigger updates
        if self.current_pore_data:
            self.create_ultra_realistic_view()

    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics"""
        if not self.current_pore_data or not self.current_bond_data:
            return {}

        pore_sizes = [pore['diameter'] for pore in self.current_pore_data]
        bond_lengths = [
            np.linalg.norm(bond['start'] - bond['end'])
            for bond in self.current_bond_data
        ]

        return {
            'total_pores': len(self.current_pore_data),
            'total_bonds': len(self.current_bond_data),
            'average_pore_size': np.mean(pore_sizes),
            'pore_size_std': np.std(pore_sizes),
            'average_bond_length': np.mean(bond_lengths),
            'connectivity': len(self.current_bond_data) / len(self.current_pore_data) if self.current_pore_data else 0,
            'network_density': len(self.current_bond_data) / (len(self.current_pore_data) ** 2) if len(self.current_pore_data) > 1 else 0
        }
