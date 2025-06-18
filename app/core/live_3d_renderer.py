#!/usr/bin/env python3
"""
Live Interactive 3D Pore Visualization Widget
Real-time updates with density filling and bonded connections
Enhanced with realistic rendering and no popup windows
"""

import time
from typing import Dict, Any, List, Tuple
import pandas as pd
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QPushButton, QCheckBox, QSpinBox, QComboBox
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import sys
import numpy as np
import matplotlib
# Force non-interactive backend to prevent popups
matplotlib.use('Agg')
# Disable all interactive features
plt.ioff()
matplotlib.rcParams['interactive'] = False


class LivePore3DRenderer(QWidget):
    """
    Live 3D pore network renderer with real-time updates
    """

    parameters_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Disable matplotlib interactive mode
        plt.ioff()

        # Data
        self.pore_data = None
        self.current_df = None
        self.spheres = []
        self.bonds = []

        # Visualization parameters
        self.params = {
            'bond_threshold': 2.0,
            'min_pore_size': 10.0,
            'max_pore_size': 1000.0,
            'density_fill': 0.5,
            'sphere_opacity': 0.7,
            'bond_opacity': 0.6,
            'color_scheme': 'viridis',
            'show_bonds': True,
            'animation_speed': 1.0,
            'board_type': 'T1'
        }

        # Setup UI
        self.setup_ui()

        # Live update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_visualization)
        self.auto_update = False

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout()

        # Create matplotlib figure
        self.figure = Figure(figsize=(10, 8), facecolor='black')
        self.canvas = FigureCanvas(self.figure)
        self.ax = None

        layout.addWidget(self.canvas, 1)

        # Controls panel
        controls_layout = QHBoxLayout()

        # Bond threshold
        controls_layout.addWidget(QLabel("Bond Threshold:"))
        self.bond_slider = QSlider(Qt.Horizontal)
        self.bond_slider.setRange(5, 50)
        self.bond_slider.setValue(20)
        self.bond_slider.valueChanged.connect(self.on_bond_threshold_changed)
        controls_layout.addWidget(self.bond_slider)

        self.bond_label = QLabel("2.0")
        controls_layout.addWidget(self.bond_label)

        # Density fill
        controls_layout.addWidget(QLabel("Density Fill:"))
        self.density_slider = QSlider(Qt.Horizontal)
        self.density_slider.setRange(10, 100)
        self.density_slider.setValue(50)
        self.density_slider.valueChanged.connect(self.on_density_changed)
        controls_layout.addWidget(self.density_slider)

        self.density_label = QLabel("50%")
        controls_layout.addWidget(self.density_label)

        # Pore size range
        controls_layout.addWidget(QLabel("Min Size:"))
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(1, 500)
        self.min_size_spin.setValue(10)
        self.min_size_spin.valueChanged.connect(self.on_size_range_changed)
        controls_layout.addWidget(self.min_size_spin)

        controls_layout.addWidget(QLabel("Max Size:"))
        self.max_size_spin = QSpinBox()
        self.max_size_spin.setRange(100, 5000)
        self.max_size_spin.setValue(1000)
        self.max_size_spin.valueChanged.connect(self.on_size_range_changed)
        controls_layout.addWidget(self.max_size_spin)

        # Board type
        controls_layout.addWidget(QLabel("Board:"))
        self.board_combo = QComboBox()
        self.board_combo.addItems(['T1', 'T2', 'T3'])
        self.board_combo.currentTextChanged.connect(self.on_board_changed)
        controls_layout.addWidget(self.board_combo)

        # Show bonds checkbox
        self.show_bonds_check = QCheckBox("Show Bonds")
        self.show_bonds_check.setChecked(True)
        self.show_bonds_check.toggled.connect(self.on_show_bonds_changed)
        controls_layout.addWidget(self.show_bonds_check)

        # Auto-update checkbox
        self.auto_update_check = QCheckBox("Live Update")
        self.auto_update_check.toggled.connect(self.toggle_auto_update)
        controls_layout.addWidget(self.auto_update_check)

        # Update button
        update_btn = QPushButton("Update")
        update_btn.clicked.connect(self.force_update)
        controls_layout.addWidget(update_btn)

        layout.addLayout(controls_layout)
        self.setLayout(layout)

    def load_data(self, data_path: str) -> bool:
        """Load pore data from file"""
        try:
            if data_path.endswith('.csv'):
                self.current_df = pd.read_csv(data_path)
            else:
                return False

            # Clean the data
            self.current_df = self.current_df.dropna()

            self.generate_pore_network()
            return True

        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def generate_pore_network(self):
        """Generate 3D pore network from data"""
        if self.current_df is None:
            return

        # Extract data for current board type
        # Try to find board-specific columns
        board_type = self.params['board_type']
        board_cols = [
            col for col in self.current_df.columns if board_type in col]
        if len(board_cols) >= 2:
            pore_sizes = self.current_df[board_cols[0]].dropna()
            pore_volumes = self.current_df[board_cols[1]].dropna()
        else:
            # Use first available columns
            pore_sizes = self.current_df.iloc[:, 0].dropna()
            pore_volumes = self.current_df.iloc[:, 1].dropna(
            ) if self.current_df.shape[1] > 1 else pore_sizes * 0.1

        # Convert to numeric, replacing non-numeric values with NaN
        pore_sizes = pd.to_numeric(pore_sizes, errors='coerce').dropna()
        pore_volumes = pd.to_numeric(pore_volumes, errors='coerce').dropna()

        # Ensure we have valid data
        if len(pore_sizes) == 0:
            print("Warning: No valid numeric pore size data found")
            return

        # Filter by size range
        min_size = self.params['min_pore_size']
        max_size = self.params['max_pore_size']

        size_mask = (pore_sizes >= min_size) & (pore_sizes <= max_size)
        pore_sizes = pore_sizes[size_mask]

        # Ensure volumes match
        if len(pore_volumes) >= len(pore_sizes):
            pore_volumes = pore_volumes[size_mask][:len(pore_sizes)]
        else:
            pore_volumes = np.tile(pore_volumes, len(
                pore_sizes) // len(pore_volumes) + 1)[:len(pore_sizes)]

        # Generate 3D positions
        n_pores = min(len(pore_sizes), 100)  # Limit for performance
        positions = self.generate_positions(n_pores, pore_sizes[:n_pores])

        # Create sphere data
        self.spheres = []
        for i in range(n_pores):
            size = pore_sizes.iloc[i] if i < len(pore_sizes) else 50
            volume = pore_volumes.iloc[i] if i < len(
                pore_volumes) else size * 0.1

            # Size factor for visualization
            radius = np.sqrt(size / 100.0) * 0.5

            # Color based on size and density
            color_value = (size - min_size) / (max_size -
                                               min_size) if max_size > min_size else 0.5

            self.spheres.append({
                'position': positions[i],
                'radius': radius,
                'size': size,
                'volume': volume,
                'color_value': color_value
            })

        # Generate bonds
        self.generate_bonds()

    def generate_positions(self, n_pores: int, sizes: pd.Series) -> np.ndarray:
        """Generate realistic 3D positions"""
        positions = []
        np.random.seed(42)  # For reproducible results

        for i in range(n_pores):
            # Size-based distribution
            size = sizes.iloc[i] if i < len(sizes) else 50

            if size > sizes.median():
                # Large pores tend to be central
                pos = np.random.normal(0, 1.5, 3)
            else:
                # Small pores more distributed
                pos = np.random.uniform(-3, 3, 3)

            # Avoid overlaps
            attempts = 0
            while attempts < 20 and len(positions) > 0:
                too_close = False
                for existing_pos in positions:
                    distance = np.linalg.norm(pos - existing_pos)
                    if distance < 0.5:  # Minimum separation
                        too_close = True
                        break

                if not too_close:
                    break

                # Try new position
                if size > sizes.median():
                    pos = np.random.normal(0, 1.5, 3)
                else:
                    pos = np.random.uniform(-3, 3, 3)
                attempts += 1

            positions.append(pos)

        return np.array(positions)

    def generate_bonds(self):
        """Generate bonds between pores"""
        self.bonds = []
        if not self.spheres or not self.params['show_bonds']:
            return

        threshold = self.params['bond_threshold']

        for i, sphere1 in enumerate(self.spheres):
            for j, sphere2 in enumerate(self.spheres[i+1:], i+1):
                distance = np.linalg.norm(
                    sphere1['position'] - sphere2['position'])
                max_dist = threshold * (sphere1['radius'] + sphere2['radius'])

                if distance <= max_dist:                    # Bond strength based on distance and sizes
                    strength = 1.0 - (distance / max_dist)
                    bond_radius = min(
                        sphere1['radius'], sphere2['radius']) * 0.2

                    self.bonds.append({
                        'start': sphere1['position'],
                        'end': sphere2['position'],
                        'radius': bond_radius,
                        'strength': strength
                    })

    def update_visualization(self):
        """Update the 3D visualization with enhanced realism"""
        if not self.spheres:
            return

        # Clear the figure completely
        self.figure.clear()

        # Ensure no pyplot interaction
        plt.ioff()

        # Create 3D subplot with black background
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.set_facecolor('black')
        self.figure.patch.set_facecolor('black')

        # Get density fill value
        density = self.params['density_fill']
        n_show = max(1, int(len(self.spheres) * density))

        # Sort spheres by size for density filling
        sorted_spheres = sorted(
            self.spheres, key=lambda x: x['size'], reverse=True)
        spheres_to_show = sorted_spheres[:n_show]

        # Enhanced sphere rendering with atomic-style appearance
        if spheres_to_show:
            positions = np.array([s['position'] for s in spheres_to_show])
            # Enhanced size scaling for better visibility
            sizes = np.array([s['radius'] * 2000 for s in spheres_to_show])
            colors = np.array([s['color_value'] for s in spheres_to_show])

            # Create multiple layers for depth effect
            for layer_alpha in [0.3, 0.6, 1.0]:
                layer_sizes = sizes * (1.0 - (1.0 - layer_alpha) * 0.3)
                scatter = self.ax.scatter(
                    positions[:, 0], positions[:, 1], positions[:, 2],
                    s=layer_sizes,
                    c=colors,
                    cmap=self.params['color_scheme'],
                    alpha=self.params['sphere_opacity'] * layer_alpha,
                    edgecolors='lightblue' if layer_alpha == 1.0 else 'none',
                    linewidth=1.0 if layer_alpha == 1.0 else 0,
                    depthshade=True
                )

        # Enhanced bond rendering with atomic-style connections
        if self.params['show_bonds'] and self.bonds:
            bond_density = max(1, int(len(self.bonds) * density))
            bonds_to_show = self.bonds[:bond_density]

            for bond in bonds_to_show:
                start, end = bond['start'], bond['end']

                # Enhanced bond appearance with gradient effect
                alpha = bond['strength'] * self.params['bond_opacity']

                # Create gradient along bond
                n_segments = 10
                for i in range(n_segments):
                    t1 = i / n_segments
                    t2 = (i + 1) / n_segments

                    seg_start = start + t1 * (end - start)
                    seg_end = start + t2 * (end - start)

                    # Vary color and thickness along bond
                    seg_strength = bond['strength'] * \
                        (1.0 - 0.3 * abs(0.5 - t1))
                    seg_alpha = alpha * seg_strength

                    color = plt.cm.plasma(seg_strength)[:3] + (seg_alpha,)
                    line_width = bond['radius'] * 15 * seg_strength

                    self.ax.plot(
                        [seg_start[0], seg_end[0]],
                        [seg_start[1], seg_end[1]],
                        [seg_start[2], seg_end[2]],
                        color=color,
                        linewidth=line_width,
                        alpha=seg_alpha,
                        solid_capstyle='round'
                    )

        # Enhanced styling with professional appearance
        self.ax.set_xlabel('X (μm)', color='white', fontsize=10)
        self.ax.set_ylabel('Y (μm)', color='white', fontsize=10)
        self.ax.set_zlabel('Z (μm)', color='white', fontsize=10)

        # Set axis limits for better view
        if spheres_to_show:
            positions = np.array([s['position'] for s in spheres_to_show])
            margin = 1.0
            self.ax.set_xlim(positions[:, 0].min() -
                             margin, positions[:, 0].max() + margin)
            self.ax.set_ylim(positions[:, 1].min() -
                             margin, positions[:, 1].max() + margin)
            self.ax.set_zlim(positions[:, 2].min() -
                             margin, positions[:, 2].max() + margin)

        # Professional grid and appearance
        self.ax.grid(True, alpha=0.3, color='gray')
        self.ax.tick_params(colors='white', labelsize=8)

        # Set background colors
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False

        # Make pane edges more subtle
        self.ax.xaxis.pane.set_edgecolor('gray')
        self.ax.yaxis.pane.set_edgecolor('gray')
        self.ax.zaxis.pane.set_edgecolor('gray')
        self.ax.xaxis.pane.set_alpha(0.1)
        self.ax.yaxis.pane.set_alpha(0.1)
        self.ax.zaxis.pane.set_alpha(0.1)

        # Add title with current parameters
        title = f"Live Pore Network | Density: {int(density*100)}% | "
        title += f"Bonds: {len(self.bonds) if self.bonds else 0} | "
        title += f"Board: {self.params['board_type']}"

        self.ax.set_title(title, color='white', fontsize=11, pad=20)

        # Force canvas update without showing
        self.canvas.draw_idle()

        # Emit parameters changed signal for other components
        self.parameters_changed.emit(self.params.copy())
        self.ax.set_zlabel('Z', color='white')
        self.ax.tick_params(colors='white')

        # Set equal aspect ratio
        max_range = 4
        self.ax.set_xlim([-max_range, max_range])
        self.ax.set_ylim([-max_range, max_range])
        self.ax.set_zlim([-max_range, max_range])

        # Title with current stats
        n_pores_shown = len(
            spheres_to_show) if 'spheres_to_show' in locals() else 0
        n_bonds_shown = len(
            self.bonds[:int(len(self.bonds) * density)]) if self.bonds else 0
        title = f"Live 3D Pore Network - {n_pores_shown} pores, {n_bonds_shown} bonds ({density*100:.0f}% density)"
        self.ax.set_title(title, color='white', fontsize=10)

        # Refresh canvas
        self.canvas.draw()

    def on_bond_threshold_changed(self, value):
        """Handle bond threshold change"""
        self.params['bond_threshold'] = value / 10.0
        self.bond_label.setText(f"{value/10.0:.1f}")
        self.generate_bonds()
        if self.auto_update:
            self.update_visualization()

    def on_density_changed(self, value):
        """Handle density change"""
        self.params['density_fill'] = value / 100.0
        self.density_label.setText(f"{value}%")
        if self.auto_update:
            self.update_visualization()

    def on_size_range_changed(self):
        """Handle size range change"""
        self.params['min_pore_size'] = self.min_size_spin.value()
        self.params['max_pore_size'] = self.max_size_spin.value()
        if self.auto_update:
            self.generate_pore_network()
            self.update_visualization()

    def on_board_changed(self, board_type):
        """Handle board type change"""
        self.params['board_type'] = board_type
        if self.auto_update:
            self.generate_pore_network()
            self.update_visualization()

    def on_show_bonds_changed(self, checked):
        """Handle show bonds toggle"""
        self.params['show_bonds'] = checked
        self.generate_bonds()
        if self.auto_update:
            self.update_visualization()

    def toggle_auto_update(self, checked):
        """Toggle auto-update mode"""
        self.auto_update = checked
        if checked:
            self.update_timer.start(100)  # Update every 100ms
        else:
            self.update_timer.stop()

    def force_update(self):
        """Force immediate update"""
        self.generate_pore_network()
        self.update_visualization()

    def set_parameters(self, **params):
        """Set visualization parameters"""
        self.params.update(params)
        self.generate_pore_network()
        self.update_visualization()


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = LivePore3DRenderer()
    widget.show()
    widget.resize(1000, 800)
    sys.exit(app.exec_())
