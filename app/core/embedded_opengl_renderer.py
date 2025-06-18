#!/usr/bin/env python3
"""
Advanced Embedded 3D Pore Visualization Widget
PyQt5 + PyOpenGL + Modern Shaders for MIST-like realistic rendering
"""

import sys
import numpy as np
import moderngl as mgl
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QPushButton, QFrame
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtOpenGL import QOpenGLWidget
from PyQt5.QtGui import QMatrix4x4, QVector3D
import pyrr
from OpenGL.GL import *
import math
import time


class ModernPoreRenderer(QOpenGLWidget):
    """
    High-performance embedded 3D pore visualization using ModernGL and shaders
    Similar to MIST rendering quality with atomic-style connections
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ctx = None
        self.program = None
        self.sphere_program = None
        self.bond_program = None

        # Camera controls
        self.camera_pos = np.array([0.0, 0.0, 10.0], dtype=np.float32)
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Mouse interaction
        self.last_mouse_pos = None
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.zoom_factor = 1.0

        # Data
        self.pore_data = []
        self.bond_data = []
        self.sphere_vao = None
        self.bond_vao = None

        # Animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(16)  # ~60 FPS

        # Enable mouse tracking
        self.setMouseTracking(True)

    def initializeGL(self):
        """Initialize OpenGL context and shaders"""
        try:
            self.ctx = mgl.create_context()

            # Enable depth testing and blending
            self.ctx.enable(mgl.DEPTH_TEST | mgl.BLEND)
            self.ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA

            # Create shader programs
            self._create_sphere_shader()
            self._create_bond_shader()

            # Initialize with sample data
            self._create_sample_pore_data()
            self._setup_sphere_geometry()
            self._setup_bond_geometry()

        except Exception as e:
            print(f"OpenGL initialization error: {e}")

    def _create_sphere_shader(self):
        """Create realistic sphere rendering shader with lighting"""
        vertex_shader = """
        #version 330 core
        
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec3 normal;
        layout (location = 2) in vec3 sphere_center;
        layout (location = 3) in float sphere_radius;
        layout (location = 4) in vec4 sphere_color;
        
        uniform mat4 mvp_matrix;
        uniform mat4 model_matrix;
        uniform mat4 view_matrix;
        uniform vec3 light_pos;
        uniform vec3 camera_pos;
        
        out vec3 frag_pos;
        out vec3 frag_normal;
        out vec4 frag_color;
        out vec3 light_dir;
        out vec3 view_dir;
        
        void main() {
            // Scale and translate sphere
            vec3 world_pos = sphere_center + position * sphere_radius;
            frag_pos = world_pos;
            frag_normal = normal;
            frag_color = sphere_color;
            
            // Lighting calculations
            light_dir = normalize(light_pos - world_pos);
            view_dir = normalize(camera_pos - world_pos);
            
            gl_Position = mvp_matrix * vec4(world_pos, 1.0);
        }
        """

        fragment_shader = """
        #version 330 core
        
        in vec3 frag_pos;
        in vec3 frag_normal;
        in vec4 frag_color;
        in vec3 light_dir;
        in vec3 view_dir;
        
        out vec4 final_color;
        
        uniform float ambient_strength;
        uniform float specular_strength;
        uniform float shininess;
        
        void main() {
            // Ambient lighting
            vec3 ambient = ambient_strength * frag_color.rgb;
            
            // Diffuse lighting
            float diff = max(dot(normalize(frag_normal), light_dir), 0.0);
            vec3 diffuse = diff * frag_color.rgb;
            
            // Specular lighting (Phong)
            vec3 reflect_dir = reflect(-light_dir, normalize(frag_normal));
            float spec = pow(max(dot(view_dir, reflect_dir), 0.0), shininess);
            vec3 specular = specular_strength * spec * vec3(1.0);
            
            // Rim lighting for depth
            float rim = 1.0 - max(dot(view_dir, normalize(frag_normal)), 0.0);
            rim = smoothstep(0.6, 1.0, rim);
            vec3 rim_color = rim * vec3(0.2, 0.4, 0.8) * 0.5;
            
            // Final color with transparency
            vec3 result = ambient + diffuse + specular + rim_color;
            final_color = vec4(result, frag_color.a);
        }
        """

        try:
            self.sphere_program = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader
            )
        except Exception as e:
            print(f"Sphere shader compilation error: {e}")

    def _create_bond_shader(self):
        """Create realistic bond/cylinder rendering shader"""
        vertex_shader = """
        #version 330 core
        
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec3 normal;
        layout (location = 2) in vec3 bond_start;
        layout (location = 3) in vec3 bond_end;
        layout (location = 4) in float bond_radius;
        layout (location = 5) in vec4 bond_color;
        
        uniform mat4 mvp_matrix;
        uniform vec3 light_pos;
        uniform vec3 camera_pos;
        
        out vec3 frag_pos;
        out vec3 frag_normal;
        out vec4 frag_color;
        out vec3 light_dir;
        out vec3 view_dir;
        
        void main() {
            // Calculate bond transformation
            vec3 bond_dir = normalize(bond_end - bond_start);
            float bond_length = length(bond_end - bond_start);
            vec3 bond_center = (bond_start + bond_end) * 0.5;
            
            // Transform cylinder to bond orientation
            vec3 local_pos = position;
            local_pos.y *= bond_length * 0.5;
            local_pos.xz *= bond_radius;
            
            // Rotate to align with bond direction
            vec3 up = vec3(0, 1, 0);
            vec3 right = normalize(cross(up, bond_dir));
            vec3 forward = cross(right, bond_dir);
            
            mat3 rotation_matrix = mat3(right, bond_dir, forward);
            vec3 world_pos = bond_center + rotation_matrix * local_pos;
            
            frag_pos = world_pos;
            frag_normal = rotation_matrix * normal;
            frag_color = bond_color;
            
            light_dir = normalize(light_pos - world_pos);
            view_dir = normalize(camera_pos - world_pos);
            
            gl_Position = mvp_matrix * vec4(world_pos, 1.0);
        }
        """

        fragment_shader = """
        #version 330 core
        
        in vec3 frag_pos;
        in vec3 frag_normal;
        in vec4 frag_color;
        in vec3 light_dir;
        in vec3 view_dir;
        
        out vec4 final_color;
        
        uniform float ambient_strength;
        uniform float specular_strength;
        uniform float shininess;
        
        void main() {
            // Similar lighting to spheres but more metallic look
            vec3 ambient = ambient_strength * frag_color.rgb;
            
            float diff = max(dot(normalize(frag_normal), light_dir), 0.0);
            vec3 diffuse = diff * frag_color.rgb;
            
            vec3 reflect_dir = reflect(-light_dir, normalize(frag_normal));
            float spec = pow(max(dot(view_dir, reflect_dir), 0.0), shininess * 2.0);
            vec3 specular = specular_strength * 1.5 * spec * vec3(1.0);
            
            vec3 result = ambient + diffuse + specular;
            final_color = vec4(result, frag_color.a);
        }
        """

        try:
            self.bond_program = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader
            )
        except Exception as e:
            print(f"Bond shader compilation error: {e}")

    def _create_sample_pore_data(self):
        """Create sample pore network data with realistic connections"""
        np.random.seed(42)

        # Generate pore centers
        n_pores = 50
        self.pore_data = []
        positions = []

        for i in range(n_pores):
            pos = np.random.uniform(-5, 5, 3)
            radius = np.random.uniform(0.1, 0.4)
            # Color based on size (small=blue, large=red)
            size_factor = (radius - 0.1) / 0.3
            color = [1.0 - size_factor, 0.3, size_factor, 0.8]

            self.pore_data.append({
                'position': pos,
                'radius': radius,
                'color': color
            })
            positions.append(pos)

        # Generate bonds between nearby pores
        self.bond_data = []
        positions = np.array(positions)

        for i in range(n_pores):
            for j in range(i + 1, n_pores):
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance < 2.5:  # Connection threshold
                    # Bond color based on distance (close=bright, far=dim)
                    intensity = 1.0 - (distance / 2.5)
                    color = [0.8, 0.8, 0.2, 0.7 * intensity]

                    self.bond_data.append({
                        'start': positions[i],
                        'end': positions[j],
                        'radius': 0.02 + 0.03 * intensity,
                        'color': color
                    })

    def _setup_sphere_geometry(self):
        """Create sphere geometry for instanced rendering"""
        if not self.pore_data:
            return

        # Create sphere mesh (icosphere for smooth appearance)
        vertices, indices = self._create_icosphere(2)  # 2 subdivisions

        # Prepare instance data
        instance_data = []
        for pore in self.pore_data:
            instance_data.extend(pore['position'])  # 3 floats: center
            instance_data.append(pore['radius'])    # 1 float: radius
            instance_data.extend(pore['color'])     # 4 floats: color

        try:
            # Create vertex buffer
            vertex_buffer = self.ctx.buffer(
                np.array(vertices, dtype=np.float32))
            index_buffer = self.ctx.buffer(np.array(indices, dtype=np.uint32))
            instance_buffer = self.ctx.buffer(
                np.array(instance_data, dtype=np.float32))

            # Create VAO
            self.sphere_vao = self.ctx.vertex_array(
                self.sphere_program,
                [(vertex_buffer, '3f 3f', 'position', 'normal')],
                index_buffer
            )

            # Bind instance buffer
            self.sphere_vao.instance(instance_buffer, '3f f 4f',
                                     'sphere_center', 'sphere_radius', 'sphere_color')

        except Exception as e:
            print(f"Sphere geometry setup error: {e}")

    def _setup_bond_geometry(self):
        """Create bond/cylinder geometry for instanced rendering"""
        if not self.bond_data:
            return

        # Create cylinder mesh
        vertices, indices = self._create_cylinder(16)  # 16 segments

        # Prepare instance data
        instance_data = []
        for bond in self.bond_data:
            instance_data.extend(bond['start'])    # 3 floats: start
            instance_data.extend(bond['end'])      # 3 floats: end
            instance_data.append(bond['radius'])   # 1 float: radius
            instance_data.extend(bond['color'])    # 4 floats: color

        try:
            # Create vertex buffer
            vertex_buffer = self.ctx.buffer(
                np.array(vertices, dtype=np.float32))
            index_buffer = self.ctx.buffer(np.array(indices, dtype=np.uint32))
            instance_buffer = self.ctx.buffer(
                np.array(instance_data, dtype=np.float32))

            # Create VAO
            self.bond_vao = self.ctx.vertex_array(
                self.bond_program,
                [(vertex_buffer, '3f 3f', 'position', 'normal')],
                index_buffer
            )

            # Bind instance buffer
            self.bond_vao.instance(instance_buffer, '3f 3f f 4f',
                                   'bond_start', 'bond_end', 'bond_radius', 'bond_color')

        except Exception as e:
            print(f"Bond geometry setup error: {e}")

    def _create_icosphere(self, subdivisions=2):
        """Create icosphere vertices and indices"""
        # Golden ratio
        phi = (1.0 + math.sqrt(5.0)) / 2.0

        # Initial icosahedron vertices
        vertices = [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ]

        # Normalize to unit sphere
        vertices = [np.array(v) / np.linalg.norm(v) for v in vertices]

        # Initial indices (triangles)
        indices = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ]

        # Subdivide
        for _ in range(subdivisions):
            new_indices = []
            edge_vertices = {}

            for tri in indices:
                # Get midpoints
                mid = []
                for i in range(3):
                    edge = tuple(sorted([tri[i], tri[(i + 1) % 3]]))
                    if edge not in edge_vertices:
                        midpoint = (vertices[tri[i]] +
                                    vertices[tri[(i + 1) % 3]]) / 2
                        midpoint = midpoint / np.linalg.norm(midpoint)
                        edge_vertices[edge] = len(vertices)
                        vertices.append(midpoint)
                    mid.append(edge_vertices[edge])

                # Create 4 new triangles
                new_indices.extend([
                    [tri[0], mid[0], mid[2]],
                    [tri[1], mid[1], mid[0]],
                    [tri[2], mid[2], mid[1]],
                    [mid[0], mid[1], mid[2]]
                ])

            indices = new_indices

        # Add normals (same as positions for unit sphere)
        final_vertices = []
        for v in vertices:
            final_vertices.extend(
                [v[0], v[1], v[2], v[0], v[1], v[2]])  # pos + normal

        return final_vertices, [idx for tri in indices for idx in tri]

    def _create_cylinder(self, segments=16):
        """Create cylinder vertices and indices"""
        vertices = []
        indices = []

        # Create vertices
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = math.cos(angle)
            z = math.sin(angle)

            # Bottom circle
            vertices.extend([x, -1, z, x, 0, z])  # position + normal
            # Top circle
            vertices.extend([x, 1, z, x, 0, z])   # position + normal

        # Create indices for sides
        for i in range(segments):
            next_i = (i + 1) % segments
            bottom_current = i * 2
            top_current = i * 2 + 1
            bottom_next = next_i * 2
            top_next = next_i * 2 + 1

            # Two triangles per quad
            indices.extend([
                bottom_current, top_current, bottom_next,
                bottom_next, top_current, top_next
            ])

        return vertices, indices

    def resizeGL(self, width, height):
        """Handle window resize"""
        if height == 0:
            height = 1

        self.ctx.viewport = (0, 0, width, height)

    def paintGL(self):
        """Render the scene"""
        if not self.ctx:
            return

        # Clear buffers
        self.ctx.clear(0.1, 0.1, 0.2, 1.0)  # Dark blue background

        # Calculate matrices
        view_matrix = self._get_view_matrix()
        projection_matrix = self._get_projection_matrix()
        mvp_matrix = projection_matrix @ view_matrix

        # Set common uniforms
        current_time = time.time()
        light_pos = [
            5 * math.sin(current_time * 0.5),
            5,
            5 * math.cos(current_time * 0.5)
        ]

        # Render spheres
        if self.sphere_vao and self.sphere_program:
            self.sphere_program['mvp_matrix'] = mvp_matrix.flatten()
            self.sphere_program['light_pos'] = light_pos
            self.sphere_program['camera_pos'] = self.camera_pos
            self.sphere_program['ambient_strength'] = 0.3
            self.sphere_program['specular_strength'] = 0.8
            self.sphere_program['shininess'] = 32.0

            self.sphere_vao.render(instances=len(self.pore_data))

        # Render bonds
        if self.bond_vao and self.bond_program:
            self.bond_program['mvp_matrix'] = mvp_matrix.flatten()
            self.bond_program['light_pos'] = light_pos
            self.bond_program['camera_pos'] = self.camera_pos
            self.bond_program['ambient_strength'] = 0.4
            self.bond_program['specular_strength'] = 1.0
            self.bond_program['shininess'] = 64.0

            self.bond_vao.render(instances=len(self.bond_data))

    def _get_view_matrix(self):
        """Calculate view matrix"""
        # Apply rotations
        eye = self.camera_pos * self.zoom_factor

        # Rotate around target
        rotation_y = pyrr.matrix44.create_from_y_rotation(self.rotation_y)
        rotation_x = pyrr.matrix44.create_from_x_rotation(self.rotation_x)
        rotation = rotation_y @ rotation_x

        rotated_eye = (rotation @ np.append(eye, 1))[:3]

        return pyrr.matrix44.create_look_at(
            rotated_eye, self.camera_target, self.camera_up
        )

    def _get_projection_matrix(self):
        """Calculate projection matrix"""
        width = self.width() or 1
        height = self.height() or 1
        aspect = width / height

        return pyrr.matrix44.create_perspective_projection(
            45.0, aspect, 0.1, 100.0
        )

    def mousePressEvent(self, event):
        """Handle mouse press"""
        self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        """Handle mouse movement for camera control"""
        if self.last_mouse_pos is not None:
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()

            if event.buttons() & Qt.LeftButton:
                # Rotate camera
                self.rotation_y += dx * 0.01
                self.rotation_x += dy * 0.01
                self.rotation_x = max(-math.pi/2,
                                      min(math.pi/2, self.rotation_x))

            self.last_mouse_pos = event.pos()
            self.update()

    def wheelEvent(self, event):
        """Handle mouse wheel for zoom"""
        delta = event.angleDelta().y()
        zoom_speed = 0.001
        self.zoom_factor *= (1.0 + delta * zoom_speed)
        self.zoom_factor = max(0.1, min(10.0, self.zoom_factor))
        self.update()

    def update_pore_data(self, pore_data, bond_data=None):
        """Update visualization with new data"""
        self.pore_data = pore_data
        if bond_data:
            self.bond_data = bond_data

        # Recreate geometry
        self._setup_sphere_geometry()
        if bond_data:
            self._setup_bond_geometry()

        self.update()


class EmbeddedPoreVisualizationWidget(QWidget):
    """
    Complete embedded pore visualization widget with controls
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Setup the UI layout"""
        layout = QVBoxLayout()

        # OpenGL renderer
        self.renderer = ModernPoreRenderer()
        layout.addWidget(self.renderer, 1)

        # Control panel
        controls_frame = QFrame()
        controls_frame.setMaximumHeight(100)
        controls_layout = QHBoxLayout(controls_frame)

        # Zoom control
        controls_layout.addWidget(QLabel("Zoom:"))
        zoom_slider = QSlider(Qt.Horizontal)
        zoom_slider.setRange(10, 1000)
        zoom_slider.setValue(100)
        zoom_slider.valueChanged.connect(self.on_zoom_changed)
        controls_layout.addWidget(zoom_slider)

        # Animation control
        animate_btn = QPushButton("Toggle Animation")
        animate_btn.clicked.connect(self.toggle_animation)
        controls_layout.addWidget(animate_btn)

        # Reset view
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.reset_view)
        controls_layout.addWidget(reset_btn)

        layout.addWidget(controls_frame)
        self.setLayout(layout)

    def on_zoom_changed(self, value):
        """Handle zoom slider change"""
        self.renderer.zoom_factor = value / 100.0
        self.renderer.update()

    def toggle_animation(self):
        """Toggle animation timer"""
        if self.renderer.timer.isActive():
            self.renderer.timer.stop()
        else:
            self.renderer.timer.start(16)

    def reset_view(self):
        """Reset camera view"""
        self.renderer.rotation_x = 0.0
        self.renderer.rotation_y = 0.0
        self.renderer.zoom_factor = 1.0
        self.renderer.update()

    def update_visualization(self, pore_data, bond_data=None):
        """Update the visualization with new data"""
        self.renderer.update_pore_data(pore_data, bond_data)


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = EmbeddedPoreVisualizationWidget()
    widget.show()
    widget.resize(800, 600)
    sys.exit(app.exec_())
