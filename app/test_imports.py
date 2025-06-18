#!/usr/bin/env python3
"""Test script to check imports"""

try:
    from core.dem_visualizer import DEMParticleVisualizer
    print('dem_visualizer: OK')
except Exception as e:
    print(f'dem_visualizer: ERROR - {e}')

try:
    from core.live_renderer import Live3DRenderer, LiveSegmentationProcessor
    print('live_renderer: OK')
except Exception as e:
    print(f'live_renderer: ERROR - {e}')

try:
    from core.mist_analyzer import MISTAnalyzer
    print('mist_analyzer: OK')
except Exception as e:
    print(f'mist_analyzer: ERROR - {e}')

try:
    from plot.volumetric_3d_visualization import create_volumetric_pore_visualization
    print('volumetric_3d_visualization: OK')
except Exception as e:
    print(f'volumetric_3d_visualization: ERROR - {e}')
