#!/usr/bin/env python3
"""
Main script for pore analysis and visualization.
This script coordinates all data processing and visualization modules.
"""

import os
import numpy as np

# Import our modules
from data_processing import load_and_clean_data, extract_sample_data, extract_thermal_data
from utils import sort_by_diameter
# from basic_plots import plot_differential_intrusion, plot_cumulative_intrusion
# from surface_plots import plot_3d_differential_surfaces
# from cumulative_plots import plot_3d_cumulative_surfaces
# from thermal_visualization import plot_3d_thermal_boards
# from scientific_visualization import create_3d_pore_visualization
# from volumetric_visualization import create_3d_volumetric_visualization
from plot.network_visualization import create_3d_pore_network
from enhanced_3d_visualizer_pro import create_enhanced_3d_visualization


def ensure_output_dir(output_dir="out"):
    """
    Create the output directory if it doesn't exist.

    Parameters:
    -----------
    output_dir : str
        Path to the output directory

    Returns:
    --------
    str
        The full path to the output directory
    """
    # Get the directory of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the full output path
    output_path = os.path.join(base_dir, output_dir)

    # Create the directory if it doesn't exist
    if not os.path.exists(output_path):
        print(f"Creating output directory: {output_path}")
        os.makedirs(output_path)
    else:
        print(f"Output directory already exists: {output_path}")

    return output_path


def resolve_path(file_path):
    """
    Resolve a path relative to the script's location.

    Parameters:
    -----------
    file_path : str
        Path to resolve relative to the script

    Returns:
    --------
    str
        The full absolute path
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, file_path)


def main(data_file="pore_data.csv", output_dir="out"):
    """
    Main function to process data and generate all visualizations.

    Parameters:
    -----------
    data_file : str
        Path to the pore data CSV file
    output_dir : str
        Directory to save output files
    """
    print("Starting pore analysis and visualization...")

    # Ensure output directory exists
    out_path = ensure_output_dir(output_dir)

    # Try multiple possible locations for the data file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        data_file,  # Try as provided
        os.path.join(base_dir, data_file),  # Relative to script
        os.path.join(base_dir, "dataset", data_file),  # In dataset folder
        os.path.join(os.path.dirname(base_dir),
                     "dataset", data_file)  # Up one level
    ]

    data_file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_file_path = path
            break

    if not data_file_path:
        raise FileNotFoundError(
            f"Could not find {data_file} in any of these locations: {possible_paths}")

    print(f"Found data file at: {data_file_path}")

    # Helper to create full output paths
    def outfile(filename):
        return os.path.join(out_path, filename)

    # 1. Load and clean data
    print("\nLoading and cleaning data...")
    df = load_and_clean_data(data_file_path)

    # 2. Extract and sort sample data
    print("\nExtracting and sorting sample data...")
    diam1, intr1, diam2, intr2, diam3, intr3 = extract_sample_data(
        df, sort_by_diameter)

    # 2.1. Extract thermal conductivity data if available
    print("\nExtracting thermal conductivity data...")
    thermal_data = extract_thermal_data(df)
    if thermal_data:
        thermal1, thermal2, thermal3 = thermal_data
        print(f"Thermal conductivity data found:")
        print(
            f"  T1: {len(thermal1)} values (avg: {np.mean(thermal1):.3f} W/m·K)")
        print(
            f"  T2: {len(thermal2)} values (avg: {np.mean(thermal2):.3f} W/m·K)")
        print(
            f"  T3: {len(thermal3)} values (avg: {np.mean(thermal3):.3f} W/m·K)")
    else:
        print("No thermal conductivity data found in dataset")

    # # 3. Compute cumulative intrusion for all samples
    # print("\nComputing cumulative intrusion values...")
    # cum1 = compute_cumulative_intrusion(diam1, intr1)
    # cum2 = compute_cumulative_intrusion(diam2, intr2)
    # cum3 = compute_cumulative_intrusion(diam3, intr3)

    # # 4. Generate basic 2D plots
    # print("\nGenerating basic 2D plots...")
    # plot_differential_intrusion(
    #     diam1, intr1, diam2, intr2, diam3, intr3, outfile("figure_a.png"))
    # plot_cumulative_intrusion(diam1, cum1, diam2, cum2,
    #                           diam3, cum3, outfile("figure_b.png"))

    # # 5. Generate scientific 3D visualization
    # print("\nCreating scientific 3D visualizations of pore distributions...")
    # create_3d_pore_visualization(
    #     diam1, intr1, diam3, intr3, outfile("3D_pore_structure_scientific.png"))

    # # 6. Generate 3D differential intrusion visualization
    # print("\nCreating 3D differential intrusion visualization...")
    # plot_3d_differential_surfaces(
    #     diam1, intr1, diam2, intr2, diam3, intr3, outfile("3D_differential_intrusion.png"))

    # # 7. Generate 3D cumulative intrusion visualization
    # print("\nCreating 3D cumulative intrusion visualization...")
    # plot_3d_cumulative_surfaces(
    #     diam1, cum1, diam2, cum2, diam3, cum3, outfile("3D_cumulative_intrusion.png"))

    # # 8. Generate 3D thermal boards visualization
    # print("\nCreating 3D thermal boards visualization...")
    # plot_3d_thermal_boards(diam1, intr1, cum1, diam2, intr2,
    #                        cum2, diam3, intr3, cum3, outfile("3D_thermal_boards.png"))

    # # 9. Generate 3D volumetric visualization (vertical and properly oriented horizontal)
    # print("\nCreating 3D volumetric visualizations of pore distribution...")
    # # This function generates both vertical and horizontal versions
    # create_3d_volumetric_visualization(
    #     diam1, intr1, diam2, intr2, diam3, intr3, outfile("3D_volumetric_pore_distribution.png"))

    # 10. Generate 3D network-like visualization of pore distributions
    print("\nCreating 3D network-like visualization of pore distributions...")
    create_3d_pore_network(
        diam1, intr1, diam2, intr2, diam3, intr3, outfile("3D_pore_network.png"))

    # 11. Generate enhanced 3D volumetric visualizations with atomic-style networks
    print("\nCreating enhanced 3D volumetric visualizations...")
    create_enhanced_3d_visualization(
        diam1, intr1, diam2, intr2, diam3, intr3, out_path)

    print(f"\nDone. All figures have been created in {out_path}:")
    print(f"- {os.path.join(output_dir, 'figure_a.png')} and {os.path.join(output_dir, 'figure_b.png')} (2D plots)")
    print(f"- {os.path.join(output_dir, '3D_pore_structure_scientific.png')} (3D scientific visualization)")
    print(f"- {os.path.join(output_dir, '3D_differential_intrusion.png')} (3D differential intrusion)")
    print(f"- {os.path.join(output_dir, '3D_cumulative_intrusion.png')} (3D cumulative intrusion)")
    print(f"- {os.path.join(output_dir, '3D_thermal_boards.png')} (3D thermal boards visualization)")
    print(f"- {os.path.join(output_dir, '3D_volumetric_pore_distribution.png')} (3D volumetric visualization - vertical)")
    print(f"- {os.path.join(output_dir, '3D_volumetric_pore_distribution_horizontal.png')} (3D volumetric visualization - horizontal)")
    print(f"- {os.path.join(output_dir, '3D_pore_network.png')} (3D network-like visualization)")
    print(f"- {os.path.join(output_dir, 'enhanced_3d_T1_volumetric.png')} (Enhanced T1 volumetric visualization)")
    print(f"- {os.path.join(output_dir, 'enhanced_3d_T2_volumetric.png')} (Enhanced T2 volumetric visualization)")
    print(f"- {os.path.join(output_dir, 'enhanced_3d_T3_volumetric.png')} (Enhanced T3 volumetric visualization)")
    print(f"- {os.path.join(output_dir, 'enhanced_3d_T1_volumetric.html')} (Interactive T1 visualization)")
    print(f"- {os.path.join(output_dir, 'enhanced_3d_T2_volumetric.html')} (Interactive T2 visualization)")
    print(f"- {os.path.join(output_dir, 'enhanced_3d_T3_volumetric.html')} (Interactive T3 visualization)")
    print(f"- {os.path.join(output_dir, 'enhanced_3d_combined.png')} (Enhanced combined comparison)")


if __name__ == "__main__":
    main()
