import matplotlib.pyplot as plt
import numpy as np
from utils import get_pore_cutoff_values


def plot_differential_intrusion(diam1, intr1, diam2, intr2, diam3, intr3, save_path="figure_a.png"):
    """
    Create and save figure (a): "Log Differential Intrusion vs. Pore Size"

    Parameters:
    -----------
    diam1, intr1, diam2, intr2, diam3, intr3 : arrays
        Diameter and intrusion data for each sample
    save_path : str
        Path to save the figure
    """
    plt.figure(figsize=(8, 5))

    # Plot T1, T2, T3 with distinct markers and colors
    plt.plot(diam1, intr1,
             marker='s', linestyle='-', color='red',
             label="T$_1$", markersize=5, linewidth=1.2)
    plt.plot(diam2, intr2,
             marker='^', linestyle='-', color='blue',
             label="T$_2$", markersize=5, linewidth=1.2)
    plt.plot(diam3, intr3,
             marker='D', linestyle='-', color='orange',
             label="T$_3$", markersize=5, linewidth=1.2)

    # Log‐scale on x‐axis
    plt.xscale('log')

    # Vertical cutoff lines at 100 nm and 2000 nm
    cutoffs = get_pore_cutoff_values()
    v_cut1 = cutoffs['mesopores']
    v_cut2 = cutoffs['macropores']
    plt.axvline(v_cut1, color='k', ls='--', linewidth=1)
    plt.axvline(v_cut2, color='k', ls='--', linewidth=1)

    # Axis labels
    plt.xlabel("Pore size diameter (nm)", fontsize=12)
    plt.ylabel("Log Differential Intrusion (mL/g)", fontsize=12)

    # Text annotations
    plt.text(20, 0.25, "Mesopores", fontsize=11, ha='center')
    plt.text(300, 0.95, "Larger mesopores/\nsmaller macropores",
             fontsize=11, ha='center')
    plt.text(30000, 1.3, "Macropores", fontsize=11, ha='center')

    # Legend, grid, and limits
    plt.legend(loc='upper left', fontsize=10)
    plt.ylim(-0.1, 1.5)
    plt.xlim(10, 3e5)
    plt.grid(which='both', linestyle=':', linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_cumulative_intrusion(diam1, cum1, diam2, cum2, diam3, cum3, save_path="figure_b.png"):
    """
    Create and save figure (b): "Cumulative Intrusion vs. Pore Size"

    Parameters:
    -----------
    diam1, cum1, diam2, cum2, diam3, cum3 : arrays
        Diameter and cumulative intrusion data for each sample
    save_path : str
        Path to save the figure
    """
    plt.figure(figsize=(8, 5))

    plt.plot(diam1, cum1,
             marker='s', linestyle='-', color='red',
             label="T$_1$", markersize=5, linewidth=1.2)
    plt.plot(diam2, cum2,
             marker='^', linestyle='-', color='blue',
             label="T$_2$", markersize=5, linewidth=1.2)
    plt.plot(diam3, cum3,
             marker='D', linestyle='-', color='orange',
             label="T$_3$", markersize=5, linewidth=1.2)

    plt.xscale('log')

    # Vertical cutoff lines
    cutoffs = get_pore_cutoff_values()
    v_cut1 = cutoffs['mesopores']
    v_cut2 = cutoffs['macropores']
    plt.axvline(v_cut1, color='k', ls='--', linewidth=1)
    plt.axvline(v_cut2, color='k', ls='--', linewidth=1)

    # Axis labels
    plt.xlabel("Pore size diameter (nm)", fontsize=12)
    plt.ylabel("Cumulative Intrusion (mL/g)", fontsize=12)

    # Text annotations
    plt.text(20, 1.4, "Mesopores", fontsize=11, ha='center')
    plt.text(300, 1.2, "Larger mesopores/\nsmaller macropores",
             fontsize=11, ha='center')
    plt.text(30000, 0.7, "Macropores", fontsize=11, ha='center')

    plt.legend(loc='upper left', fontsize=10)
    plt.ylim(-0.2, 1.6)
    plt.xlim(1, 3e5)
    plt.grid(which='both', linestyle=':', linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
