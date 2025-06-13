#!/usr/bin/env python3
"""
Pore Analyzer - Advanced analysis tools for pore structure characterization
Provides statistical analysis and pore network metrics
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional, Any
import logging


class PoreAnalyzer:
    """
    Advanced pore analysis and characterization tools.
    Provides statistical metrics, clustering, and network analysis.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.analysis_cache = {}

    def analyze_pore_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive analysis of pore size distribution.

        Args:
            df: DataFrame with pore data

        Returns:
            Dictionary with statistical analysis results
        """
        try:
            pore_diameters = df['Pore Diameter [nm]'].values
            volumes = df['Cumulative Intrusion Volume [mL/g]'].values

            # Basic statistics
            stats_basic = {
                'mean_diameter': np.mean(pore_diameters),
                'median_diameter': np.median(pore_diameters),
                'std_diameter': np.std(pore_diameters),
                'min_diameter': np.min(pore_diameters),
                'max_diameter': np.max(pore_diameters),
                'total_volume': np.max(volumes),
                'pore_count': len(pore_diameters)
            }

            # Distribution analysis
            stats_distribution = self._analyze_distribution(pore_diameters)

            # Volume analysis
            stats_volume = self._analyze_volume_distribution(df)

            # Clustering analysis
            stats_clustering = self._perform_clustering_analysis(df)

            results = {
                'basic_stats': stats_basic,
                'distribution': stats_distribution,
                'volume_analysis': stats_volume,
                'clustering': stats_clustering
            }

            self.logger.info("Completed pore distribution analysis")
            return results

        except Exception as e:
            self.logger.error(f"Failed to analyze pore distribution: {e}")
            raise

    def _analyze_distribution(self, diameters: np.ndarray) -> Dict[str, Any]:
        """Analyze statistical distribution of pore diameters."""
        # Histogram analysis
        hist, bins = np.histogram(diameters, bins=50)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Peak detection
        peaks = self._find_distribution_peaks(hist, bin_centers)

        # Skewness and kurtosis
        skewness = stats.skew(diameters)
        kurtosis_val = stats.kurtosis(diameters)

        # Percentiles
        percentiles = np.percentile(diameters, [10, 25, 50, 75, 90])

        return {
            'histogram': {'counts': hist, 'bin_centers': bin_centers},
            'peaks': peaks,
            'skewness': skewness,
            'kurtosis': kurtosis_val,
            'percentiles': {
                'p10': percentiles[0], 'p25': percentiles[1],
                'p50': percentiles[2], 'p75': percentiles[3],
                'p90': percentiles[4]
            }
        }

    def _analyze_volume_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume distribution characteristics."""
        diameters = df['Pore Diameter [nm]'].values
        volumes = df['Cumulative Intrusion Volume [mL/g]'].values

        # Calculate differential volumes
        diff_volumes = np.diff(volumes)
        diff_diameters = diameters[1:]  # Corresponding diameters

        # Volume-weighted statistics
        total_volume = np.max(volumes)
        volume_weights = diff_volumes / np.sum(diff_volumes)

        weighted_mean = np.average(diff_diameters, weights=volume_weights)
        weighted_std = np.sqrt(np.average(
            (diff_diameters - weighted_mean)**2, weights=volume_weights))

        # Porosity metrics
        porosity_metrics = self._calculate_porosity_metrics(df)

        return {
            'total_volume': total_volume,
            'differential_volumes': diff_volumes,
            'volume_weighted_mean': weighted_mean,
            'volume_weighted_std': weighted_std,
            'porosity_metrics': porosity_metrics
        }

    def _perform_clustering_analysis(self, df: pd.DataFrame, n_clusters: int = 3) -> Dict[str, Any]:
        """Perform clustering analysis on pore data."""
        # Prepare data for clustering
        X = df[['Pore Diameter [nm]', 'Cumulative Intrusion Volume [mL/g]']].values

        # Normalize data
        X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_normalized)

        # Analyze clusters
        cluster_analysis = {}
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_data = df[cluster_mask]

            cluster_analysis[f'cluster_{i}'] = {
                'size': np.sum(cluster_mask),
                'mean_diameter': cluster_data['Pore Diameter [nm]'].mean(),
                'diameter_range': [
                    cluster_data['Pore Diameter [nm]'].min(),
                    cluster_data['Pore Diameter [nm]'].max()
                ],
                'volume_contribution': cluster_data['Cumulative Intrusion Volume [mL/g]'].iloc[-1] if len(cluster_data) > 0 else 0
            }

        return {
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels,
            'cluster_centers': kmeans.cluster_centers_,
            'cluster_analysis': cluster_analysis,
            'inertia': kmeans.inertia_
        }

    def _find_distribution_peaks(self, hist: np.ndarray, bin_centers: np.ndarray) -> List[Dict[str, float]]:
        """Find peaks in the distribution histogram."""
        from scipy.signal import find_peaks

        peaks, properties = find_peaks(
            hist, height=0.1*np.max(hist), distance=5)

        peak_info = []
        for peak_idx in peaks:
            peak_info.append({
                'position': bin_centers[peak_idx],
                'height': hist[peak_idx],
                'prominence': properties.get('prominences', [0])[0] if 'prominences' in properties else 0
            })

        return peak_info

    def _calculate_porosity_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate various porosity-related metrics."""
        diameters = df['Pore Diameter [nm]'].values
        volumes = df['Cumulative Intrusion Volume [mL/g]'].values

        # Assume bulk density for calculations (typical for concrete/cement)
        bulk_density = 2.3  # g/cm³

        # Convert volumes to actual pore volumes
        pore_volumes = volumes * bulk_density  # cm³/g * g/cm³ = dimensionless

        # Microporosity (< 2 nm)
        micro_mask = diameters < 2
        microporosity = np.max(
            pore_volumes[micro_mask]) if np.any(micro_mask) else 0

        # Mesoporosity (2-50 nm)
        meso_mask = (diameters >= 2) & (diameters <= 50)
        mesoporosity = np.max(
            pore_volumes[meso_mask]) - microporosity if np.any(meso_mask) else 0

        # Macroporosity (> 50 nm)
        macro_mask = diameters > 50
        macroporosity = np.max(
            pore_volumes[macro_mask]) - microporosity - mesoporosity if np.any(macro_mask) else 0

        total_porosity = np.max(pore_volumes)

        return {
            'total_porosity': total_porosity,
            'microporosity': microporosity,
            'mesoporosity': mesoporosity,
            'macroporosity': macroporosity,
            'micro_fraction': microporosity / total_porosity if total_porosity > 0 else 0,
            'meso_fraction': mesoporosity / total_porosity if total_porosity > 0 else 0,
            'macro_fraction': macroporosity / total_porosity if total_porosity > 0 else 0
        }

    def generate_analysis_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a comprehensive analysis report."""
        report = []
        report.append("=" * 60)
        report.append("PORE STRUCTURE ANALYSIS REPORT")
        report.append("=" * 60)

        # Basic statistics
        basic = analysis_results['basic_stats']
        report.append(f"\nBASIC STATISTICS:")
        report.append(f"  Mean Diameter: {basic['mean_diameter']:.2f} nm")
        report.append(f"  Median Diameter: {basic['median_diameter']:.2f} nm")
        report.append(f"  Standard Deviation: {basic['std_diameter']:.2f} nm")
        report.append(
            f"  Diameter Range: {basic['min_diameter']:.2f} - {basic['max_diameter']:.2f} nm")
        report.append(f"  Total Volume: {basic['total_volume']:.4f} mL/g")
        report.append(f"  Pore Count: {basic['pore_count']}")

        # Distribution analysis
        dist = analysis_results['distribution']
        report.append(f"\nDISTRIBUTION CHARACTERISTICS:")
        report.append(f"  Skewness: {dist['skewness']:.3f}")
        report.append(f"  Kurtosis: {dist['kurtosis']:.3f}")
        report.append(f"  Number of Peaks: {len(dist['peaks'])}")

        # Porosity breakdown
        porosity = analysis_results['volume_analysis']['porosity_metrics']
        report.append(f"\nPOROSITY ANALYSIS:")
        report.append(f"  Total Porosity: {porosity['total_porosity']:.4f}")
        report.append(
            f"  Microporosity (<2nm): {porosity['micro_fraction']*100:.1f}%")
        report.append(
            f"  Mesoporosity (2-50nm): {porosity['meso_fraction']*100:.1f}%")
        report.append(
            f"  Macroporosity (>50nm): {porosity['macro_fraction']*100:.1f}%")

        # Clustering results
        clustering = analysis_results['clustering']
        report.append(f"\nCLUSTERING ANALYSIS:")
        report.append(f"  Number of Clusters: {clustering['n_clusters']}")
        for i in range(clustering['n_clusters']):
            cluster = clustering['cluster_analysis'][f'cluster_{i}']
            report.append(f"  Cluster {i+1}: {cluster['size']} pores, "
                          f"mean diameter {cluster['mean_diameter']:.2f} nm")

        report.append("\n" + "=" * 60)

        return "\n".join(report)
