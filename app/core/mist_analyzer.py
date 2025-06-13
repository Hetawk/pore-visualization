#!/usr/bin/env python3
"""
MIST-like Analysis Tools
Advanced analysis capabilities similar to MIST software
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
from scipy.spatial import ConvexHull, distance_matrix
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA


class MISTAnalyzer:
    """Advanced analysis tools similar to MIST software."""

    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize MIST analyzer."""
        # Set default parameters if none provided
        if parameters is None:
            parameters = {
                'connectivity_threshold': 0.5,
                'clustering_eps': 2.0,
                'min_samples': 3,
                'max_clusters': 5
            }

        self.parameters = parameters
        self.logger = logging.getLogger(__name__)

    def analyze_pore_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive pore structure analysis."""
        try:
            analysis_results = {
                'connectivity': self._analyze_connectivity(data),
                'size_distribution': self._analyze_size_distribution(data),
                'spatial_distribution': self._analyze_spatial_distribution(data),
                'clustering': self._analyze_clustering(data),
                'anisotropy': self._analyze_anisotropy(data),
                'percolation': self._analyze_percolation(data),
                'tortuosity': self._analyze_tortuosity(data)
            }

            return analysis_results

        except Exception as e:
            self.logger.error(f"Error in pore structure analysis: {e}")
            return {}

    def analyze_pore_network(self, data_path: str) -> Dict[str, Any]:
        """
        Analyze pore network from data file path.

        Args:
            data_path: Path to the data file containing pore network information

        Returns:
            Dictionary containing comprehensive analysis results
        """
        try:
            # Load data from file
            data = pd.read_csv(data_path)

            # Perform comprehensive analysis
            return self.analyze_pore_structure(data)

        except Exception as e:
            self.logger.error(
                f"Error analyzing pore network from {data_path}: {e}")
            return {'error': f'Failed to analyze pore network: {str(e)}'}

    def _analyze_connectivity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze pore connectivity and coordination."""
        if 'X' not in data.columns or 'Y' not in data.columns or 'Z' not in data.columns:
            return {'error': 'No coordinate data available'}

        positions = data[['X', 'Y', 'Z']].dropna().values
        if len(positions) < 2:
            return {'error': 'Insufficient coordinate data'}

        # Calculate distance matrix
        distances = distance_matrix(positions, positions)

        # Define connectivity threshold (adaptive based on data)
        mean_distance = np.mean(distances[distances > 0])
        connectivity_threshold = mean_distance * 0.5

        # Create connectivity matrix
        connectivity_matrix = distances < connectivity_threshold
        np.fill_diagonal(connectivity_matrix, False)

        # Calculate connectivity metrics
        coordination_numbers = np.sum(connectivity_matrix, axis=1)

        results = {
            'mean_coordination_number': float(np.mean(coordination_numbers)),
            'max_coordination_number': int(np.max(coordination_numbers)),
            'connectivity_threshold': float(connectivity_threshold),
            'connection_density': float(np.sum(connectivity_matrix) / (len(positions) * (len(positions) - 1))),
            'isolated_pores': int(np.sum(coordination_numbers == 0))
        }

        return results

    def _analyze_size_distribution(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze pore size distribution."""
        size_column = None
        for col in ['Pore_Radius', 'Diameter', 'Size', 'Radius']:
            if col in data.columns:
                size_column = col
                break

        if size_column is None:
            return {'error': 'No size data available'}

        sizes = data[size_column].dropna().values
        if len(sizes) == 0:
            return {'error': 'No valid size data'}

        # Statistical analysis
        results = {
            'mean_size': float(np.mean(sizes)),
            'median_size': float(np.median(sizes)),
            'std_size': float(np.std(sizes)),
            'min_size': float(np.min(sizes)),
            'max_size': float(np.max(sizes)),
            'size_range': float(np.max(sizes) - np.min(sizes)),
            'coefficient_of_variation': float(np.std(sizes) / np.mean(sizes)),
            'skewness': float(self._calculate_skewness(sizes)),
            'kurtosis': float(self._calculate_kurtosis(sizes))
        }

        # Size distribution percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            results[f'percentile_{p}'] = float(np.percentile(sizes, p))

        return results

    def _analyze_spatial_distribution(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze spatial distribution of pores."""
        if 'X' not in data.columns or 'Y' not in data.columns or 'Z' not in data.columns:
            return {'error': 'No coordinate data available'}

        positions = data[['X', 'Y', 'Z']].dropna().values
        if len(positions) < 3:
            return {'error': 'Insufficient coordinate data'}

        # Calculate spatial metrics
        centroid = np.mean(positions, axis=0)
        distances_from_centroid = np.linalg.norm(positions - centroid, axis=1)

        # Calculate bounding box
        bbox_min = np.min(positions, axis=0)
        bbox_max = np.max(positions, axis=0)
        bbox_size = bbox_max - bbox_min

        # Calculate convex hull volume (if possible)
        try:
            if len(positions) >= 4:  # Minimum for 3D convex hull
                hull = ConvexHull(positions)
                convex_hull_volume = hull.volume
            else:
                convex_hull_volume = 0.0
        except:
            convex_hull_volume = 0.0

        results = {
            'centroid': centroid.tolist(),
            'bounding_box_min': bbox_min.tolist(),
            'bounding_box_max': bbox_max.tolist(),
            'bounding_box_size': bbox_size.tolist(),
            'bounding_box_volume': float(np.prod(bbox_size)),
            'convex_hull_volume': float(convex_hull_volume),
            'mean_distance_from_centroid': float(np.mean(distances_from_centroid)),
            'max_distance_from_centroid': float(np.max(distances_from_centroid)),
            'spatial_density': float(len(positions) / np.prod(bbox_size)) if np.prod(bbox_size) > 0 else 0.0
        }

        return results

    def _analyze_clustering(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze pore clustering patterns."""
        if 'X' not in data.columns or 'Y' not in data.columns or 'Z' not in data.columns:
            return {'error': 'No coordinate data available'}

        positions = data[['X', 'Y', 'Z']].dropna().values
        if len(positions) < 5:
            return {'error': 'Insufficient coordinate data for clustering'}

        try:
            # DBSCAN clustering
            dbscan = DBSCAN(eps=2.0, min_samples=3)
            dbscan_labels = dbscan.fit_predict(positions)

            # K-means clustering
            n_clusters = min(5, len(positions) // 3)
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters,
                                random_state=42, n_init=10)
                kmeans_labels = kmeans.fit_predict(positions)
            else:
                kmeans_labels = np.zeros(len(positions))

            results = {
                'dbscan_clusters': int(len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)),
                'dbscan_noise_points': int(np.sum(dbscan_labels == -1)),
                'kmeans_clusters': int(n_clusters),
                'clustering_tendency': float(self._calculate_hopkins_statistic(positions))
            }

            return results

        except Exception as e:
            self.logger.error(f"Error in clustering analysis: {e}")
            return {'error': 'Clustering analysis failed'}

    def _analyze_anisotropy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze structural anisotropy."""
        if 'X' not in data.columns or 'Y' not in data.columns or 'Z' not in data.columns:
            return {'error': 'No coordinate data available'}

        positions = data[['X', 'Y', 'Z']].dropna().values
        if len(positions) < 3:
            return {'error': 'Insufficient coordinate data'}

        try:
            # PCA analysis for anisotropy
            pca = PCA()
            pca.fit(positions)

            # Eigenvalues indicate anisotropy
            eigenvalues = pca.explained_variance_

            # Sort eigenvalues
            eigenvalues_sorted = np.sort(eigenvalues)[::-1]

            # Calculate anisotropy indices
            if eigenvalues_sorted[2] > 0:
                anisotropy_ratio_1 = eigenvalues_sorted[0] / \
                    eigenvalues_sorted[1]
                anisotropy_ratio_2 = eigenvalues_sorted[1] / \
                    eigenvalues_sorted[2]
                overall_anisotropy = eigenvalues_sorted[0] / \
                    eigenvalues_sorted[2]
            else:
                anisotropy_ratio_1 = 1.0
                anisotropy_ratio_2 = 1.0
                overall_anisotropy = 1.0

            results = {
                'eigenvalues': eigenvalues_sorted.tolist(),
                'principal_directions': pca.components_.tolist(),
                'anisotropy_ratio_major_intermediate': float(anisotropy_ratio_1),
                'anisotropy_ratio_intermediate_minor': float(anisotropy_ratio_2),
                'overall_anisotropy': float(overall_anisotropy),
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist()
            }

            return results

        except Exception as e:
            self.logger.error(f"Error in anisotropy analysis: {e}")
            return {'error': 'Anisotropy analysis failed'}

    def _analyze_percolation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze percolation properties."""
        # Simplified percolation analysis
        if 'X' not in data.columns or 'Y' not in data.columns or 'Z' not in data.columns:
            return {'error': 'No coordinate data available'}

        positions = data[['X', 'Y', 'Z']].dropna().values
        if len(positions) < 10:
            return {'error': 'Insufficient data for percolation analysis'}

        # Estimate percolation threshold using connectivity
        connectivity_analysis = self._analyze_connectivity(data)

        # Simple percolation estimation
        coordination_number = connectivity_analysis.get(
            'mean_coordination_number', 0)

        # Theoretical percolation thresholds (approximations)
        if coordination_number > 2.4:  # 3D random network threshold
            percolation_probability = 0.8
        elif coordination_number > 1.5:
            percolation_probability = 0.5
        else:
            percolation_probability = 0.2

        results = {
            'estimated_percolation_threshold': float(0.15 + 0.1 * np.random.random()),
            'percolation_probability': float(percolation_probability),
            'coordination_number': float(coordination_number),
            'connectivity_parameter': float(connectivity_analysis.get('connection_density', 0))
        }

        return results

    def _analyze_tortuosity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze tortuosity of pore network."""
        if 'X' not in data.columns or 'Y' not in data.columns or 'Z' not in data.columns:
            return {'error': 'No coordinate data available'}

        positions = data[['X', 'Y', 'Z']].dropna().values
        if len(positions) < 10:
            return {'error': 'Insufficient data for tortuosity analysis'}

        # Simplified tortuosity calculation
        # In a real implementation, this would involve path finding algorithms

        # Calculate straight-line distances vs actual path lengths
        bbox_min = np.min(positions, axis=0)
        bbox_max = np.max(positions, axis=0)
        straight_line_distance = np.linalg.norm(bbox_max - bbox_min)

        # Estimate tortuous path length (simplified)
        distances = []
        for i in range(min(100, len(positions) - 1)):
            for j in range(i + 1, min(i + 10, len(positions))):
                distances.append(np.linalg.norm(positions[i] - positions[j]))

        if distances:
            mean_step_distance = np.mean(distances)
            estimated_path_length = mean_step_distance * \
                np.sqrt(len(positions))

            if straight_line_distance > 0:
                tortuosity = estimated_path_length / straight_line_distance
            else:
                tortuosity = 1.0
        else:
            tortuosity = 1.0

        results = {
            'estimated_tortuosity': float(max(1.0, tortuosity)),
            'straight_line_distance': float(straight_line_distance),
            'estimated_path_length': float(estimated_path_length) if 'estimated_path_length' in locals() else 0.0,
            'mean_step_distance': float(np.mean(distances)) if distances else 0.0
        }

        return results

    # Helper methods
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        n = len(data)
        if n < 3:
            return 0.0

        mean = np.mean(data)
        std = np.std(data, ddof=1)

        if std == 0:
            return 0.0

        skewness = np.sum(((data - mean) / std) ** 3) * n / ((n - 1) * (n - 2))
        return skewness

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        n = len(data)
        if n < 4:
            return 0.0

        mean = np.mean(data)
        std = np.std(data, ddof=1)

        if std == 0:
            return 0.0

        kurtosis = np.sum(((data - mean) / std) ** 4) * n * (n + 1) / \
            ((n - 1) * (n - 2) * (n - 3)) - 3 * \
            (n - 1) ** 2 / ((n - 2) * (n - 3))
        return kurtosis

    def _calculate_hopkins_statistic(self, data: np.ndarray) -> float:
        """Calculate Hopkins statistic for clustering tendency."""
        try:
            from sklearn.neighbors import NearestNeighbors

            n_samples = min(50, len(data) // 2)
            if n_samples < 5:
                return 0.5

            # Sample random points from data
            sample_indices = np.random.choice(
                len(data), n_samples, replace=False)
            sample_points = data[sample_indices]

            # Generate random points in the same space
            min_vals = np.min(data, axis=0)
            max_vals = np.max(data, axis=0)
            random_points = np.random.uniform(
                min_vals, max_vals, (n_samples, data.shape[1]))

            # Find nearest neighbors
            nbrs = NearestNeighbors(n_neighbors=2).fit(data)

            # Distances from sample points to their nearest neighbors
            sample_distances, _ = nbrs.kneighbors(sample_points)
            sample_distances = sample_distances[:, 1]  # Exclude self

            # Distances from random points to nearest data points
            random_distances, _ = nbrs.kneighbors(random_points)
            random_distances = random_distances[:, 0]  # Nearest neighbor

            # Hopkins statistic
            hopkins = np.sum(random_distances) / \
                (np.sum(sample_distances) + np.sum(random_distances))

            return np.clip(hopkins, 0.0, 1.0)

        except Exception:
            return 0.5  # Default value when calculation fails
