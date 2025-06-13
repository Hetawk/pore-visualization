#!/usr/bin/env python3
"""
Data Manager - Handles all data operations with robust error handling and caching
Follows DRY principle with reusable data processing methods
"""

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging


class DataManager:
    """
    Centralized data management system for pore analysis data.
    Provides caching, validation, and preprocessing capabilities.
    """

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self._data_cache = {}

    def load_pore_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load and validate pore data from CSV file with caching."""
        file_path = Path(file_path)
        cache_key = f"pore_data_{file_path.stem}_{file_path.stat().st_mtime}"

        if cache_key in self._data_cache:
            self.logger.info(f"Loading data from cache: {file_path}")
            return self._data_cache[cache_key]

        try:
            self.logger.info(f"Loading pore data from: {file_path}")

            # Try custom CSV format first
            df = self._load_custom_csv_format(file_path)

            if df is None:
                # Fallback to standard CSV loading
                df = pd.read_csv(file_path)
                self.logger.info(f"Raw data shape: {df.shape}")
                self.logger.info(f"Available columns: {list(df.columns)}")
                # Auto-detect and map column names
                df = self._auto_map_columns(df)

            self._validate_pore_data(df)

            self._data_cache[cache_key] = df
            self.logger.info(
                f"Successfully loaded and mapped data: {file_path}")
            self.logger.info(f"Final data shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load data from {file_path}: {e}")
            raise

    def _auto_map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically map columns to standard names."""
        self.logger.info("Starting column mapping...")

        # Column mapping patterns
        diameter_patterns = ['pore size diameter',
                             'diameter', 'pore diameter', 'size']
        volume_patterns = ['intrusion', 'volume', 'log differential']
        conductivity_patterns = ['thermal conductivity', 'conductivity']

        # Get all column names (case insensitive)
        columns = [col.lower() for col in df.columns]
        self.logger.info(f"Columns (lowercase): {columns}")

        # Find best matches
        diameter_col = None
        volume_col = None
        conductivity_col = None

        for i, col in enumerate(columns):
            if any(pattern in col for pattern in diameter_patterns):
                if diameter_col is None:  # Take first match
                    diameter_col = df.columns[i]
                    self.logger.info(f"Mapped diameter column: {diameter_col}")

            if any(pattern in col for pattern in volume_patterns):
                if volume_col is None:  # Take first match
                    volume_col = df.columns[i]
                    self.logger.info(f"Mapped volume column: {volume_col}")

            if any(pattern in col for pattern in conductivity_patterns):
                if conductivity_col is None:  # Take first match
                    conductivity_col = df.columns[i]
                    self.logger.info(
                        f"Mapped conductivity column: {conductivity_col}")

        # Create standardized dataframe
        if diameter_col and volume_col:
            mapped_df = pd.DataFrame()
            mapped_df['Pore Diameter [nm]'] = pd.to_numeric(
                df[diameter_col], errors='coerce')
            mapped_df['Cumulative Intrusion Volume [mL/g]'] = pd.to_numeric(
                df[volume_col], errors='coerce')

            if conductivity_col:
                mapped_df['Thermal Conductivity [W/m·K]'] = pd.to_numeric(
                    df[conductivity_col], errors='coerce')

            # Remove rows with NaN values
            initial_rows = len(mapped_df)
            mapped_df = mapped_df.dropna()
            final_rows = len(mapped_df)

            if initial_rows > final_rows:
                self.logger.warning(
                    f"Removed {initial_rows - final_rows} rows with invalid data")

            self.logger.info(f"Successfully mapped columns and cleaned data")
            return mapped_df
        else:
            self.logger.error(
                f"Could not find required columns. Available: {list(df.columns)}")
            raise ValueError(
                f"Required columns not found. Need diameter and volume columns.")

    def _validate_pore_data(self, df: pd.DataFrame) -> None:
        """Validate pore data structure and content."""
        self.logger.info("Validating pore data...")

        required_cols = ['Pore Diameter [nm]',
                         'Cumulative Intrusion Volume [mL/g]']

        # Check required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            self.logger.error(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check data types and ranges
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                self.logger.warning(
                    f"Column {col} is not numeric, attempting conversion")
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Check for negative values
            if (df[col] < 0).any():
                self.logger.warning(f"Found negative values in {col}")

            # Check for missing values
            null_count = df[col].isnull().sum()
            if null_count > 0:
                self.logger.warning(f"Found {null_count} null values in {col}")

        # Data quality checks
        data_points = len(df)
        self.logger.info(
            f"Data validation complete. {data_points} data points validated")

        if data_points < 10:
            self.logger.warning("Dataset has fewer than 10 data points")

        # Log data ranges
        for col in required_cols:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                self.logger.info(
                    f"{col}: range [{min_val:.3f} - {max_val:.3f}]")
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

    def extract_sample_data(self, df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
        """Extract and clean sample data for all samples."""
        samples = {}

        # Check if we have multi-sample data (T1, T2, T3 columns)
        if 'Pore Diameter [nm].1' in df.columns and 'Pore Diameter [nm].2' in df.columns:
            # Define column patterns for different samples
            sample_patterns = {
                'T1': ['Pore Diameter [nm]', 'Cumulative Intrusion Volume [mL/g]'],
                'T2': ['Pore Diameter [nm].1', 'Cumulative Intrusion Volume [mL/g].1'],
                'T3': ['Pore Diameter [nm].2', 'Cumulative Intrusion Volume [mL/g].2']
            }

            for sample_id, (diam_col, intr_col) in sample_patterns.items():
                if diam_col in df.columns and intr_col in df.columns:
                    # Extract data
                    diameters = df[diam_col].values
                    intrusions = df[intr_col].values

                    # Clean data (remove NaN values)
                    mask = ~(pd.isna(diameters) | pd.isna(intrusions))
                    diameters = diameters[mask]
                    intrusions = intrusions[mask]

                    if len(diameters) > 0:  # Only add if we have valid data
                        # Sort by diameter
                        sort_indices = np.argsort(diameters)
                        diameters = diameters[sort_indices]
                        intrusions = intrusions[sort_indices]

                        samples[sample_id] = {
                            'diameters': diameters,
                            'intrusions': intrusions,
                            'count': len(diameters)
                        }
        else:
            # Single sample data - use as T1
            if 'Pore Diameter [nm]' in df.columns and 'Cumulative Intrusion Volume [mL/g]' in df.columns:
                diameters = df['Pore Diameter [nm]'].values
                intrusions = df['Cumulative Intrusion Volume [mL/g]'].values

                # Clean data (remove NaN values)
                mask = ~(pd.isna(diameters) | pd.isna(intrusions))
                diameters = diameters[mask]
                intrusions = intrusions[mask]

                if len(diameters) > 0:
                    # Sort by diameter
                    sort_indices = np.argsort(diameters)
                    diameters = diameters[sort_indices]
                    intrusions = intrusions[sort_indices]

                    samples['T1'] = {
                        'diameters': diameters,
                        'intrusions': intrusions,
                        'count': len(diameters)
                    }

        return samples

    def categorize_pores(self, diameters: np.ndarray) -> np.ndarray:
        """Categorize pores based on diameter ranges."""
        categories = np.zeros(len(diameters), dtype='<U20')

        # IUPAC classification
        mesopore_mask = diameters < 50  # 2-50 nm
        small_macro_mask = (diameters >= 50) & (diameters < 1000)
        medium_macro_mask = (diameters >= 1000) & (diameters < 10000)
        large_macro_mask = diameters >= 10000

        categories[mesopore_mask] = 'mesopore'
        categories[small_macro_mask] = 'small_macropore'
        categories[medium_macro_mask] = 'medium_macropore'
        categories[large_macro_mask] = 'large_macropore'

        return categories

    def save_analysis_results(self, results: Dict, filename: str) -> None:
        """Save analysis results to cache."""
        cache_file = self.cache_dir / f"{filename}.json"
        with open(cache_file, 'w') as f:
            json.dump(results, f, indent=2, default=self._json_serializer)

    def load_analysis_results(self, filename: str) -> Optional[Dict]:
        """Load analysis results from cache."""
        cache_file = self.cache_dir / f"{filename}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None

    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy arrays."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def get_data_statistics(self, samples: Dict) -> Dict:
        """Generate comprehensive statistics for loaded data."""
        stats = {}

        for sample_id, data in samples.items():
            diameters = data['diameters']
            intrusions = data['intrusions']

            stats[sample_id] = {
                'count': len(diameters),
                'diameter_range': [float(np.min(diameters)), float(np.max(diameters))],
                'diameter_mean': float(np.mean(diameters)),
                'diameter_std': float(np.std(diameters)),
                'intrusion_range': [float(np.min(intrusions)), float(np.max(intrusions))],
                'intrusion_mean': float(np.mean(intrusions)),
                'intrusion_std': float(np.std(intrusions)),
                'total_pore_volume': float(np.max(intrusions))
            }

        return stats

    def _detect_mixed_data_types(self, file_path: Path) -> bool:
        """Detect if the CSV file contains mixed data types that need cleaning."""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()[:10]  # Check first 10 lines

            text_rows = 0
            numeric_rows = 0

            for line in lines:
                if not line.strip():
                    continue

                first_cell = line.split(',')[0].strip()
                if not first_cell:
                    continue

                try:
                    float(first_cell)
                    numeric_rows += 1
                except ValueError:
                    text_rows += 1

            # If we have both text and numeric rows, we need cleaning
            return text_rows > 0 and numeric_rows > 0

        except Exception:
            return False

    def _load_with_interactive_cleaning(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load data using interactive cleaning dialog."""
        try:
            # Import here to avoid circular imports
            from gui.data_cleaning_dialog import show_data_cleaning_dialog

            self.logger.info("Launching interactive data cleaning dialog...")

            # Show the cleaning dialog
            cleaned_data = show_data_cleaning_dialog(str(file_path))

            if cleaned_data is not None:
                # Assign proper column names based on shape
                if cleaned_data.shape[1] == 9:
                    cleaned_data.columns = [
                        'Pore Diameter [nm]', 'Cumulative Intrusion Volume [mL/g]', 'Thermal Conductivity [W/m·K]',
                        'Pore Diameter [nm].1', 'Cumulative Intrusion Volume [mL/g].1', 'Thermal Conductivity [W/m·K].1',
                        'Pore Diameter [nm].2', 'Cumulative Intrusion Volume [mL/g].2', 'Thermal Conductivity [W/m·K].2'
                    ]
                elif cleaned_data.shape[1] == 6:
                    cleaned_data.columns = [
                        'Pore Diameter [nm]', 'Cumulative Intrusion Volume [mL/g]',
                        'Pore Diameter [nm].1', 'Cumulative Intrusion Volume [mL/g].1',
                        'Pore Diameter [nm].2', 'Cumulative Intrusion Volume [mL/g].2'
                    ]
                elif cleaned_data.shape[1] == 3:
                    cleaned_data.columns = [
                        'Pore Diameter [nm]', 'Cumulative Intrusion Volume [mL/g]', 'Thermal Conductivity [W/m·K]'
                    ]
                elif cleaned_data.shape[1] == 2:
                    cleaned_data.columns = [
                        'Pore Diameter [nm]', 'Cumulative Intrusion Volume [mL/g]'
                    ]

                self.logger.info(
                    f"Successfully cleaned data with shape: {cleaned_data.shape}")
                return cleaned_data
            else:
                self.logger.info("User cancelled data cleaning")
                return None

        except ImportError as e:
            self.logger.error(f"Could not import data cleaning dialog: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Interactive cleaning failed: {e}")
            return None

    def _load_custom_csv_format(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load CSV data using custom format for pore data with T1, T2, T3 headers."""
        try:
            self.logger.info("Attempting to load custom CSV format...")

            # First, try to detect if we have mixed data types
            mixed_data_detected = self._detect_mixed_data_types(file_path)

            if mixed_data_detected:
                self.logger.info(
                    "Mixed data types detected - attempting interactive cleaning")
                return self._load_with_interactive_cleaning(file_path)

            # Read the file line by line to filter out non-numeric rows
            with open(file_path, 'r') as f:
                all_lines = f.readlines()

            # Keep only lines that start with numeric data
            clean_lines = []
            for line in all_lines:
                stripped = line.strip()
                if stripped == "":
                    continue

                first_token = stripped.split(',')[0]
                try:
                    # If first token can be converted to float, it's a data row
                    float(first_token)
                    clean_lines.append(line)
                except ValueError:
                    # Skip header rows and non-numeric rows
                    continue

            if not clean_lines:
                self.logger.warning("No numeric data found in custom format")
                return None

            # Join clean lines and parse with pandas
            from io import StringIO
            csv_data = "".join(clean_lines)
            df = pd.read_csv(StringIO(csv_data), header=None)

            # Determine column structure based on number of columns
            if df.shape[1] == 9:
                # Format: T1_diam, T1_intr, T1_thermal, T2_diam, T2_intr, T2_thermal, T3_diam, T3_intr, T3_thermal
                df.columns = [
                    'Pore Diameter [nm]', 'Cumulative Intrusion Volume [mL/g]', 'Thermal Conductivity [W/m·K]',
                    'Pore Diameter [nm].1', 'Cumulative Intrusion Volume [mL/g].1', 'Thermal Conductivity [W/m·K].1',
                    'Pore Diameter [nm].2', 'Cumulative Intrusion Volume [mL/g].2', 'Thermal Conductivity [W/m·K].2'
                ]
            elif df.shape[1] == 6:
                # Format: T1_diam, T1_intr, T2_diam, T2_intr, T3_diam, T3_intr
                df.columns = [
                    'Pore Diameter [nm]', 'Cumulative Intrusion Volume [mL/g]',
                    'Pore Diameter [nm].1', 'Cumulative Intrusion Volume [mL/g].1',
                    'Pore Diameter [nm].2', 'Cumulative Intrusion Volume [mL/g].2'
                ]
            else:
                self.logger.warning(
                    f"Unexpected number of columns in custom format: {df.shape[1]}")
                return None

            # Convert all columns to numeric, replacing errors with NaN
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove rows where all values are NaN
            df = df.dropna(how='all')

            self.logger.info(
                f"Successfully loaded custom CSV format with shape: {df.shape}")
            return df

        except Exception as e:
            self.logger.warning(f"Custom CSV format loading failed: {e}")
            return None
