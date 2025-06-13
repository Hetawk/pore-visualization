#!/usr/bin/env python3
"""
Data Cleaning Dialog - Interactive interface for cleaning mixed data types
Allows users to preview data and select rows to ignore
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QPushButton, QCheckBox, QTextEdit, QSplitter, QGroupBox, QScrollArea,
    QMessageBox, QProgressBar, QComboBox, QSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor
from typing import List, Dict, Any, Optional, Tuple


class DataPreviewTable(QTableWidget):
    """Enhanced table widget for data preview with row selection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.ignored_rows = set()

    def set_data(self, data: List[List[str]], headers: List[str] = None):
        """Set data in the table with optional headers."""
        if not data:
            return

        self.setRowCount(len(data))
        self.setColumnCount(len(data[0]) if data else 0)

        if headers:
            self.setHorizontalHeaderLabels(headers)

        # Fill the table
        for row_idx, row_data in enumerate(data):
            for col_idx, cell_data in enumerate(row_data):
                item = QTableWidgetItem(str(cell_data))

                # Color code based on data type with better contrast
                if self.is_numeric(cell_data):
                    # Green background with dark text for numeric
                    item.setBackground(QColor(200, 255, 200))  # Brighter green
                    # Dark green text
                    item.setForeground(QColor(0, 100, 0))
                elif str(cell_data).strip() == "":
                    # Yellow background with dark text for empty
                    item.setBackground(QColor(255, 255, 180)
                                       )  # Brighter yellow
                    # Dark yellow/brown text
                    item.setForeground(QColor(150, 150, 0))
                else:
                    # Red background with dark text for text
                    item.setBackground(QColor(255, 200, 200))  # Brighter red
                    item.setForeground(QColor(150, 0, 0))      # Dark red text

                # Set font for better readability
                font = item.font()
                font.setBold(True)
                item.setFont(font)

                self.setItem(row_idx, col_idx, item)

        self.resizeColumnsToContents()

    def is_numeric(self, value) -> bool:
        """Check if a value can be converted to float."""
        try:
            float(str(value))
            return True
        except (ValueError, TypeError):
            return False

    def toggle_row_ignored(self, row: int):
        """Toggle whether a row should be ignored."""
        if row in self.ignored_rows:
            self.ignored_rows.remove(row)
            self.mark_row_included(row)
        else:
            self.ignored_rows.add(row)
            self.mark_row_ignored(row)

    def mark_row_ignored(self, row: int):
        """Mark a row as ignored visually."""
        for col in range(self.columnCount()):
            item = self.item(row, col)
            if item:
                # Dark gray background with strikethrough for ignored rows
                item.setBackground(QColor(120, 120, 120))  # Darker gray
                item.setForeground(QColor(200, 200, 200))  # Light gray text
                font = item.font()
                font.setStrikeOut(True)
                font.setBold(True)
                item.setFont(font)

    def mark_row_included(self, row: int):
        """Mark a row as included visually."""
        for col in range(self.columnCount()):
            item = self.item(row, col)
            if item:
                # Restore original color coding with better contrast
                cell_data = item.text()
                if self.is_numeric(cell_data):
                    item.setBackground(QColor(200, 255, 200))  # Brighter green
                    # Dark green text
                    item.setForeground(QColor(0, 100, 0))
                elif str(cell_data).strip() == "":
                    item.setBackground(QColor(255, 255, 180)
                                       )  # Brighter yellow
                    # Dark yellow/brown text
                    item.setForeground(QColor(150, 150, 0))
                else:
                    item.setBackground(QColor(255, 200, 200))  # Brighter red
                    item.setForeground(QColor(150, 0, 0))      # Dark red text

                font = item.font()
                font.setStrikeOut(False)
                font.setBold(True)
                item.setFont(font)


class DataCleaningDialog(QDialog):
    """Dialog for interactive data cleaning and row selection."""

    data_cleaned = pyqtSignal(pd.DataFrame)

    def __init__(self, file_path: str, raw_data: List[List[str]], parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.raw_data = raw_data
        self.cleaned_data = None

        self.setWindowTitle(f"Data Cleaning - {Path(file_path).name}")
        self.setMinimumSize(1000, 700)
        self.setup_ui()
        self.analyze_data()

    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)

        # Title and info
        title = QLabel("Data Cleaning & Row Selection")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)

        info = QLabel("Preview your data and select rows to ignore. Color coding: "
                      "🟢 Green = Numeric data, 🟡 Yellow = Empty cells, 🔴 Red = Text data, ⚫ Gray = Ignored rows")
        info.setWordWrap(True)
        layout.addWidget(info)

        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left panel - Controls
        self.setup_control_panel(splitter)

        # Right panel - Data preview
        self.setup_preview_panel(splitter)

        # Buttons
        self.setup_buttons(layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

    def setup_control_panel(self, parent):
        """Setup the control panel."""
        control_widget = QGroupBox("Cleaning Options")
        control_layout = QVBoxLayout(control_widget)

        # Auto-detection section
        auto_group = QGroupBox("Auto-Detection")
        auto_layout = QVBoxLayout(auto_group)

        self.auto_ignore_headers = QCheckBox("Auto-ignore header rows")
        self.auto_ignore_headers.setChecked(True)
        auto_layout.addWidget(self.auto_ignore_headers)

        self.auto_ignore_empty = QCheckBox("Auto-ignore empty rows")
        self.auto_ignore_empty.setChecked(True)
        auto_layout.addWidget(self.auto_ignore_empty)

        self.auto_ignore_text = QCheckBox(
            "Auto-ignore rows with text in first column")
        self.auto_ignore_text.setChecked(True)
        auto_layout.addWidget(self.auto_ignore_text)

        control_layout.addWidget(auto_group)

        # Manual selection section
        manual_group = QGroupBox("Manual Selection")
        manual_layout = QVBoxLayout(manual_group)

        ignore_first_label = QLabel("Ignore first N rows:")
        manual_layout.addWidget(ignore_first_label)

        self.ignore_first_spin = QSpinBox()
        self.ignore_first_spin.setRange(0, 10)
        self.ignore_first_spin.setValue(2)  # Default to ignore first 2 rows
        manual_layout.addWidget(self.ignore_first_spin)

        apply_auto_btn = QPushButton("Apply Auto-Detection")
        apply_auto_btn.clicked.connect(self.apply_auto_detection)
        manual_layout.addWidget(apply_auto_btn)

        control_layout.addWidget(manual_group)

        # Statistics section
        stats_group = QGroupBox("Data Statistics")
        stats_layout = QVBoxLayout(stats_group)

        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)

        control_layout.addWidget(stats_group)

        # Instructions
        instructions = QLabel(
            "Instructions:\n"
            "1. Review the data preview\n"
            "2. Click on rows to toggle ignore/include\n"
            "3. Use auto-detection for quick cleaning\n"
            "4. Click 'Apply Cleaning' when ready"
        )
        instructions.setWordWrap(True)
        control_layout.addWidget(instructions)

        control_layout.addStretch()
        parent.addWidget(control_widget)

    def setup_preview_panel(self, parent):
        """Setup the data preview panel."""
        preview_widget = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout(preview_widget)

        # Table
        self.preview_table = DataPreviewTable()
        self.preview_table.cellClicked.connect(self.on_cell_clicked)
        preview_layout.addWidget(self.preview_table)

        parent.addWidget(preview_widget)

    def setup_buttons(self, layout):
        """Setup action buttons."""
        button_layout = QHBoxLayout()

        self.preview_btn = QPushButton("Preview Cleaned Data")
        self.preview_btn.clicked.connect(self.preview_cleaned_data)
        button_layout.addWidget(self.preview_btn)

        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        self.apply_btn = QPushButton("Apply Cleaning")
        self.apply_btn.clicked.connect(self.apply_cleaning)
        self.apply_btn.setDefault(True)
        button_layout.addWidget(self.apply_btn)

        layout.addLayout(button_layout)

    def analyze_data(self):
        """Analyze the raw data and populate the preview."""
        if not self.raw_data:
            return

        # Show data in preview table
        headers = [f"Column {i+1}" for i in range(len(self.raw_data[0]))]
        self.preview_table.set_data(self.raw_data, headers)

        # Generate statistics
        self.update_statistics()

        # Apply initial auto-detection
        self.apply_auto_detection()

    def update_statistics(self):
        """Update the statistics display."""
        if not self.raw_data:
            return

        total_rows = len(self.raw_data)
        total_cols = len(self.raw_data[0]) if self.raw_data else 0

        # Count different types of data
        numeric_cells = 0
        text_cells = 0
        empty_cells = 0

        for row in self.raw_data:
            for cell in row:
                if str(cell).strip() == "":
                    empty_cells += 1
                elif self.preview_table.is_numeric(cell):
                    numeric_cells += 1
                else:
                    text_cells += 1

        ignored_rows = len(self.preview_table.ignored_rows)
        usable_rows = total_rows - ignored_rows

        stats_text = f"""Data Overview:
Total Rows: {total_rows}
Total Columns: {total_cols}
Ignored Rows: {ignored_rows}
Usable Rows: {usable_rows}

Cell Types:
Numeric: {numeric_cells}
Text: {text_cells}
Empty: {empty_cells}

Data Quality: {(numeric_cells / (total_rows * total_cols) * 100):.1f}% numeric
"""

        self.stats_text.setText(stats_text)

    def apply_auto_detection(self):
        """Apply automatic detection rules."""
        if not self.raw_data:
            return

        # Clear previous selections
        self.preview_table.ignored_rows.clear()

        # Ignore first N rows if requested
        ignore_first = self.ignore_first_spin.value()
        for i in range(min(ignore_first, len(self.raw_data))):
            self.preview_table.ignored_rows.add(i)

        # Auto-ignore rules
        for row_idx, row in enumerate(self.raw_data):
            if row_idx in self.preview_table.ignored_rows:
                continue

            # Check if row is mostly empty
            if self.auto_ignore_empty.isChecked():
                empty_count = sum(1 for cell in row if str(cell).strip() == "")
                if empty_count >= len(row) * 0.8:  # 80% empty
                    self.preview_table.ignored_rows.add(row_idx)
                    continue

            # Check if first column contains text (likely header or label)
            if self.auto_ignore_text.isChecked() and row:
                first_cell = str(row[0]).strip()
                if first_cell and not self.preview_table.is_numeric(first_cell):
                    # Skip if it looks like a header
                    if any(keyword in first_cell.lower() for keyword in
                           ['pore', 'diameter', 'size', 'intrusion', 'volume', 't1', 't2', 't3']):
                        self.preview_table.ignored_rows.add(row_idx)

        # Update visual indicators
        for row_idx in range(len(self.raw_data)):
            if row_idx in self.preview_table.ignored_rows:
                self.preview_table.mark_row_ignored(row_idx)
            else:
                self.preview_table.mark_row_included(row_idx)

        self.update_statistics()

    def on_cell_clicked(self, row: int, col: int):
        """Handle cell click to toggle row ignore status."""
        self.preview_table.toggle_row_ignored(row)
        self.update_statistics()

    def preview_cleaned_data(self):
        """Preview what the cleaned data will look like."""
        cleaned_data = self.get_cleaned_data()
        if cleaned_data is None:
            QMessageBox.warning(
                self, "Warning", "No valid data remaining after cleaning!")
            return

        # Show preview in a new dialog
        preview_dialog = QDialog(self)
        preview_dialog.setWindowTitle("Cleaned Data Preview")
        preview_dialog.setMinimumSize(600, 400)

        layout = QVBoxLayout(preview_dialog)

        info_label = QLabel(
            f"Cleaned data shape: {cleaned_data.shape[0]} rows × {cleaned_data.shape[1]} columns")
        layout.addWidget(info_label)

        preview_table = QTableWidget()
        preview_table.setRowCount(
            min(20, len(cleaned_data)))  # Show first 20 rows
        preview_table.setColumnCount(len(cleaned_data.columns))
        preview_table.setHorizontalHeaderLabels(cleaned_data.columns.tolist())

        for i in range(min(20, len(cleaned_data))):
            for j in range(len(cleaned_data.columns)):
                item = QTableWidgetItem(str(cleaned_data.iloc[i, j]))
                preview_table.setItem(i, j, item)

        preview_table.resizeColumnsToContents()
        layout.addWidget(preview_table)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(preview_dialog.accept)
        layout.addWidget(close_btn)

        preview_dialog.exec_()

    def get_cleaned_data(self) -> Optional[pd.DataFrame]:
        """Get the cleaned data as a pandas DataFrame."""
        if not self.raw_data:
            return None

        # Filter out ignored rows
        cleaned_rows = []
        for row_idx, row in enumerate(self.raw_data):
            if row_idx not in self.preview_table.ignored_rows:
                cleaned_rows.append(row)

        if not cleaned_rows:
            return None

        # Create DataFrame
        try:
            df = pd.DataFrame(cleaned_rows)

            # Convert to numeric where possible
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove any remaining rows with all NaN
            df = df.dropna(how='all')

            return df

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to create cleaned data: {e}")
            return None

    def apply_cleaning(self):
        """Apply the cleaning and emit the cleaned data."""
        self.cleaned_data = self.get_cleaned_data()

        if self.cleaned_data is None or len(self.cleaned_data) == 0:
            QMessageBox.warning(self, "Warning",
                                "No valid data remaining after cleaning!\n"
                                "Please adjust your selection and try again.")
            return

        # Emit the cleaned data
        self.data_cleaned.emit(self.cleaned_data)
        self.accept()


def show_data_cleaning_dialog(file_path: str, parent=None) -> Optional[pd.DataFrame]:
    """
    Show data cleaning dialog for a CSV file.
    Returns cleaned DataFrame or None if cancelled.
    """
    try:
        # Read raw file data
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Parse CSV manually to preserve all data
        raw_data = []
        for line in lines:
            if line.strip():
                row = [cell.strip() for cell in line.strip().split(',')]
                raw_data.append(row)

        if not raw_data:
            QMessageBox.warning(parent, "Error", "No data found in file!")
            return None

        # Show dialog
        dialog = DataCleaningDialog(file_path, raw_data, parent)

        result_data = None

        def on_data_cleaned(data):
            nonlocal result_data
            result_data = data

        dialog.data_cleaned.connect(on_data_cleaned)

        if dialog.exec_() == QDialog.Accepted:
            return result_data
        else:
            return None

    except Exception as e:
        QMessageBox.critical(
            parent, "Error", f"Failed to load file for cleaning: {e}")
        return None
