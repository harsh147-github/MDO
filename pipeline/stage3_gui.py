"""
Stage 3 — PyQt6 Desktop GUI for Live Screening Progress
=========================================================
A native desktop application showing live screening progress with
real-time plots for aero (L/D), structures (stress), and stealth (RCS).

Launch:
    python -m pipeline.stage3_gui
    python -m pipeline.stage3_gui --csv data/ccav_feasible.csv --samples 100

Architecture:
    - QThread runs the screening loop, emits signals per design
    - Main thread updates QTableView and pyqtgraph plots
    - Three stacked real-time charts: L/D histogram, stress scatter, RCS scatter
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSpinBox, QLabel, QFileDialog, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QStatusBar,
    QSplitter, QGroupBox, QLineEdit, QMessageBox,
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QElapsedTimer,
)
from PyQt6.QtGui import QColor, QFont, QBrush, QPalette

import numpy as np

# ── Try pyqtgraph for fast real-time plots, fallback to basic ─────
try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
    pg.setConfigOptions(antialias=True, background="#1a1d23", foreground="#c8ccd4")
except ImportError:
    HAS_PYQTGRAPH = False

# ── Pipeline imports ──────────────────────────────────────────────
from pipeline.ccav_sampler import (
    generate_doe, load_design_space, compute_derived, get_baseline_vector,
)
from pipeline.stage3_screening import (
    evaluate_single_design, STRESS_LIMIT_MPA, RCS_LIMIT_DBSM, LD_MIN,
    export_screening_results, load_feasible_samples,
)

# ═══════════════════════════════════════════════════════════════════
#  SCREENING WORKER THREAD
# ═══════════════════════════════════════════════════════════════════

class ScreeningWorker(QThread):
    """Runs the screening loop in a background thread."""

    screening_started  = pyqtSignal(int)           # n_total
    sample_complete    = pyqtSignal(dict)           # single result
    screening_finished = pyqtSignal(list)           # all results
    screening_error    = pyqtSignal(str)            # error message

    def __init__(self, samples: list[dict], parent=None):
        super().__init__(parent)
        self._samples = samples
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            n = len(self._samples)
            self.screening_started.emit(n)
            results = []

            for i, sample in enumerate(self._samples):
                if not self._running:
                    break

                sample["sample_id"] = sample.get("sample_id", i)
                result = evaluate_single_design(sample)
                results.append(result)
                self.sample_complete.emit(result)

            # Sort by J_norm
            results.sort(key=lambda r: r.get("J_norm", 999))
            for i, r in enumerate(results):
                r["rank"] = i + 1

            self.screening_finished.emit(results)

        except Exception as e:
            self.screening_error.emit(str(e))


# ═══════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════

# Colour palette
C_BG       = "#1a1d23"
C_PANEL    = "#1e2128"
C_ACCENT   = "#6dba8a"
C_RED      = "#d47272"
C_YELLOW   = "#c9a84e"
C_BLUE     = "#72a8d4"
C_TEXT     = "#c8ccd4"
C_DIM      = "#5a6070"


class ScreeningGUI(QMainWindow):
    """Main window for the Stage 3 screening GUI."""

    def __init__(self, csv_path: str | None = None, max_samples: int = 0):
        super().__init__()
        self.setWindowTitle("CCAV Stage 3 — Screening Monitor")
        self.setMinimumSize(1400, 800)
        self._apply_dark_theme()

        self._csv_path = csv_path
        self._max_samples = max_samples
        self._worker: ScreeningWorker | None = None
        self._results: list[dict] = []
        self._ld_data: list[float] = []
        self._stress_data: list[float] = []
        self._rcs_data: list[float] = []
        self._n_total = 0
        self._n_done = 0
        self._n_feasible = 0
        self._elapsed_timer = QElapsedTimer()

        self._build_ui()
        self._setup_status_timer()

        # Pre-fill CSV path if provided
        if csv_path:
            self._csv_input.setText(csv_path)
        if max_samples > 0:
            self._samples_spin.setValue(max_samples)

    def _apply_dark_theme(self):
        """Apply a dark aerospace theme."""
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(C_BG))
        pal.setColor(QPalette.ColorRole.WindowText, QColor(C_TEXT))
        pal.setColor(QPalette.ColorRole.Base, QColor(C_PANEL))
        pal.setColor(QPalette.ColorRole.AlternateBase, QColor(C_BG))
        pal.setColor(QPalette.ColorRole.Text, QColor(C_TEXT))
        pal.setColor(QPalette.ColorRole.Button, QColor(C_PANEL))
        pal.setColor(QPalette.ColorRole.ButtonText, QColor(C_TEXT))
        pal.setColor(QPalette.ColorRole.Highlight, QColor(C_ACCENT))
        self.setPalette(pal)
        self.setStyleSheet(f"""
            QMainWindow {{ background: {C_BG}; }}
            QGroupBox {{
                border: 1px solid {C_DIM};
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 14px;
                font-weight: 600;
                color: {C_ACCENT};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }}
            QPushButton {{
                background: {C_PANEL};
                border: 1px solid {C_DIM};
                border-radius: 4px;
                padding: 8px 16px;
                color: {C_TEXT};
                font-weight: 500;
            }}
            QPushButton:hover {{
                border-color: {C_ACCENT};
                color: {C_ACCENT};
            }}
            QPushButton:disabled {{
                color: {C_DIM};
                border-color: #2a2e36;
            }}
            QPushButton#startBtn {{
                background: rgba(109,186,138,0.1);
                border-color: {C_ACCENT};
                color: {C_ACCENT};
                font-size: 14px;
            }}
            QPushButton#stopBtn {{
                background: rgba(212,114,114,0.1);
                border-color: {C_RED};
                color: {C_RED};
            }}
            QSpinBox, QLineEdit {{
                background: {C_PANEL};
                border: 1px solid {C_DIM};
                border-radius: 4px;
                padding: 4px 8px;
                color: {C_TEXT};
            }}
            QLabel {{ color: {C_TEXT}; }}
            QProgressBar {{
                border: 1px solid {C_DIM};
                border-radius: 4px;
                text-align: center;
                color: {C_TEXT};
                background: {C_PANEL};
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {C_ACCENT}, stop:1 #4ecdc4);
                border-radius: 3px;
            }}
            QTableWidget {{
                background: {C_BG};
                gridline-color: #2a2e36;
                color: {C_TEXT};
                border: 1px solid {C_DIM};
                border-radius: 4px;
            }}
            QTableWidget::item {{
                padding: 2px 6px;
            }}
            QHeaderView::section {{
                background: {C_PANEL};
                color: {C_DIM};
                border: none;
                border-bottom: 1px solid {C_DIM};
                padding: 4px 6px;
                font-weight: 600;
                font-size: 11px;
            }}
            QStatusBar {{
                background: {C_PANEL};
                color: {C_DIM};
                border-top: 1px solid {C_DIM};
            }}
        """)

    def _build_ui(self):
        """Construct the 3-panel layout."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # ── LEFT PANEL: Controls ──────────────────────────────────
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)

        # Input group
        input_group = QGroupBox("INPUT")
        input_layout = QVBoxLayout(input_group)

        csv_row = QHBoxLayout()
        self._csv_input = QLineEdit()
        self._csv_input.setPlaceholderText("CSV path (or generate fresh DOE)")
        csv_browse_btn = QPushButton("Browse")
        csv_browse_btn.clicked.connect(self._browse_csv)
        csv_row.addWidget(self._csv_input, stretch=1)
        csv_row.addWidget(csv_browse_btn)
        input_layout.addLayout(csv_row)

        samples_row = QHBoxLayout()
        samples_row.addWidget(QLabel("Max Samples:"))
        self._samples_spin = QSpinBox()
        self._samples_spin.setRange(0, 10000)
        self._samples_spin.setValue(0)
        self._samples_spin.setSpecialValueText("All")
        samples_row.addWidget(self._samples_spin)
        input_layout.addLayout(samples_row)

        left_layout.addWidget(input_group)

        # Control buttons
        ctrl_group = QGroupBox("CONTROLS")
        ctrl_layout = QVBoxLayout(ctrl_group)

        self._start_btn = QPushButton("▶  Start Screening")
        self._start_btn.setObjectName("startBtn")
        self._start_btn.clicked.connect(self._start_screening)
        ctrl_layout.addWidget(self._start_btn)

        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setObjectName("stopBtn")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_screening)
        ctrl_layout.addWidget(self._stop_btn)

        left_layout.addWidget(ctrl_group)

        # Progress
        prog_group = QGroupBox("PROGRESS")
        prog_layout = QVBoxLayout(prog_group)

        self._progress_bar = QProgressBar()
        self._progress_bar.setTextVisible(True)
        prog_layout.addWidget(self._progress_bar)

        self._elapsed_label = QLabel("Elapsed: —")
        self._throughput_label = QLabel("Throughput: —")
        self._feasible_label = QLabel("Feasible: —")
        prog_layout.addWidget(self._elapsed_label)
        prog_layout.addWidget(self._throughput_label)
        prog_layout.addWidget(self._feasible_label)

        left_layout.addWidget(prog_group)

        # Export
        export_group = QGroupBox("EXPORT")
        export_layout = QVBoxLayout(export_group)
        self._export_btn = QPushButton("💾  Save Results CSV")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_results)
        export_layout.addWidget(self._export_btn)
        left_layout.addWidget(export_group)

        left_layout.addStretch()
        splitter.addWidget(left_panel)

        # ── CENTER PANEL: Results Table ───────────────────────────
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)

        table_group = QGroupBox("SCREENING RESULTS")
        table_layout = QVBoxLayout(table_group)

        self._table = QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels([
            "ID", "L/D", "Stress (MPa)", "RCS (dBsm)",
            "J_norm", "Status", "Time (s)",
        ])
        header = self._table.horizontalHeader()
        for i in range(7):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table_layout.addWidget(self._table)
        center_layout.addWidget(table_group)
        splitter.addWidget(center_panel)

        # ── RIGHT PANEL: Real-Time Plots ──────────────────────────
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        if HAS_PYQTGRAPH:
            self._build_pyqtgraph_plots(right_layout)
        else:
            self._build_fallback_plots(right_layout)

        splitter.addWidget(right_panel)

        # Splitter proportions: 20% / 40% / 40%
        splitter.setSizes([280, 560, 560])

        # ── STATUS BAR ────────────────────────────────────────────
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready — select input and click Start")

    def _build_pyqtgraph_plots(self, layout):
        """Build the 3 stacked real-time plots using pyqtgraph."""
        plots_group = QGroupBox("REAL-TIME PLOTS")
        plots_layout = QVBoxLayout(plots_group)

        # Plot 1: L/D histogram
        self._ld_plot = pg.PlotWidget(title="L/D Distribution")
        self._ld_plot.setLabel("bottom", "L/D")
        self._ld_plot.setLabel("left", "Count")
        self._ld_plot.setMinimumHeight(180)
        plots_layout.addWidget(self._ld_plot)

        # Plot 2: Stress scatter
        self._stress_plot = pg.PlotWidget(title="Max Stress vs Sample")
        self._stress_plot.setLabel("bottom", "Sample Index")
        self._stress_plot.setLabel("left", "Stress (MPa)")
        self._stress_plot.setMinimumHeight(180)
        # Add threshold line
        self._stress_threshold = pg.InfiniteLine(
            pos=STRESS_LIMIT_MPA, angle=0,
            pen=pg.mkPen(color="#d47272", width=2, style=Qt.PenStyle.DashLine))
        self._stress_plot.addItem(self._stress_threshold)
        plots_layout.addWidget(self._stress_plot)

        # Plot 3: RCS scatter
        self._rcs_plot = pg.PlotWidget(title="RCS vs Sample")
        self._rcs_plot.setLabel("bottom", "Sample Index")
        self._rcs_plot.setLabel("left", "RCS (dBsm)")
        self._rcs_plot.setMinimumHeight(180)
        # Add threshold line
        self._rcs_threshold = pg.InfiniteLine(
            pos=RCS_LIMIT_DBSM, angle=0,
            pen=pg.mkPen(color="#d47272", width=2, style=Qt.PenStyle.DashLine))
        self._rcs_plot.addItem(self._rcs_threshold)
        plots_layout.addWidget(self._rcs_plot)

        layout.addWidget(plots_group)

    def _build_fallback_plots(self, layout):
        """Build placeholder labels when pyqtgraph is not available."""
        plots_group = QGroupBox("PLOTS (install pyqtgraph for live charts)")
        plots_layout = QVBoxLayout(plots_group)

        self._ld_label = QLabel("L/D: waiting for data...")
        self._stress_label = QLabel("Stress: waiting for data...")
        self._rcs_label = QLabel("RCS: waiting for data...")

        for lbl in [self._ld_label, self._stress_label, self._rcs_label]:
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setMinimumHeight(120)
            lbl.setStyleSheet(f"border: 1px solid {C_DIM}; border-radius: 4px;")
            plots_layout.addWidget(lbl)

        layout.addWidget(plots_group)

    def _setup_status_timer(self):
        """Timer to update elapsed time and throughput."""
        self._status_timer = QTimer()
        self._status_timer.setInterval(500)
        self._status_timer.timeout.connect(self._update_timing)

    # ── Actions ───────────────────────────────────────────────────

    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select DOE CSV", str(_REPO_ROOT / "data"),
            "CSV files (*.csv);;All files (*)")
        if path:
            self._csv_input.setText(path)

    def _start_screening(self):
        """Load samples and start the screening worker thread."""
        csv_text = self._csv_input.text().strip()
        csv_path = Path(csv_text) if csv_text else None

        # Load samples
        try:
            if csv_path and csv_path.exists():
                samples = load_feasible_samples(csv_path)
            else:
                # Generate fresh DOE
                self._status_bar.showMessage("Generating DOE samples...")
                QApplication.processEvents()
                all_s, feas_m, _ = generate_doe(
                    n_samples=500, seed=42, include_baseline=True, verbose=False)
                samples = [s for s, ok in zip(all_s, feas_m) if ok]
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load samples:\n{e}")
            return

        max_n = self._samples_spin.value()
        if max_n > 0:
            samples = samples[:max_n]

        if not samples:
            QMessageBox.warning(self, "Warning", "No samples to screen.")
            return

        # Reset state
        self._results.clear()
        self._ld_data.clear()
        self._stress_data.clear()
        self._rcs_data.clear()
        self._n_done = 0
        self._n_feasible = 0
        self._table.setRowCount(0)

        if HAS_PYQTGRAPH:
            self._ld_plot.clear()
            self._stress_plot.clear()
            self._stress_plot.addItem(self._stress_threshold)
            self._rcs_plot.clear()
            self._rcs_plot.addItem(self._rcs_threshold)

        # UI state
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._export_btn.setEnabled(False)

        # Start worker
        self._worker = ScreeningWorker(samples, self)
        self._worker.screening_started.connect(self._on_started)
        self._worker.sample_complete.connect(self._on_sample_complete)
        self._worker.screening_finished.connect(self._on_finished)
        self._worker.screening_error.connect(self._on_error)
        self._elapsed_timer.start()
        self._status_timer.start()
        self._worker.start()

    def _stop_screening(self):
        """Stop the screening worker."""
        if self._worker:
            self._worker.stop()
            self._status_bar.showMessage("Stopping...")

    def _export_results(self):
        """Save results to CSV."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", str(_REPO_ROOT / "data" / "ccav_screening_results.csv"),
            "CSV files (*.csv)")
        if path:
            export_screening_results(self._results, Path(path))
            self._status_bar.showMessage(f"Results saved to {path}")

    # ── Signal Handlers ───────────────────────────────────────────

    def _on_started(self, n_total: int):
        self._n_total = n_total
        self._progress_bar.setMaximum(n_total)
        self._progress_bar.setValue(0)
        self._status_bar.showMessage(f"Screening {n_total} designs...")

    def _on_sample_complete(self, result: dict):
        """Handle a single completed design evaluation."""
        self._results.append(result)
        self._n_done += 1

        is_feas = result.get("Status") == "Feasible"
        if is_feas:
            self._n_feasible += 1

        ld = result.get("L_over_D", 0)
        stress = result.get("stress_max_MPa", 0)
        rcs = result.get("rcs_dbsm", 0)

        self._ld_data.append(ld)
        self._stress_data.append(stress)
        self._rcs_data.append(rcs)

        # Update progress bar
        self._progress_bar.setValue(self._n_done)

        # Update table
        row = self._table.rowCount()
        self._table.insertRow(row)

        items = [
            str(result.get("sample_id", "")),
            f"{ld:.2f}",
            f"{stress:.1f}",
            f"{rcs:.1f}",
            f"{result.get('J_norm', 0):.4f}",
            result.get("Status", ""),
            f"{result.get('wall_time_s', 0):.3f}",
        ]

        color = QColor(C_ACCENT) if is_feas else QColor(C_RED)
        for col, text in enumerate(items):
            item = QTableWidgetItem(text)
            item.setForeground(QBrush(color))
            self._table.setItem(row, col, item)

        # Auto-scroll to bottom
        self._table.scrollToBottom()

        # Update plots (every 5 samples for performance)
        if self._n_done % 5 == 0 or self._n_done == self._n_total:
            self._update_plots()

    def _on_finished(self, results: list):
        """Handle screening completion."""
        self._results = results
        self._status_timer.stop()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._export_btn.setEnabled(True)
        self._update_plots()
        self._update_timing()

        elapsed = self._elapsed_timer.elapsed() / 1000.0
        self._status_bar.showMessage(
            f"Complete — {self._n_feasible}/{self._n_done} feasible | "
            f"{elapsed:.1f}s | {self._n_done / elapsed:.1f} designs/s")

    def _on_error(self, msg: str):
        """Handle screening error."""
        self._status_timer.stop()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        QMessageBox.critical(self, "Screening Error", msg)
        self._status_bar.showMessage(f"Error: {msg}")

    # ── Plot Updates ──────────────────────────────────────────────

    def _update_plots(self):
        """Update all 3 real-time plots."""
        if not HAS_PYQTGRAPH:
            self._update_fallback_plots()
            return

        # Plot 1: L/D histogram
        if self._ld_data:
            self._ld_plot.clear()
            y, x = np.histogram(self._ld_data, bins=min(30, max(5, len(self._ld_data) // 5)))
            self._ld_plot.addItem(pg.BarGraphItem(
                x0=x[:-1], width=np.diff(x), height=y,
                brush=pg.mkBrush(C_ACCENT + "88"),
                pen=pg.mkPen(C_ACCENT, width=1)))

        # Plot 2: Stress scatter
        if self._stress_data:
            self._stress_plot.clear()
            self._stress_plot.addItem(self._stress_threshold)
            xs = np.arange(len(self._stress_data))
            colors = [
                pg.mkBrush(C_ACCENT if s <= STRESS_LIMIT_MPA else C_RED)
                for s in self._stress_data
            ]
            scatter = pg.ScatterPlotItem(
                x=xs, y=np.array(self._stress_data),
                size=5, brush=colors, pen=pg.mkPen(None))
            self._stress_plot.addItem(scatter)

        # Plot 3: RCS scatter
        if self._rcs_data:
            self._rcs_plot.clear()
            self._rcs_plot.addItem(self._rcs_threshold)
            xs = np.arange(len(self._rcs_data))
            colors = [
                pg.mkBrush(C_ACCENT if r <= RCS_LIMIT_DBSM else C_RED)
                for r in self._rcs_data
            ]
            scatter = pg.ScatterPlotItem(
                x=xs, y=np.array(self._rcs_data),
                size=5, brush=colors, pen=pg.mkPen(None))
            self._rcs_plot.addItem(scatter)

    def _update_fallback_plots(self):
        """Update text-based fallback when pyqtgraph is unavailable."""
        if self._ld_data:
            self._ld_label.setText(
                f"L/D — min: {min(self._ld_data):.1f}, "
                f"mean: {np.mean(self._ld_data):.1f}, "
                f"max: {max(self._ld_data):.1f}")
        if self._stress_data:
            self._stress_label.setText(
                f"Stress — min: {min(self._stress_data):.0f} MPa, "
                f"max: {max(self._stress_data):.0f} MPa, "
                f">{STRESS_LIMIT_MPA:.0f}: "
                f"{sum(1 for s in self._stress_data if s > STRESS_LIMIT_MPA)}")
        if self._rcs_data:
            self._rcs_label.setText(
                f"RCS — min: {min(self._rcs_data):.1f} dBsm, "
                f"max: {max(self._rcs_data):.1f} dBsm, "
                f">{RCS_LIMIT_DBSM:.0f}: "
                f"{sum(1 for r in self._rcs_data if r > RCS_LIMIT_DBSM)}")

    def _update_timing(self):
        """Update elapsed time, throughput, and feasibility labels."""
        if not self._elapsed_timer.isValid():
            return

        elapsed = self._elapsed_timer.elapsed() / 1000.0
        rate = self._n_done / elapsed if elapsed > 0 else 0
        remaining = self._n_total - self._n_done
        eta = remaining / rate if rate > 0 else 0

        self._elapsed_label.setText(f"Elapsed: {elapsed:.1f}s")
        self._throughput_label.setText(f"Throughput: {rate:.1f} designs/s")

        pct = 100 * self._n_feasible / self._n_done if self._n_done > 0 else 0
        self._feasible_label.setText(
            f"Feasible: {self._n_feasible}/{self._n_done} ({pct:.1f}%)")

        self._status_bar.showMessage(
            f"Processing sample {self._n_done}/{self._n_total} | "
            f"{self._n_feasible} feasible ({pct:.1f}%) | "
            f"{rate:.1f} designs/s | ETA: {eta:.0f}s")


# ═══════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3 — CCAV Screening Monitor (PyQt6 GUI)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to feasible DOE CSV")
    parser.add_argument("--samples", "-n", type=int, default=0,
                        help="Max samples to evaluate (0 = all)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("CCAV Stage 3 Screening Monitor")

    # Set application font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = ScreeningGUI(csv_path=args.csv, max_samples=args.samples)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
