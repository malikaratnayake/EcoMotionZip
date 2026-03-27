#!/usr/bin/env python3
"""
EcoMotionZip PySide6
====================
Single-file desktop application for the EcoMotionZip motion-based video
compression pipeline.

Features
--------
• Selectable dark (slate-950) / light (slate-50) themes via QSS,
  mapped directly from Tailwind CSS v3 colour tokens.
• Tab 1 — Configuration: file pickers, motion slider, codec / transparency
  controls, collapsible advanced settings, live CLI command preview.
• Tab 2 — Pipeline Monitor: three-thread architecture visualisation,
  real-time queue progress bars, frame counter, HTML log panel.
• Actual EcoMotionZip_lite.py subprocess launched via QProcess; stdout is
  forwarded to the log panel. A QTimer simulation provides visual activity
  for queue levels and detector status independent of log output cadence.

Run
---
    pip install PySide6
    python src/EcoMotionZip_PySide6.py
"""

from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, QTimer, QSize
from PySide6.QtGui import QFont, QIcon, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget,
    QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QSlider, QSpinBox,
    QCheckBox, QComboBox, QPlainTextEdit, QTextEdit, QProgressBar,
    QFileDialog, QGroupBox, QFrame, QButtonGroup,
    QSizePolicy,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).parent
_LITE_SCRIPT = _HERE / "EcoMotionZip_lite.py"

# ═══════════════════════════════════════════════════════════════════════════════
# Tailwind CSS v3 colour tokens
# ═══════════════════════════════════════════════════════════════════════════════

# Dark theme — slate-950 base ───────────────────────────────────────────────────
DARK = dict(
    bg            = "#020617",   # slate-950
    surface       = "#0F172A",   # slate-900
    panel         = "#1E293B",   # slate-800
    border        = "#334155",   # slate-700
    muted         = "#475569",   # slate-600
    placeholder   = "#64748B",   # slate-500
    text_dim      = "#94A3B8",   # slate-400
    text          = "#CBD5E1",   # slate-300
    text_bright   = "#F1F5F9",   # slate-100
    primary       = "#059669",   # emerald-600
    primary_hv    = "#047857",   # emerald-700
    primary_lt    = "#34D399",   # emerald-400
    focus         = "#3B82F6",   # blue-500
    focus_hv      = "#2563EB",   # blue-600
    warn          = "#FBBF24",   # amber-400
    error         = "#F87171",   # red-400
    stop          = "#DC2626",   # red-600
    stop_hv       = "#B91C1C",   # red-700
    reader_clr    = "#3B82F6",   # blue-500
    writer_clr    = "#F97316",   # orange-500
    detector_clr  = "#10B981",   # emerald-500
    terminal_bg   = "#020617",   # slate-950
    terminal_fg   = "#4ADE80",   # green-400
    log_bg        = "#0F172A",   # slate-900
    log_info      = "#CBD5E1",   # slate-300
    log_warn      = "#FBBF24",   # amber-400
    log_err       = "#F87171",   # red-400
    log_ts        = "#475569",   # slate-600
    log_lvl_info  = "#64748B",   # slate-500
    log_lvl_warn  = "#D97706",   # amber-600
    log_lvl_err   = "#DC2626",   # red-600
    banner_ok_bg     = "#052e16",  # green-950
    banner_ok_border = "#16a34a",  # green-600
    banner_ok_text   = "#4ade80",  # green-400
    banner_err_bg    = "#450a0a",  # red-950
)

# Light theme — slate-50 base ──────────────────────────────────────────────────
LIGHT = dict(
    bg            = "#F8FAFC",   # slate-50
    surface       = "#FFFFFF",   # white
    panel         = "#F1F5F9",   # slate-100
    border        = "#CBD5E1",   # slate-300
    muted         = "#94A3B8",   # slate-400
    placeholder   = "#94A3B8",   # slate-400
    text_dim      = "#64748B",   # slate-500
    text          = "#334155",   # slate-700
    text_bright   = "#0F172A",   # slate-900
    primary       = "#059669",   # emerald-600
    primary_hv    = "#047857",   # emerald-700
    primary_lt    = "#10B981",   # emerald-500
    focus         = "#3B82F6",   # blue-500
    focus_hv      = "#2563EB",   # blue-600
    warn          = "#D97706",   # amber-600  (darker for light bg)
    error         = "#DC2626",   # red-600
    stop          = "#DC2626",   # red-600
    stop_hv       = "#B91C1C",   # red-700
    reader_clr    = "#3B82F6",   # blue-500
    writer_clr    = "#F97316",   # orange-500
    detector_clr  = "#059669",   # emerald-600
    terminal_bg   = "#1E293B",   # slate-800  (dark terminal in light mode)
    terminal_fg   = "#4ADE80",   # green-400
    log_bg        = "#F1F5F9",   # slate-100
    log_info      = "#334155",   # slate-700
    log_warn      = "#92400E",   # amber-800
    log_err       = "#991B1B",   # red-800
    log_ts        = "#94A3B8",   # slate-400
    log_lvl_info  = "#94A3B8",   # slate-400
    log_lvl_warn  = "#B45309",   # amber-700
    log_lvl_err   = "#B91C1C",   # red-700
    banner_ok_bg     = "#dcfce7",  # green-100
    banner_ok_border = "#16a34a",  # green-600
    banner_ok_text   = "#15803d",  # green-700
    banner_err_bg    = "#fee2e2",  # red-100
)


# ═══════════════════════════════════════════════════════════════════════════════
# QSS generator — produces the full stylesheet from a colour token dict
# ═══════════════════════════════════════════════════════════════════════════════

def build_qss(c: dict) -> str:
    """Return a complete application QSS string from a Tailwind colour dict."""
    return f"""
/* ═══════════════════════════════════════════════════════
   EcoMotionZip  —  Tailwind-inspired Qt Stylesheet
   ═══════════════════════════════════════════════════════ */

/* ── Base ──────────────────────────────────────────────────────────────── */
QMainWindow, QDialog {{
    background-color: {c['bg']};
}}
QWidget {{
    background-color: {c['bg']};
    color: {c['text']};
    font-family: "Segoe UI", "SF Pro Text", "Ubuntu", Arial, sans-serif;
    font-size: 13px;
}}
QScrollArea, QScrollArea > QWidget > QWidget {{
    background-color: {c['bg']};
    border: none;
}}

/* ── Transparent backgrounds ───────────────────────────────────────────
   All non-input widgets must be transparent so they don't show the global
   QWidget bg over the surface-coloured GroupBox/header cards.            */
QLabel, QSlider, QCheckBox, QFrame, QAbstractScrollArea {{ background-color: transparent; }}

/* Descendants of GroupBoxes: make all layout containers transparent,
   then explicitly restore widgets that need their own fill.              */
QGroupBox QWidget     {{ background-color: transparent; }}
QGroupBox QLineEdit,
QGroupBox QSpinBox,
QGroupBox QDoubleSpinBox,
QGroupBox QComboBox   {{ background-color: {c['panel']}; }}
QGroupBox QProgressBar {{ background: {c['panel']}; }}
QGroupBox QTextEdit,
QGroupBox QPlainTextEdit {{ background-color: {c['log_bg']}; }}

/* ── Header ────────────────────────────────────────────────────────────── */
QWidget#header {{
    background-color: {c['surface']};
    border-bottom: 1px solid {c['border']};
}}

/* ── Group boxes ───────────────────────────────────────────────────────── */
QGroupBox {{
    background-color: {c['surface']};
    border: 1px solid {c['border']};
    border-radius: 8px;
    margin-top: 16px;
    padding: 14px 12px 12px 12px;
    font-size: 11px;
    font-weight: 700;
    color: {c['text_dim']};
    letter-spacing: 0.8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    top: -1px;
    padding: 2px 8px;
    background-color: transparent;
    color: {c['text_dim']};
    text-transform: uppercase;
}}
/* Detector group: default border, and green when active="true" */
QGroupBox#detector_group {{
    border: 2px solid {c['border']};
}}
QGroupBox#detector_group[active="true"] {{
    border: 2px solid {c['detector_clr']};
}}

/* ── Inputs ────────────────────────────────────────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {c['panel']};
    color: {c['text_bright']};
    border: 1px solid {c['border']};
    border-radius: 6px;
    padding: 7px 10px;
    font-size: 13px;
    selection-background-color: {c['focus']};
    selection-color: #FFFFFF;
    min-height: 20px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1.5px solid {c['focus']};
}}
QWidget:disabled {{
    color: {c['muted']};
}}
QLabel:disabled {{
    color: {c['muted']};
}}
QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {c['muted']};
    background-color: {c['bg']};
    border-color: {c['border']};
}}
QSpinBox:disabled::up-button, QSpinBox:disabled::down-button {{
    background: {c['bg']};
}}
QLineEdit::placeholder-text {{
    color: {c['placeholder']};
    font-style: italic;
}}
QSpinBox::up-button, QSpinBox::down-button {{
    width: 20px;
    background: {c['border']};
    border-radius: 3px;
    margin: 1px;
}}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
    background: {c['muted']};
}}
QComboBox::drop-down {{
    border: none;
    width: 28px;
    background: {c['border']};
    border-radius: 0 5px 5px 0;
}}
QComboBox QAbstractItemView {{
    background-color: {c['panel']};
    color: {c['text_bright']};
    border: 1px solid {c['border']};
    selection-background-color: {c['focus']};
    selection-color: #FFFFFF;
    outline: none;
    padding: 3px;
}}

/* ── Sliders ───────────────────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 6px;
    background: {c['panel']};
    border: 1px solid {c['border']};
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: {c['focus']};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {c['surface']};
    border: 2.5px solid {c['focus']};
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{
    background: {c['focus']};
    border-color: {c['focus_hv']};
}}

/* ── Checkboxes ────────────────────────────────────────────────────────── */
QCheckBox {{
    color: {c['text']};
    spacing: 8px;
    font-size: 13px;
}}
QCheckBox::indicator {{
    width: 17px;
    height: 17px;
    border-radius: 4px;
    border: 1.5px solid {c['border']};
    background: {c['panel']};
}}
QCheckBox::indicator:hover    {{ border-color: {c['focus']}; }}
QCheckBox::indicator:checked  {{ background: {c['focus']}; border-color: {c['focus']}; }}

/* ── Buttons (base) ────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {c['panel']};
    color: {c['text']};
    border: 1px solid {c['border']};
    border-radius: 6px;
    padding: 7px 14px;
    font-size: 13px;
    font-weight: 500;
}}
QPushButton:hover   {{ background-color: {c['border']}; color: {c['text_bright']}; }}
QPushButton:pressed {{ background-color: {c['muted']};  color: {c['text_bright']}; }}
QPushButton:disabled {{
    background-color: {c['panel']};
    color: {c['muted']};
    border-color: {c['border']};
}}

/* Primary — emerald-600 */
QPushButton#primary_btn {{
    background-color: {c['primary']};
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 13px 32px;
    font-weight: 700;
    font-size: 14px;
    letter-spacing: 0.4px;
}}
QPushButton#primary_btn:hover   {{ background-color: {c['primary_hv']}; }}
QPushButton#primary_btn:pressed {{ background-color: {c['primary_hv']}; }}

/* Stop — red-600 */
QPushButton#stop_btn {{
    background-color: {c['stop']};
    color: #FFFFFF;
    border: none;
    border-radius: 6px;
    padding: 8px 20px;
    font-weight: 700;
}}
QPushButton#stop_btn:hover {{ background-color: {c['stop_hv']}; }}

/* Resume — emerald-600 */
QPushButton#resume_btn {{
    background-color: {c['primary']};
    color: #FFFFFF;
    border: none;
    border-radius: 6px;
    padding: 8px 20px;
    font-weight: 700;
}}
QPushButton#resume_btn:hover {{ background-color: {c['primary_hv']}; }}

/* Header tab buttons */
QPushButton#tab_btn {{
    background-color: transparent;
    color: {c['text_dim']};
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    padding: 10px 22px;
    font-weight: 500;
    font-size: 13px;
}}
QPushButton#tab_btn:checked {{
    color: {c['primary']};
    border-bottom: 2px solid {c['primary']};
    font-weight: 700;
    background-color: transparent;
}}
QPushButton#tab_btn:hover:!checked {{
    color: {c['text']};
    background-color: transparent;
}}

/* Browse */
QPushButton#browse_btn {{
    background-color: {c['border']};
    color: {c['text']};
    border: 1px solid {c['muted']};
    padding: 7px 14px;
    font-size: 12px;
    font-weight: 600;
    border-radius: 6px;
}}
QPushButton#browse_btn:hover {{ background-color: {c['muted']}; color: {c['text_bright']}; }}

/* Advanced settings toggle switch */
QCheckBox#adv_toggle {{
    spacing: 6px;
    color: {c['text_dim']};
    font-size: 12px;
}}
QCheckBox#adv_toggle::indicator {{
    width: 38px;
    height: 20px;
    border-radius: 10px;
    border: 2px solid {c['border']};
    background-color: {c['panel']};
}}
QCheckBox#adv_toggle::indicator:checked {{
    background-color: {c['focus']};
    border-color: {c['focus']};
}}
QCheckBox#adv_toggle::indicator:hover {{
    border-color: {c['focus']};
}}

/* Copy */
QPushButton#copy_btn {{
    background: {c['border']};
    color: {c['text']};
    border: none;
    border-radius: 5px;
    padding: 5px 16px;
    font-size: 12px;
    font-weight: 500;
}}
QPushButton#copy_btn:hover {{ background: {c['muted']}; color: {c['text_bright']}; }}

/* Theme toggle */
QPushButton#theme_btn {{
    background: {c['panel']};
    color: {c['text_dim']};
    border: 1px solid {c['border']};
    border-radius: 14px;
    padding: 4px 14px;
    font-size: 12px;
    min-width: 72px;
}}
QPushButton#theme_btn:hover {{ border-color: {c['focus']}; color: {c['text']}; }}

/* ── Progress bars ─────────────────────────────────────────────────────── */
QProgressBar {{
    background: {c['panel']};
    border: 1px solid {c['border']};
    border-radius: 5px;
    color: transparent;
    min-height: 10px;
    max-height: 10px;
    text-align: center;
}}
QProgressBar#reader_bar::chunk {{
    background: {c['reader_clr']};
    border-radius: 5px;
}}
QProgressBar#writer_bar::chunk {{
    background: {c['writer_clr']};
    border-radius: 5px;
}}
QProgressBar#overall_bar {{
    background: {c['panel']};
    border: 1px solid {c['border']};
    border-radius: 5px;
    color: transparent;
    min-height: 8px;
    max-height: 8px;
}}
QProgressBar#overall_bar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {c['reader_clr']}, stop:1 {c['primary']});
    border-radius: 5px;
}}

/* ── CLI terminal preview ──────────────────────────────────────────────── */
QPlainTextEdit#cli_preview {{
    background-color: {c['terminal_bg']};
    color: {c['terminal_fg']};
    border: 1px solid {c['border']};
    border-radius: 6px;
    font-family: "Menlo", "Consolas", "Courier New", "DejaVu Sans Mono", monospace;
    font-size: 12px;
    padding: 10px;
    selection-background-color: {c['focus']};
    selection-color: #FFFFFF;
}}

/* ── Log view ──────────────────────────────────────────────────────────── */
QTextEdit#log_view {{
    background-color: {c['log_bg']};
    color: {c['text']};
    border: 1px solid {c['border']};
    border-radius: 6px;
    font-family: "Menlo", "Consolas", "Courier New", monospace;
    font-size: 12px;
    padding: 8px;
    selection-background-color: {c['focus']};
}}

/* ── Named labels ──────────────────────────────────────────────────────── */
QLabel#title_lbl {{
    color: {c['text_bright']};
    font-size: 20px;
    font-weight: 700;
    background: transparent;
}}
QLabel#subtitle_lbl {{
    color: {c['text_dim']};
    font-size: 12px;
    background: transparent;
}}
QLabel#section_lbl {{
    color: {c['text_bright']};
    font-size: 13px;
    font-weight: 600;
    background: transparent;
}}
QLabel#hint_lbl {{
    color: {c['text_dim']};
    font-size: 11px;
    line-height: 1.5;
    background: transparent;
}}
QLabel#val_badge {{
    color: {c['focus']};
    font-weight: 700;
    font-size: 15px;
    background: transparent;
    padding: 1px 8px;
    min-width: 40px;
}}
QLabel#frame_counter {{
    color: {c['primary_lt']};
    font-size: 34px;
    font-weight: 700;
    font-family: "Menlo", "Consolas", "Courier New", monospace;
    background: transparent;
}}
QLabel#status_active {{
    color: {c['primary_lt']};
    font-weight: 700;
    font-size: 13px;
    background: transparent;
}}
QLabel#status_stopped {{
    color: {c['text_dim']};
    font-weight: 600;
    font-size: 13px;
    background: transparent;
}}
QLabel#queue_lbl {{
    color: {c['text']};
    font-size: 12px;
    background: transparent;
}}
QLabel#detector_active {{
    color: {c['detector_clr']};
    font-weight: 700;
    font-size: 14px;
    background: transparent;
}}
QLabel#detector_idle {{
    color: {c['muted']};
    font-size: 14px;
    background: transparent;
}}

/* ── Scroll bars ───────────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 0;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {c['border']};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: {c['muted']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
    border: none;
}}
QScrollBar::handle:horizontal {{ background: {c['border']}; border-radius: 4px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ── H-line separators ─────────────────────────────────────────────────── */
QFrame[frameShape="4"] {{
    background: {c['border']};
    border: none;
    max-height: 1px;
    min-height: 1px;
}}

/* ── Result banner ─────────────────────────────────────────────────────── */
QFrame#result_success {{
    background-color: {c['banner_ok_bg']};
    border: 1.5px solid {c['banner_ok_border']};
    border-radius: 10px;
}}
QFrame#result_error {{
    background-color: {c['banner_err_bg']};
    border: 1.5px solid {c['stop']};
    border-radius: 10px;
}}
QLabel#banner_icon_ok  {{ color: {c['banner_ok_text']}; font-size: 28px; background: transparent; }}
QLabel#banner_icon_err {{ color: {c['error']}; font-size: 28px; background: transparent; }}
QLabel#banner_title_ok  {{ color: {c['banner_ok_text']}; font-size: 15px; font-weight: 700; background: transparent; }}
QLabel#banner_title_err {{ color: {c['error']}; font-size: 15px; font-weight: 700; background: transparent; }}
QLabel#banner_body  {{ color: {c['text']}; font-size: 12px; background: transparent; }}
QPushButton#dismiss_btn {{
    background: transparent;
    color: {c['muted']};
    border: none;
    font-size: 18px;
    padding: 0 6px;
}}
QPushButton#dismiss_btn:hover {{ color: {c['text_bright']}; }}

/* ── Friendly status panel ─────────────────────────────────────────────── */
QLabel#friendly_status {{
    color: {c['text_bright']};
    font-size: 13px;
    font-weight: 600;
    background: transparent;
}}
QPushButton#log_toggle_btn {{
    background: transparent;
    color: {c['focus']};
    border: 1px solid {c['border']};
    border-radius: 5px;
    padding: 4px 14px;
    font-size: 12px;
    font-weight: 500;
}}
QPushButton#log_toggle_btn:hover {{ border-color: {c['focus']}; background: {c['panel']}; }}
"""


# ── Pre-built QSS strings (module-level, as required by spec) ──────────────────
DARK_QSS  = build_qss(DARK)
LIGHT_QSS = build_qss(LIGHT)


# ── Mock pipeline log message templates ────────────────────────────────────────
_MOCK_LOGS: list[tuple[str, str]] = [
    ("INFO",  "ReaderThread   | Decoded frame batch ({b} frames in reading queue)"),
    ("INFO",  "MotionDetector | Frame difference computed — {p:,} pixels exceed threshold"),
    ("INFO",  "MotionDetector | Dilation applied (kernel={k}px, downscale={d}×)"),
    ("INFO",  "WriterThread   | Compressed frame {f:,} written to encoder buffer"),
    ("INFO",  "WriterThread   | JPEG snapshot saved → frame_{f}.jpg"),
    ("INFO",  "MotionDetector | Background blend applied (transparency={t:.1f})"),
    ("INFO",  "ReaderThread   | Smart-sleep engaged ({s}ms, queue at {q}%)"),
    ("WARN",  "ReaderThread   | Reading queue at {q}% capacity — reader sleeping"),
    ("INFO",  "MotionDetector | Full keyframe captured at interval boundary"),
    ("WARN",  "WriterThread   | Writing queue at {q}% capacity"),
    ("INFO",  "WriterThread   | CSV sidecar updated: {f:,} frame mappings written"),
    ("INFO",  "MotionDetector | Post-motion buffer: {b} frames remaining"),
    ("INFO",  "ReaderThread   | Batch read in {s}ms — {b} frames queued"),
    ("ERROR", "WriterThread   | Frame decode warning — retrying (attempt 1/3)"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class ConfigTab(QWidget):
    """
    Configuration panel.

    Every input widget is connected to ``_update_cli_preview``, which
    dynamically rebuilds the CLI command string whenever any value changes.
    """

    def __init__(self, on_launch, parent=None):
        super().__init__(parent)
        self._on_launch = on_launch
        self._adv_spins: dict[str, QSpinBox] = {}

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(24, 16, 24, 24)
        vbox.setSpacing(14)

        vbox.addWidget(self._build_io_group())
        vbox.addWidget(self._build_split_section())
        vbox.addLayout(self._build_advanced_section())

        # ── Launch button ──────────────────────────────────────────────────
        self._launch_btn = QPushButton("▶   Run")
        self._launch_btn.setObjectName("primary_btn")
        self._launch_btn.setFixedHeight(50)
        self._launch_btn.clicked.connect(self._launch)
        vbox.addWidget(self._launch_btn)
        vbox.addStretch()

    # ── Directories ────────────────────────────────────────────────────────

    def _build_io_group(self) -> QGroupBox:
        g = QGroupBox("Directories")
        layout = QVBoxLayout(g)
        layout.setSpacing(12)

        # Video source row — file(s) or folder
        vs_row = QHBoxLayout()
        vs_lbl = QLabel("Video Source")
        vs_lbl.setObjectName("section_lbl")
        vs_lbl.setFixedWidth(136)
        self.video_source = QLineEdit()
        self.video_source.setPlaceholderText("Select a video file, multiple files, or a folder…")
        btn_files = QPushButton("File(s)")
        btn_files.setObjectName("browse_btn")
        btn_files.setFixedWidth(68)
        btn_files.setToolTip("Select one or more video files")
        btn_files.clicked.connect(self._browse_video_files)
        btn_folder = QPushButton("Folder")
        btn_folder.setObjectName("browse_btn")
        btn_folder.setFixedWidth(68)
        btn_folder.setToolTip("Select a folder — all videos inside will be processed")
        btn_folder.clicked.connect(self._browse_video_folder)
        vs_row.addWidget(vs_lbl)
        vs_row.addWidget(self.video_source)
        vs_row.addWidget(btn_files)
        vs_row.addWidget(btn_folder)
        layout.addLayout(vs_row)

        # Output directory row
        od_row = QHBoxLayout()
        od_lbl = QLabel("Output Directory")
        od_lbl.setObjectName("section_lbl")
        od_lbl.setFixedWidth(136)
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("Select an output folder…")
        btn_od = QPushButton("Browse")
        btn_od.setObjectName("browse_btn")
        btn_od.setFixedWidth(82)
        btn_od.clicked.connect(self._browse_output)
        od_row.addWidget(od_lbl)
        od_row.addWidget(self.output_dir)
        od_row.addWidget(btn_od)
        layout.addLayout(od_row)

        self.delete_original = QCheckBox("Delete original file after processing is complete")
        layout.addWidget(self.delete_original)

        # Wire every signal → CLI preview
        self.video_source.textChanged.connect(self._update_cli_preview)
        self.output_dir.textChanged.connect(self._update_cli_preview)
        self.delete_original.toggled.connect(self._update_cli_preview)

        return g

    # ── Motion Tuning + Encoding Settings (side-by-side) ──────────────────

    def _build_split_section(self) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        row.addWidget(self._build_motion_group(), stretch=1)
        row.addWidget(self._build_encoding_group(), stretch=1)
        return container

    def _build_motion_group(self) -> QGroupBox:
        g = QGroupBox("Motion Tuning")
        layout = QVBoxLayout(g)
        layout.setSpacing(8)

        # Label + live value badge
        top = QHBoxLayout()
        lbl = QLabel("Movement Threshold")
        lbl.setObjectName("section_lbl")
        self._thresh_badge = QLabel("30")
        self._thresh_badge.setObjectName("val_badge")
        self._thresh_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thresh_badge.setFixedWidth(52)
        top.addWidget(lbl)
        top.addStretch()
        top.addWidget(self._thresh_badge)
        layout.addLayout(top)

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(30)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        layout.addWidget(self.threshold_slider)

        hint = QLabel(
            "Pixel brightness change (0–255) required to flag a pixel as moving.\n"
            "Low values detect subtle movement; high values ignore minor variation."
        )
        hint.setObjectName("hint_lbl")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # Downscale Factor (moved here from Advanced)
        ds_row = QHBoxLayout()
        ds_lbl = QLabel("Downscale Factor")
        ds_lbl.setObjectName("section_lbl")
        downscale_spin = QSpinBox()
        downscale_spin.setRange(1, 64)
        downscale_spin.setValue(16)
        downscale_spin.setFixedWidth(72)
        downscale_spin.setToolTip(
            "Frame is shrunk by this factor before motion detection.\n"
            "Higher = faster but less precise; lower = slower but detects finer movement."
        )
        downscale_spin.valueChanged.connect(self._update_cli_preview)
        ds_row.addWidget(ds_lbl)
        ds_row.addStretch()
        ds_row.addWidget(downscale_spin)
        layout.addLayout(ds_row)
        ds_hint = QLabel("Higher values speed up processing; lower values detect finer motion.")
        ds_hint.setObjectName("hint_lbl")
        ds_hint.setWordWrap(True)
        layout.addWidget(ds_hint)

        self._adv_spins["downscale_factor"] = downscale_spin
        layout.addStretch()
        return g

    def _on_threshold_changed(self, v: int):
        self._thresh_badge.setText(str(v))
        self._update_cli_preview()

    def _build_encoding_group(self) -> QGroupBox:
        g = QGroupBox("Encoding Settings")
        layout = QVBoxLayout(g)
        layout.setSpacing(10)

        # Codec
        codec_row = QHBoxLayout()
        codec_lbl = QLabel("Video Codec")
        codec_lbl.setObjectName("section_lbl")
        self.codec_combo = QComboBox()
        self.codec_combo.addItems(["DIVX", "X264", "FFV1", "HEVC", "H265"])
        self.codec_combo.setCurrentText("X264")
        self.codec_combo.setFixedWidth(110)
        codec_row.addWidget(codec_lbl)
        codec_row.addStretch()
        codec_row.addWidget(self.codec_combo)
        layout.addLayout(codec_row)

        # Background transparency slider
        bg_top = QHBoxLayout()
        bg_lbl = QLabel("Background Transparency")
        bg_lbl.setObjectName("section_lbl")
        self._bg_badge = QLabel("100 %")
        self._bg_badge.setObjectName("val_badge")
        self._bg_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._bg_badge.setFixedWidth(52)
        bg_top.addWidget(bg_lbl)
        bg_top.addStretch()
        bg_top.addWidget(self._bg_badge)
        layout.addLayout(bg_top)

        self.bg_slider = QSlider(Qt.Orientation.Horizontal)
        self.bg_slider.setRange(0, 100)   # 0–100 % → passed as 0.0–1.0
        self.bg_slider.setValue(100)
        self.bg_slider.valueChanged.connect(self._on_bg_changed)
        layout.addWidget(self.bg_slider)

        # Save JPEGs checkbox + hidden spinbox
        jpeg_row = QHBoxLayout()
        self.save_jpegs = QCheckBox("Save individual JPEGs")
        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(1, 500_000)
        self.max_frames_spin.setValue(2000)
        self.max_frames_spin.setFixedWidth(90)
        self.max_frames_spin.setVisible(False)
        jpeg_row.addWidget(self.save_jpegs)
        jpeg_row.addStretch()
        jpeg_row.addWidget(self.max_frames_spin)
        layout.addLayout(jpeg_row)

        hint2 = QLabel("Max frames visible when 'Save JPEGs' is checked.")
        hint2.setObjectName("hint_lbl")
        layout.addWidget(hint2)

        # Timestamp overlay checkbox
        self.embed_timestamps = QCheckBox("Embed timestamps in video")
        layout.addWidget(self.embed_timestamps)
        ts_hint = QLabel(
            "Burns frame number and time into each output frame.\n"
            "Useful for identifying exact moments of activity in footage."
        )
        ts_hint.setObjectName("hint_lbl")
        ts_hint.setWordWrap(True)
        layout.addWidget(ts_hint)

        layout.addStretch()

        # Wire signals
        self.codec_combo.currentTextChanged.connect(self._update_cli_preview)
        self.save_jpegs.toggled.connect(self._on_jpeg_toggle)
        self.max_frames_spin.valueChanged.connect(self._update_cli_preview)
        self.embed_timestamps.toggled.connect(self._update_cli_preview)

        return g

    def _on_bg_changed(self, v: int):
        self._bg_badge.setText(f"{v} %")
        self._update_cli_preview()

    def _on_jpeg_toggle(self, checked: bool):
        self.max_frames_spin.setVisible(checked)
        self._update_cli_preview()

    # ── Advanced Settings (always visible, enabled by toggle) ─────────────

    def _build_advanced_section(self) -> QVBoxLayout:
        outer = QVBoxLayout()
        outer.setSpacing(6)

        # Header row: toggle switch + label
        hdr = QHBoxLayout()
        self._adv_toggle = QCheckBox()
        self._adv_toggle.setObjectName("adv_toggle")
        self._adv_toggle.setChecked(False)
        self._adv_toggle.toggled.connect(self._toggle_advanced)
        lbl = QLabel("Advanced Settings")
        lbl.setObjectName("section_lbl")
        hdr.addWidget(self._adv_toggle)
        hdr.addWidget(lbl)
        hdr.addStretch()
        outer.addLayout(hdr)

        # Settings row — always shown, disabled until toggle is on
        self._adv_panel = QWidget()
        self._adv_panel.setEnabled(False)
        row_layout = QHBoxLayout(self._adv_panel)
        row_layout.setSpacing(16)
        row_layout.setContentsMargins(0, 4, 0, 4)

        specs = [
            ("Dilate Kernel Size",    "dilate_kernel_size",        1,  255, 128),
            ("Post-Motion Frames",    "post_motion_record_frames", 0,  500,   0),
            ("OpenCV Threads",        "num_opencv_threads",        1,   64,  10),
        ]
        for label, key, mn, mx, default in specs:
            cell = QWidget()
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(4)
            fl = QLabel(label)
            fl.setObjectName("section_lbl")
            spin = QSpinBox()
            spin.setRange(mn, mx)
            spin.setValue(default)
            spin.valueChanged.connect(self._update_cli_preview)
            cl.addWidget(fl)
            cl.addWidget(spin)
            row_layout.addWidget(cell, stretch=1)
            self._adv_spins[key] = spin
        row_layout.addStretch()

        outer.addWidget(self._adv_panel)
        return outer

    def _toggle_advanced(self, enabled: bool):
        self._adv_panel.setEnabled(enabled)

    def _update_cli_preview(self):
        """No-op — CLI preview panel has been removed from the Config tab."""

    # ── File browse helpers ────────────────────────────────────────────────

    def _browse_video_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Video File(s)", str(Path.home()),
            "Video Files (*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV);;All Files (*)",
        )
        if paths:
            # Single path stored as-is; multiple paths joined with the OS separator
            self.video_source.setText(paths[0] if len(paths) == 1 else os.pathsep.join(paths))

    def _browse_video_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Video Folder", str(Path.home())
        )
        if path:
            self.video_source.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", str(Path.home())
        )
        if path:
            self.output_dir.setText(path)

    # ── Launch ────────────────────────────────────────────────────────────

    def _launch(self):
        self._on_launch(self._collect_cfg())

    def _collect_cfg(self) -> dict:
        return {
            "video_source":                      self.video_source.text().strip(),
            "output_directory":                  self.output_dir.text().strip(),
            "movement_threshold":                self.threshold_slider.value(),
            "video_codec":                       self.codec_combo.currentText(),
            "background_transparency":           self.bg_slider.value() / 100,
            "save_frames":                       self.save_jpegs.isChecked(),
            "frames_to_save":                    self.max_frames_spin.value(),
            "delete_original_after_processing":  self.delete_original.isChecked(),
            "embed_timestamps":                  self.embed_timestamps.isChecked(),
            **{k: s.value() for k, s in self._adv_spins.items()},
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Pipeline Monitor
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineMonitorTab(QWidget):
    """
    Live pipeline monitor.

    Architecture layout:
        ┌──────────────┐   ┌──────────────────┐   ┌──────────────┐
        │ Reader Thread│ → │ Motion Detector  │ → │ Writer Thread│
        │ queue bar    │   │ status + border  │   │ queue bar    │
        └──────────────┘   └──────────────────┘   └──────────────┘

    A 500 ms QTimer drives the visual simulation independent of log output
    cadence, keeping the UI responsive even when the subprocess produces
    no output for several seconds.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._running        = False
        self._completed      = False   # True after successful exit
        self._reader_q       = 0
        self._writer_q       = 0
        self._frames         = 0
        self._proc: QProcess | None = None
        self._video_source   = ""
        self._output_dir     = ""
        self._stdout_buffer: list[str] = []
        self._log_visible    = False   # raw log hidden by default
        self._start_time     = 0.0
        self._on_new_run_cb  = None    # set by MainWindow
        self._video_files:   list[str] = []
        self._total_videos   = 0
        self._current_video_idx = 0
        self._on_next_video_cb = None  # set by MainWindow when more files are queued

        # Simulation timer — 500 ms as per spec
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._tick)

        # Colour tokens for the log panel (updated on theme switch via MainWindow)
        self._log_colors: dict = DARK

        self._build_ui()

    # ── Construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 20)
        root.setSpacing(10)

        root.addWidget(self._build_control_bar())   # status + what's happening + progress bar

        # Result banner — hidden until pipeline finishes
        self._result_banner = self._build_result_banner()
        root.addWidget(self._result_banner)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(sep)

        root.addWidget(self._build_arch_panel())    # compact single-line rows
        root.addWidget(self._build_console(), stretch=1)  # console always at bottom

    # ── Control bar ───────────────────────────────────────────────────────

    def _build_control_bar(self) -> QWidget:
        bar = QWidget()
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # ── Top row: status | frame counter | button ───────────────────────
        top_row = QWidget()
        top_layout = QHBoxLayout(top_row)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(24)

        status_col = QVBoxLayout()
        status_col.setSpacing(2)
        status_col.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_cap = QLabel("Pipeline Status")
        status_cap.setObjectName("hint_lbl")
        self.status_lbl = QLabel("● Stopped")
        self.status_lbl.setObjectName("status_stopped")
        status_col.addWidget(status_cap)
        status_col.addWidget(self.status_lbl)
        top_layout.addLayout(status_col)

        vline = QFrame()
        vline.setFrameShape(QFrame.Shape.VLine)
        vline.setStyleSheet("background: #334155; border: none; max-width: 1px;")
        top_layout.addWidget(vline)

        fc_col = QVBoxLayout()
        fc_col.setSpacing(2)
        fc_col.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fc_cap = QLabel("Frames Encoded")
        fc_cap.setObjectName("hint_lbl")
        fc_cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_counter = QLabel("0")
        self.frame_counter.setObjectName("frame_counter")
        self.frame_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._current_file_lbl = QLabel("")
        self._current_file_lbl.setObjectName("hint_lbl")
        self._current_file_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._current_file_lbl.setVisible(False)
        self._video_progress_lbl = QLabel("")
        self._video_progress_lbl.setObjectName("hint_lbl")
        self._video_progress_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_progress_lbl.setVisible(False)
        fc_col.addWidget(fc_cap)
        fc_col.addWidget(self.frame_counter)
        fc_col.addWidget(self._current_file_lbl)
        fc_col.addWidget(self._video_progress_lbl)
        top_layout.addLayout(fc_col, stretch=1)

        vline2 = QFrame()
        vline2.setFrameShape(QFrame.Shape.VLine)
        vline2.setStyleSheet("background: #334155; border: none; max-width: 1px;")
        top_layout.addWidget(vline2)

        self.toggle_btn = QPushButton("■   Stop")
        self.toggle_btn.setObjectName("stop_btn")
        self.toggle_btn.setFixedSize(QSize(140, 44))
        self.toggle_btn.clicked.connect(self._toggle_pipeline)
        top_layout.addWidget(self.toggle_btn)
        layout.addWidget(top_row)

        # ── "What's happening" label — sits between status row and progress bar
        self._friendly_status = QLabel("Waiting for pipeline to start…")
        self._friendly_status.setObjectName("friendly_status")
        self._friendly_status.setWordWrap(True)
        layout.addWidget(self._friendly_status)

        # ── Progress bar ───────────────────────────────────────────────────
        pb_row = QWidget()
        pb_layout = QHBoxLayout(pb_row)
        pb_layout.setContentsMargins(0, 0, 0, 0)
        pb_layout.setSpacing(8)
        pb_cap = QLabel("Progress")
        pb_cap.setObjectName("hint_lbl")
        pb_cap.setFixedWidth(56)
        self._overall_bar = QProgressBar()
        self._overall_bar.setObjectName("overall_bar")
        self._overall_bar.setRange(0, 0)
        self._overall_bar.setTextVisible(False)
        self._overall_bar.setVisible(False)
        pb_layout.addWidget(pb_cap)
        pb_layout.addWidget(self._overall_bar, stretch=1)
        layout.addWidget(pb_row)

        return bar

    # ── Architecture panel (compact single-line rows) ─────────────────────

    def _build_arch_panel(self) -> QGroupBox:
        g = QGroupBox("Processing Pipeline")
        layout = QVBoxLayout(g)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(10)

        def _bar(obj_name: str) -> QProgressBar:
            b = QProgressBar()
            b.setObjectName(obj_name)
            b.setRange(0, 100)
            b.setValue(0)
            b.setTextVisible(False)
            b.setFixedWidth(120)
            return b

        # ── Reader row ─────────────────────────────────────────────────
        reader_row = QHBoxLayout()
        reader_row.setSpacing(8)
        r_lbl = QLabel("Reader Thread")
        r_lbl.setObjectName("section_lbl")
        r_lbl.setFixedWidth(138)
        self.reader_bar = _bar("reader_bar")
        self.reader_pct = QLabel("0 %")
        self.reader_pct.setObjectName("queue_lbl")
        self.reader_pct.setFixedWidth(34)
        self._reader_act = QLabel("Waiting…")
        self._reader_act.setObjectName("hint_lbl")
        reader_row.addWidget(r_lbl)
        reader_row.addWidget(self.reader_bar)
        reader_row.addWidget(self.reader_pct)
        reader_row.addWidget(self._reader_act, stretch=1)
        layout.addLayout(reader_row)

        # ── Motion Detector row ────────────────────────────────────────
        det_row = QHBoxLayout()
        det_row.setSpacing(8)
        d_lbl = QLabel("Motion Detector")
        d_lbl.setObjectName("section_lbl")
        d_lbl.setFixedWidth(138)
        self.detector_status = QLabel("● Idle")
        self.detector_status.setObjectName("detector_idle")
        self.detector_status.setFixedWidth(120)   # aligns with bars above/below
        self.detector_detail = QLabel("Waiting for pipeline to start…")
        self.detector_detail.setObjectName("hint_lbl")
        det_row.addWidget(d_lbl)
        det_row.addWidget(self.detector_status)
        det_row.addSpacing(42)                    # aligns with pct label column
        det_row.addWidget(self.detector_detail, stretch=1)
        layout.addLayout(det_row)

        # ── Writer row ─────────────────────────────────────────────────
        writer_row = QHBoxLayout()
        writer_row.setSpacing(8)
        w_lbl = QLabel("Writer Thread")
        w_lbl.setObjectName("section_lbl")
        w_lbl.setFixedWidth(138)
        self.writer_bar = _bar("writer_bar")
        self.writer_pct = QLabel("0 %")
        self.writer_pct.setObjectName("queue_lbl")
        self.writer_pct.setFixedWidth(34)
        self._writer_act = QLabel("Waiting…")
        self._writer_act.setObjectName("hint_lbl")
        writer_row.addWidget(w_lbl)
        writer_row.addWidget(self.writer_bar)
        writer_row.addWidget(self.writer_pct)
        writer_row.addWidget(self._writer_act, stretch=1)
        layout.addLayout(writer_row)

        return g

    # ── Result banner (shown after completion) ────────────────────────────

    def _build_result_banner(self) -> QFrame:
        """Card shown after the pipeline exits — green (success) or red (error)."""
        frame = QFrame()
        frame.setObjectName("result_success")   # default; toggled in _show_result
        frame.setVisible(False)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(14)

        self._banner_icon = QLabel("✓")
        self._banner_icon.setObjectName("banner_icon_ok")
        self._banner_icon.setFixedWidth(36)
        self._banner_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        text_col = QVBoxLayout()
        text_col.setSpacing(3)
        self._banner_title = QLabel("Processing complete!")
        self._banner_title.setObjectName("banner_title_ok")
        self._banner_body  = QLabel("")
        self._banner_body.setObjectName("banner_body")
        self._banner_body.setWordWrap(True)
        text_col.addWidget(self._banner_title)
        text_col.addWidget(self._banner_body)

        dismiss = QPushButton("✕")
        dismiss.setObjectName("dismiss_btn")
        dismiss.setFixedWidth(32)
        dismiss.clicked.connect(lambda: frame.setVisible(False))

        layout.addWidget(self._banner_icon)
        layout.addLayout(text_col, stretch=1)
        layout.addWidget(dismiss)
        return frame

    def _show_result(self, success: bool, body: str):
        """Update and reveal the result banner."""
        if success:
            self._result_banner.setObjectName("result_success")
            self._banner_icon.setObjectName("banner_icon_ok")
            self._banner_icon.setText("✓")
            self._banner_title.setObjectName("banner_title_ok")
            self._banner_title.setText("Processing complete!")
        else:
            self._result_banner.setObjectName("result_error")
            self._banner_icon.setObjectName("banner_icon_err")
            self._banner_icon.setText("✗")
            self._banner_title.setObjectName("banner_title_err")
            self._banner_title.setText("Processing failed")
        self._banner_body.setText(body)
        # Force QSS re-evaluation after objectName change
        for w in (self._result_banner, self._banner_icon, self._banner_title):
            w.style().unpolish(w)
            w.style().polish(w)
        self._result_banner.setVisible(True)

    # ── Console output (always visible at bottom) ─────────────────────────

    def _build_console(self) -> QGroupBox:
        g = QGroupBox("Console Output")
        layout = QVBoxLayout(g)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.setSpacing(0)
        self.log_view = QTextEdit()
        self.log_view.setObjectName("log_view")
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view)
        return g

    def _update_friendly_status(self, line: str):
        """Map a raw stdout line to a plain-English status message."""
        lu = line.upper()
        if "READER" in lu or "READING" in lu:
            msg = "Reading video frames from source file…"
        elif "MOTION" in lu or "DETECTOR" in lu or "THRESHOLD" in lu:
            msg = "Analysing frames for motion…"
        elif "JPEG" in lu or "SNAPSHOT" in lu:
            msg = "Saving snapshot images to disk…"
        elif "CSV" in lu or "SIDECAR" in lu:
            msg = "Writing motion index file…"
        elif "WRITER" in lu or "ENCOD" in lu or "WRITING" in lu:
            msg = "Encoding compressed video to disk…"
        elif "FINISH" in lu or "COMPLETE" in lu or "DONE" in lu:
            msg = "Finishing up — closing output files…"
        else:
            return   # don't overwrite a good message with a generic one
        self._friendly_status.setText(msg)

    # ── Public API ────────────────────────────────────────────────────────

    def set_source_paths(self, video_source: str, output_dir: str):
        """Allow MainWindow to pass file paths for compression-ratio calculation."""
        self._video_source = video_source
        self._output_dir   = output_dir

        # Resolve the list of videos for filename/progress display
        self._video_files = []
        self._current_video_idx = 0
        if not video_source:
            self._current_file_lbl.setVisible(False)
            self._video_progress_lbl.setVisible(False)
            return

        src = Path(video_source)
        if src.is_dir():
            _exts = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"}
            self._video_files = sorted(
                str(f) for f in src.iterdir() if f.suffix in _exts
            )
        elif os.pathsep in video_source:
            self._video_files = video_source.split(os.pathsep)
        else:
            self._video_files = [video_source]

        self._total_videos = len(self._video_files)
        if self._total_videos > 0:
            self._current_file_lbl.setText(Path(self._video_files[0]).name)
            self._current_file_lbl.setVisible(True)
        if self._total_videos > 1:
            self._video_progress_lbl.setText(f"Video 1 of {self._total_videos}")
            self._video_progress_lbl.setVisible(True)
        else:
            self._video_progress_lbl.setVisible(False)

    def start_pipeline(self, proc: QProcess | None = None):
        """Called by MainWindow. Starts simulation timer + optional QProcess relay."""
        import time
        self._running        = True
        self._completed      = False
        self._frames         = 0
        self._reader_q       = 0
        self._writer_q       = 0
        self._proc           = proc
        self._stdout_buffer  = []
        self._start_time     = time.monotonic()
        self._result_banner.setVisible(False)
        self._friendly_status.setText("Starting up — initialising threads…")
        self._overall_bar.setRange(0, 0)   # indeterminate
        self._overall_bar.setVisible(True)
        self._timer.start()
        self._set_running_ui(True)

        self.log_msg("Pipeline started — threads initialising.", "INFO")
        self.log_msg("Reader   → reading_queue (maxsize=512 frames)", "INFO")
        self.log_msg("Detector → writing_queue (maxsize=256 frames)", "INFO")
        self.log_msg("Writer   → encoding output to disk", "INFO")

        if proc:
            proc.readyRead.connect(self._on_proc_output)
            proc.finished.connect(self._on_proc_finished)

    def stop_pipeline(self):
        """Pause the timer and update all UI labels to reflect a stopped state."""
        self._running = False
        self._timer.stop()
        self._overall_bar.setVisible(False)
        self._set_running_ui(False)
        if self._proc and self._proc.state() != QProcess.ProcessState.NotRunning:
            self._proc.terminate()
        self._friendly_status.setText("Pipeline stopped by user.")
        self.log_msg("─── Pipeline stopped ───────────────────────────────────", "WARN")

    def set_log_colors(self, colors: dict):
        """Allow MainWindow to push updated colour tokens after a theme switch."""
        self._log_colors = colors

    def log_msg(self, msg: str, level: str = "INFO"):
        """
        Append a colour-coded, timestamped line to the log panel.

        Parameters
        ----------
        msg:   Message text.
        level: One of ``"INFO"`` (white/slate), ``"WARN"`` (amber), ``"ERROR"`` (red).
        """
        c  = self._log_colors
        ts = datetime.now().strftime("%H:%M:%S")

        text_color = {
            "INFO":  c.get("log_info",  "#CBD5E1"),
            "WARN":  c.get("log_warn",  "#FBBF24"),
            "ERROR": c.get("log_err",   "#F87171"),
        }.get(level, c.get("log_info", "#CBD5E1"))

        lvl_color = {
            "INFO":  c.get("log_lvl_info", "#64748B"),
            "WARN":  c.get("log_lvl_warn", "#D97706"),
            "ERROR": c.get("log_lvl_err",  "#DC2626"),
        }.get(level, c.get("log_lvl_info", "#64748B"))

        ts_color = c.get("log_ts", "#475569")

        html = (
            f'<span style="color:{ts_color};">[{ts}]</span>&nbsp;'
            f'<span style="color:{lvl_color};font-weight:600;">[{level:5s}]</span>&nbsp;'
            f'<span style="color:{text_color};">{msg}</span><br>'
        )
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)
        self.log_view.insertHtml(html)
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)

    # ── Simulation timer tick ─────────────────────────────────────────────

    def _tick(self):
        """Fired every 500 ms. Updates queue levels, frame counter, and detector."""
        if not self._running:
            return

        # Random-walk queue levels
        self._reader_q = max(0, min(100, self._reader_q + random.randint(-8,  14)))
        self._writer_q = max(0, min(100, self._writer_q + random.randint(-12,  8)))

        # Only simulate frame increment when no real process is attached.
        # When a real process is running, _frames is updated from stdout.
        if not self._proc:
            self._frames += random.randint(2, 7)
            self.frame_counter.setText(f"{self._frames:,}")

        # Update queue bars and compact activity labels
        self.reader_bar.setValue(self._reader_q)
        self.writer_bar.setValue(self._writer_q)
        self.reader_pct.setText(f"{self._reader_q} %")
        self.writer_pct.setText(f"{self._writer_q} %")
        self._reader_act.setText(
            f"Queue {self._reader_q}% full — reading frames from source"
            if self._reader_q > 0 else "Waiting…"
        )
        self._writer_act.setText(
            f"Queue {self._writer_q}% full — encoding to disk"
            if self._writer_q > 0 else "Waiting…"
        )

        # Detector: active ~70% of ticks
        active = random.random() < 0.70
        status_text = "● Processing…" if active else "● Idle"
        obj_name    = "detector_active" if active else "detector_idle"
        detail_text = (
            f"{random.randint(200, 50_000):,} px above threshold"
            if active else "No motion — frame discarded"
        )
        self.detector_status.setText(status_text)
        self.detector_status.setObjectName(obj_name)
        self.detector_detail.setText(detail_text)
        self.detector_status.style().unpolish(self.detector_status)
        self.detector_status.style().polish(self.detector_status)

        # Mock log ~30% of ticks (only when no real process)
        if not self._proc and random.random() < 0.30:
            self._emit_mock_log()

        # Queue spike warnings
        if self._reader_q > 88:
            self.log_msg(f"ReaderThread | Queue at {self._reader_q}% — reader sleeping", "WARN")
        elif self._writer_q > 88:
            self.log_msg(f"WriterThread | Queue at {self._writer_q}% capacity", "WARN")

    def _emit_mock_log(self):
        """Pick a random mock log template and emit a filled-in version."""
        level, tmpl = random.choice(_MOCK_LOGS)
        msg = tmpl.format(
            b=random.randint(10, 80),
            p=random.randint(500, 50_000),
            k=random.randint(32, 200),
            d=random.choice([4, 8, 16]),
            f=self._frames,
            t=round(random.uniform(0, 1), 1),
            s=random.randint(50, 1_000),
            q=random.randint(0, 100),
        )
        self.log_msg(msg, level)

    # ── UI state helpers ──────────────────────────────────────────────────

    def _set_running_ui(self, running: bool):
        if running:
            self.status_lbl.setText("● Active")
            self.status_lbl.setObjectName("status_active")
            self.toggle_btn.setText("■   Stop")
            self.toggle_btn.setObjectName("stop_btn")
        elif self._completed:
            self.status_lbl.setText("● Complete")
            self.status_lbl.setObjectName("status_active")
            self.toggle_btn.setText("↺   New Run")
            self.toggle_btn.setObjectName("resume_btn")
        else:
            self.status_lbl.setText("● Stopped")
            self.status_lbl.setObjectName("status_stopped")
            self.toggle_btn.setText("▶   Resume")
            self.toggle_btn.setObjectName("resume_btn")

        for w in (self.status_lbl, self.toggle_btn):
            w.style().unpolish(w)
            w.style().polish(w)

    def _toggle_pipeline(self):
        if self._running:
            self.stop_pipeline()
        elif self._completed:
            # "New Run" — go back to config tab so user sets up a new job
            self._completed = False
            self._overall_bar.setVisible(False)
            self._overall_bar.setRange(0, 0)
            if self._on_new_run_cb:
                self._on_new_run_cb()
        else:
            # Resume simulation (does not restart the QProcess)
            self._running = True
            self._timer.start()
            self._overall_bar.setRange(0, 0)
            self._overall_bar.setVisible(True)
            self._set_running_ui(True)
            self._friendly_status.setText("Pipeline resumed…")
            self.log_msg("Pipeline resumed.", "INFO")

    # ── QProcess stdout relay ─────────────────────────────────────────────

    def _on_proc_output(self):
        import re
        if not self._proc:
            return
        raw = self._proc.readAll().data().decode("utf-8", errors="replace")
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            self._stdout_buffer.append(line)
            level = ("ERROR" if "ERROR" in line.upper() else
                     "WARN"  if "WARNING" in line.upper() else "INFO")
            self.log_msg(line, level)
            self._update_friendly_status(line)

            # Parse "Written N frames so far" — logged every 50 frames by Writer
            m = re.search(r"Written\s+(\d+)\s+frames", line, re.IGNORECASE)
            if m:
                self._frames = int(m.group(1))
                self.frame_counter.setText(f"{self._frames:,}")

            # Detect video switches: lite.py logs "Running main() with Config: {...}"
            # at the start of each video; parse the video_source field from it.
            if "Running main() with Config:" in line:
                vm = re.search(r"'video_source':\s*'([^']+)'", line)
                if vm:
                    current_path = vm.group(1)
                    self._current_file_lbl.setText(Path(current_path).name)
                    self._current_file_lbl.setVisible(True)
                    # Update x/N counter if processing multiple videos
                    if self._total_videos > 1:
                        try:
                            idx = self._video_files.index(current_path)
                            self._current_video_idx = idx
                        except ValueError:
                            self._current_video_idx += 1
                        self._video_progress_lbl.setText(
                            f"Video {self._current_video_idx + 1} of {self._total_videos}"
                        )
                        self._video_progress_lbl.setVisible(True)

    def _on_proc_finished(self, code: int, _):
        import time
        elapsed = time.monotonic() - self._start_time
        self._running = False
        self._timer.stop()

        if code == 0:
            self._completed = True
            # Zero queues and mark progress bar 100%
            self._reader_q = 0
            self._writer_q = 0
            self.reader_bar.setValue(0)
            self.writer_bar.setValue(0)
            self.reader_pct.setText("0 %")
            self.writer_pct.setText("0 %")
            self._overall_bar.setRange(0, 100)
            self._overall_bar.setValue(100)
            # Reset pipeline rows to idle
            self.detector_status.setText("● Done")
            self.detector_status.setObjectName("detector_active")
            self.detector_status.style().unpolish(self.detector_status)
            self.detector_status.style().polish(self.detector_status)
            self.detector_detail.setText("Processing complete.")
            self._reader_act.setText("Done")
            self._writer_act.setText("Done")

            self.log_msg(f"Video finished in {elapsed:.1f}s.", "INFO")

            # If more videos are queued, hand off to MainWindow without
            # showing the completion banner or changing the button state.
            if self._on_next_video_cb:
                cb = self._on_next_video_cb
                self._on_next_video_cb = None
                self._overall_bar.setRange(0, 0)   # back to indeterminate
                cb()
                return

            self._set_running_ui(False)   # shows "New Run" button
            self._friendly_status.setText(
                f"Done! {self._frames:,} frames encoded in {elapsed:.0f} s."
            )
            self._show_result(True, self._build_success_body(elapsed))
        else:
            self._completed = False
            self._overall_bar.setVisible(False)
            self._set_running_ui(False)
            # Find the last ERROR line for context
            last_err = next(
                (l for l in reversed(self._stdout_buffer) if "ERROR" in l.upper()),
                f"Process exited with code {code}."
            )
            self._friendly_status.setText("Processing failed — see the technical log for details.")
            self.log_msg(f"Process exited with code {code}.", "ERROR")
            self._show_result(False, last_err)

    def _build_success_body(self, elapsed: float) -> str:
        """Build the completion message, including compression ratio if available."""
        parts: list[str] = []
        parts.append(f"{self._frames:,} frames encoded  ·  {elapsed:.0f} s processing time")

        try:
            in_path = Path(self._video_source)
            if in_path.is_file():
                in_size = in_path.stat().st_size
                stem = in_path.stem
                out_dir = Path(self._output_dir) / "EcoMotionZip" / stem
                # Find any video file the writer produced
                out_files = [
                    f for ext in ("*.avi", "*.mp4", "*.mkv")
                    for f in out_dir.glob(ext)
                ]
                if out_files:
                    out_size = sum(f.stat().st_size for f in out_files)
                    ratio = in_size / out_size if out_size > 0 else 0
                    saved_mb = (in_size - out_size) / 1_048_576
                    parts.append(
                        f"Compression ratio: {ratio:.1f}×  "
                        f"({in_size/1_048_576:.1f} MB → {out_size/1_048_576:.1f} MB, "
                        f"saved {saved_mb:.1f} MB)"
                    )
        except Exception:
            pass   # silently skip if paths unavailable

        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Window
# ═══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """Application shell — header, theme toggle, stacked content."""

    def __init__(self, dark_mode: bool = True):
        super().__init__()
        self.setWindowTitle("EcoMotionZip")
        self.setMinimumWidth(780)
        _thumb = _HERE.parent / "docs" / "assets" / "EcoMotionZip_thumbnail.png"
        self.setWindowIcon(QIcon(str(_thumb)))
        self._dark_mode = dark_mode
        self._pending_cfgs: list[dict] = []

        self._build_ui()
        # Size to content after layout is built; cap at screen dimensions
        self.adjustSize()

    # ── Construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())
        root.addWidget(self._build_stack(), stretch=1)

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("header")
        header.setFixedHeight(64)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 0, 20, 0)
        layout.setSpacing(0)

        # Logo (left)
        _logo_path = _HERE.parent / "docs" / "assets" / "EcoMotionZip_logo_.png"
        logo_lbl = QLabel()
        logo_lbl.setFixedHeight(44)
        px = QPixmap(str(_logo_path))
        if not px.isNull():
            logo_lbl.setPixmap(
                px.scaledToHeight(44, Qt.TransformationMode.SmoothTransformation)
            )
        else:
            # Fallback to text if image not found
            logo_lbl.setText("EcoMotionZip")
            logo_lbl.setObjectName("title_lbl")
        layout.addWidget(logo_lbl)

        layout.addStretch()

        # Tab buttons (right of header) — exclusive checkable group
        self._tab_group = QButtonGroup(self)
        self._tab_group.setExclusive(True)

        self.btn_config = QPushButton("  Configuration  ")
        self.btn_config.setObjectName("tab_btn")
        self.btn_config.setCheckable(True)
        self.btn_config.setChecked(True)
        self.btn_config.clicked.connect(lambda: self._switch_tab(0))

        self.btn_monitor = QPushButton("  Monitor  ")
        self.btn_monitor.setObjectName("tab_btn")
        self.btn_monitor.setCheckable(True)
        self.btn_monitor.clicked.connect(lambda: self._switch_tab(1))

        self._tab_group.addButton(self.btn_config,  0)
        self._tab_group.addButton(self.btn_monitor, 1)

        layout.addWidget(self.btn_config)
        layout.addWidget(self.btn_monitor)
        layout.addSpacing(16)

        # Theme toggle button
        self.theme_btn = QPushButton("☀  Light" if self._dark_mode else "🌙  Dark")
        self.theme_btn.setObjectName("theme_btn")
        self.theme_btn.setFixedWidth(84)
        self.theme_btn.clicked.connect(self._toggle_theme)
        layout.addWidget(self.theme_btn)

        return header

    def _build_stack(self) -> QStackedWidget:
        self.stack = QStackedWidget()
        self.config_tab  = ConfigTab(on_launch=self._on_launch)
        self.monitor_tab = PipelineMonitorTab()
        self.monitor_tab._on_new_run_cb = lambda: self._switch_tab(0)
        self.stack.addWidget(self.config_tab)
        self.stack.addWidget(self.monitor_tab)
        return self.stack

    # ── Tab switching ─────────────────────────────────────────────────────

    def _switch_tab(self, idx: int):
        self.stack.setCurrentIndex(idx)
        self._tab_group.button(idx).setChecked(True)
        # Make hidden tab widgets report zero size so adjustSize() uses only the
        # current tab's sizeHint (QStackedWidget otherwise returns the max of all).
        for i in range(self.stack.count()):
            w = self.stack.widget(i)
            if i == idx:
                w.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
            else:
                w.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        QTimer.singleShot(0, self.adjustSize)

    # ── Launch ────────────────────────────────────────────────────────────

    def _on_launch(self, cfg: dict):
        """Split multi-file sources into a queue then start the first video."""
        video_source = cfg.get("video_source", "")
        if os.pathsep in video_source:
            files = video_source.split(os.pathsep)
            # Store remaining files as per-video cfgs; override set_source_paths
            # with the full list so x/N tracking works correctly.
            self._pending_cfgs = [dict(cfg, video_source=f) for f in files[1:]]
            cfg = dict(cfg, video_source=files[0])
            # Pre-populate monitor with the full file list for x/N display
            self.monitor_tab.set_source_paths(video_source, cfg.get("output_directory", ""))
        else:
            self._pending_cfgs = []
        self._switch_tab(1)
        self._start_one(cfg, first=True)

    def _start_one(self, cfg: dict, first: bool = False):
        """Launch EcoMotionZip_lite.py for a single video file."""
        # Write embed_timestamps to config.json so EcoMotionZip_lite.py picks it up
        _config_path = _HERE.parent / "config.json"
        try:
            existing: dict = {}
            if _config_path.exists():
                with open(_config_path, "r", encoding="utf-8") as fh:
                    existing = json.load(fh)
            existing["embed_timestamps"] = bool(cfg.get("embed_timestamps", False))
            with open(_config_path, "w", encoding="utf-8") as fh:
                json.dump(existing, fh, indent=4)
        except Exception:
            pass   # non-fatal — lite.py will fall back to its own default

        # Build CLI args matching EcoMotionZip_lite.py's argparse
        args: list[str] = []
        if cfg.get("video_source"):
            args += ["--video_source",    cfg["video_source"]]
        if cfg.get("output_directory"):
            args += ["--output_directory", cfg["output_directory"]]
        args += ["--movement_threshold",      str(cfg.get("movement_threshold", 30))]
        args += ["--video_codec",             cfg.get("video_codec", "DIVX")]
        args += ["--background_transparency", str(cfg.get("background_transparency", 1.0))]
        args += ["--downscale_factor",        str(cfg.get("downscale_factor", 16))]
        args += ["--dilate_kernel_size",      str(cfg.get("dilate_kernel_size", 128))]
        args += ["--post_motion_record_frames", str(cfg.get("post_motion_record_frames", 0))]
        args += ["--num_opencv_threads",      str(cfg.get("num_opencv_threads", 10))]
        if cfg.get("save_frames"):
            args += ["--save_frames",    "True",
                     "--frames_to_save", str(cfg.get("frames_to_save", 2000))]

        # For single-file (or no pending queue), let set_source_paths run normally
        if first and not self._pending_cfgs:
            self.monitor_tab.set_source_paths(
                cfg.get("video_source", ""),
                cfg.get("output_directory", ""),
            )

        more_pending = bool(self._pending_cfgs)
        self.monitor_tab._on_next_video_cb = self._on_video_done if more_pending else None

        proc: QProcess | None = None
        if _LITE_SCRIPT.exists():
            proc = QProcess(self)
            proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            proc.start(sys.executable, [str(_LITE_SCRIPT)] + args)

        self.monitor_tab.start_pipeline(proc)

    def _on_video_done(self):
        """Called after each successful video when more files remain in the queue."""
        if not self._pending_cfgs:
            return
        cfg = self._pending_cfgs.pop(0)
        more_pending = bool(self._pending_cfgs)
        self.monitor_tab._on_next_video_cb = self._on_video_done if more_pending else None
        self.monitor_tab.log_msg(
            f"─── Starting next video ({cfg['video_source'].split('/')[-1]}) ───", "INFO"
        )
        self._start_one(cfg)

    # ── Theme toggle ──────────────────────────────────────────────────────

    def _toggle_theme(self):
        self._dark_mode = not self._dark_mode
        qss    = DARK_QSS  if self._dark_mode else LIGHT_QSS
        colors = DARK       if self._dark_mode else LIGHT
        QApplication.instance().setStyleSheet(qss)
        self.monitor_tab.set_log_colors(colors)
        self.theme_btn.setText("☀  Light" if self._dark_mode else "🌙  Dark")

    # ── Close guard ───────────────────────────────────────────────────────

    def closeEvent(self, event):
        proc = self.monitor_tab._proc
        if proc and proc.state() != QProcess.ProcessState.NotRunning:
            proc.kill()
        event.accept()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("EcoMotionZip")
    app.setOrganizationName("EcoMotionZip")

    # Set application icon — required at QApplication level for the macOS Dock
    _thumb = _HERE.parent / "docs" / "assets" / "EcoMotionZip_thumbnail.png"
    app.setWindowIcon(QIcon(str(_thumb)))

    # Detect system dark/light preference (Qt 6.5+; falls back to dark)
    dark_mode = True
    try:
        dark_mode = (app.styleHints().colorScheme() == Qt.ColorScheme.Dark)
    except AttributeError:
        pass

    app.setStyleSheet(DARK_QSS if dark_mode else LIGHT_QSS)

    # Use best available system font
    f = app.font()
    f.setFamily({"win32": "Segoe UI", "darwin": "SF Pro Text"}.get(sys.platform, "Ubuntu"))
    app.setFont(f)

    window = MainWindow(dark_mode=dark_mode)
    window.show()
    window.activateWindow()
    window.raise_()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
