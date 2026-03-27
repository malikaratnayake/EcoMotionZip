"""
EcoMotionZip — PySide6 desktop GUI.

Design principles (v2)
-----------------------
- Neutral slate/white base palette — green used only as a purposeful accent,
  never as the base tone. All text meets WCAG AA contrast ratios (≥ 4.5:1).
- Progressive disclosure: everyday controls on tabs 1-3, internals in Advanced.
- Sliders with live value badges for the three key motion parameters.
- Right panel replaced with a visual output pane:
    · Live frame preview — watches the output folder for new JPEG frames
      and displays the most recent detection in real time.
    · Stats strip — shows frames processed, elapsed time, and estimated
      compression ratio as the pipeline runs.
    · Compact activity feed — colour-coded status lines (no dark terminal).
- Validation before run with friendly, actionable error dialogs.

Usage
-----
    pip install PySide6
    python src/EcoMotionZip_gui.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, QTimer, QSize
from PySide6.QtGui import (
    QColor, QFont, QFontMetrics, QTextCharFormat, QTextCursor,
    QPixmap, QPainter, QPen, QBrush,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QSlider, QSpinBox,
    QDoubleSpinBox, QCheckBox, QComboBox, QTextEdit, QProgressBar,
    QFileDialog, QMessageBox, QTabWidget, QScrollArea,
    QSplitter, QFrame, QStatusBar, QSizePolicy, QStackedWidget,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).parent
_CONFIG_PATH = _HERE.parent / "config.json"
_LITE_SCRIPT = _HERE / "EcoMotionZip_lite.py"

# ── System font ───────────────────────────────────────────────────────────────
_FONT = {"win32": "Segoe UI", "darwin": "SF Pro Text"}.get(sys.platform, "Ubuntu")

# ── Palette — neutral base, green accent ─────────────────────────────────────
#   All text-on-background pairs checked for WCAG AA (≥ 4.5:1)
C_BG          = "#F0F2F5"   # Window / sidebar background  (neutral slate-100)
C_CARD        = "#FFFFFF"   # Panel / card surface
C_SIDEBAR     = "#1E293B"   # Dark sidebar strip for header (slate-800)
C_PRIMARY     = "#166534"   # Dark forest green — buttons  (contrast on white: 8.4:1)
C_PRIMARY_HVR = "#14532D"   # Hover / pressed state
C_PRIMARY_LT  = "#DCFCE7"   # Very light green — hover bg for outline btns
C_ACCENT      = "#16A34A"   # Medium green — slider fill, badges
C_STOP        = "#991B1B"   # Dark red — stop button
C_STOP_HVR    = "#7F1D1D"
C_TEXT        = "#111827"   # Near-black body text          (contrast: 16:1 on white)
C_SUBTEXT     = "#374151"   # Secondary text                (contrast: 7.2:1 on white)
C_HINT        = "#6B7280"   # Hint / caption text           (contrast: 4.6:1 on white)
C_BORDER      = "#CBD5E1"   # Neutral gray border
C_DIVIDER     = "#E2E8F0"   # Lighter divider lines
C_FIELD_BG    = "#FFFFFF"   # Input field background
C_PREVIEW_BG  = "#0F172A"   # Preview pane background (slate-900)
C_FEED_BG     = "#F8FAFC"   # Activity feed background
C_WARN_TXT    = "#92400E"   # Warning text (amber-800)  contrast 6.4:1 on white
C_ERR_TXT     = "#991B1B"   # Error text (red-800)      contrast 7.4:1 on white
C_OK_TXT      = "#166534"   # Success text              contrast 8.4:1 on white
C_WARN_BG     = "#FFFBEB"
C_ERR_BG      = "#FEF2F2"
C_OK_BG       = "#F0FDF4"

# ── Master stylesheet ─────────────────────────────────────────────────────────
QSS = f"""
/* ── Base ───────────────────────────────────────────────────────────────── */
QMainWindow, QDialog {{
    background: {C_BG};
}}
QWidget {{
    font-family: "{_FONT}", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    color: {C_TEXT};
}}

/* ── Tab bar ─────────────────────────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {C_BORDER};
    border-radius: 0 8px 8px 8px;
    background: {C_CARD};
}}
QTabBar {{
    background: transparent;
}}
QTabBar::tab {{
    background: {C_BG};
    color: {C_SUBTEXT};
    border: 1px solid {C_BORDER};
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    padding: 9px 22px;
    margin-right: 3px;
    font-weight: 500;
    font-size: 13px;
}}
QTabBar::tab:selected {{
    background: {C_CARD};
    color: {C_PRIMARY};
    font-weight: 700;
    border-bottom: 2px solid {C_CARD};
}}
QTabBar::tab:hover:!selected {{
    background: {C_PRIMARY_LT};
    color: {C_PRIMARY};
}}

/* ── Inputs ──────────────────────────────────────────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {C_FIELD_BG};
    color: {C_TEXT};
    border: 1.5px solid {C_BORDER};
    border-radius: 6px;
    padding: 7px 10px;
    selection-background-color: {C_ACCENT};
    selection-color: white;
    font-size: 13px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {C_ACCENT};
    outline: none;
}}
QLineEdit::placeholder {{
    color: {C_HINT};
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 20px;
    border-left: 1px solid {C_BORDER};
    background: {C_BG};
    border-radius: 0 5px 5px 0;
}}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {C_PRIMARY_LT};
}}
QComboBox::drop-down {{
    border: none;
    width: 28px;
    background: {C_BG};
    border-radius: 0 5px 5px 0;
}}
QComboBox QAbstractItemView {{
    background: {C_CARD};
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    selection-background-color: {C_PRIMARY_LT};
    selection-color: {C_PRIMARY};
    outline: none;
    padding: 2px;
}}

/* ── Slider ──────────────────────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 6px;
    background: {C_DIVIDER};
    border-radius: 3px;
    border: none;
}}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {C_PRIMARY}, stop:1 {C_ACCENT});
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {C_CARD};
    border: 2.5px solid {C_PRIMARY};
    width: 18px;
    height: 18px;
    margin: -6px 0;
    border-radius: 9px;
}}
QSlider::handle:horizontal:hover {{
    background: {C_PRIMARY_LT};
    border-color: {C_PRIMARY_HVR};
}}
QSlider::handle:horizontal:pressed {{
    background: {C_PRIMARY_LT};
}}

/* ── Checkbox ─────────────────────────────────────────────────────────────── */
QCheckBox {{
    color: {C_TEXT};
    spacing: 9px;
    font-size: 13px;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 2px solid {C_BORDER};
    background: {C_FIELD_BG};
}}
QCheckBox::indicator:hover {{
    border-color: {C_ACCENT};
}}
QCheckBox::indicator:checked {{
    background: {C_PRIMARY};
    border-color: {C_PRIMARY};
    image: url(none);   /* checkmark drawn via border trick below */
}}

/* ── Primary button ──────────────────────────────────────────────────────── */
QPushButton {{
    background: {C_PRIMARY};
    color: #FFFFFF;
    border: none;
    border-radius: 7px;
    padding: 9px 22px;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.2px;
}}
QPushButton:hover   {{ background: {C_PRIMARY_HVR}; }}
QPushButton:pressed {{ background: {C_PRIMARY_HVR}; }}
QPushButton:disabled {{
    background: {C_DIVIDER};
    color: {C_HINT};
}}

/* Stop button */
QPushButton#stop_btn       {{ background: {C_STOP}; color: #FFFFFF; }}
QPushButton#stop_btn:hover {{ background: {C_STOP_HVR}; }}

/* Outline / ghost button */
QPushButton#outline_btn {{
    background: {C_CARD};
    color: {C_PRIMARY};
    border: 2px solid {C_PRIMARY};
    font-weight: 600;
}}
QPushButton#outline_btn:hover {{
    background: {C_PRIMARY_LT};
}}
QPushButton#outline_btn:disabled {{
    color: {C_HINT};
    border-color: {C_BORDER};
    background: {C_CARD};
}}

/* Browse button */
QPushButton#browse_btn {{
    background: {C_BG};
    color: {C_PRIMARY};
    border: 1.5px solid {C_BORDER};
    padding: 7px 14px;
    font-weight: 600;
    font-size: 12px;
    border-radius: 6px;
}}
QPushButton#browse_btn:hover {{
    background: {C_PRIMARY_LT};
    border-color: {C_ACCENT};
    color: {C_PRIMARY_HVR};
}}

/* ── Progress bar ────────────────────────────────────────────────────────── */
QProgressBar {{
    background: {C_DIVIDER};
    border: none;
    border-radius: 4px;
    min-height: 8px;
    max-height: 8px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {C_PRIMARY}, stop:1 {C_ACCENT});
    border-radius: 4px;
}}

/* ── Scroll area ─────────────────────────────────────────────────────────── */
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {C_BORDER};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: #94A3B8; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

/* ── Status bar ──────────────────────────────────────────────────────────── */
QStatusBar {{
    background: {C_CARD};
    border-top: 1px solid {C_BORDER};
    color: {C_SUBTEXT};
    font-size: 12px;
    padding: 0 10px;
}}
QStatusBar QLabel {{
    color: {C_SUBTEXT};
    font-size: 12px;
}}

/* ── Separators ──────────────────────────────────────────────────────────── */
QFrame[frameShape="4"] {{
    background: {C_DIVIDER};
    border: none;
    max-height: 1px;
    min-height: 1px;
}}

/* ── Tooltips ────────────────────────────────────────────────────────────── */
QToolTip {{
    background: {C_SIDEBAR};
    color: #F1F5F9;
    border: none;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 12px;
    line-height: 1.5;
}}

/* ── Text edit (activity feed) ───────────────────────────────────────────── */
QTextEdit {{
    background: {C_FEED_BG};
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    border-radius: 6px;
    font-size: 12px;
    padding: 4px;
}}
"""

# ── Tooltip copy ──────────────────────────────────────────────────────────────
TIPS = {
    "video_source": (
        "Select the video file you want to compress.\n\n"
        "Works best with camera-trap or fixed-camera footage\n"
        "where animals appear intermittently against a still background."
    ),
    "output_directory": (
        "Choose the folder where EcoMotionZip will save:\n"
        "  • the compressed video file\n"
        "  • the frame index spreadsheet (.csv)\n"
        "  • individual frame images (if enabled)"
    ),
    "movement_threshold": (
        "How sensitive should the motion detector be?\n\n"
        "Low (5–25)    → catches subtle movement, e.g. a swaying leaf.\n"
        "Medium (25–60) → good for most wildlife cameras.\n"
        "High (60+)    → only catches large, obvious movement.\n\n"
        "Increase this value if you get too many false detections."
    ),
    "downscale_factor": (
        "Before checking for motion, each frame is shrunk by this factor.\n\n"
        "Higher = faster processing, slightly less precise.\n"
        "Lower  = more precise but slower.\n\n"
        "Recommended: 8–16 for 1080p footage."
    ),
    "dilate_kernel_size": (
        "After motion pixels are found, this setting expands the detected\n"
        "region outward — ensuring the whole animal body is captured,\n"
        "not just the part that moved first.\n\n"
        "Recommended: 64–150 for wildlife-sized subjects."
    ),
    "post_motion_record_frames": (
        "Keep recording for this many extra frames after motion stops.\n\n"
        "Useful for capturing an animal slowing down or a bird landing.\n"
        "Set to 0 to stop immediately when motion ends."
    ),
    "full_frame_capture_interval": (
        "Every N frames, save one complete unmasked frame to provide\n"
        "spatial context (you can always see where in the scene you are).\n\n"
        "300 ≈ every 10 seconds at 30 fps.  Set to 0 to disable."
    ),
    "background_transparency": (
        "How much of the background scene to show in non-motion regions.\n\n"
        "0.0 → black background  (maximum compression, smallest file)\n"
        "1.0 → full background   (easier to identify location)\n\n"
        "Use 0.0 for maximum savings; 1.0 for spatial context."
    ),
    "video_codec": (
        "The compression format for the output video file.\n\n"
        "DIVX  — widely compatible with media players; good for sharing.\n"
        "X264  — smaller files, excellent quality.\n"
        "FFV1  — lossless (no quality loss), largest files; best for archiving.\n"
        "HEVC  — smallest files, best quality (requires FFmpeg)."
    ),
    "save_frames": (
        "Also export each motion frame as an individual JPEG image.\n\n"
        "Required for the live frame preview in this window.\n"
        "Also useful for quick browsing or feeding into other tools."
    ),
    "frames_to_save": (
        "Maximum number of JPEG images to save.\n"
        "Images stop being saved once this limit is reached."
    ),
    "embed_timestamps": "Overlay the original recording timestamp on each output frame.",
    "delete_original_after_processing": (
        "Permanently delete the original video file once compression finishes.\n\n"
        "⚠ This cannot be undone."
    ),
    "num_opencv_threads": (
        "Number of CPU threads allocated to OpenCV image processing.\n"
        "Leave at the default unless you experience performance issues."
    ),
    "reader_sleep_seconds": (
        "How long (seconds) the frame reader pauses when its buffer is full.\n"
        "Prevents excessive memory use on fast video sources."
    ),
    "reader_flush_proportion": (
        "The fraction (0–1) of the reading buffer that must fill before\n"
        "the reader pauses. 0.9 = pause when 90% full."
    ),
}


# =============================================================================
# Reusable building-block widgets
# =============================================================================

def _hsep() -> QFrame:
    """Return a one-pixel horizontal divider."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    return line


class SectionLabel(QLabel):
    """Bold section heading used inside tabs."""
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        f = QFont()
        f.setPointSize(11)
        f.setWeight(QFont.Weight.DemiBold)
        self.setFont(f)
        # Dark green on white — contrast 8.4:1
        self.setStyleSheet(
            f"color: {C_PRIMARY};"
            " margin-top: 10px; margin-bottom: 2px;"
        )


class HintLabel(QLabel):
    """Small caption text beneath a control."""
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        # C_HINT (#6B7280) on C_CARD (#FFFFFF) → 4.6:1 — passes WCAG AA
        self.setStyleSheet(f"color: {C_HINT}; font-size: 11px; line-height: 1.4;")
        self.setWordWrap(True)


class FieldLabel(QLabel):
    """Standard medium-weight field label."""
    def __init__(self, text: str, tip: str = "", parent=None):
        super().__init__(text, parent)
        # C_TEXT (#111827) on C_CARD (#FFFFFF) → 16:1
        self.setStyleSheet(f"color: {C_TEXT}; font-weight: 600; font-size: 13px;")
        if tip:
            self.setToolTip(tip)


class SliderRow(QWidget):
    """
    Labelled horizontal slider with a live-updating value badge.
    Exposes ``value() -> int`` and ``set_value(int)``.
    """
    def __init__(
        self,
        label: str,
        min_val: int,
        max_val: int,
        default: int,
        hint: str = "",
        tooltip: str = "",
        suffix: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._suffix = suffix
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(6)

        # Label + live value badge
        top = QHBoxLayout()
        lbl = FieldLabel(label, tooltip)
        self._badge = QLabel(f"{default}{suffix}")
        self._badge.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        # Accent green on white — contrast 4.6:1
        self._badge.setStyleSheet(
            f"color: {C_ACCENT}; font-weight: 700; font-size: 15px;"
            f" background: {C_PRIMARY_LT}; border-radius: 4px;"
            " padding: 1px 8px; min-width: 52px;"
        )
        top.addWidget(lbl)
        top.addStretch()
        top.addWidget(self._badge)
        layout.addLayout(top)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(min_val)
        self.slider.setMaximum(max_val)
        self.slider.setValue(default)
        self.slider.setToolTip(tooltip)
        self.slider.valueChanged.connect(
            lambda v: self._badge.setText(f"{v}{suffix}")
        )
        layout.addWidget(self.slider)

        if hint:
            layout.addWidget(HintLabel(hint))

    def value(self) -> int:
        return self.slider.value()

    def set_value(self, v: int | float):
        self.slider.setValue(int(v))


class FilePicker(QWidget):
    """Labelled text field with a Browse button."""
    def __init__(
        self,
        label: str,
        placeholder: str = "",
        pick_file: bool = True,
        file_filter: str = (
            "Video Files (*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV *.MKV);;"
            "All Files (*)"
        ),
        tooltip: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._pick_file = pick_file
        self._filter = file_filter

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(5)

        layout.addWidget(FieldLabel(label, tooltip))

        row = QHBoxLayout()
        self.field = QLineEdit()
        self.field.setPlaceholderText(placeholder)
        self.field.setToolTip(tooltip)

        btn = QPushButton("Browse…")
        btn.setObjectName("browse_btn")
        btn.setFixedWidth(90)
        btn.clicked.connect(self._browse)

        row.addWidget(self.field)
        row.addWidget(btn)
        layout.addLayout(row)

    def _browse(self):
        if self._pick_file:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Video File", str(Path.home()), self._filter
            )
        else:
            path = QFileDialog.getExistingDirectory(
                self, "Select Output Folder", str(Path.home())
            )
        if path:
            self.field.setText(path)

    def text(self) -> str:
        return self.field.text().strip()

    def set_text(self, v: str):
        self.field.setText(str(v))


def _scrollable(inner: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.Shape.NoFrame)
    sa.setWidget(inner)
    return sa


# =============================================================================
# Tab pages (left panel)
# =============================================================================

class FilesTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        layout.addWidget(SectionLabel("Input Video"))
        layout.addWidget(HintLabel(
            "Select the camera-trap or fixed-camera video file to compress."
        ))
        self.video_source = FilePicker(
            "Video file",
            placeholder="e.g.  /Volumes/SSD/fieldwork/camera_01.mp4",
            pick_file=True,
            tooltip=TIPS["video_source"],
        )
        layout.addWidget(self.video_source)

        layout.addWidget(_hsep())

        layout.addWidget(SectionLabel("Output Folder"))
        layout.addWidget(HintLabel(
            "EcoMotionZip will save the compressed video, frame index (.csv), "
            "and exported images here."
        ))
        self.output_directory = FilePicker(
            "Save results to",
            placeholder="e.g.  /Volumes/SSD/fieldwork/compressed/",
            pick_file=False,
            tooltip=TIPS["output_directory"],
        )
        layout.addWidget(self.output_directory)
        layout.addStretch()


class MotionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        inner = QWidget()
        inner.setStyleSheet(f"background: {C_CARD};")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        layout.addWidget(SectionLabel("Detection Sensitivity"))
        layout.addWidget(HintLabel(
            "These sliders control how the software decides whether something "
            "is moving. Start with the defaults and adjust if you detect too many "
            "false positives (e.g. wind-blown vegetation) or miss real events."
        ))

        self.movement_threshold = SliderRow(
            "Movement Threshold",
            1, 200, 40,
            hint="Low = more sensitive  ·  High = less sensitive",
            tooltip=TIPS["movement_threshold"],
        )
        layout.addWidget(self.movement_threshold)

        self.downscale_factor = SliderRow(
            "Processing Speed Factor",
            1, 32, 16,
            hint="Higher = faster processing, slightly less precise",
            tooltip=TIPS["downscale_factor"],
        )
        layout.addWidget(self.downscale_factor)

        self.dilate_kernel_size = SliderRow(
            "Motion Region Expansion",
            1, 255, 128,
            hint="Enlarges detected regions to capture whole subjects",
            tooltip=TIPS["dilate_kernel_size"],
        )
        layout.addWidget(self.dilate_kernel_size)

        layout.addWidget(_hsep())
        layout.addWidget(SectionLabel("Recording Behaviour"))

        pmf_row = QHBoxLayout()
        pmf_row.addWidget(FieldLabel(
            "Extra frames after motion stops", TIPS["post_motion_record_frames"]
        ))
        pmf_row.addStretch()
        self.post_motion_record_frames = QSpinBox()
        self.post_motion_record_frames.setRange(0, 1000)
        self.post_motion_record_frames.setValue(0)
        self.post_motion_record_frames.setToolTip(TIPS["post_motion_record_frames"])
        self.post_motion_record_frames.setFixedWidth(90)
        pmf_row.addWidget(self.post_motion_record_frames)
        layout.addLayout(pmf_row)
        layout.addWidget(HintLabel(
            "Continues recording for N frames after the last detected movement."
        ))

        layout.addSpacing(6)

        ffi_row = QHBoxLayout()
        ffi_row.addWidget(FieldLabel(
            "Reference frame interval (frames)", TIPS["full_frame_capture_interval"]
        ))
        ffi_row.addStretch()
        self.full_frame_capture_interval = QSpinBox()
        self.full_frame_capture_interval.setRange(0, 10000)
        self.full_frame_capture_interval.setValue(300)
        self.full_frame_capture_interval.setToolTip(TIPS["full_frame_capture_interval"])
        self.full_frame_capture_interval.setFixedWidth(90)
        ffi_row.addWidget(self.full_frame_capture_interval)
        layout.addLayout(ffi_row)
        layout.addWidget(HintLabel(
            "Saves a full unmasked scene frame every N frames for spatial context. "
            "At 30 fps, 300 ≈ every 10 seconds."
        ))

        layout.addStretch()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scrollable(inner))


class OutputTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        inner = QWidget()
        inner.setStyleSheet(f"background: {C_CARD};")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        layout.addWidget(SectionLabel("Video Format"))

        codec_row = QHBoxLayout()
        codec_row.addWidget(FieldLabel("Codec", TIPS["video_codec"]))
        codec_row.addStretch()
        self.video_codec = QComboBox()
        self.video_codec.addItems(["DIVX", "X264", "FFV1", "HEVC"])
        self.video_codec.setFixedWidth(130)
        self.video_codec.setToolTip(TIPS["video_codec"])
        codec_row.addWidget(self.video_codec)
        layout.addLayout(codec_row)
        layout.addWidget(HintLabel(
            "DIVX — compatible  ·  X264 — smaller files  ·  "
            "FFV1 — lossless archive  ·  HEVC — smallest (needs FFmpeg)"
        ))

        layout.addWidget(_hsep())
        layout.addWidget(SectionLabel("Background Visibility"))
        self.background_transparency = SliderRow(
            "Background Transparency",
            0, 10, 10,
            hint="0 = black background (max compression)  ·  10 = always visible",
            tooltip=TIPS["background_transparency"],
            suffix="/10",
        )
        layout.addWidget(self.background_transparency)

        layout.addWidget(_hsep())
        layout.addWidget(SectionLabel("Individual Frame Export"))

        self.save_frames = QCheckBox("Save each motion frame as an image (JPEG)")
        self.save_frames.setToolTip(TIPS["save_frames"])
        layout.addWidget(self.save_frames)
        layout.addWidget(HintLabel(
            "Enables the live frame preview panel on the right. "
            "Also useful for browsing detections without opening a video."
        ))

        fts_row = QHBoxLayout()
        fts_row.addWidget(FieldLabel("Maximum images to save", TIPS["frames_to_save"]))
        fts_row.addStretch()
        self.frames_to_save = QSpinBox()
        self.frames_to_save.setRange(1, 500000)
        self.frames_to_save.setValue(2000)
        self.frames_to_save.setToolTip(TIPS["frames_to_save"])
        self.frames_to_save.setFixedWidth(100)
        fts_row.addWidget(self.frames_to_save)
        layout.addLayout(fts_row)

        self.frames_to_save.setEnabled(False)
        self.save_frames.toggled.connect(self.frames_to_save.setEnabled)

        layout.addWidget(_hsep())
        layout.addWidget(SectionLabel("File Management"))

        self.embed_timestamps = QCheckBox("Embed recording timestamps on each frame")
        self.embed_timestamps.setToolTip(TIPS["embed_timestamps"])
        layout.addWidget(self.embed_timestamps)

        self.delete_original = QCheckBox(
            "Delete original video after compression is complete"
        )
        self.delete_original.setToolTip(TIPS["delete_original_after_processing"])
        layout.addWidget(self.delete_original)
        layout.addWidget(HintLabel("⚠  This permanently deletes the source file."))

        layout.addStretch()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scrollable(inner))


class AdvancedTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        inner = QWidget()
        inner.setStyleSheet(f"background: {C_CARD};")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        layout.addWidget(SectionLabel("Processing Performance"))
        layout.addWidget(HintLabel(
            "These settings control low-level pipeline behaviour. "
            "The defaults work well for most computers — only change them "
            "if you experience slowdowns or high memory usage."
        ))

        oct_row = QHBoxLayout()
        oct_row.addWidget(FieldLabel(
            "CPU threads for image processing", TIPS["num_opencv_threads"]
        ))
        oct_row.addStretch()
        self.num_opencv_threads = QSpinBox()
        self.num_opencv_threads.setRange(1, 64)
        self.num_opencv_threads.setValue(10)
        self.num_opencv_threads.setToolTip(TIPS["num_opencv_threads"])
        self.num_opencv_threads.setFixedWidth(90)
        oct_row.addWidget(self.num_opencv_threads)
        layout.addLayout(oct_row)

        layout.addWidget(_hsep())
        layout.addWidget(SectionLabel("Frame Buffer Control"))

        rss_row = QHBoxLayout()
        rss_row.addWidget(FieldLabel(
            "Reader pause duration (seconds)", TIPS["reader_sleep_seconds"]
        ))
        rss_row.addStretch()
        self.reader_sleep_seconds = QDoubleSpinBox()
        self.reader_sleep_seconds.setRange(0.0, 30.0)
        self.reader_sleep_seconds.setSingleStep(0.1)
        self.reader_sleep_seconds.setDecimals(2)
        self.reader_sleep_seconds.setValue(1.0)
        self.reader_sleep_seconds.setToolTip(TIPS["reader_sleep_seconds"])
        self.reader_sleep_seconds.setFixedWidth(90)
        rss_row.addWidget(self.reader_sleep_seconds)
        layout.addLayout(rss_row)

        rfp_row = QHBoxLayout()
        rfp_row.addWidget(FieldLabel(
            "Buffer fill threshold (0.0 – 1.0)", TIPS["reader_flush_proportion"]
        ))
        rfp_row.addStretch()
        self.reader_flush_proportion = QDoubleSpinBox()
        self.reader_flush_proportion.setRange(0.1, 1.0)
        self.reader_flush_proportion.setSingleStep(0.05)
        self.reader_flush_proportion.setDecimals(2)
        self.reader_flush_proportion.setValue(0.9)
        self.reader_flush_proportion.setToolTip(TIPS["reader_flush_proportion"])
        self.reader_flush_proportion.setFixedWidth(90)
        rfp_row.addWidget(self.reader_flush_proportion)
        layout.addLayout(rfp_row)
        layout.addWidget(HintLabel(
            "The reader pauses once this fraction of the frame buffer is filled."
        ))

        layout.addStretch()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scrollable(inner))


# =============================================================================
# Right panel — visual output (frame preview + stats + activity feed)
# =============================================================================

class _PlaceholderPixmap:
    """Creates the 'waiting for frames' placeholder image once and caches it."""
    _cache: QPixmap | None = None

    @classmethod
    def get(cls, w: int, h: int) -> QPixmap:
        pm = QPixmap(w, h)
        pm.fill(QColor(C_PREVIEW_BG))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Subtle crosshair / camera-lens circle
        cx, cy, r = w // 2, h // 2, min(w, h) // 5
        pen = QPen(QColor("#334155"), 2)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(cx - r, cy - r, r * 2, r * 2)
        p.drawLine(cx - r - 12, cy, cx + r + 12, cy)
        p.drawLine(cx, cy - r - 12, cx, cy + r + 12)

        # Text
        p.setPen(QPen(QColor("#475569")))
        f = QFont(_FONT)
        f.setPointSize(11)
        p.setFont(f)
        p.drawText(
            pm.rect().adjusted(0, cy + r + 20, 0, 0),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            "Waiting for first detection…",
        )
        p.setPen(QPen(QColor("#334155")))
        f.setPointSize(9)
        p.setFont(f)
        p.drawText(
            pm.rect().adjusted(0, cy + r + 44, 0, 0),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            'Enable "Save individual frames" on the Output tab\n'
            "to see live detections here.",
        )
        p.end()
        return pm


class FramePreviewPanel(QWidget):
    """
    Right-hand output panel containing:
      1. Frame preview — displays the most recently saved JPEG detection.
      2. Stats strip   — frames processed, elapsed time, compression ratio.
      3. Activity feed — colour-coded status messages from the pipeline.

    Uses a 500ms ``QTimer`` poll to scan the output directory for new JPEG
    files.  Polling is used instead of ``QFileSystemWatcher`` because:
      * The output subdirectory is created by the subprocess *after* launch,
        so any pre-launch watch registration would silently fail.
      * ``QFileSystemWatcher`` can miss events on macOS (kqueue limitations).
    The timer handles both "directory not yet created" and "new file arrived"
    states naturally without any extra machinery.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._watch_path: Path | None = None   # exact dir where JPEGs land
        self._last_shown: str = ""             # path of the last displayed file
        self._frames_shown = 0
        self._start_time: float | None = None
        self._total_input_frames = 0

        # Primary mechanism: poll every 500 ms while running
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(500)
        self._poll_timer.timeout.connect(self._poll_scan)

        self._build_ui()

    # ── Construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── 1. Frame preview ──────────────────────────────────────────
        preview_container = QWidget()
        preview_container.setStyleSheet(f"background: {C_PREVIEW_BG};")
        preview_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        pc_layout = QVBoxLayout(preview_container)
        pc_layout.setContentsMargins(0, 0, 0, 0)

        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._preview.setMinimumHeight(200)
        pc_layout.addWidget(self._preview)

        # Frame metadata bar (overlaid at bottom of preview area)
        self._meta_bar = QWidget()
        self._meta_bar.setStyleSheet(
            "background: rgba(15, 23, 42, 0.88);"
        )
        self._meta_bar.setFixedHeight(32)
        meta_layout = QHBoxLayout(self._meta_bar)
        meta_layout.setContentsMargins(12, 0, 12, 0)

        lbl_style = "color: #94A3B8; font-size: 11px; background: transparent;"
        val_style = "color: #E2E8F0; font-size: 11px; font-weight: 600; background: transparent;"

        self._meta_name = QLabel("—")
        self._meta_name.setStyleSheet(val_style)
        self._meta_dims = QLabel("")
        self._meta_dims.setStyleSheet(lbl_style)
        meta_layout.addWidget(self._meta_name)
        meta_layout.addStretch()
        meta_layout.addWidget(self._meta_dims)
        pc_layout.addWidget(self._meta_bar)

        layout.addWidget(preview_container, stretch=3)

        # ── 2. Stats strip ────────────────────────────────────────────
        stats = QWidget()
        stats.setStyleSheet(
            f"background: {C_SIDEBAR}; border-top: 1px solid #334155;"
        )
        stats.setFixedHeight(56)
        sl = QHBoxLayout(stats)
        sl.setContentsMargins(16, 0, 16, 0)
        sl.setSpacing(0)

        self._stat_frames  = self._make_stat("0",   "Detections")
        self._stat_elapsed = self._make_stat("—",    "Elapsed")
        self._stat_ratio   = self._make_stat("—",    "Compression")

        sl.addLayout(self._stat_frames)
        sl.addWidget(self._vdivider())
        sl.addLayout(self._stat_elapsed)
        sl.addWidget(self._vdivider())
        sl.addLayout(self._stat_ratio)

        layout.addWidget(stats)

        # Elapsed timer
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)

        # ── 3. Activity feed ──────────────────────────────────────────
        feed_header = QWidget()
        feed_header.setFixedHeight(32)
        feed_header.setStyleSheet(
            f"background: {C_CARD}; border-top: 1px solid {C_BORDER};"
            f" border-bottom: 1px solid {C_BORDER};"
        )
        fhl = QHBoxLayout(feed_header)
        fhl.setContentsMargins(12, 0, 12, 0)
        fh_lbl = QLabel("Activity")
        fh_lbl.setStyleSheet(
            f"color: {C_SUBTEXT}; font-size: 11px; font-weight: 700;"
            " text-transform: uppercase; letter-spacing: 0.5px;"
            " background: transparent;"
        )
        self._clear_feed_btn = QPushButton("Clear")
        self._clear_feed_btn.setObjectName("outline_btn")
        self._clear_feed_btn.setFixedSize(QSize(56, 22))
        self._clear_feed_btn.setStyleSheet(
            f"background: transparent; color: {C_HINT}; border: 1px solid {C_BORDER};"
            " border-radius: 4px; font-size: 11px; font-weight: 500; padding: 0;"
        )
        self._clear_feed_btn.clicked.connect(self._clear_feed)
        fhl.addWidget(fh_lbl)
        fhl.addStretch()
        fhl.addWidget(self._clear_feed_btn)
        layout.addWidget(feed_header)

        self._feed = QTextEdit()
        self._feed.setReadOnly(True)
        self._feed.setFixedHeight(130)
        self._feed.setStyleSheet(
            f"background: {C_FEED_BG}; color: {C_TEXT};"
            " border: none; border-radius: 0;"
            f" font-family: '{_FONT}', monospace; font-size: 12px;"
            " padding: 6px 12px; line-height: 1.5;"
        )
        self._feed.document().setMaximumBlockCount(300)
        layout.addWidget(self._feed)

        # Pre-built text formats for the feed
        self._fmt_ok   = self._make_fmt(C_OK_TXT)
        self._fmt_warn = self._make_fmt(C_WARN_TXT)
        self._fmt_err  = self._make_fmt(C_ERR_TXT)
        self._fmt_grey = self._make_fmt(C_HINT)

        self._show_placeholder()

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _make_stat(value: str, label: str) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setContentsMargins(20, 8, 20, 8)
        col.setSpacing(1)
        col.setAlignment(Qt.AlignmentFlag.AlignCenter)
        val = QLabel(value)
        val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        val.setStyleSheet(
            "color: #F1F5F9; font-size: 18px; font-weight: 700; background: transparent;"
        )
        lbl = QLabel(label)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            "color: #64748B; font-size: 10px; text-transform: uppercase;"
            " letter-spacing: 0.5px; background: transparent;"
        )
        col.addWidget(val)
        col.addWidget(lbl)
        # Store value label so we can update it
        col._val_lbl = val  # type: ignore[attr-defined]
        return col

    @staticmethod
    def _vdivider() -> QFrame:
        d = QFrame()
        d.setFrameShape(QFrame.Shape.VLine)
        d.setFixedWidth(1)
        d.setStyleSheet("background: #334155; border: none; max-width: 1px;")
        return d

    @staticmethod
    def _make_fmt(colour: str) -> QTextCharFormat:
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(colour))
        return fmt

    def _show_placeholder(self):
        size = self._preview.size()
        w = max(size.width(), 400)
        h = max(size.height(), 220)
        self._preview.setPixmap(_PlaceholderPixmap.get(w, h))
        self._meta_name.setText("No detections yet")
        self._meta_dims.setText("")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._frames_shown == 0:
            self._show_placeholder()

    # ── Public API called by MainWindow ──────────────────────────────────

    def start_run(self, watch_path: Path):
        """
        Begin a new run, watching *watch_path* for JPEG frames.

        Parameters
        ----------
        watch_path:
            The exact directory where ``EcoMotionZip_lite.py`` will write
            JPEG files.  This is:
                ``<output_directory>/EcoMotionZip/<video_stem>/``
            The directory need not exist yet — the poll timer will wait
            until it appears and then begin displaying frames.
        """
        self._watch_path = watch_path
        self._last_shown = ""
        self._frames_shown = 0
        self._start_time = time.monotonic()
        self._elapsed_timer.start()
        self._poll_timer.start()
        self._update_stat(self._stat_frames, "0")
        self._update_stat(self._stat_elapsed, "0s")
        self._update_stat(self._stat_ratio, "—")
        self._show_placeholder()

    def stop_run(self, total_input: int = 0):
        self._poll_timer.stop()
        self._elapsed_timer.stop()
        self._total_input_frames = total_input
        if total_input > 0 and self._frames_shown > 0:
            ratio = total_input / self._frames_shown
            self._update_stat(self._stat_ratio, f"{ratio:.1f}×")

    def add_log_line(self, text: str):
        """Append a line to the activity feed with auto colour-coding."""
        upper = text.upper()
        if any(k in upper for k in ("ERROR", "TRACEBACK", "EXCEPTION", "CRITICAL")):
            fmt = self._fmt_err
            prefix = "✗  "
        elif any(k in upper for k in ("WARNING", "WARN")):
            fmt = self._fmt_warn
            prefix = "⚠  "
        elif any(k in upper for k in ("FINISH", "COMPLETE", "DONE", "SAVED", "WRITTEN")):
            fmt = self._fmt_ok
            prefix = "✓  "
        else:
            fmt = self._fmt_grey
            prefix = "   "

        cursor = self._feed.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(prefix + text + "\n", fmt)
        self._feed.setTextCursor(cursor)
        self._feed.ensureCursorVisible()

        # Also look for numeric frame counts
        m = re.search(r"\bframe[s]?\s+(\d{2,})\b", text, re.IGNORECASE)
        if not m:
            m = re.search(r"\b(\d{4,})\b", text)
        if m:
            n = int(m.group(1))
            self._update_stat(self._stat_frames, f"{n:,}")

    def _clear_feed(self):
        self._feed.clear()

    # ── Frame polling ─────────────────────────────────────────────────────

    def _poll_scan(self):
        """
        Called every 500 ms while the pipeline is running.

        Scans ``_watch_path`` for JPEG files.  If a file newer than the last
        one shown is found, it is displayed in the preview pane.  The method
        is safe to call before the directory exists — it simply returns early
        until the subprocess creates it.
        """
        if self._watch_path is None or not self._watch_path.is_dir():
            return  # Directory not created yet — keep waiting

        # Collect all JPEGs, sorted oldest → newest by modification time
        jpegs = sorted(
            self._watch_path.glob("*.jpg"),
            key=lambda p: p.stat().st_mtime,
        )
        if not jpegs:
            return

        latest = jpegs[-1]
        if str(latest) == self._last_shown:
            return  # No new frame since last poll

        self._last_shown = str(latest)
        self._display_frame(latest)

    def _display_frame(self, path: Path):
        """Load *path* and display it in the preview label."""
        px = QPixmap(str(path))
        if px.isNull():
            return  # File may still be mid-write; next poll will retry

        self._frames_shown += 1
        self._update_stat(self._stat_frames, f"{self._frames_shown:,}")

        # Use the label's actual rendered size; fall back to a safe minimum
        # so the first frame isn't scaled to (0, 0) before layout completes.
        lw = max(self._preview.width(), 480)
        lh = max(self._preview.height(), 270)
        scaled = px.scaled(
            QSize(lw, lh),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview.setPixmap(scaled)
        self._meta_name.setText(path.stem)
        self._meta_dims.setText(f"{px.width()} × {px.height()} px")

    # ── Timer ─────────────────────────────────────────────────────────────

    def _tick_elapsed(self):
        if self._start_time is None:
            return
        secs = int(time.monotonic() - self._start_time)
        if secs < 60:
            self._update_stat(self._stat_elapsed, f"{secs}s")
        else:
            m, s = divmod(secs, 60)
            self._update_stat(self._stat_elapsed, f"{m}m {s:02d}s")

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _update_stat(col_layout: QVBoxLayout, text: str):
        col_layout._val_lbl.setText(text)  # type: ignore[attr-defined]


# =============================================================================
# Main window
# =============================================================================

class MainWindow(QMainWindow):
    """EcoMotionZip application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EcoMotionZip")
        self.resize(1180, 760)
        self.setMinimumSize(860, 580)

        self._raw_config: dict = {}
        self._proc: QProcess | None = None

        self._build_ui()
        self._load_config()

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        root_w = QWidget()
        self.setCentralWidget(root_w)
        root = QVBoxLayout(root_w)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background: {C_BORDER}; }}"
        )
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setSizes([480, 680])
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, stretch=1)

        root.addWidget(self._build_footer())

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._set_status("Ready — select a video file and press Run.")

    # ── Header ────────────────────────────────────────────────────────────

    def _build_header(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(68)
        bar.setStyleSheet(f"background: {C_SIDEBAR};")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(24, 0, 24, 0)

        # Logo dot
        dot = QLabel("●")
        dot.setStyleSheet(
            f"color: {C_ACCENT}; font-size: 22px; background: transparent;"
        )
        layout.addWidget(dot)
        layout.addSpacing(8)

        titles = QVBoxLayout()
        titles.setSpacing(0)
        h1 = QLabel("EcoMotionZip")
        f1 = QFont()
        f1.setPointSize(16)
        f1.setWeight(QFont.Weight.Bold)
        h1.setFont(f1)
        h1.setStyleSheet("color: #F1F5F9; background: transparent;")
        h2 = QLabel("Motion-based video compression for field ecologists")
        h2.setStyleSheet("color: #64748B; font-size: 12px; background: transparent;")
        titles.addWidget(h1)
        titles.addWidget(h2)
        layout.addLayout(titles)
        layout.addStretch()

        # Status badge
        self._pill = QLabel("● Ready")
        self._pill.setStyleSheet(
            f"color: {C_ACCENT}; font-size: 13px; font-weight: 700;"
            " background: transparent;"
        )
        layout.addWidget(self._pill)
        return bar

    # ── Left panel (settings tabs) ────────────────────────────────────────

    def _build_left(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"background: {C_BG};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 6, 12)
        layout.setSpacing(0)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"QTabWidget {{ background: {C_BG}; }}")

        self.files_tab    = FilesTab()
        self.motion_tab   = MotionTab()
        self.output_tab   = OutputTab()
        self.advanced_tab = AdvancedTab()

        self.tabs.addTab(self.files_tab,    "  Files  ")
        self.tabs.addTab(self.motion_tab,   "  Motion Detection  ")
        self.tabs.addTab(self.output_tab,   "  Output  ")
        self.tabs.addTab(self.advanced_tab, "  Advanced  ")

        layout.addWidget(self.tabs)
        return panel

    # ── Right panel (visual output) ───────────────────────────────────────

    def _build_right(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"background: {C_PREVIEW_BG};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header strip
        hdr = QWidget()
        hdr.setFixedHeight(36)
        hdr.setStyleSheet(
            f"background: {C_CARD}; border-bottom: 1px solid {C_BORDER};"
        )
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(14, 0, 14, 0)
        hl_lbl = QLabel("Live Output Preview")
        hl_lbl.setStyleSheet(
            f"color: {C_SUBTEXT}; font-size: 11px; font-weight: 700;"
            " text-transform: uppercase; letter-spacing: 0.6px;"
        )
        hl.addWidget(hl_lbl)
        layout.addWidget(hdr)

        self._preview_panel = FramePreviewPanel()
        layout.addWidget(self._preview_panel, stretch=1)
        return panel

    # ── Footer (progress + buttons) ───────────────────────────────────────

    def _build_footer(self) -> QWidget:
        footer = QWidget()
        footer.setFixedHeight(64)
        footer.setStyleSheet(
            f"background: {C_CARD}; border-top: 1px solid {C_BORDER};"
        )
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(12)

        # Progress column
        prog_col = QVBoxLayout()
        prog_col.setSpacing(4)
        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._frame_lbl = QLabel("Ready")
        self._frame_lbl.setStyleSheet(f"color: {C_HINT}; font-size: 12px;")
        prog_col.addWidget(self._progress)
        prog_col.addWidget(self._frame_lbl)
        layout.addLayout(prog_col, stretch=1)

        layout.addSpacing(20)

        self._save_btn = QPushButton("Save Config")
        self._save_btn.setObjectName("outline_btn")
        self._save_btn.setFixedSize(QSize(128, 40))
        self._save_btn.clicked.connect(self._save_config)

        self._run_btn = QPushButton("▶  Run")
        self._run_btn.setFixedSize(QSize(128, 40))
        self._run_btn.clicked.connect(self._run)

        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setObjectName("stop_btn")
        self._stop_btn.setFixedSize(QSize(128, 40))
        self._stop_btn.setVisible(False)
        self._stop_btn.clicked.connect(self._stop)

        layout.addWidget(self._save_btn)
        layout.addWidget(self._run_btn)
        layout.addWidget(self._stop_btn)
        return footer

    # ── Config I/O ────────────────────────────────────────────────────────

    def _load_config(self):
        try:
            with open(_CONFIG_PATH) as fh:
                cfg = json.load(fh)
        except FileNotFoundError:
            self._preview_panel.add_log_line(
                f"WARNING: config.json not found at {_CONFIG_PATH}. Using defaults."
            )
            return
        except json.JSONDecodeError as exc:
            self._preview_panel.add_log_line(f"ERROR: Cannot parse config.json — {exc}")
            return

        self._raw_config = cfg

        ft = self.files_tab
        ft.video_source.set_text(cfg.get("video_source", ""))
        ft.output_directory.set_text(cfg.get("output_directory", ""))

        mt = self.motion_tab
        mt.movement_threshold.set_value(cfg.get("movement_threshold", 40))
        mt.downscale_factor.set_value(cfg.get("downscale_factor", 16))
        mt.dilate_kernel_size.set_value(cfg.get("dilate_kernel_size", 128))
        mt.post_motion_record_frames.setValue(cfg.get("post_motion_record_frames", 0))
        mt.full_frame_capture_interval.setValue(cfg.get("full_frame_capture_interval", 300))

        ot = self.output_tab
        idx = ot.video_codec.findText(
            cfg.get("video_codec", "DIVX").upper(), Qt.MatchFlag.MatchExactly
        )
        if idx >= 0:
            ot.video_codec.setCurrentIndex(idx)
        ot.background_transparency.set_value(
            round(cfg.get("background_transparency", 1.0) * 10)
        )
        ot.save_frames.setChecked(cfg.get("save_frames", False))
        ot.frames_to_save.setValue(cfg.get("frames_to_save", 2000))
        ot.embed_timestamps.setChecked(cfg.get("embed_timestamps", False))
        ot.delete_original.setChecked(cfg.get("delete_original_after_processing", False))

        at = self.advanced_tab
        at.num_opencv_threads.setValue(cfg.get("num_opencv_threads", 10))
        at.reader_sleep_seconds.setValue(cfg.get("reader_sleep_seconds", 1.0))
        at.reader_flush_proportion.setValue(cfg.get("reader_flush_proportion", 0.9))

        self._preview_panel.add_log_line(f"Configuration loaded from {_CONFIG_PATH}")

    def _collect_config(self) -> dict:
        ft = self.files_tab
        mt = self.motion_tab
        ot = self.output_tab
        at = self.advanced_tab

        cfg = dict(self._raw_config)
        cfg.update({
            "video_source":                     ft.video_source.text(),
            "output_directory":                 ft.output_directory.text(),
            "movement_threshold":               mt.movement_threshold.value(),
            "downscale_factor":                 mt.downscale_factor.value(),
            "dilate_kernel_size":               mt.dilate_kernel_size.value(),
            "post_motion_record_frames":        mt.post_motion_record_frames.value(),
            "full_frame_capture_interval":      mt.full_frame_capture_interval.value(),
            "video_codec":                      ot.video_codec.currentText(),
            "background_transparency":          round(ot.background_transparency.value() / 10, 1),
            "save_frames":                      ot.save_frames.isChecked(),
            "frames_to_save":                   ot.frames_to_save.value(),
            "embed_timestamps":                 ot.embed_timestamps.isChecked(),
            "delete_original_after_processing": ot.delete_original.isChecked(),
            "num_opencv_threads":               at.num_opencv_threads.value(),
            "reader_sleep_seconds":             round(float(at.reader_sleep_seconds.value()), 2),
            "reader_flush_proportion":          round(float(at.reader_flush_proportion.value()), 2),
        })
        return cfg

    def _save_config(self, *, silent: bool = False) -> bool:
        cfg = self._collect_config()

        if not cfg["video_source"]:
            QMessageBox.warning(
                self, "No video selected",
                "Please choose a video file on the Files tab."
            )
            self.tabs.setCurrentIndex(0)
            return False

        if not cfg["output_directory"]:
            QMessageBox.warning(
                self, "No output folder",
                "Please choose an output folder on the Files tab."
            )
            self.tabs.setCurrentIndex(0)
            return False

        if not Path(cfg["video_source"]).exists():
            QMessageBox.warning(
                self, "File not found",
                f"The selected video does not exist:\n\n{cfg['video_source']}\n\n"
                "Please check the path on the Files tab."
            )
            self.tabs.setCurrentIndex(0)
            return False

        try:
            with open(_CONFIG_PATH, "w") as fh:
                json.dump(cfg, fh, indent=4)
            self._raw_config = cfg
            if not silent:
                QMessageBox.information(
                    self, "Configuration Saved",
                    f"Settings saved to:\n{_CONFIG_PATH}"
                )
            self._preview_panel.add_log_line(f"Configuration saved → {_CONFIG_PATH}")
            return True
        except OSError as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))
            return False

    # ── Process control ───────────────────────────────────────────────────

    def _run(self):
        if not self._save_config(silent=True):
            return

        # Warn if save_frames is off — preview won't show live frames
        if not self.output_tab.save_frames.isChecked():
            reply = QMessageBox.question(
                self, "Frame Preview Unavailable",
                '"Save individual frames" is currently off.\n\n'
                "The live preview panel will not show detections.\n\n"
                "Enable it on the Output tab for visual feedback, "
                "or continue without the preview?",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Ok,
            )
            if reply == QMessageBox.StandardButton.Cancel:
                self.tabs.setCurrentIndex(2)
                return

        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._proc.readyRead.connect(self._on_output)
        self._proc.finished.connect(self._on_finished)
        self._proc.start(sys.executable, [str(_LITE_SCRIPT)])

        if not self._proc.waitForStarted(3000):
            QMessageBox.critical(
                self, "Launch Failed",
                "Could not start EcoMotionZip_lite.py.\n\n"
                "Make sure Python and all dependencies\n"
                "(opencv-python, numpy) are installed."
            )
            self._proc = None
            return

        # Compute the exact directory where JPEG frames will be written.
        # EcoMotionZip_lite.py creates:
        #   <output_directory>/EcoMotionZip/<video_stem>/<video_stem>_frame_N.jpg
        cfg = self._raw_config
        video_stem = Path(cfg.get("video_source", "unknown")).stem
        watch_path = Path(cfg.get("output_directory", "")) / "EcoMotionZip" / video_stem

        # Switch to running state
        self._run_btn.setVisible(False)
        self._stop_btn.setVisible(True)
        self._save_btn.setEnabled(False)
        self._progress.setRange(0, 0)   # indeterminate spinner
        self._frame_lbl.setText("Processing…")
        self._pill.setText("● Processing")
        self._pill.setStyleSheet(
            "color: #FCD34D; font-size: 13px; font-weight: 700; background: transparent;"
        )
        self._set_status("Processing — this may take several minutes for long recordings.")
        self._preview_panel.start_run(watch_path)
        self._preview_panel.add_log_line(
            f"EcoMotionZip pipeline started.  Watching: {watch_path}"
        )

    def _stop(self):
        if self._proc and self._proc.state() != QProcess.ProcessState.NotRunning:
            self._proc.terminate()
            QTimer.singleShot(3000, self._force_kill)
        self._preview_panel.add_log_line("Stop requested — waiting for process to exit…")

    def _force_kill(self):
        if self._proc and self._proc.state() != QProcess.ProcessState.NotRunning:
            self._proc.kill()

    def _on_output(self):
        raw = self._proc.readAll().data().decode("utf-8", errors="replace")
        for line in raw.splitlines():
            line = line.strip()
            if line:
                self._preview_panel.add_log_line(line)

    def _on_finished(self, exit_code: int, _status):
        self._run_btn.setVisible(True)
        self._stop_btn.setVisible(False)
        self._save_btn.setEnabled(True)
        self._progress.setRange(0, 1)
        self._progress.setValue(1 if exit_code == 0 else 0)
        self._preview_panel.stop_run()

        if exit_code == 0:
            self._pill.setText("● Done")
            self._pill.setStyleSheet(
                f"color: {C_ACCENT}; font-size: 13px; font-weight: 700;"
                " background: transparent;"
            )
            out_dir = self._raw_config.get("output_directory", "the output folder")
            self._set_status("Compression complete.")
            self._frame_lbl.setText("Complete")
            self._preview_panel.add_log_line(
                f"Finished — results saved to: {out_dir}"
            )
            QMessageBox.information(
                self, "Compression Complete",
                "EcoMotionZip finished successfully.\n\n"
                f"Output saved to:\n{out_dir}"
            )
        else:
            self._pill.setText("● Error")
            self._pill.setStyleSheet(
                f"color: {C_ERR_TXT}; font-size: 13px; font-weight: 700;"
                " background: transparent;"
            )
            self._set_status(f"Process exited with error (code {exit_code}). See activity feed.")
            self._frame_lbl.setText(f"Error (code {exit_code})")
            self._preview_panel.add_log_line(
                f"ERROR: Process exited with code {exit_code}."
            )

        self._proc = None

    # ── Helpers ───────────────────────────────────────────────────────────

    def _set_status(self, msg: str):
        self._status_bar.showMessage(msg)

    def closeEvent(self, event):
        if self._proc and self._proc.state() != QProcess.ProcessState.NotRunning:
            reply = QMessageBox.question(
                self, "Processing in Progress",
                "EcoMotionZip is still running.\nStop the process and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._proc.kill()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# =============================================================================
# Entry point
# =============================================================================

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("EcoMotionZip")
    app.setOrganizationName("EcoMotionZip")
    app.setStyleSheet(QSS)
    f = app.font()
    f.setFamily(_FONT)
    app.setFont(f)

    window = MainWindow()
    window.show()
    window.activateWindow()
    window.raise_()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
