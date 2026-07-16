"""Dark commercial-grade theme (warm charcoal + coral accent)."""

from __future__ import annotations

import pyqtgraph as pg

# Palette ---------------------------------------------------------------
BG = "#262624"          # window background
PANEL = "#2d2d2b"       # sidebar / cards
CARD = "#31312e"        # card background
SURFACE = "#3a3a37"     # inputs
BORDER = "#45443f"
PLOT_BG = "#1f1e1c"
TEXT = "#eceae6"
TEXT_MUTED = "#a8a49e"
TEXT_FAINT = "#7d7a74"
ACCENT = "#d97757"      # coral
ACCENT_HOVER = "#e08b6d"
ACCENT_DOWN = "#c4633f"
GREEN = "#7fbf7a"
YELLOW = "#e0b25a"
RED = "#e05a5a"
GRID = "#3c3b38"


def apply_pyqtgraph_theme() -> None:
    pg.setConfigOptions(
        background=PLOT_BG,
        foreground=TEXT_MUTED,
        antialias=True,
        useOpenGL=False,
    )


def style_plot(plot: pg.PlotItem, title: str = "", x_label: str = "", y_label: str = "") -> None:
    if title:
        plot.setTitle(title, color=TEXT_MUTED, size="10pt")
    if x_label:
        plot.setLabel("bottom", x_label, color=TEXT_FAINT)
    if y_label:
        plot.setLabel("left", y_label, color=TEXT_FAINT)
    plot.showGrid(x=True, y=True, alpha=0.18)
    for side in ("left", "bottom"):
        ax = plot.getAxis(side)
        ax.setPen(pg.mkPen(BORDER))
        ax.setTextPen(pg.mkPen(TEXT_FAINT))


STYLESHEET = f"""
* {{
    font-family: 'Segoe UI', 'Malgun Gothic', sans-serif;
    font-size: 12px;
}}
QMainWindow, QWidget {{
    background-color: {BG};
    color: {TEXT};
}}
#Sidebar {{
    background-color: {PANEL};
    border-right: 1px solid {BORDER};
}}
#AppTitle {{
    font-size: 17px;
    font-weight: 700;
    color: {TEXT};
    letter-spacing: 0.5px;
}}
#AppSubtitle {{
    font-size: 10px;
    color: {TEXT_FAINT};
    letter-spacing: 1px;
}}
#Card {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
#CardTitle {{
    font-size: 11px;
    font-weight: 600;
    color: {TEXT_MUTED};
    letter-spacing: 1.2px;
}}
QLabel {{
    background: transparent;
    color: {TEXT_MUTED};
}}
QLabel#StatValue {{
    color: {TEXT};
    font-size: 13px;
    font-weight: 600;
}}
QLabel#DeviceInfo {{
    color: {TEXT_MUTED};
    font-size: 11px;
    padding: 2px 2px 4px 2px;
}}
QLabel#DeviceInfo[connected="true"] {{
    color: {GREEN};
}}
QPushButton {{
    background-color: {SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 6px 14px;
    font-weight: 600;
}}
QPushButton:hover {{ background-color: #44443f; border-color: #56554e; }}
QPushButton:pressed {{ background-color: #3a3a35; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; background-color: {PANEL}; }}
QPushButton#Primary {{
    background-color: {ACCENT};
    color: #201914;
    border: none;
    font-size: 13px;
    font-weight: 700;
    padding: 10px 16px;
}}
QPushButton#Primary:hover {{ background-color: {ACCENT_HOVER}; }}
QPushButton#Primary:pressed {{ background-color: {ACCENT_DOWN}; }}
QPushButton#Danger {{
    background-color: #8c3a34;
    color: {TEXT};
    border: none;
    font-size: 13px;
    font-weight: 700;
    padding: 10px 16px;
}}
QPushButton#Danger:hover {{ background-color: #a34239; }}
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px 8px;
    selection-background-color: {ACCENT};
}}
QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: #56554e;
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox::down-arrow {{
    image: none; border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_MUTED}; margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    selection-color: #201914;
}}
QCheckBox {{ color: {TEXT_MUTED}; spacing: 6px; }}
QCheckBox::indicator {{
    width: 15px; height: 15px; border-radius: 4px;
    border: 1px solid {BORDER}; background: {SURFACE};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT}; border-color: {ACCENT};
}}
QTabWidget::pane {{ border: none; background: {BG}; }}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_MUTED};
    padding: 9px 18px;
    border: none;
    border-bottom: 2px solid transparent;
    font-weight: 600;
}}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover:!selected {{ color: {TEXT}; }}
QTableWidget {{
    background-color: {PLOT_BG};
    alternate-background-color: #242320;
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    gridline-color: {GRID};
}}
QTableWidget::item {{ padding: 3px; }}
QTableWidget::item:selected {{ background: #4a3529; color: {TEXT}; }}
QHeaderView::section {{
    background-color: {PANEL};
    color: {TEXT_MUTED};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 5px;
    font-weight: 600;
}}
QScrollBar:vertical {{
    background: {BG}; width: 10px; border-radius: 5px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER}; border-radius: 5px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: #56554e; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{
    background: {BG}; height: 10px; border-radius: 5px;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER}; border-radius: 5px; min-width: 30px;
}}
QStatusBar {{
    background: {PANEL};
    color: {TEXT_MUTED};
    border-top: 1px solid {BORDER};
}}
QSplitter::handle {{ background: {BORDER}; width: 1px; }}
QToolTip {{
    background-color: {PANEL}; color: {TEXT};
    border: 1px solid {BORDER}; padding: 4px;
}}
QSlider::groove:horizontal {{
    height: 4px; background: {SURFACE}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT}; width: 14px; height: 14px;
    margin: -5px 0; border-radius: 7px;
}}
QScrollArea {{ border: none; }}
"""
