"""
================================================================================
SESSION 2 UTILITIES - Data loaders and visualization helpers
================================================================================

This module provides data loading and plotting functions for Session 2.
Shared utilities (baseline correction, curve fitting) are in workshop_utils.py.

Import example:
    from session2_utils import load_fsec_data, plot_dose_response_context
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- needed for 3D projections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import curve_fit

# Import shared utilities from workshop_utils
from workshop_utils import COLOR_MAP, PANEL_CONFIGS, logistic_fn

DATA_ROOT = Path('materials/session2/data')


def load_fsec_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load size-exclusion traces for the FSEC example."""

    file_path = Path(path) if path else DATA_ROOT / 'fsec.csv'
    return pd.read_csv(file_path)


def preview_fsec_context(df: pd.DataFrame, max_traces: int = 4) -> None:
    """Plot a quick look at a subset of FSEC traces."""

    plt.figure(figsize=(8, 4))
    measurement_cols = df.columns[1:1 + max_traces]
    for col in measurement_cols:
        plt.plot(df['Timepoint'], df[col], label=col)
    plt.xlabel('Timepoint')
    plt.ylabel('Signal (a.u.)')
    plt.title('Raw FSEC traces (subset)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def load_dose_response_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load the synthetic dose-response dataset and normalize column names."""

    file_path = Path(path) if path else DATA_ROOT / 'nature_figure_gpcr_b_selectivity.csv'
    df = pd.read_csv(file_path)
    df = df.rename(columns={
        'Protein': 'Receptor',
        'Log_Concentration': 'LogConcentration',
        'Response_Percent': 'Response'
    })
    return df


def plot_dose_response_context(df: pd.DataFrame) -> None:
    """Mimic the three-panel nature figure for visual context."""

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, panel in zip(axes, ['a', 'b', 'c']):
        cfg = PANEL_CONFIGS[panel]
        ligand = cfg['ligand']
        order = cfg['order']
        panel_df = df[df['Ligand'] == ligand]
        for receptor in order:
            subset = panel_df[panel_df['Receptor'] == receptor]
            if subset.empty:
                continue
            subset = subset.sort_values('LogConcentration')
            ax.plot(
                subset['LogConcentration'],
                subset['Response'],
                marker='o',
                label=receptor,
                color=COLOR_MAP.get(receptor)
            )
        ax.set_title(cfg['title'])
        ax.set_xlabel('log[ligand] (M)')
        if panel == 'a':
            ax.set_ylabel('cAMP response (%)')
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


def load_plate_reader_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load simulated growth curves from a microplate reader."""

    file_path = Path(path) if path else DATA_ROOT / 'plate_reader_curves.csv'
    return pd.read_csv(file_path)


def plot_plate_reader_context(df: pd.DataFrame, example_cols: int = 3) -> None:
    """Display representative growth curves for discussion."""

    plt.figure(figsize=(8, 4))
    for col in df.columns[1:1 + example_cols]:
        plt.plot(df['time_hr'], df[col], label=col)
    plt.xlabel('Time (hr)')
    plt.ylabel('OD600 (a.u.)')
    plt.title('Plate reader raw signals')
    plt.legend()
    plt.tight_layout()
    plt.show()


def load_spectro_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load simulated spectrophotometer scans."""

    file_path = Path(path) if path else DATA_ROOT / 'spectro_scans.csv'
    return pd.read_csv(file_path)


def plot_spectro_context(df: pd.DataFrame) -> None:
    """Plot all opsin scans to highlight the dual-peak structure."""

    plt.figure(figsize=(8, 4))
    for col in df.columns[1:]:
        plt.plot(df['wavelength_nm'], df[col], label=col)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Absorbance (a.u.)')
    plt.title('Spectrophotometer scans (opsins)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def preview_marco_samples(root: Path | str | None = None) -> None:
    """Show one MARCO crystal image per class for discussion."""

    samples_root = Path(root) if root else DATA_ROOT / 'marco-protein-crystal-image-recognition/converted_train'
    classes = ['Clear', 'Crystals', 'Other', 'Precipitate']
    fig, axes = plt.subplots(1, len(classes), figsize=(12, 3))
    for ax, cls in zip(axes, classes):
        imgs: Iterable[Path] = list((samples_root / cls).glob('*.jpg'))
        if imgs:
            ax.imshow(Image.open(imgs[0]))
        ax.set_title(cls)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def generate_procrustes_example(noise: float = 0.01, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create paired point sets for the Procrustes cautionary tale."""

    rng = np.random.default_rng(seed)
    A = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    B = A @ rotation.T + rng.normal(scale=noise, size=A.shape)
    return A, B, rotation


def plot_procrustes_context(A: np.ndarray, B: np.ndarray) -> None:
    """Show the point clouds before aligning to motivate the task."""

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(A[:, 0], A[:, 1], A[:, 2], label='Template', color='#1f77b4')
    ax.scatter(B[:, 0], B[:, 1], B[:, 2], label='Target', color='#d62728')
    for a, b in zip(A, B):
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color='gray', linewidth=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point clouds before alignment')
    ax.legend()
    plt.tight_layout()
    plt.show()

def fit_logistic_curve(subset: pd.DataFrame):
    """
    Fit a 4-parameter logistic curve to dose-response data.

    Uses logistic_fn from workshop_utils.

    Parameters
    ----------
    subset : pd.DataFrame
        Must have 'LogConcentration' and 'Response' columns

    Returns
    -------
    log_ec50 : float
        Log of the EC50 value
    hill : float
        Hill slope coefficient
    """
    x = subset['LogConcentration'].values
    y = subset['Response'].values
    if len(subset) < 4:
        return np.nan, np.nan
    p0 = [y.min(), y.max(), np.median(x), 1.0]
    popt, _ = curve_fit(logistic_fn, x, y, p0=p0, maxfev=10000)
    return popt[2], popt[3]


# Re-export for backward compatibility
# (snip_baseline was here but is now in workshop_utils)
