#!/usr/bin/env python3
"""
================================================================================
WORKSHOP UTILITIES - Shared functions for all sessions
================================================================================

This module consolidates common utilities used across the workshop:
- Signal processing (baseline correction)
- Curve fitting functions (logistic models)
- Data loading and comparison
- Visualization constants

Import these in notebooks:
    from workshop_utils import snip_baseline, logistic_fn, compare_dataframes
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# SIGNAL PROCESSING
# =============================================================================

def snip_baseline(signal: np.ndarray, window_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove baseline drift using the SNIP algorithm.

    The Sensitive Nonlinear Iterative Peak (SNIP) algorithm estimates a
    background baseline by iteratively clipping peaks. This is commonly used
    for chromatography and spectroscopy data.

    Parameters
    ----------
    signal : np.ndarray
        Input signal with baseline drift
    window_size : int, default=100
        Number of iterations for baseline estimation. Larger values give
        smoother baselines but may clip broad peaks.

    Returns
    -------
    corrected : np.ndarray
        Signal with baseline subtracted
    baseline : np.ndarray
        Estimated baseline

    Example
    -------
    >>> signal = np.array([1.0, 1.1, 5.0, 1.2, 1.0])  # spike at index 2
    >>> corrected, baseline = snip_baseline(signal, window_size=2)
    >>> corrected[2] > corrected[0]  # spike preserved
    True

    References
    ----------
    Ryan et al., Nuclear Instruments and Methods in Physics Research (1988)
    """
    # LLS (Log-Log-Square root) transform to stabilize variance
    s_lls = np.log(np.log(np.sqrt(signal + 1) + 1) + 1)
    s_lls_filt = s_lls.copy()

    # Iterative clipping: replace each point with min of itself and
    # average of neighbors at distance m
    for m in range(1, window_size + 1):
        for i in range(m, len(s_lls_filt) - m):
            neighbor_avg = (s_lls_filt[i - m] + s_lls_filt[i + m]) / 2
            s_lls_filt[i] = min(s_lls_filt[i], neighbor_avg)

    # Inverse LLS transform
    baseline = (np.exp(np.exp(s_lls_filt) - 1) - 1) ** 2 - 1
    corrected = signal - baseline

    return corrected, baseline


# =============================================================================
# CURVE FITTING FUNCTIONS
# =============================================================================

def logistic_fn(x: np.ndarray, bottom: float, top: float,
                log_ec50: float, hill: float) -> np.ndarray:
    """
    4-parameter logistic function for dose-response curves.

    The standard sigmoidal model used for fitting dose-response data
    and calculating EC50/IC50 values.

    Parameters
    ----------
    x : np.ndarray
        Log concentration values (e.g., log10[M])
    bottom : float
        Lower asymptote (baseline response)
    top : float
        Upper asymptote (maximum response)
    log_ec50 : float
        Log of the EC50 (concentration at 50% response)
    hill : float
        Hill slope (steepness of the curve)

    Returns
    -------
    np.ndarray
        Predicted response values

    Example
    -------
    >>> import numpy as np
    >>> x = np.linspace(-10, -4, 20)
    >>> y = logistic_fn(x, bottom=0, top=100, log_ec50=-7, hill=1)
    >>> y[10]  # Response at EC50 should be ~50
    """
    return bottom + (top - bottom) / (1 + 10 ** ((log_ec50 - x) * hill))


def logistic_growth(t: np.ndarray, max_od: float, rate: float,
                    lag: float) -> np.ndarray:
    """
    Logistic growth model for microbial growth curves.

    Models the characteristic S-shaped growth curve with lag phase,
    exponential phase, and stationary phase.

    Parameters
    ----------
    t : np.ndarray
        Time points (typically in hours)
    max_od : float
        Maximum OD (carrying capacity)
    rate : float
        Growth rate constant
    lag : float
        Lag time (time at inflection point)

    Returns
    -------
    np.ndarray
        Predicted OD values

    Example
    -------
    >>> t = np.linspace(0, 24, 100)
    >>> od = logistic_growth(t, max_od=1.5, rate=0.5, lag=5)
    >>> od[-1] > 1.4  # Approaches max_od
    True
    """
    return max_od / (1 + np.exp(-rate * (t - lag)))


# =============================================================================
# DATA LOADING
# =============================================================================

def load_extracted_data(
    ai_path: Optional[str] = None,
    corrected_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load AI-extracted and hand-corrected CSV files for comparison.

    Designed for the Session 1 exercise comparing AI data extraction
    against manually verified values.

    Parameters
    ----------
    ai_path : str, optional
        Path to AI-extracted CSV. Defaults to Gemini extraction.
    corrected_path : str, optional
        Path to hand-corrected CSV. Defaults to by-hand version.

    Returns
    -------
    df_ai : pd.DataFrame
        AI-extracted data
    df_corrected : pd.DataFrame or None
        Hand-corrected data (None if file doesn't exist)

    Example
    -------
    >>> df_ai, df_corrected = load_extracted_data()
    >>> df_ai.shape == df_corrected.shape
    True
    """
    base = Path('materials/session1/solution')

    ai_file = Path(ai_path) if ai_path else base / 'nature_figure_gpcr_b_gemini.csv'
    corrected_file = Path(corrected_path) if corrected_path else base / 'nature_figure_gpcr_b_by_hand.csv'

    df_ai = pd.read_csv(ai_file)
    df_corrected = pd.read_csv(corrected_file) if corrected_file.exists() else None

    return df_ai, df_corrected


def compare_dataframes(
    df_ai: pd.DataFrame,
    df_corrected: pd.DataFrame,
    show_differences: int = 10
) -> Tuple[float, List[Dict]]:
    """
    Compare two dataframes by position, handling mismatched column names.

    This function compares values cell-by-cell using positional indices,
    which is robust when AI extraction produces different column names
    than the reference data.

    Parameters
    ----------
    df_ai : pd.DataFrame
        AI-extracted data
    df_corrected : pd.DataFrame
        Hand-corrected reference data
    show_differences : int, default=10
        Number of differences to print

    Returns
    -------
    match_rate : float
        Percentage of cells that match (0-100)
    differences : list of dict
        Details of each difference found

    Example
    -------
    >>> import pandas as pd
    >>> df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> df2 = pd.DataFrame({'X': [1, 2], 'Y': [3, 5]})  # Different col names!
    >>> rate, diffs = compare_dataframes(df1, df2)
    >>> rate  # 3 out of 4 match
    75.0
    """
    # Check shape compatibility
    if df_ai.shape != df_corrected.shape:
        print(f"Shape mismatch: AI {df_ai.shape} vs Corrected {df_corrected.shape}")
        print("Cannot compare dataframes with different shapes.")
        return 0.0, []

    matches = 0
    total = df_ai.size
    differences = []

    for i in range(df_ai.shape[0]):
        for j in range(df_ai.shape[1]):
            ai_val = df_ai.iloc[i, j]
            corr_val = df_corrected.iloc[i, j]

            # Handle NaN comparison
            both_nan = pd.isna(ai_val) and pd.isna(corr_val)
            values_equal = str(ai_val).strip() == str(corr_val).strip()

            if both_nan or values_equal:
                matches += 1
            else:
                differences.append({
                    'row': i,
                    'col': j,
                    'ai_col_name': df_ai.columns[j],
                    'corr_col_name': df_corrected.columns[j],
                    'ai_value': ai_val,
                    'corrected_value': corr_val
                })

    match_rate = (matches / total) * 100

    # Print summary
    print(f"{'='*60}")
    print(f"COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Total cells compared: {total}")
    print(f"Matching cells: {matches}")
    print(f"Different cells: {len(differences)}")
    print(f"Match rate: {match_rate:.1f}%")

    if differences and show_differences > 0:
        print(f"\n{'='*60}")
        print(f"DIFFERENCES FOUND (showing first {min(show_differences, len(differences))})")
        print(f"{'='*60}")
        for d in differences[:show_differences]:
            print(f"  Row {d['row']}, Col {d['col']}:")
            print(f"    AI value:        '{d['ai_value']}'")
            print(f"    Corrected value: '{d['corrected_value']}'")

    return match_rate, differences


# =============================================================================
# VISUALIZATION CONSTANTS
# =============================================================================

# Color palette for GPCR dose-response plots
COLOR_MAP: Dict[str, str] = {
    'WT PTH1R': '#1f77b4',      # Blue
    'PTH1R(P415A)': '#d62728',  # Red
    'WT PTH2R': '#2ca02c',      # Green
    'PTH2R(L370P)': '#9467bd',  # Purple
    'WT GLP1R': '#8c564b',      # Brown
    'GLP1R-2M': '#e377c2',      # Pink
    'GLP1R-4M': '#7f7f7f',      # Gray
    'GLP1R-5M': '#bcbd22'       # Olive
}

# Panel configurations for Nature-style selectivity plots
PANEL_CONFIGS: Dict[str, Dict] = {
    'a': {
        'ligand': 'PCO371',
        'order': ['WT PTH1R', 'PTH1R(P415A)', 'WT PTH2R', 'PTH2R(L370P)'],
        'title': 'a  PCO371 on PTH receptors'
    },
    'b': {
        'ligand': 'PCO371',
        'order': ['WT PTH1R', 'WT GLP1R', 'GLP1R-2M', 'GLP1R-4M', 'GLP1R-5M'],
        'title': 'b  PCO371 on GLP1 receptors'
    },
    'c': {
        'ligand': 'PTH',
        'order': [
            'WT PTH1R', 'PTH1R(P415A)', 'WT PTH2R', 'PTH2R(L370P)',
            'WT GLP1R', 'GLP1R-2M', 'GLP1R-4M', 'GLP1R-5M'
        ],
        'title': 'c  PTH on PTH/GLP receptors'
    }
}


# =============================================================================
# MODULE INFO
# =============================================================================

__all__ = [
    'snip_baseline',
    'logistic_fn',
    'logistic_growth',
    'load_extracted_data',
    'compare_dataframes',
    'COLOR_MAP',
    'PANEL_CONFIGS'
]

if __name__ == '__main__':
    # Quick self-test
    print("Workshop Utils - Self Test")
    print("-" * 40)

    # Test snip_baseline
    test_signal = np.linspace(10, 15, 100) + np.random.normal(0, 0.1, 100)
    corrected, baseline = snip_baseline(test_signal, window_size=20)
    print(f"snip_baseline: input shape {test_signal.shape}, output shape {corrected.shape}")

    # Test logistic functions
    x = np.linspace(-10, -4, 20)
    y = logistic_fn(x, 0, 100, -7, 1)
    print(f"logistic_fn: EC50 response = {y[len(y)//2]:.1f}% (expected ~50%)")

    t = np.linspace(0, 24, 100)
    od = logistic_growth(t, 1.5, 0.5, 5)
    print(f"logistic_growth: final OD = {od[-1]:.2f} (expected ~1.5)")

    print("-" * 40)
    print("All tests passed!")
