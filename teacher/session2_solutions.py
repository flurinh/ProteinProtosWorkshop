#!/usr/bin/env python3
"""
================================================================================
SESSION 2 SOLUTIONS - Reference implementations for all exercises
================================================================================

This script provides complete, runnable solutions for all Session 2 exercises:
1. FSEC trace analysis (baseline correction, peak detection)
2. Dose-response curves (EC50 fitting, selectivity analysis)
3. Plate reader kinetics (logistic growth fitting)
4. Spectrophotometer scans (peak detection)

Run this script to regenerate all analysis outputs:
    python session2_solutions.py

Outputs are saved to:
    materials/session2/results/
    materials/session2/figures/
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

# Import shared utilities (no more duplication!)
from workshop_utils import (
    snip_baseline,
    logistic_fn,
    logistic_growth,
    COLOR_MAP,
    PANEL_CONFIGS
)

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "materials" / "session2" / "data"
RESULTS_DIR = BASE_DIR / "materials" / "session2" / "results"
FIG_DIR = BASE_DIR / "materials" / "session2" / "figures"

# Create output directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# EXERCISE 1: FSEC TRACES
# =============================================================================

def analyze_fsec() -> pd.DataFrame:
    """
    Analyze FSEC traces: baseline correction and peak detection.

    This function demonstrates:
    - Loading chromatography data
    - SNIP baseline correction
    - Signal normalization
    - Peak detection with scipy

    Returns
    -------
    pd.DataFrame
        Summary of detected peaks for each trace
    """
    print("\n" + "=" * 60)
    print("EXERCISE 1: FSEC TRACE ANALYSIS")
    print("=" * 60)

    # Load data
    df = pd.read_csv(DATA_DIR / 'fsec.csv')
    print(f"Loaded {len(df)} timepoints x {len(df.columns)-1} traces")

    # Process each trace
    peak_rows = []
    normalized_df = df.copy()

    for col in df.columns[1:]:  # Skip 'Timepoint' column
        signal = df[col].values

        # Baseline correction
        corrected, baseline = snip_baseline(signal, window_size=100)

        # Normalize to 0-1 range
        norm = (corrected - corrected.min()) / (corrected.max() - corrected.min() + 1e-8)
        normalized_df[col] = norm

        # Detect peaks
        peaks, properties = find_peaks(corrected, prominence=0.02)
        peak_times = df['Timepoint'].iloc[peaks].tolist()

        # Record results
        peak_rows.append({
            'trace': col,
            'num_peaks': len(peaks),
            'peak_timepoints': ';'.join(f"{t:.4f}" for t in peak_times),
            'mean_prominence': float(properties['prominences'].mean()) if len(peaks) else 0.0
        })

    # Create summary DataFrame
    peak_summary = pd.DataFrame(peak_rows)
    peak_summary.to_csv(RESULTS_DIR / 'fsec_peak_summary.csv', index=False)
    print(f"Saved: {RESULTS_DIR / 'fsec_peak_summary.csv'}")

    # Visualization: Raw traces
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax1 = axes[0]
    for col in df.columns[1:5]:  # First 4 traces
        ax1.plot(df['Timepoint'], df[col], label=col)
    ax1.set_xlabel('Timepoint')
    ax1.set_ylabel('Signal (a.u.)')
    ax1.set_title('Raw FSEC Traces')
    ax1.legend(fontsize=8)

    ax2 = axes[1]
    for col in normalized_df.columns[1:5]:
        ax2.plot(normalized_df['Timepoint'], normalized_df[col], label=col)
    ax2.set_xlabel('Timepoint')
    ax2.set_ylabel('Normalized Signal')
    ax2.set_title('Baseline-Corrected & Normalized')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fsec_analysis.png', dpi=150)
    plt.close()
    print(f"Saved: {FIG_DIR / 'fsec_analysis.png'}")

    return peak_summary


# =============================================================================
# EXERCISE 2: DOSE-RESPONSE CURVES
# =============================================================================

def analyze_dose_response() -> pd.DataFrame:
    """
    Fit dose-response curves and calculate EC50 values.

    This function demonstrates:
    - Loading dose-response data
    - 4-parameter logistic curve fitting
    - EC50 extraction
    - Publication-quality multi-panel plots

    Returns
    -------
    pd.DataFrame
        Fitted parameters (EC50, Hill slope) for each receptor/ligand pair
    """
    print("\n" + "=" * 60)
    print("EXERCISE 2: DOSE-RESPONSE ANALYSIS")
    print("=" * 60)

    # Load data
    df = pd.read_csv(DATA_DIR / 'nature_figure_gpcr_b_selectivity.csv')

    # Normalize column names (handle different naming conventions)
    rename_map = {
        'Protein': 'Receptor',
        'Log_Concentration': 'LogConcentration',
        'Response_Percent': 'Response'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    print(f"Loaded {len(df)} data points")
    print(f"Receptors: {df['Receptor'].unique().tolist()}")
    print(f"Ligands: {df['Ligand'].unique().tolist()}")

    # Fit logistic curves
    fit_results = []

    for (receptor, ligand), subset in df.groupby(['Receptor', 'Ligand']):
        subset = subset.sort_values('LogConcentration')
        x = subset['LogConcentration'].values
        y = subset['Response'].values

        if len(subset) < 4:
            continue

        # Initial parameter guesses
        p0 = [y.min(), y.max(), np.median(x), 1.0]

        try:
            popt, _ = curve_fit(logistic_fn, x, y, p0=p0, maxfev=10000)
            fit_results.append({
                'Receptor': receptor,
                'Ligand': ligand,
                'Bottom': popt[0],
                'Top': popt[1],
                'logEC50': popt[2],
                'EC50_nM': 10 ** (popt[2] + 9),  # Convert to nM
                'Hill': popt[3]
            })
        except RuntimeError:
            fit_results.append({
                'Receptor': receptor,
                'Ligand': ligand,
                'Bottom': np.nan,
                'Top': np.nan,
                'logEC50': np.nan,
                'EC50_nM': np.nan,
                'Hill': np.nan
            })

    fits_df = pd.DataFrame(fit_results)
    fits_df.to_csv(RESULTS_DIR / 'dose_response_ec50.csv', index=False)
    print(f"Saved: {RESULTS_DIR / 'dose_response_ec50.csv'}")

    # Create 3-panel Nature-style figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for ax, panel_key in zip(axes, ['a', 'b', 'c']):
        config = PANEL_CONFIGS[panel_key]
        ligand = config['ligand']
        receptor_order = config['order']

        panel_data = df[(df['Ligand'] == ligand) & (df['Receptor'].isin(receptor_order))]

        for receptor in receptor_order:
            subset = panel_data[panel_data['Receptor'] == receptor].sort_values('LogConcentration')
            if subset.empty:
                continue

            color = COLOR_MAP.get(receptor, '#333333')
            ax.plot(
                subset['LogConcentration'],
                subset['Response'],
                marker='o',
                color=color,
                label=receptor,
                linewidth=2,
                markersize=6
            )

        ax.set_title(config['title'], fontsize=12)
        ax.set_xlabel('log[ligand] (M)')
        ax.legend(fontsize=7, loc='lower right')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)

    axes[0].set_ylabel('cAMP response (% of WT PTH1R)')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'dose_response_selectivity.png', dpi=150)
    plt.close()
    print(f"Saved: {FIG_DIR / 'dose_response_selectivity.png'}")

    # Selectivity summary: WT vs mutant
    print("\nSelectivity Summary:")
    for ligand in ['PCO371', 'PTH']:
        subset = df[df['Ligand'] == ligand]
        wt_max = subset[subset['Receptor'] == 'WT PTH1R']['Response'].max()
        mut_max = subset[subset['Receptor'] == 'PTH1R(P415A)']['Response'].max()
        if wt_max and not np.isnan(wt_max):
            pct = (mut_max / wt_max) * 100
            print(f"  {ligand}: P415A mutant = {pct:.1f}% of WT max response")

    return fits_df


# =============================================================================
# EXERCISE 3: PLATE READER KINETICS
# =============================================================================

def analyze_plate_reader() -> pd.DataFrame:
    """
    Fit logistic growth curves to plate reader data.

    This function demonstrates:
    - Loading kinetic data
    - Logistic growth model fitting
    - Extracting biological parameters (max OD, growth rate, lag time)
    - Comparing fitted vs ground truth parameters

    Returns
    -------
    pd.DataFrame
        Fitted growth parameters for each sample
    """
    print("\n" + "=" * 60)
    print("EXERCISE 3: PLATE READER KINETICS")
    print("=" * 60)

    # Load data
    df = pd.read_csv(DATA_DIR / 'plate_reader_curves.csv')
    time = df['time_hr'].values

    # Load ground truth if available
    truth_path = DATA_DIR / 'plate_reader_truth.csv'
    truth = pd.read_csv(truth_path) if truth_path.exists() else None

    print(f"Loaded {len(df.columns)-1} growth curves, {len(df)} time points")

    # Fit each curve
    fit_results = []

    for col in df.columns[1:]:
        y = df[col].values

        # Initial guesses based on data
        p0 = [y.max(), 0.4, time.mean()]
        bounds = ([0, 0.05, -5], [3, 2, 30])

        try:
            popt, _ = curve_fit(
                logistic_growth, time, y,
                p0=p0, bounds=bounds, maxfev=10000
            )
            fit_results.append({
                'sample': col,
                'max_od': popt[0],
                'growth_rate': popt[1],
                'lag_hr': popt[2]
            })
        except RuntimeError:
            fit_results.append({
                'sample': col,
                'max_od': np.nan,
                'growth_rate': np.nan,
                'lag_hr': np.nan
            })

    fits_df = pd.DataFrame(fit_results)

    # Compare to ground truth if available
    if truth is not None:
        fits_df = fits_df.merge(truth, on='sample', suffixes=('_fit', '_true'))
        output_file = RESULTS_DIR / 'plate_reader_comparison.csv'
    else:
        output_file = RESULTS_DIR / 'plate_reader_parameters.csv'

    fits_df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot raw curves
    ax1 = axes[0]
    for col in df.columns[1:]:
        ax1.plot(df['time_hr'], df[col], label=col, linewidth=1.5)
    ax1.set_xlabel('Time (hr)')
    ax1.set_ylabel('OD600')
    ax1.set_title('Growth Curves')
    ax1.legend(fontsize=8)

    # Plot fitted vs data for one example
    ax2 = axes[1]
    example_col = df.columns[1]
    example_fit = fits_df[fits_df['sample'] == example_col].iloc[0]

    ax2.scatter(df['time_hr'], df[example_col], alpha=0.7, label='Data')
    t_fit = np.linspace(time.min(), time.max(), 200)
    y_fit = logistic_growth(t_fit, example_fit['max_od'], example_fit['growth_rate'], example_fit['lag_hr'])
    ax2.plot(t_fit, y_fit, 'r-', linewidth=2, label='Fit')
    ax2.axvline(example_fit['lag_hr'], color='gray', linestyle='--', alpha=0.5, label=f"Lag = {example_fit['lag_hr']:.1f} hr")
    ax2.set_xlabel('Time (hr)')
    ax2.set_ylabel('OD600')
    ax2.set_title(f'{example_col}: Logistic Fit')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'plate_reader_analysis.png', dpi=150)
    plt.close()
    print(f"Saved: {FIG_DIR / 'plate_reader_analysis.png'}")

    return fits_df


# =============================================================================
# EXERCISE 4: SPECTROPHOTOMETER SCANS
# =============================================================================

def analyze_spectro() -> pd.DataFrame:
    """
    Detect absorption peaks in spectrophotometer scans.

    This function demonstrates:
    - Loading spectral data
    - Savitzky-Golay smoothing
    - Peak detection with prominence threshold
    - Comparing detected vs expected peak positions

    Returns
    -------
    pd.DataFrame
        Detected peak centers for each sample
    """
    print("\n" + "=" * 60)
    print("EXERCISE 4: SPECTROPHOTOMETER ANALYSIS")
    print("=" * 60)

    # Load data
    df = pd.read_csv(DATA_DIR / 'spectro_scans.csv')
    wavelength = df['wavelength_nm'].values

    # Load ground truth if available
    truth_path = DATA_DIR / 'spectro_truth.csv'
    truth = pd.read_csv(truth_path) if truth_path.exists() else None

    print(f"Loaded {len(df.columns)-1} spectra, {len(df)} wavelength points")
    print(f"Wavelength range: {wavelength.min():.0f} - {wavelength.max():.0f} nm")

    # Detect peaks in each spectrum
    results = []

    for col in df.columns[1:]:
        signal = df[col].values

        # Smooth the spectrum (window=11, polynomial order=3)
        smoothed = savgol_filter(signal, 11, 3)

        # Find peaks with prominence threshold
        peaks, properties = find_peaks(smoothed, prominence=0.05)
        peak_centers = wavelength[peaks]

        results.append({
            'sample': col,
            'num_peaks': len(peaks),
            'peak_centers_nm': ';'.join(f"{c:.1f}" for c in peak_centers)
        })

    results_df = pd.DataFrame(results)

    # Compare to ground truth if available
    if truth is not None:
        results_df = results_df.merge(truth, on='sample', suffixes=('_detected', '_true'))
        output_file = RESULTS_DIR / 'spectro_comparison.csv'
    else:
        output_file = RESULTS_DIR / 'spectro_peaks.csv'

    results_df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.tab10.colors
    for i, col in enumerate(df.columns[1:]):
        ax.plot(wavelength, df[col], label=col, color=colors[i % len(colors)], linewidth=1.5)

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Absorbance (a.u.)')
    ax.set_title('Opsin Absorption Spectra')
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'spectro_analysis.png', dpi=150)
    plt.close()
    print(f"Saved: {FIG_DIR / 'spectro_analysis.png'}")

    # Print detected peaks
    print("\nDetected Peaks:")
    for _, row in results_df.iterrows():
        print(f"  {row['sample']}: {row['peak_centers_nm']}")

    return results_df


# =============================================================================
# MAIN DRIVER
# =============================================================================

def main():
    """Run all Session 2 analyses and generate outputs."""
    print("=" * 60)
    print("SESSION 2 SOLUTIONS")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Figures directory: {FIG_DIR}")

    # Run all analyses
    fsec_results = analyze_fsec()
    dose_response_results = analyze_dose_response()
    plate_reader_results = analyze_plate_reader()
    spectro_results = analyze_spectro()

    print("\n" + "=" * 60)
    print("ALL ANALYSES COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Figures saved to: {FIG_DIR}")

    return {
        'fsec': fsec_results,
        'dose_response': dose_response_results,
        'plate_reader': plate_reader_results,
        'spectro': spectro_results
    }


if __name__ == '__main__':
    main()
