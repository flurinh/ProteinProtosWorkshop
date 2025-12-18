#!/usr/bin/env python3
"""Generate synthetic datasets for Session 2 (plate reader + spectrophotometer).
Run this script whenever you want to refresh the data in materials/session2/data.
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'materials' / 'session2' / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(123)

# Plate reader data ---------------------------------------------------------
time_hr = np.linspace(0, 24, 49)
plate_params = [
    {'sample': 'culture_A', 'max_od': 1.3, 'growth_rate': 0.42, 'lag_hr': 6.0},
    {'sample': 'culture_B', 'max_od': 1.1, 'growth_rate': 0.55, 'lag_hr': 4.5},
    {'sample': 'culture_C', 'max_od': 1.5, 'growth_rate': 0.35, 'lag_hr': 7.5}
]
plate_df = pd.DataFrame({'time_hr': time_hr})
for params in plate_params:
    logistic = params['max_od'] / (1 + np.exp(-params['growth_rate'] * (time_hr - params['lag_hr'])))
    noise = np.random.normal(scale=0.03, size=time_hr.size)
    plate_df[params['sample']] = logistic + noise
plate_df.to_csv(DATA_DIR / 'plate_reader_curves.csv', index=False)
pd.DataFrame(plate_params).to_csv(DATA_DIR / 'plate_reader_truth.csv', index=False)

# Spectrophotometer data ----------------------------------------------------
wavelength = np.arange(250, 701, 2)

def gaussian(x, center, width, amplitude):
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)

# Opsin-like spectra: two characteristic peaks (e.g., retinal + second band)
spectro_specs = [
    {'sample': 'opsin_A', 'peaks': [(365, 14, 0.55), (515, 20, 0.85)], 'baseline': 0.05},
    {'sample': 'opsin_B', 'peaks': [(380, 16, 0.60), (540, 22, 0.80)], 'baseline': 0.04},
    {'sample': 'opsin_C', 'peaks': [(400, 18, 0.58), (560, 24, 0.78)], 'baseline': 0.04}
]

spectro_df = pd.DataFrame({'wavelength_nm': wavelength})
truth_records = []
for spec in spectro_specs:
    signal = np.zeros_like(wavelength, dtype=float) + spec['baseline']
    centers = []
    for (center, width, amplitude) in spec['peaks']:
        signal += gaussian(wavelength, center, width, amplitude)
        centers.append(center)
    signal += np.random.normal(scale=0.01, size=wavelength.size)
    spectro_df[spec['sample']] = signal
    truth_records.append({'sample': spec['sample'], 'peak_centers_nm': ';'.join(str(c) for c in centers)})

spectro_df.to_csv(DATA_DIR / 'spectro_scans.csv', index=False)
pd.DataFrame(truth_records).to_csv(DATA_DIR / 'spectro_truth.csv', index=False)

print('Synthetic data written to', DATA_DIR)
