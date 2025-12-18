"""Utility functions for Session 1 warmup notebook."""

from __future__ import annotations
import numpy as np

def snip_baseline(signal, window_size=100):
    s_lls = np.log(np.log(np.sqrt(signal + 1) + 1) + 1)
    s_lls_filt = s_lls.copy()
    for m in range(1, window_size + 1):
        for i in range(m, len(s_lls_filt) - m):
            s_lls_filt[i] = min(s_lls_filt[i], (s_lls_filt[i - m] + s_lls_filt[i + m]) / 2)
    baseline = (np.exp(np.exp(s_lls_filt) - 1) - 1)**2 - 1
    corrected = signal - baseline
    return corrected, baseline

import pandas as pd
from pathlib import Path

def load_extracted_data(ai_path=None, corrected_path=None):
    base = Path('materials/session1/solution')
    ai_file = Path(ai_path) if ai_path else base / 'nature_figure_gpcr_b_gemini.csv'
    corrected_file = Path(corrected_path) if corrected_path else base / 'nature_figure_gpcr_b_by_hand.csv'
    df_ai = pd.read_csv(ai_file)
    df_corrected = pd.read_csv(corrected_file) if corrected_file.exists() else None
    return df_ai, df_corrected
