# ProteinProtos Workshop

> **Python for Protein Scientists: From Basics to Structural Data Analysis**

A hands-on workshop teaching Python fundamentals, data analysis techniques, and structural biology workflows relevant to protein science research.

---

## Workshop Overview

| Session | Duration | Focus |
|:--------|:--------:|:------|
| **Session 1** | ~1.5 hrs | Python basics, pandas, AI-assisted data extraction |
| **Session 2** | ~1.5 hrs | Signal processing, curve fitting, experimental data analysis |
| **Session 3** | ~1.5 hrs | Structural biology, PDB analysis, GRN mapping |

---

## Quick Start

```bash
# 1. Clone or download this repository
cd ProteinProtosWorkshop

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook

# 4. Open session1.ipynb and begin!
```

---

## Repository Structure

```
ProteinProtosWorkshop/
│
├── session1.ipynb                 # Python fundamentals
├── session2.ipynb                 # Experimental data analysis
├── session3.ipynb                 # Structural biology workflows
│
├── python_cheatsheet.py           # Quick reference card
├── requirements.txt               # Python dependencies
│
├── materials/
│   ├── session1/
│   │   ├── data/                  # Practice datasets
│   │   └── solution/              # AI extraction reference files
│   │
│   ├── session2/
│   │   └── data/                  # FSEC, dose-response, spectroscopy data
│   │
│   └── session3/
│       └── data/                  # PDB structures, GRN mappings, mutant data
│
└── protos/                        # Structural biology processing library
```

---

## Session Details

### Session 1: Python Basics

**Target audience:** Complete beginners with no programming experience

**Topics:**
- Jupyter notebook orientation
- Variables and data types (`str`, `int`, `float`, `bool`)
- Operators and comparisons
- Strings and sequences
- Lists and dictionaries
- Loops and control flow
- Functions
- Pandas DataFrames
- AI data extraction validation

**Outcome:** Load, inspect, and manipulate tabular data with pandas.

---

### Session 2: Experimental Data Analysis

**Prerequisites:** Session 1 or equivalent Python/pandas knowledge

| Section | Technique | Application |
|:--------|:----------|:------------|
| FSEC | Baseline correction, peak detection | Chromatography QC |
| Dose-Response | 4-parameter logistic fitting | GPCR pharmacology |
| Growth Curves | Logistic growth modeling | Expression optimization |
| Spectroscopy | Smoothing, peak finding | Protein characterization |

**Outcome:** Apply signal processing and curve fitting to real experimental data.

---

### Session 3: Structural Biology & Data Management

**Prerequisites:** Sessions 1-2 or equivalent experience

| Section | Technique | Application |
|:--------|:----------|:------------|
| Structure Loading | PDB/mmCIF parsing | Coordinate extraction |
| Ligand Analysis | Distance calculations | Binding site identification |
| GRN Mapping | Generic residue numbering | Cross-receptor comparison |
| Structure-Function | Data joining | Mutation impact analysis |

**Case Study:** PCO371/PTH1R binding (Zhao et al. Nature 2023)

**Outcome:** Extract structural data, calculate interactions, and compare with published findings.

---

## For Instructors

Teaching guides and solution files are available on the `teacher` branch:

```bash
git checkout notes
```

The `teacher/` folder contains:
- Teaching guides for each session
- Solution scripts and utilities
- Reference implementations

---

## Dependencies

**Core:**
- Python 3.8+
- numpy
- pandas
- matplotlib
- scipy
- jupyter

**Session 3 (structural biology):**
- biopython

**Optional:**
- torch, torchvision (crystal classifier)
- rdkit (ligand analysis)

---

## Contributing

This workshop is actively being developed. Feedback and contributions are welcome.
