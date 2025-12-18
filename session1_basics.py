#!/usr/bin/env python3
"""
================================================================================
SESSION 1: Python Fundamentals for Structural Biologists
================================================================================

A gentle introduction to Python for experimental biologists.
Run this file cell-by-cell in Jupyter/Spyder, or copy blocks to try interactively.

No prior coding experience required!
"""

# ==============================================================================
# PART 0: YOUR FIRST PYTHON COMMANDS
# ==============================================================================
# In Python, the '#' symbol marks a comment - Python ignores everything after it.
# Comments help you (and others) understand what your code does.

# Let's store a protein name in a "variable" - think of it as a labeled container
protein_name = "Lysozyme"
print("Our protein:", protein_name)

# The print() function displays output - you'll use it constantly to check results

# Curious what something is? Use type() to find out:
print("Type of protein_name:", type(protein_name))  # It's a string (text)

# Need help with any function? Use help():
# help(print)  # Uncomment this line to see documentation


# ==============================================================================
# PART 1: IMPORTING PACKAGES (Your Scientific Toolbox)
# ==============================================================================
# Python's power comes from "packages" - pre-written code for specific tasks.
# Think of them like specialized lab equipment you bring in when needed.

# Three ways to import:

# Way 1: Import the whole package
import math  # Mathematical functions (sin, cos, log, etc.)

# Way 2: Import with a nickname (alias) - very common for long package names
import pandas as pd      # Data analysis - like Excel on steroids
import numpy as np       # Numerical computing - handles arrays of numbers

# Way 3: Import specific items from a package
from math import pi, sqrt

print("Value of pi:", pi)
print("Square root of 2:", sqrt(2))
print("Pandas version:", pd.__version__)


# ==============================================================================
# PART 2: WORKING WITH YOUR FILES
# ==============================================================================
# You'll often need to load data files. First, know where Python is "looking":

import os

# Where is Python currently looking for files?
print("Current working directory:", os.getcwd())

# To change directories (like 'cd' in terminal):
# os.chdir("/path/to/your/data")  # Uncomment and modify as needed

# List files in current directory:
# print(os.listdir())  # Uncomment to see files


# ==============================================================================
# PART 3: BASIC MATH OPERATIONS
# ==============================================================================
# Python works like a powerful calculator. Essential for data processing!

# --- Arithmetic ---
print("\n--- Arithmetic Operations ---")
print("Addition: 102 + 37 =", 102 + 37)
print("Subtraction: 102 - 37 =", 102 - 37)
print("Multiplication: 4 × 6 =", 4 * 6)
print("Division: 22 / 7 =", 22 / 7)           # Regular division (gives decimal)
print("Integer division: 22 // 7 =", 22 // 7)  # Drops the decimal part
print("Power: 3^4 =", 3 ** 4)                  # Exponentiation
print("Remainder: 22 % 7 =", 22 % 7)           # Modulo (remainder after division)

# --- Real-world example: Molecular weight calculation ---
mw_glycine = 75.07      # Daltons
num_residues = 129      # Lysozyme has ~129 residues
water_loss = 18.015     # Water lost per peptide bond

# Rough MW estimate (actual MW depends on exact sequence)
estimated_mw = (mw_glycine * num_residues) - (water_loss * (num_residues - 1))
print(f"\nEstimated MW of {num_residues}-residue protein: {estimated_mw:.1f} Da")

# --- Comparisons (return True or False) ---
print("\n--- Comparisons ---")
print("Is 3 equal to 3?", 3 == 3)       # Double equals for comparison!
print("Is 3 not equal to 1?", 3 != 1)
print("Is pH 7.4 > 7.0?", 7.4 > 7.0)
print("Is 4°C ≤ 4?", 4 <= 4)

# --- Logical operators (combining conditions) ---
print("\n--- Logical Operators ---")
pH = 7.4
temp_celsius = 25
# Is this physiological conditions?
physiological = (7.0 <= pH <= 7.6) and (20 <= temp_celsius <= 40)
print(f"pH={pH}, Temp={temp_celsius}°C → Physiological?", physiological)


# ==============================================================================
# PART 4: LISTS - Ordered Collections of Items
# ==============================================================================
# Lists store multiple items in order. Use square brackets [].
# Perfect for: sample names, wavelengths, time points, etc.

print("\n--- Lists ---")

# A list of amino acid codes
amino_acids = ["Ala", "Cys", "Asp", "Glu", "Phe"]
print("Amino acids:", amino_acids)

# Accessing items by position (INDEX) - Python counts from 0!
print("First amino acid (index 0):", amino_acids[0])   # Ala
print("Third amino acid (index 2):", amino_acids[2])   # Asp
print("Last amino acid (index -1):", amino_acids[-1])  # Phe (negative = from end)

# SLICING: Get a range of items [start:stop] - stop is NOT included
print("Items 1-3 (indices 1,2):", amino_acids[1:3])    # ['Cys', 'Asp']
print("First three:", amino_acids[:3])                  # ['Ala', 'Cys', 'Asp']
print("From index 2 onward:", amino_acids[2:])         # ['Asp', 'Glu', 'Phe']

# Combining lists
hydrophobic = ["Ala", "Val", "Leu"]
charged = ["Asp", "Glu", "Lys"]
combined = hydrophobic + charged
print("Combined list:", combined)

# Useful list operations
wavelengths = [280, 260, 230, 340, 214]
print("Sorted wavelengths:", sorted(wavelengths))
print("How many wavelengths?", len(wavelengths))
print("Is 280 in the list?", 280 in wavelengths)


# ==============================================================================
# PART 5: DICTIONARIES - Labeled Data Storage
# ==============================================================================
# Dictionaries store key-value pairs. Use curly braces {}.
# Think: lookup tables, sample metadata, protein properties.

print("\n--- Dictionaries ---")

# Extinction coefficients for common chromophores (M⁻¹ cm⁻¹ at 280 nm)
extinction_coeff = {
    "Trp": 5500,
    "Tyr": 1490,
    "Cystine": 125    # Disulfide bond
}

print("Extinction coefficients:", extinction_coeff)
print("ε for Tryptophan:", extinction_coeff["Trp"], "M⁻¹ cm⁻¹")

# A more complex example: protein metadata
lysozyme = {
    "name": "Hen egg-white lysozyme",
    "pdb_id": "1LYZ",
    "residues": 129,
    "mw_kda": 14.3,
    "disulfides": 4,
    "trp_count": 6,
    "tyr_count": 3
}

print(f"\nProtein: {lysozyme['name']}")
print(f"PDB ID: {lysozyme['pdb_id']}")
print(f"Molecular weight: {lysozyme['mw_kda']} kDa")

# Calculate extinction coefficient at 280 nm
epsilon_280 = (lysozyme["trp_count"] * 5500 +
               lysozyme["tyr_count"] * 1490 +
               lysozyme["disulfides"] * 125)
print(f"Calculated ε₂₈₀: {epsilon_280} M⁻¹ cm⁻¹")


# ==============================================================================
# PART 6: STRINGS - Working with Text
# ==============================================================================
# Strings are text data. Essential for: sequences, file names, labels.

print("\n--- Strings ---")

# Creating strings (single or double quotes both work)
sequence = "KVFGRCELAAAMKR"
print("Sequence:", sequence)

# Strings work like lists - you can slice them!
print("First 5 residues:", sequence[:5])
print("Last 3 residues:", sequence[-3:])
print("Sequence length:", len(sequence))

# Including quotes in strings
description = 'The "active site" is crucial'  # Use opposite quote type
print(description)

# Or use escape character \
description2 = "The \"active site\" is crucial"
print(description2)

# Useful string methods
pdb_id = "  1lyz  "
print("Uppercase:", pdb_id.upper())           # "  1LYZ  "
print("Stripped whitespace:", pdb_id.strip()) # "1lyz"
print("Clean PDB ID:", pdb_id.strip().upper()) # "1LYZ"

# Splitting strings (great for parsing!)
header = "1LYZ_A_chain|Lysozyme|Gallus gallus"
parts = header.split("|")
print("Split by '|':", parts)

# Joining strings
residues = ["ALA", "GLY", "SER"]
joined = "-".join(residues)
print("Joined with '-':", joined)


# ==============================================================================
# PART 7: NUMPY - Numerical Computing
# ==============================================================================
# NumPy is essential for scientific data. Arrays are like lists but:
# - Much faster for math operations
# - Apply operations to ALL elements at once
# - Required by most scientific packages

print("\n--- NumPy Arrays ---")

import numpy as np

# Create arrays from lists
absorbance_values = np.array([0.123, 0.456, 0.789, 0.234, 0.567])
print("Absorbance readings:", absorbance_values)

# Math on entire arrays at once (no loops needed!)
# Beer-Lambert: A = ε × c × l  →  c = A / (ε × l)
epsilon = 38000  # M⁻¹ cm⁻¹
path_length = 1  # cm
concentrations_M = absorbance_values / (epsilon * path_length)
concentrations_uM = concentrations_M * 1e6
print("Concentrations (μM):", np.round(concentrations_uM, 2))

# Create sequences of numbers
time_points = np.arange(0, 60, 5)  # 0 to 60 in steps of 5
print("Time points (min):", time_points)

wavelengths = np.linspace(200, 400, 5)  # 5 evenly spaced points from 200-400
print("Wavelengths (nm):", wavelengths)

# Statistical functions - crucial for data analysis!
data = np.array([14.2, 14.5, 14.1, 14.8, 14.3, 14.4])
print(f"\nMolecular weight measurements: {data} kDa")
print(f"  Mean: {np.mean(data):.2f} kDa")
print(f"  Std Dev: {np.std(data):.2f} kDa")
print(f"  Min: {np.min(data):.2f} kDa")
print(f"  Max: {np.max(data):.2f} kDa")
print(f"  Median: {np.median(data):.2f} kDa")


# ==============================================================================
# PART 8: PANDAS - Data Analysis (Your New Best Friend)
# ==============================================================================
# Pandas handles tabular data (like Excel/CSV files).
# The DataFrame is like a spreadsheet with labeled rows and columns.

print("\n--- Pandas DataFrames ---")

import pandas as pd

# Create a DataFrame from a dictionary (each key becomes a column)
experiment_data = {
    "Sample": ["WT", "Mutant_A", "Mutant_B", "Control"],
    "A280": [0.523, 0.612, 0.498, 0.052],
    "Tm_celsius": [62.3, 58.1, 64.7, None],     # None = missing data
    "Active": [True, True, True, False]
}

df = pd.DataFrame(experiment_data)
print("Our experiment data:")
print(df)

# Basic DataFrame info
print(f"\nShape (rows, cols): {df.shape}")
print(f"Column names: {list(df.columns)}")

# --- SELECTING DATA ---
print("\n--- Selecting Data ---")

# Select a single column (returns a Series)
print("A280 values:")
print(df["A280"])

# Select multiple columns (returns a DataFrame)
print("\nSample and Tm:")
print(df[["Sample", "Tm_celsius"]])

# Select rows by position with .iloc[row, column]
print("\nFirst row (index 0):")
print(df.iloc[0])

print("\nRows 0-2, columns 0-2:")
print(df.iloc[0:2, 0:2])

# Select by label with .loc[row_label, column_name]
print("\nA280 of row 1:")
print(df.loc[1, "A280"])

# --- FILTERING DATA ---
print("\n--- Filtering ---")

# Find samples with Tm > 60°C
stable_proteins = df[df["Tm_celsius"] > 60]
print("Thermostable samples (Tm > 60°C):")
print(stable_proteins)

# Find active samples only
active_samples = df[df["Active"] == True]
print("\nActive samples:")
print(active_samples)


# ==============================================================================
# PART 9: LOADING REAL DATA FILES
# ==============================================================================
# In practice, you'll load data from files. Here are the common patterns:

print("\n--- Loading Data from Files ---")

# CSV files (most common for exported instrument data)
# df = pd.read_csv("my_data.csv")

# Excel files
# df = pd.read_excel("my_data.xlsx", sheet_name="Sheet1")

# Tab-separated files (common from some instruments)
# df = pd.read_csv("my_data.txt", sep="\t")

# Example: Create a sample CSV and read it back
sample_csv_content = """Wavelength_nm,Absorbance,Sample
280,0.523,Lysozyme
260,0.312,Lysozyme
280,0.612,BSA
260,0.287,BSA"""

# Save to file
with open("sample_uv_data.csv", "w") as f:
    f.write(sample_csv_content)

# Read it back
uv_data = pd.read_csv("sample_uv_data.csv")
print("Loaded UV-Vis data:")
print(uv_data)

# Quick data exploration
print("\nBasic statistics:")
print(uv_data.describe())

# Clean up the demo file
os.remove("sample_uv_data.csv")


# ==============================================================================
# QUICK REFERENCE CARD
# ==============================================================================
"""
ESSENTIAL PYTHON FOR BIOLOGISTS - CHEAT SHEET

VARIABLES & TYPES:
    protein = "Lysozyme"      # String (text)
    mw = 14.3                 # Float (decimal number)  
    residues = 129            # Integer (whole number)
    is_active = True          # Boolean (True/False)
    type(protein)             # Check the type

LISTS (ordered, mutable):
    samples = ["A", "B", "C"]
    samples[0]                # First item → "A"
    samples[-1]               # Last item → "C"
    samples[1:3]              # Slice → ["B", "C"]
    len(samples)              # Length → 3
    samples.append("D")       # Add item

DICTIONARIES (key-value pairs):
    protein = {"name": "Lysozyme", "mw": 14.3}
    protein["name"]           # Access value → "Lysozyme"
    protein.keys()            # All keys
    protein.values()          # All values

NUMPY (numerical arrays):
    import numpy as np
    arr = np.array([1, 2, 3])
    arr * 2                   # Element-wise → [2, 4, 6]
    np.mean(arr)              # Statistics
    np.arange(0, 10, 2)       # Range → [0, 2, 4, 6, 8]

PANDAS (data tables):
    import pandas as pd
    df = pd.read_csv("file.csv")
    df.head()                 # First 5 rows
    df["column"]              # Select column
    df.iloc[0]                # Select row by position
    df[df["col"] > 5]         # Filter rows

COMMON MISTAKES:
    ✗ samples[1:3]  includes index 1 and 2, NOT 3
    ✗ = is assignment, == is comparison
    ✗ Python counts from 0, not 1!
    ✗ Don't forget to import packages before using them
"""

print("\n" + "="*60)
print("🎉 Session 1 Complete!")
print("="*60)
print("""
Next steps to practice:
1. Try modifying the protein examples with your own data
2. Create a list of your sample names
3. Build a dictionary with your experiment metadata
4. Challenge: Calculate concentration from A280 for your protein

Questions? That's normal! Programming is learned by doing (and Googling).
""")