#!/usr/bin/env python3
"""
================================================================================
PYTHON FOR PROTEIN SCIENTISTS - QUICK REFERENCE CARD
================================================================================

Keep this file open while working through the workshop!
Each section shows syntax with protein science examples.

Tip: In Jupyter, type a function name followed by ? to see help:
     >>> pd.read_csv?
"""

# =============================================================================
# VARIABLES & TYPES
# =============================================================================

protein_name = "Lysozyme"     # str   - text in quotes
molecular_weight = 14.3       # float - decimal numbers
residue_count = 129           # int   - whole numbers
is_active = True              # bool  - True or False

# Check type with type()
type(protein_name)            # -> <class 'str'>
type(molecular_weight)        # -> <class 'float'>

# f-strings for formatting (the f before the quote is important!)
print(f"{protein_name} has {residue_count} residues")
print(f"MW = {molecular_weight:.2f} kDa")  # .2f = 2 decimal places


# =============================================================================
# MATH OPERATORS
# =============================================================================

# Basic math
5 + 3                         # 8     - addition
10 - 4                        # 6     - subtraction
6 * 7                         # 42    - multiplication
22 / 7                        # 3.14  - division (always gives float)
22 // 7                       # 3     - integer division (drops decimal)
22 % 7                        # 1     - modulo (remainder)
2 ** 10                       # 1024  - exponentiation

# Comparisons (return True or False)
5 == 5                        # True  - equal to (double equals!)
5 != 3                        # True  - not equal
pH = 7.4
7.0 <= pH <= 7.6              # True  - range check


# =============================================================================
# STRINGS
# =============================================================================

sequence = "MKTLLILAVVLAV"

# Indexing (Python counts from 0!)
sequence[0]                   # 'M'   - first character
sequence[-1]                  # 'V'   - last character
sequence[2]                   # 'T'   - third character

# Slicing [start:stop] - stop is NOT included!
sequence[:3]                  # 'MKT' - first 3 characters
sequence[3:6]                 # 'LLI' - characters 3, 4, 5
sequence[-4:]                 # 'VLAV' - last 4 characters
len(sequence)                 # 13    - length

# Useful methods
"  1LYZ  ".strip()            # '1LYZ'    - remove whitespace
"1lyz".upper()                # '1LYZ'    - uppercase
"MKTL".lower()                # 'mktl'    - lowercase
"1LYZ_A_chain".split("_")     # ['1LYZ', 'A', 'chain']
"-".join(["A", "B", "C"])     # 'A-B-C'


# =============================================================================
# LISTS - ordered, mutable collections
# =============================================================================

samples = ["WT", "K52A", "D101N"]

# Indexing (same as strings)
samples[0]                    # 'WT'      - first item
samples[-1]                   # 'D101N'   - last item
samples[1:3]                  # ['K52A', 'D101N']

# Modifying lists
samples.append("E35Q")        # Add to end: ['WT', 'K52A', 'D101N', 'E35Q']
samples.insert(0, "Ref")      # Insert at position 0
samples.remove("WT")          # Remove by value
del samples[0]                # Remove by index

# Useful functions
len(samples)                  # Number of items
sorted(samples)               # Return sorted copy (original unchanged)
samples.sort()                # Sort in-place (modifies original)
"WT" in samples               # True if "WT" is in list


# =============================================================================
# DICTIONARIES - key-value pairs
# =============================================================================

protein = {
    "name": "Lysozyme",
    "pdb_id": "1LYZ",
    "mw_kda": 14.3,
    "organism": "Gallus gallus"
}

# Accessing values
protein["name"]               # 'Lysozyme'
protein["mw_kda"]             # 14.3
protein.get("missing", None)  # None (default if key missing)

# Modifying
protein["resolution"] = 1.8   # Add new key
protein["mw_kda"] = 14.4      # Update existing key

# Getting keys and values
protein.keys()                # All keys
protein.values()              # All values
protein.items()               # Key-value pairs


# =============================================================================
# LOOPS
# =============================================================================

samples = ["WT", "K52A", "D101N"]

# For loop - iterate over items
for sample in samples:
    print(sample)

# For loop with index
for i, sample in enumerate(samples):
    print(f"{i}: {sample}")

# Loop over dictionary
for key, value in protein.items():
    print(f"{key} = {value}")

# Range-based loop
for i in range(5):            # 0, 1, 2, 3, 4
    print(i)

for i in range(2, 10, 2):     # 2, 4, 6, 8 (start, stop, step)
    print(i)


# =============================================================================
# FUNCTIONS
# =============================================================================

def calculate_mw(n_residues, avg_mw=110):
    """
    Estimate molecular weight from residue count.

    Parameters
    ----------
    n_residues : int
        Number of amino acid residues
    avg_mw : float
        Average residue mass (default 110 Da)

    Returns
    -------
    float
        Estimated MW in kDa
    """
    return n_residues * avg_mw / 1000

# Call the function
mw = calculate_mw(129)        # Uses default avg_mw
mw = calculate_mw(129, 111)   # Override avg_mw


# =============================================================================
# NUMPY - numerical arrays
# =============================================================================

import numpy as np

# Creating arrays
arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
zeros = np.zeros(10)          # Array of 10 zeros
ones = np.ones(10)            # Array of 10 ones

# Ranges
np.arange(0, 10, 2)           # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)          # [0, 0.25, 0.5, 0.75, 1.0]

# Element-wise operations (no loops needed!)
arr * 2                       # [0.2, 0.4, 0.6, 0.8, 1.0]
arr + 1                       # [1.1, 1.2, 1.3, 1.4, 1.5]
np.sqrt(arr)                  # Square root of each element

# Statistics
np.mean(arr)                  # Mean
np.std(arr)                   # Standard deviation
np.min(arr), np.max(arr)      # Min and max
np.median(arr)                # Median


# =============================================================================
# PANDAS - tabular data
# =============================================================================

import pandas as pd

# Load data
df = pd.read_csv("data.csv")

# Inspect data
df.head()                     # First 5 rows
df.tail()                     # Last 5 rows
df.shape                      # (rows, columns)
df.columns                    # Column names
df.info()                     # Column types and memory
df.describe()                 # Statistics for numeric columns

# Select columns
df["Sample"]                  # Single column (Series)
df[["Sample", "OD280"]]       # Multiple columns (DataFrame)

# Select rows
df.iloc[0]                    # First row by position
df.iloc[0:5]                  # Rows 0-4
df.loc[0, "Sample"]           # Row 0, column "Sample"

# Filter rows
df[df["OD280"] > 0.5]         # Rows where OD280 > 0.5
df[df["Sample"] == "WT"]      # Rows where Sample is "WT"
df[df["Sample"].isin(["WT", "K52A"])]  # Multiple values

# Aggregate
df.groupby("Sample").mean()   # Mean by group
df["OD280"].value_counts()    # Count occurrences


# =============================================================================
# COMMON ISSUES
# =============================================================================

# 1. Python counts from 0, not 1!
#    samples[1] is the SECOND item

# 2. = assigns, == compares
#    x = 5     # set x to 5
#    x == 5    # check if x equals 5

# 3. Slicing excludes the end index
#    seq[0:3] gives indices 0, 1, 2 (NOT 3)

# 4. Indentation matters!
#    Code inside loops/functions must be indented (4 spaces)

# 5. Don't forget to import libraries
#    import pandas as pd  # before using pd.read_csv()

# 6. Strings are immutable
#    seq[0] = "A"  # ERROR! Can't modify strings
#    seq = "A" + seq[1:]  # Create new string instead


