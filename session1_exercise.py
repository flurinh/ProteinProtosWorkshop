#!/usr/bin/env python3
"""
================================================================================
SESSION 1 EXERCISE: Extracting Data from Paper Figures/Tables
================================================================================

Learning Objectives:
1. Use Claude to extract numerical data from published figures/tables
2. Understand the limitations of AI-based data extraction
3. Validate extracted data against original sources
4. Create visualizations to compare extracted vs. original data

This script is designed to work with whatever CSV data you provide!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# PART 1: LOADING AND INSPECTING AI-EXTRACTED DATA
# ==============================================================================
"""
EXERCISE STEP 1:
----------------
First, let's load the AI-extracted data and see what we're working with.
The script will adapt to whatever columns are present!
"""

# === CONFIGURE YOUR FILE PATHS HERE ===
ai_extracted_path = "materials/session1/solution/nature_figure_gpcr_b_gemini.csv"
corrected_path = "materials/session1/solution/nature_figure_gpcr_b_by_hand.csv"
original_figure_path = "materials/session1/nature_gpcr_b_abc_figure.jpg"

# Check if file exists
if not os.path.exists(ai_extracted_path):
    print(f"❌ File not found: {ai_extracted_path}")
    print("\nPlease update the 'ai_extracted_path' variable to point to your CSV file.")
    exit(1)

# Load the AI-extracted data
df_ai = pd.read_csv(ai_extracted_path)

print("=" * 70)
print("📊 AI-EXTRACTED DATA - INITIAL INSPECTION")
print("=" * 70)
print(f"\n📁 File: {ai_extracted_path}")
print(f"📐 Shape: {df_ai.shape[0]} rows × {df_ai.shape[1]} columns")
print(f"\n📋 Columns found:")
for i, col in enumerate(df_ai.columns, 1):
    print(f"   {i:2d}. {col}")

print("\n" + "-" * 70)
print("👀 FIRST 5 ROWS:")
print("-" * 70)
print(df_ai.head().to_string())

print("\n" + "-" * 70)
print("📈 DATA TYPES:")
print("-" * 70)
print(df_ai.dtypes)

# ==============================================================================
# PART 2: DETERMINE DATA TYPE AND ANALYZE ACCORDINGLY
# ==============================================================================
"""
The script will detect what kind of data you have and analyze it appropriately.
"""

# Check for dose-response curve data (expected columns)
dose_response_cols = {'Protein', 'Ligand', 'Log_Concentration', 'Response_Percent'}
has_dose_response = dose_response_cols.issubset(set(df_ai.columns))

# Check for receptor/residue table data
receptor_table_cols = {'Family', 'Receptor'}
has_receptor_table = receptor_table_cols.issubset(set(df_ai.columns))

print("\n" + "=" * 70)
print("🔍 DATA TYPE DETECTION")
print("=" * 70)

if has_dose_response:
    print("✅ Detected: DOSE-RESPONSE CURVE DATA")
    print("   Found columns: Protein, Ligand, Log_Concentration, Response_Percent")
    data_type = "dose_response"

elif has_receptor_table:
    print("✅ Detected: RECEPTOR/RESIDUE TABLE DATA")
    print("   Found columns: Family, Receptor, and residue information")
    data_type = "receptor_table"

else:
    print("ℹ️  Detected: GENERIC TABLE DATA")
    print("   Will perform basic analysis on available columns")
    data_type = "generic"

# ==============================================================================
# PART 3: ANALYZE BASED ON DATA TYPE
# ==============================================================================

if data_type == "dose_response":
    # ----- DOSE-RESPONSE ANALYSIS -----
    print("\n" + "=" * 70)
    print("📊 DOSE-RESPONSE DATA ANALYSIS")
    print("=" * 70)

    print(f"\n🧬 Unique proteins: {df_ai['Protein'].unique().tolist()}")
    print(f"💊 Unique ligands: {df_ai['Ligand'].unique().tolist()}")
    print(f"📏 Concentration range: {df_ai['Log_Concentration'].min():.1f} to {df_ai['Log_Concentration'].max():.1f}")
    print(f"📈 Response range: {df_ai['Response_Percent'].min():.1f}% to {df_ai['Response_Percent'].max():.1f}%")

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    for protein in df_ai['Protein'].unique():
        for ligand in df_ai['Ligand'].unique():
            subset = df_ai[(df_ai['Protein'] == protein) & (df_ai['Ligand'] == ligand)]
            if not subset.empty:
                subset = subset.sort_values('Log_Concentration')
                ax.plot(subset['Log_Concentration'], subset['Response_Percent'],
                        'o-', label=f"{protein} + {ligand}", markersize=6)

    ax.set_xlabel("log[Concentration] (M)", fontsize=12)
    ax.set_ylabel("Response (%)", fontsize=12)
    ax.set_title("Dose-Response Curves (from AI-extracted data)", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=100, color='black', linewidth=0.5, linestyle=':')

    plt.tight_layout()
    plt.savefig("extracted_dose_response.png", dpi=150, bbox_inches='tight')
    print("\n✅ Saved: extracted_dose_response.png")
    plt.show()


elif data_type == "receptor_table":
    # ----- RECEPTOR TABLE ANALYSIS -----
    print("\n" + "=" * 70)
    print("📊 RECEPTOR TABLE ANALYSIS")
    print("=" * 70)

    print(f"\n🏷️  Unique families: {df_ai['Family'].dropna().unique().tolist()}")
    print(f"🧬 Unique receptors: {df_ai['Receptor'].dropna().unique().tolist()}")

    # Count conserved vs non-conserved columns
    conserved_cols = [c for c in df_ai.columns if 'Conserved' in c and 'Non' not in c]
    non_conserved_cols = [c for c in df_ai.columns if 'Non-Conserved' in c]

    print(f"\n📊 Column breakdown:")
    print(f"   Conserved residue columns: {len(conserved_cols)}")
    print(f"   Non-conserved residue columns: {len(non_conserved_cols)}")

    # Try to reshape if it's a wide-format table
    print("\n" + "-" * 70)
    print("📋 DATA PREVIEW BY RECEPTOR:")
    print("-" * 70)

    for idx, row in df_ai.head(5).iterrows():
        receptor = row.get('Receptor', f'Row {idx}')
        family = row.get('Family', 'Unknown')
        print(f"\n{receptor} ({family}):")

        # Print conserved residues
        conserved = [str(row[c]) for c in conserved_cols if pd.notna(row[c]) and str(row[c]) != 'nan']
        if conserved:
            print(f"   Conserved: {', '.join(conserved[:6])}...")

        # Print non-conserved residues
        non_conserved = [str(row[c]) for c in non_conserved_cols if pd.notna(row[c]) and str(row[c]) != 'nan']
        if non_conserved:
            print(f"   Non-conserved: {', '.join(non_conserved[:6])}...")

    # Create a simple visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    # Count residues per receptor
    receptor_data = []
    for idx, row in df_ai.iterrows():
        receptor = row.get('Receptor', f'Row {idx}')
        n_conserved = sum(1 for c in conserved_cols if pd.notna(row[c]) and str(row[c]) != 'nan')
        n_non_conserved = sum(1 for c in non_conserved_cols if pd.notna(row[c]) and str(row[c]) != 'nan')
        receptor_data.append({'Receptor': receptor, 'Conserved': n_conserved, 'Non-Conserved': n_non_conserved})

    plot_df = pd.DataFrame(receptor_data)

    if len(plot_df) > 0:
        x = np.arange(len(plot_df))
        width = 0.35

        ax.bar(x - width / 2, plot_df['Conserved'], width, label='Conserved', color='#2ecc71')
        ax.bar(x + width / 2, plot_df['Non-Conserved'], width, label='Non-Conserved', color='#e74c3c')

        ax.set_xlabel('Receptor', fontsize=12)
        ax.set_ylabel('Number of Residues', fontsize=12)
        ax.set_title('Binding Pocket Residues by Receptor', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df['Receptor'], rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig("extracted_receptor_residues.png", dpi=150, bbox_inches='tight')
        print("\n✅ Saved: extracted_receptor_residues.png")
        plt.show()


else:
    # ----- GENERIC TABLE ANALYSIS -----
    print("\n" + "=" * 70)
    print("📊 GENERIC TABLE ANALYSIS")
    print("=" * 70)

    print("\n📈 Summary statistics for numeric columns:")
    numeric_cols = df_ai.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) > 0:
        print(df_ai[numeric_cols].describe())
    else:
        print("   No numeric columns found.")

    print("\n📝 Value counts for categorical columns:")
    categorical_cols = df_ai.select_dtypes(include=['object']).columns

    for col in categorical_cols[:3]:  # First 3 categorical columns
        print(f"\n   {col}:")
        print(df_ai[col].value_counts().head(5).to_string())

# ==============================================================================
# PART 4: COMPARE WITH MANUALLY CORRECTED DATA (if available)
# ==============================================================================

print("\n" + "=" * 70)
print("🔄 COMPARING WITH CORRECTED DATA")
print("=" * 70)

if os.path.exists(corrected_path):
    df_corrected = pd.read_csv(corrected_path)

    print(f"\n📁 Corrected file: {corrected_path}")
    print(f"📐 AI-extracted shape: {df_ai.shape}")
    print(f"📐 Corrected shape: {df_corrected.shape}")

    # Check if columns match
    ai_cols = set(df_ai.columns)
    corrected_cols = set(df_corrected.columns)

    if ai_cols == corrected_cols:
        print("✅ Column names match!")

        # Compare values
        if df_ai.shape == df_corrected.shape:
            # Find differences
            differences = (df_ai != df_corrected)
            n_different_cells = differences.sum().sum()
            total_cells = df_ai.shape[0] * df_ai.shape[1]

            print(f"\n📊 Comparison results:")
            print(f"   Total cells: {total_cells}")
            print(f"   Different cells: {n_different_cells}")
            print(f"   Match rate: {(1 - n_different_cells / total_cells) * 100:.1f}%")

            if n_different_cells > 0:
                print("\n⚠️  DIFFERENCES FOUND:")
                # Show which cells differ
                for col in df_ai.columns:
                    col_diff = df_ai[col] != df_corrected[col]
                    if col_diff.any():
                        print(f"\n   Column '{col}':")
                        diff_rows = df_ai[col_diff].index.tolist()
                        for row in diff_rows[:5]:  # Show first 5 differences
                            print(
                                f"      Row {row}: AI='{df_ai.loc[row, col]}' vs Corrected='{df_corrected.loc[row, col]}'")
                        if len(diff_rows) > 5:
                            print(f"      ... and {len(diff_rows) - 5} more differences")
        else:
            print(f"⚠️  Different number of rows - detailed comparison not possible")
    else:
        print("⚠️  Column names don't match!")
        print(f"   Only in AI: {ai_cols - corrected_cols}")
        print(f"   Only in corrected: {corrected_cols - ai_cols}")
else:
    print(f"\nℹ️  No corrected file found at: {corrected_path}")
    print("   To compare, create a manually-corrected version of your data.")

# ==============================================================================
# PART 5: KEY TAKEAWAYS
# ==============================================================================

print("\n" + "=" * 70)
print("🎓 KEY TAKEAWAYS")
print("=" * 70)
print("""
1. ✅ AI CAN extract data from figures and tables
   - Saves hours of manual work
   - Works on plots, tables, even handwritten data

2. ⚠️  BUT extraction isn't perfect
   - Check column names (may have duplicates like '.1', '.2')
   - Verify numerical values
   - Look for missing or extra data

3. 🔬 ALWAYS validate your extracted data
   - Reconstruct figures and compare visually
   - Compare against paper text where possible
   - When in doubt, verify key values manually

4. 📝 DOCUMENT your data provenance
   - Note: "Data extracted from Figure X using Claude"
   - Keep both raw extraction and corrected versions
""")

print("\n" + "=" * 70)
print("🎉 EXERCISE COMPLETE!")
print("=" * 70)
print("""
NEXT STEPS:
1. Review the extracted data carefully
2. Create your own visualization
3. Try extracting data from YOUR OWN paper!

Upload a figure to Claude and ask:
"Please extract the data from this figure as a CSV file"
""")