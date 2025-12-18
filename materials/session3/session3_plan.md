# Session 3: ProtOS - Protein Data Management & Rational Design

## Overview

**Duration**: ~1.5 hours
**Prerequisites**: Sessions 1 & 2 (Python basics, data analysis concepts)
**Theme**: From experimental data to computational protein/ligand design

### Learning Objectives
By the end of this session, participants will understand:
1. How ProtOS manages protein data (structures, sequences, ligands) with zero configuration
2. The relationship between different data types and processors
3. How to connect experimental data (from sessions 1&2) to structural analysis
4. How to prepare inputs for structure prediction (Boltz-2) for rational design

### Session 4 Preview
This session prepares users for Session 4 (ProtOS-MCP), where an LLM interface can orchestrate these workflows automatically.

---

## Notebook Structure

### Part 0: Introduction & Framework Overview (10 min)

#### 0.1 Display Overview Diagram
```python
# Display the ProtOS architecture diagram
from IPython.display import Image, display
display(Image(filename='../../protos/resources/overview2.png', width=800))
```

**Key Points to Explain:**
- **Zero-config philosophy**: No environment variables, no manual path management
- **Processor ecosystem**: Each data type has a dedicated processor
- **Data flow**: Structure → Sequence → GRN → Properties → Embeddings → Models
- **Universal entity registry**: All entities tracked across formats with human-readable names

#### 0.2 The PCO371/Class B GPCR Story
Brief recap of the Nature paper:
- PCO371: First orally available small-molecule agonist for PTH1R
- Unique binding site at receptor-G protein interface
- Conserved pocket across Class B GPCRs
- **Our goal**: Use ProtOS to analyze this data and design improved variants

---

### Part 1: Easy Usage - Download & Explore Structures (15 min)

#### 1.1 Initialize ProtOS
```python
# One-line initialization
import protos
from protos.processing.structure import StructureProcessor
from protos.processing.sequence import SequenceProcessor

# Initialize processor - that's it! Zero config.
struct_proc = StructureProcessor(name="gpcr_workshop")
```

#### 1.2 Download Class B GPCR Structures
```python
# PCO371-PTH1R-Gs complex from the Nature paper
from protos.io.ingest.download_structures import download_protein_structures

class_b_structures = {
    '8JR9': 'PCO371-PTH1R-Gs (Nature paper)',
    '6NBF': 'LA-PTH-PTH1R-Gs (reference)',
    '7LCI': 'GLP1R-Gs active state',
    '6X18': 'PTH2R-Gs',
}

# Download all - automatically registered!
pdb_ids = list(class_b_structures.keys())
downloaded = download_protein_structures(pdb_ids)
print(f"Downloaded {len(downloaded)} structures")
```

#### 1.3 Basic Structure Inspection
```python
# Load a structure by name (not path!)
structure = struct_proc.load_entity('8jr9')  # PCO371-bound PTH1R

# Inspect the data
print(f"Total atoms: {len(structure)}")
print(f"Chains: {structure['auth_chain_id'].unique()}")
print(f"Ligands: {structure[structure['record_type'] == 'HETATM']['res_name3l'].unique()}")
```

#### 1.4 Create a Dataset
```python
# Group structures into a named dataset
struct_proc.create_dataset(
    name="class_b_gpcr_active",
    entities=['8jr9', '6nbf', '7lci', '6x18'],
    metadata={
        "description": "Active Class B GPCR structures",
        "source": "PDB",
        "workshop": "session3"
    }
)

# Load entire dataset
struct_proc.load_dataset("class_b_gpcr_active")
print(f"Loaded {len(struct_proc.pdb_ids)} structures")
```

**Takeaway**: With 5 lines of code, we downloaded, registered, and organized 4 protein structures.

---

### Part 2: Medium Usage - Sequence Extraction & Alignment (20 min)

#### 2.1 Extract Sequences from Structures
```python
# Extract sequences from all structures in dataset
sequences = struct_proc.collect_chain_sequences(['8jr9', '6nbf', '7lci', '6x18'])

# Display extracted sequences
for pdb_id, chains in sequences.items():
    for chain_id, data in chains.items():
        print(f"{pdb_id}_{chain_id}: {len(data['sequence'])} residues")
        print(f"  First 50: {data['sequence'][:50]}...")
```

#### 2.2 Save Sequences to SequenceProcessor
```python
seq_proc = SequenceProcessor(name="gpcr_workshop")

# Register receptor sequences (chain R typically)
receptor_sequences = {}
for pdb_id, chains in sequences.items():
    for chain_id, data in chains.items():
        if len(data['sequence']) > 300:  # Likely receptor chain
            name = f"{pdb_id}_receptor"
            receptor_sequences[name] = data['sequence']
            seq_proc.save_entity(name, data['sequence'])

print(f"Registered {len(receptor_sequences)} receptor sequences")
```

#### 2.3 Create Sequence Dataset
```python
seq_proc.create_dataset(
    name="class_b_receptors",
    entities=list(receptor_sequences.keys()),
    metadata={"family": "Class B GPCR", "type": "receptor"}
)
```

#### 2.4 Multiple Sequence Alignment (MSA)
```python
# Perform alignment using built-in tools
from protos.processing.sequence import align_blosum62, init_aligner, format_alignment

aligner = init_aligner()
seq_list = list(receptor_sequences.values())
names = list(receptor_sequences.keys())

# Pairwise alignment example
alignment = align_blosum62(seq_list[0], seq_list[1], aligner)
print(format_alignment(alignment))
```

---

### Part 3: Medium-Advanced - GRN (Generic Residue Numbering) (20 min)

#### 3.1 Why GRN Matters
```markdown
**The Problem**: Position 415 in PTH1R is different from position 415 in GLP1R
**The Solution**: GRN provides universal numbering (e.g., 6.47b = P415 in PTH1R)

From the Nature paper:
- P415^6.47b in PTH1R is CRITICAL for PCO371 binding
- PTH2R has L370^6.47b (Leu instead of Pro) → no PCO371 response
- L370F mutation in PTH2R restores PCO371 activity!
```

#### 3.2 Load GRN Reference
```python
from protos.processing.grn import GRNProcessor

grn_proc = GRNProcessor(name="gpcr_workshop")

# Load Class B GPCR reference (or create from session 1 data)
# The binding pocket table from session 1 IS a GRN table!
```

#### 3.3 Connect Session 1 Data - Binding Pocket Residues
```python
import pandas as pd

# Load the data we extracted/validated in session 1
binding_pocket = pd.read_csv('../../materials/session1/solution/nature_figure_gpcr_b_by_hand.csv')
print(binding_pocket)

# This table contains GRN positions for the PCO371 binding pocket:
# 2x46, 2x50, 3x47, 3x50, 6x43-6x49, 7x56, 7x57, 8x47, 8x49
```

#### 3.4 Map GRN to Structure
```python
# Key binding pocket residues from the paper
pco371_pocket_grns = {
    '2.46b': 'R219',   # Hydrogen bond with PCO371
    '2.50b': 'H223',   # Part of polar network
    '3.47b': 'I299',   # Hydrophobic contact
    '3.50b': 'E302',   # Sulfonamide interaction
    '6.45b': 'L413',   # Hydrophobic pocket
    '6.47b': 'P415',   # CRITICAL - defines selectivity
    '6.49b': 'F417',   # Pi stacking
    '7.56b': 'I458',   # Hydrophobic contact
    '7.57b': 'Y459',   # Pi stacking with PCO371
}

# Display on structure (conceptual)
for grn, residue in pco371_pocket_grns.items():
    print(f"GRN {grn}: {residue} in PTH1R")
```

---

### Part 4: Advanced - Connect Experimental Data (15 min)

#### 4.1 Load Session 2 Selectivity Data
```python
# Load the dose-response data from session 2
selectivity_df = pd.read_csv('../../materials/session2/data/nature_figure_gpcr_b_selectivity.csv')

# This data shows:
# - WT PTH1R responds to PCO371 (EC50 ~200 nM)
# - PTH1R(P415A) is DEAD - no PCO371 response
# - PTH2R(L370F) GAINS PCO371 response
# - GLP1R-2M (2 mutations) gains partial PCO371 response
```

#### 4.2 Mutant Activity Table (from paper Extended Data Table 2)
```python
# Create the mutant activity table provided by user
mutant_data = pd.DataFrame({
    'Mutant': ['WT', 'R219A', 'H223A', 'L226A', 'I299A', 'E302A', 'L306A',
               'K408A', 'V412A', 'L413A', 'M414A', 'P415A', 'L416A', 'F417A',
               'I458A', 'Y459A', 'C462A', 'N463A', 'G464A', 'E465A'],
    'pEC50': [6.74, 6.23, 6.31, 6.07, 6.09, 5.85, 6.78, 6.63, 7.49, 6.01,
              7.34, None, 7.01, 7.14, 6.51, 6.38, 6.32, 6.63, 7.07, 6.76],
    'PCO371_pct_WT': [100, 46.17, 40.93, 32.35, 60.52, 47.56, 78.32, 77.45,
                      97.10, 72.94, 88.78, 96.36, 75.75, 89.05, 80.93, 94.38,
                      86.41, 103.10, 100.36, 90.05]
})

# P415A is the only mutation that COMPLETELY abolishes activity
# V412A and M414A ENHANCE activity!
```

#### 4.3 Integrate Property Data into ProtOS
```python
from protos.processing.property import PropertyProcessor

prop_proc = PropertyProcessor(name="gpcr_workshop")

# Import the mutant activity as a property dataset
prop_proc.import_properties(
    name="pth1r_mutant_activity",
    data=mutant_data,
    key_column="Mutant"
)

# Now we can query: which mutations improve activity?
enhanced = mutant_data[mutant_data['pEC50'] > 7.0]
print("Mutations that ENHANCE PCO371 potency:")
print(enhanced[['Mutant', 'pEC50']])
```

---

### Part 5: Advanced - Rational Design with Boltz-2 (20 min)

#### 5.1 The Design Strategy
```markdown
**Goal 1: Design an improved receptor**
- Based on the mutant data, V412A, M414A, L416A, F417A enhance PCO371 activity
- Combine beneficial mutations for super-responder

**Goal 2: Design a new ligand**
- Inverse approach: modify PCO371 to work on GLP1R
- Target the non-conserved positions: 3x47, 6x43-6x47

**Goal 3: Predict structures with Boltz-2**
- Validate our designs computationally before experiments
```

#### 5.2 Initialize ModelManager
```python
from protos.models.model_manager import ModelManager

manager = ModelManager()

# List available models
print("Available models:")
for model in manager.list_models():
    print(f"  - {model}")
```

#### 5.3 Design Enhanced PTH1R Variant
```python
# Get WT PTH1R sequence
pth1r_seq = seq_proc.load_entity("8jr9_receptor")

# Define beneficial mutations (from paper data)
beneficial_mutations = [
    {"position": 412, "original": "V", "mutant": "A", "name": "V412A"},  # pEC50: 7.49
    {"position": 414, "original": "M", "mutant": "A", "name": "M414A"},  # pEC50: 7.34
]

# Create mutant sequences
from protos.processing.sequence.mutations import apply_mutations

for mutation in beneficial_mutations:
    mutant_seq = apply_mutations(pth1r_seq, [mutation])
    seq_proc.save_entity(f"PTH1R_{mutation['name']}", mutant_seq)
    print(f"Created: PTH1R_{mutation['name']}")

# Create double mutant
double_mutant = apply_mutations(pth1r_seq, beneficial_mutations)
seq_proc.save_entity("PTH1R_V412A_M414A", double_mutant)
print("Created: PTH1R_V412A_M414A (enhanced PCO371 responder)")
```

#### 5.4 Prepare Boltz-2 Submission
```python
# Prepare input for Boltz-2 structure prediction
config = {
    "recycling": 5,
    "num_samples": 3,
    "crop_size": 512,
    "device": "cuda"
}

# Prepare wild-type prediction
wt_input = manager.prepare_input(
    model_name="boltz2",
    entity_name="8jr9_receptor",
    entity_format="sequence",
    config=config
)

# Prepare mutant predictions
mutant_inputs = []
for name in ["PTH1R_V412A", "PTH1R_M414A", "PTH1R_V412A_M414A"]:
    mutant_input = manager.prepare_input(
        model_name="boltz2",
        entity_name=name,
        entity_format="sequence",
        config=config
    )
    mutant_inputs.append(mutant_input)

print(f"Prepared {1 + len(mutant_inputs)} Boltz-2 predictions")
```

#### 5.5 Generate Submission Scripts
```python
# Get the command to run Boltz-2
print("Boltz-2 commands to run:")
print(f"\n# Wild-type:")
print(f"  {' '.join(wt_input.get_command())}")

print(f"\n# Mutants:")
for inp in mutant_inputs:
    print(f"  {' '.join(inp.get_command())}")
```

#### 5.6 Inverse Design: Engineer GLP1R to Respond to PCO371
```python
# From the paper: GLP1R-2M (two mutations) gains PCO371 response
# The key positions are at 3.47 and 6.47

glp1r_seq = seq_proc.load_entity("7lci_receptor")

# Mutations to convert GLP1R to PCO371-responsive
glp1r_mutations = [
    {"position": "L3.47I", "note": "Match PTH1R 3x47"},
    {"position": "P6.47", "note": "Already conserved"},
    # Additional mutations from paper...
]

# This creates a GLP1R variant that might respond to PCO371
# allowing treatment of diabetes with oral drugs!
```

---

### Part 6: Summary & Connection to Session 4 (10 min)

#### 6.1 What We Accomplished
```python
# Summary of entities created
print("\n=== ProtOS Data Created ===")
print("\nStructure Datasets:")
for ds in struct_proc.list_datasets():
    print(f"  - {ds}")

print("\nSequence Datasets:")
for ds in seq_proc.list_datasets():
    print(f"  - {ds}")

print("\nProperty Datasets:")
for ds in prop_proc.list_datasets():
    print(f"  - {ds}")
```

#### 6.2 The Rational Design Pipeline
```markdown
**Session 1**: Extracted binding pocket data from Nature paper
**Session 2**: Analyzed dose-response/selectivity curves
**Session 3**: Connected data → structures → sequences → designs
**Session 4**: LLM orchestrates this entire pipeline automatically

The key insight:
- P415^6.47b is the "master switch" for PCO371 selectivity
- V412A and M414A ENHANCE activity (larger binding pocket)
- This suggests PCO371 is slightly too bulky for WT PTH1R
- A smaller PCO371 derivative might be even more potent!
```

#### 6.3 Preview: ProtOS-MCP (Session 4)
```markdown
In Session 4, you'll see how an LLM can:
1. "Load the PCO371 structure and find binding pocket residues"
2. "Compare sequences of responders vs non-responders"
3. "Design mutations to enhance PCO371 binding"
4. "Submit Boltz-2 predictions for all designs"

All using natural language - the LLM calls ProtOS under the hood!
```

---

## Data Files Needed

1. **From Session 1**:
   - `materials/session1/solution/nature_figure_gpcr_b_by_hand.csv` (binding pocket GRN table)

2. **From Session 2**:
   - `materials/session2/data/nature_figure_gpcr_b_selectivity.csv` (dose-response data)
   - `materials/session2/data/nature_gpcr_b_abc_figure.jpg` (reference figure)

3. **ProtOS Resources**:
   - `protos/resources/overview2.png` (architecture diagram)

4. **Mutant Activity Table** (create in notebook from user-provided data)

---

## Key Code Patterns to Follow

### Pattern 1: Zero-Config Initialization
```python
from protos.processing.structure import StructureProcessor
proc = StructureProcessor(name="my_analysis")
# That's it - no paths, no env vars
```

### Pattern 2: Entity Operations
```python
# Load by name
entity = proc.load_entity("8jr9")

# Save and auto-register
proc.save_entity("my_protein", data)

# Create dataset
proc.create_dataset("my_dataset", ["entity1", "entity2"])
```

### Pattern 3: Cross-Processor Workflows
```python
# Structure → Sequence
sequences = struct_proc.collect_chain_sequences(["8jr9"])

# Sequence → GRN
grn_table = seq_proc.annotate_with_grn("my_seqs", "gpcrdb_ref")

# Sequence → Embeddings
embeddings = emb_proc.embed_sequences(sequences)

# Any → Model
model_input = manager.prepare_input("boltz2", "my_entity", "sequence")
```

---

## Teaching Notes

1. **Don't overwhelm with code** - Focus on concepts, show minimal working examples
2. **Connect to biology** - Always explain WHY each step matters for drug design
3. **Reference the paper** - The Nature paper provides real-world validation
4. **Build anticipation** - Session 4 will automate all of this with LLM
5. **Encourage exploration** - Participants can modify mutations and see effects

---

## Exercises (Optional)

1. **Find another enhanced mutant**: Which other mutations from the table might enhance activity?
2. **Design a triple mutant**: Combine 3 beneficial mutations
3. **Compare binding pockets**: Which residues differ between responders/non-responders?
4. **Predict conservation**: Which binding pocket positions are most conserved?
