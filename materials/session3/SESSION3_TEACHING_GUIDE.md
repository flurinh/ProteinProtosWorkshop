# Session 3: Teaching Guide
## Data Management in Structural Biology

---

## AT A GLANCE

| Aspect | Details |
|--------|---------|
| **Duration** | 60-90 minutes |
| **Prerequisites** | Session 1 (GRN table), Session 2 (dose-response) |
| **Key Deliverable** | Understanding that structural biology IS data management |
| **Tools Used** | StructureProcessor, LigandInteractionAnalyzer, PropertyProcessor |

---

## LEARNING OBJECTIVES

By the end of this session, participants will:

1. **Understand** why data management matters in structural biology
2. **Apply** the concept of "data flows" between processors
3. **Calculate** real residue-ligand distances from coordinates
4. **Connect** structural data to experimental functional data
5. **Validate** computational findings against published claims

---

## KEY STATIONS (Teaching Checkpoints)

### STATION 1: The Mindset Shift (5 min)
**Location in script:** Part 0-1

**Key Message:**
> "The difference between a good and great scientist is often not the experiments they run, but HOW THEY ORGANIZE THEIR DATA."

**Teaching Points:**
- Frame the biological question FIRST (How does PCO371 bind?)
- Then ask: What data do we need? How should we organize it?
- Introduce the concept: Each step is a DATA TRANSFORMATION

**Student Activity:**
Ask: "What would you need to answer this question manually?"
(Structure file, coordinates, distance calculations, paper lookup...)

---

### STATION 2: Abstraction Saves Time (10 min)
**Location in script:** Part 2

**Key Message:**
> "You ask for '8jr9' - the processor handles download, parsing, caching."

**Teaching Points:**
- Compare: `struct_proc.load_entity('8jr9')` vs manual download + parsing
- Explain what happens "under the hood" (download, parse, cache)
- Emphasize: Same interface regardless of file format (PDB vs mmCIF)

**Live Demo:**
```python
# Show this is ONE line, not 50 lines of file handling
df = struct_proc.load_entity('8jr9')
print(f"Loaded {len(df)} atoms - no file paths needed!")
```

**Bridge to Biology:**
"Just like a lab protocol abstracts complex techniques into steps, the processor abstracts data handling."

---

### STATION 3: Filtering = Finding What Matters (10 min)
**Location in script:** Part 3

**Key Message:**
> "8000 atoms -> 30 ligand atoms. This is DATA REDUCTION."

**Teaching Points:**
- Raw data is rarely usable directly
- Filtering by criteria (HETATM, res_name) extracts relevant subset
- Show the reduction: 7924 atoms -> ~30 atoms

**Bridge to Biology:**
"In the lab, you don't analyze every molecule - you purify first. This is the computational equivalent."

**Student Activity:**
"If you had to find the ligand manually, what would you search for?"

---

### STATION 4: Distance Calculations (15 min) **CORE CONCEPT**
**Location in script:** Part 4

**Key Message:**
> "We compute REAL distances from atomic coordinates - this is QUANTITATIVE analysis."

**Teaching Points:**
- Explain the distance formula: sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)
- Show the distance matrix concept: N ligand atoms x M protein atoms
- Cutoff filtering: keep residues within 5 Angstroms

**Live Demo:**
```python
# Show actual computed distances
for row in binding_residues_sorted.head(5).iterrows():
    print(f"{row['res_name']}{row['res_id']}: {row['min_distance']:.2f} A")
```

**Bridge to Biology:**
"This is exactly what crystallographers measure - we're just automating it."

**Key Output:** Binding site residue list with distances

---

### STATION 5: Interaction Types (10 min)
**Location in script:** Part 5

**Key Message:**
> "Different forces require different calculations - the analyzer handles all of them."

**Teaching Points:**
- H-bonds: distance + angle criteria
- Hydrophobic: carbon-carbon contacts within 4.5 A
- Pi-stacking: ring centroid distance + normal vector angle
- Salt bridges: charged group distances

**Visual Aid:**
```
Interaction Type    Criteria              Distance
--------------      --------              --------
H-bond              D-H...A, angle>120    2.5-3.5 A
Hydrophobic         C...C contacts        3.5-4.5 A
Pi-stacking         Ring centroids        4.0-5.5 A
Salt bridge         +/- groups            <4.0 A
```

**Bridge to Biology:**
"These are the same interactions you learn in biochemistry - we're detecting them computationally."

---

### STATION 6: GRN = Universal Language (10 min)
**Location in script:** Part 6

**Key Message:**
> "Position 6.47b is important for ALL Class B GPCRs, not just PTH1R."

**Teaching Points:**
- Sequence positions are receptor-specific (P415 in PTH1R)
- GRN positions are universal (6.47b across all Class B)
- This enables cross-receptor comparisons

**Key Example:**
```
PTH1R: P415 = 6.47b = Proline
PTH2R: L370 = 6.47b = Leucine
Same structural position, different residues!
```

**Bridge to Biology:**
"GRN is like a GPS coordinate system for proteins - same position, any receptor."

**Student Activity:**
"Why doesn't PCO371 work on PTH2R?" (Answer: L vs P at 6.47b)

---

### STATION 7: Connecting Structure to Function (10 min)
**Location in script:** Part 7

**Key Message:**
> "We're JOINING datasets - structure + function = mechanistic understanding."

**Teaching Points:**
- Structural data: which residues contact ligand
- Functional data: which mutations affect activity
- JOIN KEY: residue identity

**Key Insight:**
```
P415 contacts ligand (structure)
P415A abolishes activity (function)
P is conserved in PTH1R (evolution)
L is present in PTH2R (comparison)
= COMPLETE MECHANISTIC PICTURE
```

**Bridge to Biology:**
"In the lab, you'd do mutagenesis. We're connecting that data to structure computationally."

---

### STATION 8: Reproducibility (5 min)
**Location in script:** Part 8

**Key Message:**
> "A dataset is not just files - it's a reproducible record with metadata."

**Teaching Points:**
- Datasets bundle entities + metadata + provenance
- Can be reloaded months later
- Enables sharing and publication

**Bridge to Biology:**
"Like a lab notebook - but machine-readable and version-controlled."

---

### STATION 9: Visualization (5 min)
**Location in script:** Part 9

**Key Message:**
> "Computational results should be visualizable - always."

**Teaching Points:**
- Generate PyMOL scripts for consistent visualization
- Export CSV for further analysis
- Create reproducible views

---

### STATION 10: Validation (10 min) **CRITICAL**
**Location in script:** Part 10

**Key Message:**
> "Do our computational results MATCH the published findings?"

**Teaching Points:**
- Check each paper claim against our data
- Quantitative verification, not just "it looks right"
- This is how you BUILD CONFIDENCE in your analysis

**Validation Checklist:**
```
[ ] Claim 1: TM2/TM3/TM6/TM7 contacts  -> CONFIRMED
[ ] Claim 2: P415 is critical          -> CONFIRMED (no activity)
[ ] Claim 3: Pocket is conserved       -> CONFIRMED (6 positions)
[ ] Claim 4: L370P rescues PTH2R       -> CONFIRMED (GRN match)
```

---

## KEY CONCEPTS TO REINFORCE

### 1. Data Transformation Pipeline
```
Structure file (.cif)
    -> Atom coordinates (DataFrame)
    -> Distance matrix (N x M array)
    -> Contact list (residue IDs)
    -> Interaction summary (counts)
    -> Validation (claims checked)
```

### 2. Biological Concept <-> Data Concept Mapping

| Biology | Data Management |
|---------|-----------------|
| Load structure | `load_entity()` - parse + cache |
| Find ligand | Filter by group='HETATM' |
| Binding site | Distance matrix < cutoff |
| Interactions | Geometric criteria |
| Cross-receptor | GRN mapping |
| Mutagenesis | Join structure + function |
| Validation | Compare computed vs published |

### 3. The Protos Philosophy
- **Zero configuration**: Set data path once, forget file locations
- **Human-readable names**: Ask for "8jr9", not file paths
- **Universal tracking**: Same entity across formats
- **Reproducible workflows**: Datasets with metadata

---

## COMMON STUDENT QUESTIONS

**Q: Why not just look at the structure in PyMOL?**
A: Looking is subjective. Computing distances is quantitative and reproducible.

**Q: Why use protos instead of BioPython?**
A: Protos manages the DATA FLOW, not just parsing. It tracks entities across transformations.

**Q: How do I know if my binding site is correct?**
A: Validate against published data! Compare your residue list to the paper's.

**Q: What if the paper doesn't match?**
A: Great question! Investigate why - different cutoffs? Different chain? Error in paper?

---

## SESSION FLOW DIAGRAM

```
[START]
    |
    v
[Station 1: Mindset] -----> "Biology questions need data answers"
    |
    v
[Station 2: Abstraction] -> "Processors hide complexity"
    |
    v
[Station 3: Filtering] ---> "Extract what matters"
    |
    v
[Station 4: Distances] ---> "QUANTITATIVE analysis" *KEY*
    |
    v
[Station 5: Interactions] -> "Geometric criteria"
    |
    v
[Station 6: GRN] ---------> "Universal language"
    |
    v
[Station 7: Function] ----> "Join structure + data"
    |
    v
[Station 8: Datasets] ----> "Reproducibility"
    |
    v
[Station 9: Export] ------> "Visualization ready"
    |
    v
[Station 10: Validate] ---> "Match published claims" *CRITICAL*
    |
    v
[END: Understanding achieved]
```

---

## PREPARATION CHECKLIST

- [ ] Session 1 completed (GRN table available)
- [ ] Session 2 completed (selectivity data available)
- [ ] `pth1r_mutant_activity.csv` in materials/session3/data/
- [ ] protos environment activated
- [ ] PyMOL installed (for visualization demo, optional)

---

## KEY OUTPUTS

| File | Purpose |
|------|---------|
| `pco371_binding_site.csv` | Computed residue-ligand distances |
| `visualize_binding_site.pml` | PyMOL script for 3D view |
| Dataset: `pco371_binding_analysis` | Reproducible record |

---

## BRIDGE TO SESSION 4

At the end of Session 3, students understand:
- How data flows through processors
- What calculations happen at each step
- How to validate computational results

Session 4 (ProtOS-MCP) will show them:
- How to do ALL of this via natural language
- "Show me the binding site residues within 5 A of PCO371"
- The LLM understands the data flow and calls the right functions

**Key Bridge Statement:**
> "You just learned to write these data transformations in code. In Session 4, you'll learn to describe them in natural language - and the system will understand."

---

## NOTES FOR INSTRUCTOR

1. **Pace yourself** at Station 4 (distances) - this is where the "aha moment" happens
2. **Show real numbers** - don't just say "we calculated distances", show the actual values
3. **Validate everything** - Station 10 is critical for building trust
4. **Connect constantly** - every data concept has a biological analog
5. **Leave PyMOL for later** - focus on the data flow, visualization is supplementary

---

*Generated for ProteinProtosWorkshop Session 3*
