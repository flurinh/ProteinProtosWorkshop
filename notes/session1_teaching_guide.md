# Session 1 Teaching Guide

## Pre-Workshop Checklist

- [ ] Test Jupyter installation on workshop machines/laptops
- [ ] Verify all data files present in `materials/session1/data/` and `materials/session1/solution/`
- [ ] Pre-run the notebook once to cache imports (first import of pandas can be slow)
- [ ] Prepare backup: export notebook as PDF in case of technical issues
- [ ] Have `python_cheatsheet.py` open or printed for reference
- [ ] Test screen sharing if presenting remotely

## Learning Objectives

By the end of Session 1, participants will be able to:

1. Navigate Jupyter notebooks confidently (run cells, restart kernel, switch cell types)
2. Use Python's core data types: `str`, `int`, `float`, `bool`, `list`, `dict`
3. Write simple loops and functions
4. Load CSV data with pandas and perform basic inspection
5. Understand when AI-extracted data needs verification

---

## Timing Guide (1h 20min total)

| Section | Time | Key Points |
|---------|------|------------|
| 0. Welcome & Goals | 5 min | Set expectations, why Python for protein science |
| 1. Jupyter Orientation | 5 min | Run cells, cell types, restart kernel |
| 2. Variables & Types | 10 min | str, int, float, bool, f-strings |
| 3. Operators | 5 min | Math, comparisons, MW calculation |
| 4. Strings | 7 min | Indexing, slicing, methods |
| 5. Lists | 10 min | Creation, indexing, slicing, methods |
| 6. Dictionaries | 10 min | Key-value pairs, nested dicts |
| 7. Loops | 8 min | for loops, enumerate, zip |
| 8. Functions | 7 min | def, parameters, return, docstrings |
| 9. Pandas | 15 min | CRITICAL - go slow, load/inspect CSV |
| 10. AI Exercise | 15 min | Compare AI vs corrected, discussion |
| Buffer | 5-10 min | Q&A, troubleshooting |

---

## Common Pitfalls & How to Handle

### "I got an error!"

1. **Stay calm** - errors are normal and helpful
2. **Read the error message together**:
   - Point to the line number mentioned
   - Read the error type (NameError, TypeError, etc.)
   - Read the description
3. **Common errors**:
   - `NameError`: Typo in variable name, or forgot to run earlier cell
   - `IndentationError`: Inconsistent spaces/tabs
   - `SyntaxError`: Missing colon, bracket, or quote
   - `KeyError`: Wrong dictionary key or column name
4. **Demo strategy**: Intentionally trigger an error and fix it live

### "My cell won't run"

1. Check if kernel is busy (asterisk `[*]` next to cell)
2. Check for infinite loops (while True without break)
3. Try: Kernel → Restart & Clear Output
4. If stuck: Kernel → Interrupt, then Restart

### "I don't understand slicing"

Use this visual:

```
String:  M  K  T  L  L  I  L
Index:   0  1  2  3  4  5  6
        -7 -6 -5 -4 -3 -2 -1

seq[0]     → 'M'      (first)
seq[-1]    → 'L'      (last)
seq[0:3]   → 'MKT'    (indices 0, 1, 2 - NOT 3!)
seq[:3]    → 'MKT'    (same as above)
seq[3:]    → 'LLIL'   (from index 3 to end)
```

Key phrase: **"Up to but not including"**

### "Nothing happened when I ran the cell"

- Assignment doesn't print: `x = 5` produces no output
- Add `print(x)` or just `x` on the last line to see values
- In Jupyter, the last expression in a cell is automatically displayed

### "I'm lost / too fast"

- Encourage them to use `python_cheatsheet.py` as reference
- Pair struggling participants with those who are ahead
- Emphasize: "You don't need to memorize - just know where to look"

---

## Discussion Prompts

### After Section 2 (Variables):
> "What kind of data do you work with daily? Numbers? Text? Categories? All of those map to Python types we just learned."

### After Section 6 (Dictionaries):
> "Think about how you might represent a protein's properties: name, PDB ID, molecular weight, sequence... A dictionary is perfect for this!"

### After Section 9 (Pandas):
> "Think about Excel. What's tedious about it? Repetitive clicking? Copy-paste errors? Pandas automates all of that with reproducible code."

### After Section 10 (AI Exercise):
> "Would you trust this AI-extracted data for a publication? What would you check before using it? This is the 'trust but verify' mindset we need."

---

## Instructor Notes by Section

### Section 0: Welcome & Goals
- Acknowledge varied backgrounds - some may have coded before, others not
- Key message: "You don't need to become a programmer - just fluent enough to automate tedious tasks"
- Python is a tool like a pipette - we're learning to use it for our work

### Section 1: Jupyter Orientation
- Demo: Run a cell with Shift+Enter
- Demo: Create a new cell, change to Markdown, write something, run it
- Demo: Restart kernel - explain why this is sometimes needed
- Let them modify the print statement and run it themselves

### Section 2: Variables & Types
- Use protein examples they'll recognize: Lysozyme, GFP, insulin
- Show `type()` frequently - demystifies what's happening
- f-strings are extremely useful - spend time here
- Exercise should feel achievable (just creating variables)

### Section 3: Operators
- Quick section - don't dwell
- The MW calculation connects math to biology
- Show that `==` is comparison, `=` is assignment (common confusion)

### Section 4: Strings
- Sequences are strings! This connects directly to their work
- Slicing is tricky - use the visual aid above
- `split()` is incredibly useful for parsing headers

### Section 5: Lists
- Lists are ordered and changeable
- Show that list slicing works the same as string slicing
- `append()` vs `extend()` - common confusion point
- The exercise with amino acids makes it tangible

### Section 6: Dictionaries
- "Like a labeled drawer" - each key has a specific value
- Show both bracket notation `d["key"]` and `.get("key", default)`
- Nested dictionaries for protein properties - builds toward real data structures

### Section 7: Loops
- `for item in collection:` is the pattern
- `enumerate()` when you need both index and value
- Don't spend too long on `while` - `for` is more common
- The sample iteration exercise mirrors real batch processing

### Section 8: Functions
- "Reusable recipe with ingredients"
- Docstrings are comments that help others (and your future self)
- Default parameters make functions flexible
- Keep the concentration calculator simple

### Section 9: Pandas (CRITICAL)
- **THIS IS THE PAYOFF** - everything builds to here
- Go slow, let them explore `df.head()`, `df.info()`
- `df["column"]` vs `df[["col1", "col2"]]` - single vs multiple columns
- Filtering: `df[df["column"] > value]` - powerful pattern
- Connect to: "Imagine doing this to your real data"

### Section 10: AI Exercise
- Let them discover the differences themselves first
- Don't reveal the "answer" immediately
- Key insights to draw out:
  - Column names may differ
  - Some values may be truncated or misread
  - Numbers might be formatted differently
- **Message**: AI is helpful but not infallible - always verify

---

## Troubleshooting Quick Reference

| Problem | Solution |
|---------|----------|
| Kernel dies | Kernel → Restart, re-run cells from top |
| Import fails | Check virtual environment, `pip install <package>` |
| File not found | Check path, use `Path.exists()` to verify |
| Cell runs forever | Kernel → Interrupt, check for infinite loops |
| Strange output | Restart kernel, clear outputs, run fresh |

---

## Post-Session

- Encourage practice before Session 2
- Point to `python_cheatsheet.py` as reference
- Remind them Session 2 will apply these skills to real analysis
- Collect feedback: What was confusing? What clicked?
