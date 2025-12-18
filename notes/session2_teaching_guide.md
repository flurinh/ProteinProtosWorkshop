# Session 2 Teaching Guide

## Pre-Workshop Checklist

- [ ] Verify Session 1 was completed (pandas basics)
- [ ] Test all data files in `materials/session2/data/`
- [ ] Pre-run the notebook to ensure all plots render correctly
- [ ] Have scipy, numpy, matplotlib installed (`pip install -r requirements.txt`)
- [ ] Optional: Download MARCO crystal dataset if doing live classification demo
- [ ] Test 3D plots (Procrustes section) - may need `%matplotlib notebook` or `%matplotlib widget`

## Learning Objectives

By the end of Session 2, participants will be able to:

1. **See problems as data problems** - Identify what data they have and what they want to extract
2. **Formulate problems clearly** - Describe a problem with enough detail for an AI (or colleague) to help
3. **Vibe code and review solutions** - Get working code from AI tools and understand what it does
4. **Visualize to verify** - Use plots to check if results make biological sense
5. **Think critically about correctness** - Ask "how would I know if this is wrong?"
6. **Not be afraid to try** - Experiment with prompts, iterate when things don't work

**Note:** The specific biology problems (growth curves, dose-response, FSEC, etc.) are vehicles for teaching these transferable skills. Don't get bogged down in the domain details - focus on the problem-solving workflow.

---

## Timing Guide (1h 25min total)

| Section | Time | Focus |
|---------|------|-------|
| Setup & Learning Goals | 5 min | Frame the session as skill-building, not memorization |
| Problem 1: Growth Curves | 15 min | "When to induce?" - First full workflow walkthrough |
| Problem 2: Dose-Response | 15 min | "Is drug working?" - Practice formulating the question |
| Problem 3: FSEC | 15 min | "Which sample?" - Emphasis on visualization |
| Problem 4: Crystal Classification | 15 min | "Can we automate?" - Discussion + demo, no coding |
| Problem 5: Procrustes | 15 min | "Did structure change?" - Cautionary tale about verification |
| Summary & Takeaways | 10 min | Reinforce meta-skills, encourage them to try on their data |

**Instructor focus per problem:**
- Problem 1: Walk through the full workflow slowly - this is the template
- Problem 2-3: Let participants practice formulating prompts themselves
- Problem 4: More discussion, less code - when does ML help?
- Problem 5: Critical thinking - AI can be wrong, how do you catch it?

---

## Common Pitfalls & How to Handle

### "curve_fit failed" or "Optimal parameters not found"

1. **Check initial guesses (p0)** - Bad starting points cause failures
2. **Add bounds** - Constrain parameters to reasonable ranges
3. **Increase maxfev** - Allow more iterations: `maxfev=10000`
4. **Check data** - Is the data actually sigmoidal? Look at the plot first!

Example fix:
```python
# Instead of just:
popt, _ = curve_fit(logistic_fn, x, y)

# Use:
p0 = [0, y.max(), np.median(x), 1]  # Initial guesses
bounds = ([0, 0, -12, 0], [10, 200, -2, 5])  # Reasonable bounds
popt, _ = curve_fit(logistic_fn, x, y, p0=p0, bounds=bounds, maxfev=10000)
```

### "Peak detection finds too many/too few peaks"

1. **Adjust prominence** - Higher values filter out smaller peaks
2. **Smooth first** - Use `savgol_filter()` before `find_peaks()`
3. **Check signal scale** - Normalize if values are very small/large

```python
# Too many peaks? Increase prominence
peaks, _ = find_peaks(signal, prominence=0.1)  # Try 0.1 instead of 0.02

# Still noisy? Smooth first
smoothed = savgol_filter(signal, window_length=11, polyorder=3)
peaks, _ = find_peaks(smoothed, prominence=0.05)
```

### "3D plots don't display"

1. Try `%matplotlib inline` at the top of the notebook
2. Or `%matplotlib notebook` for interactive plots
3. If in VS Code, may need `%matplotlib widget`

### "Import error for workshop_utils"

1. Ensure notebook is in the correct directory
2. Check that `workshop_utils.py` exists in the same folder
3. Restart the kernel after creating new files

---

## Discussion Prompts

### Focus on Meta-Skills, Not Just Biology

The goal of discussion is to reinforce the **problem-solving workflow**, not just domain knowledge.

### After Problem 1 (Growth Curves):
> "How did we turn a biological question into a data problem?"
>
> Guide them to: "We had OD600 vs time, we wanted to find the inflection point, so we fit a model that has an inflection point parameter."

### After Problem 2 (Dose-Response):
> "What made the vibe prompt effective? What would have made it worse?"
>
> Good prompt: specific about data format, clear goal, lists steps, asks for specific output.
> Bad prompt: vague ("analyze my data"), no context about what success looks like.

### After Problem 3 (FSEC):
> "The plot told us something the numbers alone wouldn't. What?"
>
> Visualization revealed which samples were actually clean vs. had hidden problems. Numbers can hide issues.

### After Problem 4 (Crystal Classification):
> "When would you trust the AI classifier? When would you double-check?"
>
> Trust: high-confidence predictions, first-pass screening. Verify: edge cases, final decisions, anything going in a paper.

### After Problem 5 (Procrustes):
> "This is a cautionary tale. What's the lesson?"
>
> AI can generate code with subtle bugs. Always: understand what *should* happen, include sanity checks, visualize results.

### At the End:
> "What data problem from YOUR work could you tackle with this workflow?"
>
> Get them thinking about applying these skills immediately. Offer to brainstorm prompts with them.

---

## Instructor Notes by Problem

### Setup & Learning Goals (5 min)
- Frame the session clearly: **"Today is NOT about memorizing code"**
- Read through the learning goals together
- Key message: "You'll learn a workflow for solving data problems you've never seen before"
- Make sure the import cell runs without errors

### Problem 1: Growth Curves (15 min) - THE TEMPLATE
This is your chance to **model the entire workflow**. Go slow.

1. **Show the scenario** - make it relatable ("You've all done this")
2. **Look at the data together** - what do we see? what's the question?
3. **Group discussion** - "How would you approach this?" Let them suggest ideas
4. **Write the vibe prompt together** - ask "What would you tell an AI?"
5. **Run the code** - don't explain every line, focus on "what does this do?"
6. **Interpret** - "Do these numbers make sense? How do we know?"

**This sets the pattern for all other problems.**

### Problem 2: Dose-Response (15 min)
- Let participants **practice formulating** the question themselves
- Ask: "What data do we have? What do we want?"
- Have them draft a prompt before showing the example
- Emphasize: a good prompt gets good results

### Problem 3: FSEC (15 min)
- Focus on **visualization as verification**
- After running code, ask: "Would you trust this ranking? Why or why not?"
- Key insight: plots catch problems that tables hide

### Problem 4: Crystal Classification (15 min)
- **More discussion, less coding** - this is about when to use ML
- Don't explain how neural networks work in detail
- Focus on: "When would you trust this? When wouldn't you?"
- If classifier isn't trained, just discuss the images

### Problem 5: Procrustes (15 min) - THE CAUTIONARY TALE
- This is about **critical thinking**
- Key message: AI can produce code that looks right but is subtly wrong
- Walk through the reflection problem slowly
- "How would you have caught this bug?"
- Reinforce: always visualize, always include sanity checks

### Summary (10 min)
- Don't rush this - it's where the learning solidifies
- Walk through the workflow diagram together
- Ask: "What problem from YOUR work could you tackle?"
- Encourage them to try it - failure is part of learning

---

## Technical Notes

### snip_baseline() Algorithm
The SNIP (Sensitive Nonlinear Iterative Peak) algorithm:
1. Apply LLS transform to stabilize variance
2. Iteratively replace each point with minimum of itself and average of neighbors
3. Inverse transform to get baseline
4. Subtract baseline from signal

### logistic_fn() Parameters
```
Y = bottom + (top - bottom) / (1 + 10^((log_ec50 - X) * hill))

bottom:   Lower asymptote (baseline response)
top:      Upper asymptote (maximum response)
log_ec50: Log10 of EC50 concentration
hill:     Hill coefficient (curve steepness)
```

### logistic_growth() Parameters
```
OD(t) = max_od / (1 + exp(-rate * (t - lag)))

max_od: Carrying capacity
rate:   Growth rate constant (1/hour)
lag:    Time at inflection point (hours)
```

---

## Troubleshooting Quick Reference

| Problem | Solution |
|---------|----------|
| curve_fit fails | Check p0, add bounds, increase maxfev |
| Too many peaks | Increase prominence, smooth first |
| 3D plot blank | Try `%matplotlib inline` or restart kernel |
| Memory error | Restart kernel, reduce dataset size |
| Import fails | Check file paths, restart kernel |

---

## Post-Session

**Reinforce the workflow, not the specific techniques:**

- "Pick ONE data problem from your work and try the workflow this week"
- Point to `session2_solutions.py` only if they want to see more examples
- The takeaway is the **workflow**, not the code
- Remind them: "You don't need to memorize - you need to formulate clearly and verify carefully"

**Collect feedback focused on skills:**
- "Do you feel more confident approaching a new data problem?"
- "What was the hardest part of the workflow?"
- "What would you like more practice with?"

---

## Additional Resources

### For participants who want to learn more:
- scipy.optimize documentation for curve_fit
- scipy.signal documentation for find_peaks, savgol_filter
- NumPy linear algebra for SVD and matrix operations

### Relevant papers:
- SNIP algorithm: Ryan et al., NIM (1988)
- MARCO crystal dataset: Holton et al. (2019)
- Procrustes analysis: Gower & Dijksterhuis (2004)
