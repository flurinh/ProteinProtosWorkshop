# Session 2: Vibecoding - Communicating with AI

---

## The Vibecoding Process

1. **Describe your problem** clearly
2. **Provide context** (what you have, what you need)
3. **Specify constraints** (language, format, edge cases)
4. **Test the result** and iterate

---

## Bad Example

> "Write me a function to convert temperature"

### Problems:
- Which direction? (C→F? F→C? Both?)
- What programming language?
- What should the function return?
- How should it handle invalid input?

### Result: AI guesses, you get something you didn't want

---

## Good Example

> "Write a Python function called `celsius_to_fahrenheit` that:
> - Takes a single float as input (temperature in Celsius)
> - Returns the temperature in Fahrenheit as a float
> - Raises a ValueError if input is below absolute zero (-273.15°C)"

### Why it works:
- **Language**: Python
- **Function name**: specified
- **Input/Output**: clear types
- **Edge case**: handled explicitly

---

## Testing Your AI-Generated Code

Always verify the result:

```python
# Test normal case
assert celsius_to_fahrenheit(0) == 32.0
assert celsius_to_fahrenheit(100) == 212.0

# Test edge case
try:
    celsius_to_fahrenheit(-300)
    print("FAIL: Should have raised ValueError")
except ValueError:
    print("PASS: Correctly rejected invalid input")
```

---

## Key Takeaways

| Don't | Do |
|-------|-----|
| Be vague | Be specific |
| Assume AI knows your context | Provide relevant context |
| Trust blindly | Test the output |
| Give up after first try | Iterate and refine |

---

## The Prompt Formula

```
I need a [LANGUAGE] function/script that:
- Does [SPECIFIC TASK]
- Takes [INPUT TYPE/FORMAT]
- Returns [OUTPUT TYPE/FORMAT]
- Handles [EDGE CASES]
```

---
