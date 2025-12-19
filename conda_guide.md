# Conda Guide

A complete guide to setting up Python with Miniconda — from installation to running your first script.

---

## 1. Installing Miniconda

### Download

Go to: [https://www.anaconda.com/download](https://www.anaconda.com/download)

Scroll down past Anaconda Distribution to find the **Miniconda** installers.

### Install

**Windows:**
1. Run the `.exe` installer
2. Accept the license
3. Choose "Just Me"
4. Check "Add Miniconda to my PATH" (important!)

**macOS:**
1. Download the `.pkg` file (Apple Silicon or Intel)
2. Run the installer and follow prompts

**Linux:**
```bash
# Download the installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Run it
bash Miniconda3-latest-Linux-x86_64.sh
```

Follow the prompts and type `yes` to accept the license.

---

## 2. Adding Conda to Your Shell (Important!)

After installation, you need to initialize conda so it works in your terminal.

### Run conda init

```bash
conda init bash
```

Or for other shells:
```bash
conda init zsh      # macOS default
conda init fish     # fish shell
```

### What this does

This command adds the following to your `~/.bashrc` (or `~/.zshrc`):

```bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/username/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/username/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/username/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/username/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```

### Apply the changes

After running `conda init`, either:

```bash
# Option 1: Restart your terminal
# Just close and reopen it

# Option 2: Source your config file
source ~/.bashrc      # Linux
source ~/.zshrc       # macOS
```

### Verify it works

```bash
conda --version
```

You should see something like `conda 24.x.x`.

If you see `(base)` at the start of your prompt, conda is active!

### Optional: Disable auto-activation of base

If you don't want conda's `(base)` environment to activate every time you open a terminal:

```bash
conda config --set auto_activate_base false
```

---

## 3. Why Use Environments?

**The Problem:**
- Project A needs `numpy 1.2`
- Project B needs `numpy 2.0`
- Installing one breaks the other!

**The Solution: Virtual Environments**
- Isolated "boxes" for each project
- Each has its own Python + libraries
- Switch between them easily

```
┌─────────────────┐    ┌─────────────────┐
│   Project A     │    │   Project B     │
│   Python 3.9    │    │   Python 3.11   │
│   numpy 1.2     │    │   numpy 2.0     │
│   pandas 1.5    │    │   pandas 2.1    │
└─────────────────┘    └─────────────────┘
```

---

## 4. Create, Activate & Deactivate Environments

### Create an environment

```bash
conda create -n myproject python=3.11
```

- `-n myproject` → name of your environment
- `python=3.11` → Python version to use

### Activate it

```bash
conda activate myproject
```

Your prompt changes: `(base)` → `(myproject)`

### Deactivate when done

```bash
conda deactivate
```

Back to `(base)` or system Python.

---

## 5. Installing Libraries

Once your environment is activated:

```bash
# Install a single package
conda install numpy

# Install multiple packages
conda install numpy pandas matplotlib

# Install a specific version
conda install numpy=1.26

# Install from conda-forge (more packages)
conda install -c conda-forge seaborn
```

### Useful commands

```bash
conda list                     # See what's installed
conda env list                 # See all your environments
conda remove numpy             # Uninstall a package
conda env remove -n myproject  # Delete an environment
```

---

## 6. Two Ways to Run Python Code

| | Jupyter Notebook | Python Script |
|---|---|---|
| File extension | `.ipynb` | `.py` |
| Style | Interactive, cell-by-cell | Run all at once |
| Best for | Exploration, visualization, teaching | Automation, production, reusable code |
| Output | Inline (graphs, tables) | Terminal / saved files |

---

## 7. Jupyter Notebooks

**What is it?**
- Interactive coding in your browser
- Mix code, output, and notes in one document
- Run code cell by cell — see results immediately

### Install & Launch

```bash
conda activate myproject
conda install jupyter
jupyter notebook
```

This opens Jupyter in your browser.

### Basic usage

- Cells = blocks of code
- `Shift + Enter` = run a cell
- Markdown cells for notes/explanations

---

## 8. Python Scripts

**What is it?**
- Plain text file with `.py` extension
- Write in any text editor or IDE
- Run the entire file at once

### Create a simple script

Create a file called `hello.py`:

```python
print("Hello, world!")
x = 5 + 3
print(f"The answer is {x}")
```

### Run it

```bash
python hello.py
```

---

## 9. The `main` Function — Structuring Your Scripts

As your scripts grow, you'll want to organize them properly. The standard way to do this in Python is with a **main function**.

### Why use a main function?

1. **Organization** — keeps your code structured
2. **Reusability** — your script can be imported by other scripts without running automatically
3. **Clarity** — makes it obvious where your program starts

### The pattern

```python
def main():
    """This is where your program starts."""
    print("Hello, world!")
    x = 5 + 3
    print(f"The answer is {x}")


if __name__ == "__main__":
    main()
```

### What does `if __name__ == "__main__"` mean?

When Python runs a file directly, it sets a special variable `__name__` to `"__main__"`.

- **Run directly:** `python hello.py` → `__name__` is `"__main__"` → `main()` runs
- **Imported:** `import hello` → `__name__` is `"hello"` → `main()` does NOT run

This lets you write code that works both as a standalone script AND as a reusable module.

### A more complete example

```python
"""
my_analysis.py
A script that analyzes some data.
"""

import numpy as np


def load_data(filename):
    """Load data from a file."""
    print(f"Loading {filename}...")
    # Your loading code here
    return [1, 2, 3, 4, 5]


def analyze(data):
    """Perform analysis on the data."""
    average = np.mean(data)
    return average


def main():
    """Main entry point of the script."""
    # Load data
    data = load_data("measurements.csv")
    
    # Analyze
    result = analyze(data)
    
    # Report
    print(f"The average is: {result}")


if __name__ == "__main__":
    main()
```

Run it:
```bash
python my_analysis.py
```

---

## 10. Full Workflow Summary

### First-time setup (once)

```bash
# 1. Install Miniconda (see section 1)

# 2. Initialize conda for your shell
conda init bash
source ~/.bashrc

# 3. Create your environment
conda create -n workshop python=3.11

# 4. Activate and install packages
conda activate workshop
conda install numpy pandas matplotlib jupyter
```

### Daily workflow

```bash
# 1. Open terminal

# 2. Activate your environment
conda activate workshop

# 3. Work
jupyter notebook       # for exploration
python myscript.py     # for running scripts

# 4. When done
conda deactivate
```

---

## 11. Quick Reference

| Task | Command |
|------|---------|
| Initialize conda | `conda init bash` |
| Reload shell config | `source ~/.bashrc` |
| Create environment | `conda create -n envname python=3.11` |
| Activate | `conda activate envname` |
| Deactivate | `conda deactivate` |
| Install package | `conda install packagename` |
| List packages | `conda list` |
| List environments | `conda env list` |
| Delete environment | `conda env remove -n envname` |
| Launch Jupyter | `jupyter notebook` |
| Run a script | `python filename.py` |

---

## 12. Troubleshooting

### "conda: command not found"

Conda isn't in your PATH. Run:
```bash
# Find where conda is installed, then:
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### "conda activate" doesn't work

You may need to run `conda init` first (see section 2).

### Package not found in conda

Try installing from conda-forge:
```bash
conda install -c conda-forge packagename
```

Or use pip (inside your activated environment):
```bash
pip install packagename
```

### PSI network blocks download

Try downloading Miniconda from home or using a mobile hotspot.
