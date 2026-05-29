# How the BoolForge Tutorials Are Built

The BoolForge tutorials are designed to be **reproducible, maintainable, and executable end-to-end**.
Rather than maintaining Jupyter notebooks directly, we follow a **source-first workflow** that separates
content, execution, and presentation.

This document briefly explains how the tutorial build system works.

---

## Source of truth: Python scripts

Each tutorial is written as a plain Python file (`.py`) using the **Jupytext percent format**, for example:

    tutorials/src/tutorial06_boolean_networks.py

These files contain:
- Markdown cells (as commented blocks),
- Code cells (pure Python),
- No notebook metadata or execution state.

This ensures that:
- tutorials are easy to review in version control,
- diffs remain readable,
- content can be edited in any Python editor or IDE, and
- notebooks never accumulate hidden or stale state.

The `.py` files are the **single source of truth**.

---

## Conversion to Jupyter notebooks

Jupyter notebooks (`.ipynb`) are **generated**, not authored directly.

Conversion is handled using **Jupytext**, which deterministically maps each `.py` file to a notebook:

    jupytext --to notebook tutorial06_boolean_networks.py

This step:
- preserves cell structure and ordering,
- produces clean notebooks without execution artifacts,
- avoids manual notebook editing.

---

## Executing tutorials

To ensure correctness and reproducibility, every generated notebook is executed
from top to bottom using:
    
    jupyter nbconvert --execute

Execution guarantees that:
- all imports are valid,
- all examples run as written,
- figures and tables are produced correctly,
- tutorials never rely on hidden state.

If any cell fails, the build stops immediately.

---

## Incremental builds with Make

All tutorial steps are orchestrated using a `Makefile`.

Each tutorial follows the dependency chain:
    
    tutorialXX.py -> tutorialXX.ipynb -> tutorialXX.html

The build system:
- rebuilds only tutorials whose source files changed,
- skips up-to-date tutorials automatically,
- localizes errors to the specific tutorial that failed.

This makes development fast and scalable as the tutorial suite grows.

To build all tutorials:

    make tutorials

To force a full rebuild:

    make clean
    make tutorials
    
---

## Website integration

The generated notebooks and HTML files are treated as **build artifacts** and can be:
- published directly on the BoolForge website,
- rendered statically (e.g., via GitHub Pages),
- distributed alongside the source code.

Because tutorials are executed during the build,
the published results are guaranteed to match the code.

---

## Why this approach?

This workflow ensures:
- **Reproducibility** - tutorials always run from a clean state,
- **Maintainability** - no manual notebook fixes,
- **Transparency** - readable diffs and clean version history,
- **Scalability** - adding or updating tutorials is trivial,
- **Trust** - users can rely on the examples as executable documentation.

In short, the tutorials are treated as **tested scientific artifacts**, not static examples.
