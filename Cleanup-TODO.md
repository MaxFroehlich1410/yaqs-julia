
# Publication-Ready Repository Guide (Yaqs.jl / yaqs-julia)

This document is a **step-by-step** checklist to make this repository clean, reproducible, and publication-ready.

It is written for the current repository layout:

-`src/` Julia package source

-`test/` Julia tests (`runtests.jl` + per-module tests)

-`experiments/` scripts used for paper experiments/benchmarks

-`00_safety_checks_full_algorithms/` existing verification/safety-check scripts and outputs

---

## Goals and “definition of done”

You are done when all items below are true:

-**Docs**: Every exported/public function and type has a docstring; module-level docs exist; docs show minimal runnable examples.

-**Tests**: Automated tests cover all public APIs and key algorithmic behavior; tests are deterministic (seeded); `Pkg.test()` passes on CI.

-**Safety checks**: Paper-level / algorithm-level verification exists as a reproducible “verification suite” with fixed inputs and tolerances (not ad-hoc scripts).

-**Repository hygiene**: No random results, caches, `.DS_Store`, `__pycache__`, `.pyc`, large binary artifacts, or untracked outputs in the repo.

-**Paper reproducibility**: Notebooks/scripts that reproduce figures/tables are clean, runnable from a fresh checkout, and documented.

-**CI**: A continuous integration pipeline runs tests and basic hygiene checks automatically.

-**Cite + license**: `LICENSE` and `CITATION.cff` exist, and the release/tag used for the paper is unambiguous.

---

## Step 0 — Freeze scope and define what is “supported”

**Goal:** Clearly separate (A) the package API from (B) paper artifacts (experiments/notebooks) and (C) large outputs.

### Actions

1. Decide what is **public API**:

- The functions/types that you want users to rely on long-term.
- Everything else is internal implementation and can change.

2. Decide what constitutes **paper reproduction artifacts**:

- Scripts and notebooks that regenerate figures/tables.
- Small reference data (if needed) that is safe to track in git.

3. Decide what is **not tracked**:

- Large `.pkl`, `.png`, `.csv` results from runs.
- Any data derived from simulations that can be regenerated.

### Done when

- You can point to a short list of exported functions/types (e.g. by reviewing `src/Yaqs.jl` exports).
- You know where reproduction code will live (recommended: `paper/` or `repro/`).

---

## Step 1 — Add the missing “entrypoint” files (README, License, Citation)

**Goal:** A fresh user can understand what this repo is and how to run it.

### Actions

1. Create `README.md` including:

- What the package does (1–2 paragraphs).
- Installation steps (Julia + optional Python).
- Minimal example (small system run).
- “Reproduce paper results” section pointing to scripts/notebooks.
- Support statement: what is stable / what is experimental.

2. Add `LICENSE` (choose a standard license: MIT/BSD-3/Apache-2.0 are common).
3. Add `CITATION.cff`:

- Paper title/authors/DOI (or arXiv) and repo tag to cite.

### Done when

- A user can run *something* from the README on a clean machine.
- The paper citation information is present and correct.

---

## Step 2 — Environments and reproducibility (Julia + optional Python)

**Goal:** One command should set up the environment; experiments reproduce from a fresh checkout.

### Julia environment (package)

This repo already contains `Project.toml` and `Manifest.toml`. Treat them as the **canonical** environment.

Recommended commands for users:

```bash

julia--project=.-e'using Pkg; Pkg.instantiate()'

julia--project=.-e'using Pkg; Pkg.test()'

```

### Python environment (if you require Python comparisons/benchmarks)

This repo already depends on `PythonCall` and `CondaPkg`. Decide whether Python is:

-**Required** for core features, or

-**Optional** only for comparisons/benchmarks.

**Recommended structure**:

- Keep Python dependencies managed by `CondaPkg.toml` (and/or `.CondaPkg/` artifacts excluded by git).
- Document one “known-good” workflow:

-`Pkg.instantiate()` handles Python via `CondaPkg` automatically, or

- Provide explicit setup steps if you use system Python.

### Done when

- Fresh checkout → `Pkg.instantiate()` → `Pkg.test()` works.
- If Python is needed, the README states exactly how to get it working.

---

## Step 3 — Repository hygiene: remove outputs and block them forever

**Goal:** No random result files in git. Only source, configs, and small stable assets remain.

### 3.1 Identify artifacts and decide their destination

Typical artifacts to exclude:

- macOS: `.DS_Store`
- Python: `__pycache__/`, `*.pyc`
- Jupyter: `.ipynb_checkpoints/`
- Results: `*.png`, `*.pdf`, `*.csv`, `*.pkl`, `*.h5`, `*.jld2`, `*.npz`, `*.npy` (unless deliberately tracked as tiny reference fixtures)
- Temporary: `*.log`, `*.tmp`

If you **must** provide result files:

- Prefer a **small** set of curated assets in `paper/assets/` (or `docs/assets/`) with clear provenance.
- For large assets, use a release attachment or external archive (Zenodo/OSF), and document the download step.

### 3.2 Update `.gitignore`

Add patterns for:

-`.DS_Store`

-`__pycache__/`, `*.pyc`

-`.ipynb_checkpoints/`

-`results/`, `output/` folders

-`*.pkl`, large plots, and generated CSVs (unless they are curated fixtures)

### 3.3 Purge already-tracked artifacts

If artifacts were committed earlier, `.gitignore` alone is not enough. Remove them from git history tracking (keeping local copies if needed):

```bash

gitrm-r--cachedpath/to/results_or_artifacts

gitrm--cached"**/.DS_Store"

```

Then commit the removal and updated ignore rules.

### Done when

-`git status` shows no generated outputs after running experiments.

- New outputs do not show up as untracked files (because ignore rules cover them).

---

## Step 4 — Organize repository structure for readers

**Goal:** A reader instantly knows where to look.

### Recommended top-level structure

-`src/` — package code only

-`test/` — tests only

-`docs/` — package documentation (optional but strongly recommended)

-`paper/` (or `repro/`) — reproducibility scripts + notebooks + figure generation

-`paper/scripts/` — command-line scripts to run experiments

-`paper/notebooks/` — clean notebooks (optional)

-`paper/figures/` — *generated* outputs (ignored) OR curated small subset (tracked)

-`paper/data/` — small input data (tracked) OR download instructions

-`experiments/` — keep only if you clearly label it as “development/benchmark”; otherwise migrate into `paper/`

-`00_safety_checks_full_algorithms/` — migrate into `verification/` (see Step 7)

### Migration guidance

- If a script is needed to reproduce the paper: move it to `paper/scripts/`.
- If a script is a benchmark/profiling tool: keep it in `experiments/bench/` and mark “not part of reproduction”.

### Done when

- You can explain each top-level folder in one sentence.
- Paper reproduction is contained under one directory.

---

## Step 5 — Documentation of all functions (docstrings + user docs)

**Goal:** A user can understand every public API without reading the implementation.

### 5.1 Document public API in-code

For each exported function/type:

- Add a Julia docstring with:
- One-sentence purpose
- Argument meaning + expected shapes/layouts (critical in tensor code)
- Return values
- Notes on performance expectations (allocations, mutating `!` functions)
- A minimal runnable example

For tensor-network code, explicitly document:

- MPS tensor layout: `(Dl, d, Dr)`
- MPO tensor layout: `(Dl, d_out, d_in, Dr)` (or whatever you enforce—be consistent)
- Truncation rules and tolerances
- Determinism/seed requirements where randomness is used

### 5.2 Add user-facing docs (recommended)

Best practice in Julia is `Documenter.jl` in a separate `docs/` environment. This does not need to be a runtime dependency of the package.

Minimum viable docs site:

- Home: overview + install + quickstart
- “Theory/Conventions”: tensor layouts, truncation, noise models
- “API reference”: auto-generated from docstrings
- “Reproducing paper”: link to scripts/notebooks and how to run them

If you want **no extra tooling**, keep `docs/` as plain markdown and ensure README is thorough. (But CI-built docs is more professional.)

### Done when

-`?YourFunction` in Julia shows a complete docstring for every exported symbol.

- There is a single docs entrypoint (README and/or docs site) that covers installation + quickstart.

---

## Step 6 — Tests for all functions (unit + integration + determinism)

**Goal:**`Pkg.test()` meaningfully verifies correctness and prevents regressions.

### 6.1 Define testing tiers

-**Unit tests**: small, fast, local correctness of functions (tensor shapes, gate matrices, local contractions).

-**Integration tests**: run a small end-to-end simulation (small \(L\), few timesteps).

-**Regression tests**: fixed-seed outputs/observables for known scenarios.

### 6.2 Determinism rules

- Seed randomness at the start of tests using `Random.seed!(...)`.
- Avoid tests that depend on wall-clock time, thread scheduling, or non-deterministic ordering.
- Use tolerances for floating-point comparisons (`≈` / `isapprox`).

### 6.3 Coverage expectations

For publication-ready:

- All exported functions: at least one test that hits typical usage.
- All major algorithms: at least one integration test.
- Edge cases: invalid input, boundary sizes, empty systems if applicable, extreme tolerances.

### Done when

-`julia --project=. -e 'using Pkg; Pkg.test()'` passes reliably.

- Tests finish in a reasonable time (ideally minutes, not hours).

---

## Step 7 — Convert safety checks into a reproducible verification suite

**Goal:** Your `00_safety_checks_full_algorithms/` becomes a structured, repeatable “verification” pipeline, not a pile of scripts.

### 7.1 What belongs in verification vs tests?

-**Tests**: fast and run in CI on every commit.

-**Verification suite**: slower, paper-level, might run in CI on schedule or manually before release.

### 7.2 How to structure verification

Recommended:

- Create `verification/` (or `validation/`) with:

-`verification/runners/` — scripts that run each verification scenario

-`verification/reference/` — small fixed reference data (only if necessary)

-`verification/results/` — generated outputs (ignored)

Each verification scenario should:

- Have fixed parameters (and seeds if stochastic).
- Produce a small, machine-readable summary (e.g. JSON/CSV of final observables).
- Compare against a reference with tolerances.

### 7.3 Make verification callable

Provide one command to run everything, e.g.:

```bash

julia--project=.verification/run_all.jl

```

### Done when

- “Safety checks” do not generate random untracked clutter.
- You can run verification in one command and get a clear pass/fail report.

---

## Step 8 — Clean Jupyter notebooks and paper experiments

**Goal:** Notebooks are minimal, deterministic, and runnable by others.

### 8.1 Prefer scripts as the primary reproduction method

Best practice for publication:

- Make **scripts** the canonical reproduction path (easier to CI and document).
- Keep notebooks as optional, exploratory, or for pedagogy.

### 8.2 If you include notebooks

Rules:

- Clear all cell outputs before committing.
- Avoid storing large arrays or binary blobs inside `.ipynb`.
- Pin environment and kernels:
- If using IJulia, document how to install the kernel with your project.
- Alternatively, keep a separate `paper/Project.toml` for notebooks.
- Make notebooks restart-and-run-all clean.

### 8.3 Determinism

- Seed randomness in the first notebook cell.
- Keep runtime reasonable (or provide a “quick mode”).

### Done when

- A fresh user can “Restart & Run All” and obtain the documented result.
- Notebooks in git are clean (no huge diffs from output noise).

---

## Step 9 — Continuous Integration (CI) for credibility

**Goal:** Automated proof that the repo builds and tests pass.

### Minimum CI checks (GitHub Actions)

- Instantiate Julia project
- Run `Pkg.test()`
- Run a basic hygiene check (no large artifacts, no forbidden files)
- (Optional) build docs
- (Optional) run a small verification subset

### Hygiene check examples

Fail CI if repository contains:

-`.DS_Store`

-`__pycache__`

-`*.pyc`

- Jupyter checkpoints
- Large binary artifacts above some size threshold

### Done when

- Every push/PR triggers CI.
- CI passes on your supported Julia version(s).

---

## Step 10 — Release process for the paper (tag, archive, provenance)

**Goal:** The exact code used in the paper is permanently identifiable.

### Actions

1. Choose a version number (e.g. `v0.1.0` or `v1.0.0` depending on maturity).
2. Create a release tag on the commit used for the paper.
3. If using Zenodo:

- Connect GitHub repo releases to Zenodo and archive the release.

4. Update `CITATION.cff` with DOI after archival.

### Done when

- There is a tag/release used in the paper.
- Anyone can check out that tag and reproduce the documented results.

---

## Practical ordered checklist (recommended execution order)

Use this as the “do this now” list:

1.**Hygiene first**

- Add `.gitignore` rules for artifacts
- Remove already-tracked artifacts with `git rm --cached`

2.**Create README + LICENSE + CITATION**

3.**Stabilize environments**

- Ensure `Pkg.instantiate()` works from clean checkout
- Document Python as optional/required

4.**Refactor folder structure**

- Create `paper/` (or `repro/`) and move reproduction scripts there

5.**Docstrings for public API**

6.**Complete tests**

- Unit first, then integration/regression

7.**Turn safety checks into `verification/`**

8.**Clean notebooks**

9.**Add CI**

10.**Tag a release**

---

## Suggested “repo policies” (small but important)

-**No large artifacts in git**: keep only small curated examples/fixtures.

-**Mutating functions end with `!`** and document allocation behavior.

-**Every exported function must be tested** (at least one test).

-**Every paper figure/table must have a script** that regenerates it.

-**One command per workflow** (test, verify, reproduce paper).
