# Morris Method Global Sensitivity Analysis

Implementation of the Morris Method (Elementary Effects) parameter screening for
PyPSA-Earth sector-coupled energy system models.

---

## Table of Contents

1. [Background](#background)
2. [Method Summary](#method-summary)
3. [File Structure](#file-structure)
4. [Configuration](#configuration)
5. [Workflow DAG](#workflow-dag)
6. [Script Reference](#script-reference)
7. [Environment Setup](#environment-setup)
8. [Running the Analysis](#running-the-analysis)
9. [Output Files](#output-files)
10. [Interpreting Results](#interpreting-results)
11. [Example: ZA 2050 Test Run](#example-za-2050-test-run)

---

## Background

The Morris Method (1991) is a **one-at-a-time** global sensitivity analysis technique
designed for **parameter screening** — identifying which inputs have negligible,
linear, or nonlinear/interaction effects on model outputs, while keeping the number
of model evaluations manageable.

Each parameter is varied along a pre-designed trajectory through the input space.
For each trajectory, $D+1$ model evaluations are made (one per parameter change),
producing an **Elementary Effect** (EE) per parameter:

$$EE_i = \frac{f(x_1, \ldots, x_i + \Delta, \ldots, x_D) - f(x)}{\Delta}$$

Over $N$ trajectories, three sensitivity indices are computed per parameter per
output metric:

| Index | Symbol | Meaning |
|-------|--------|---------|
| Mean of EEs | $\mu$ | Overall influence (signed) |
| Mean of absolute EEs | $\mu^*$ | **Overall importance** — primary screening metric |
| Std dev of EEs | $\sigma$ | Non-linearity or interaction effects |

**Total model runs:** $K = N \times (D + 1)$, where $N$ = trajectories and $D$ =
number of parameters. For $N=4$, $D=3$: $K = 4 \times 4 = 16$ runs.

---

## Method Summary

The implementation uses [SALib](https://salib.readthedocs.io/) for both sampling
(`SALib.sample.morris`) and analysis (`SALib.analyze.morris`). The perturbation
approach is **multiplicative**: each parameter's nominal value in the PyPSA network
is scaled by a factor drawn from the Morris sample matrix. This mirrors the existing
`monte_carlo.py` implementation.

```
Sample matrix row i:  [factor_0, factor_1, ..., factor_{D-1}]
                           ↓
n.{attr} = n.{attr} * factor_j   (for each parameter j)
```

PyPSA network attributes are accessed via `exec()` using the `attr` string, allowing
arbitrary selectors such as `loads_t.p_set` or
`generators_t.p_max_pu.loc[:, n.generators.carrier == 'onwind']`.

---

## File Structure

```
scripts/
└── morris/
    ├── morris_sample.py         # Generate sample matrix with SALib
    ├── morris_perturb.py        # Apply multiplicative perturbation to prenetwork
    ├── morris_extract.py        # Extract scalar metrics from solved network
    ├── morris_analyze.py        # SALib analysis + plots
    └── morris_fill_missing.py   # Fill NaN placeholders for missing runs

envs/
└── pypsa-earth-morris.yaml  # Conda environment (PyPSA ≥ 1.1, SALib)

configs/GSA/
└── config.GSA_ZA_morris.yaml   # Test scenario: ZA, 10 clusters, 2050

config.default.yaml          # Contains default morris: section (lines ~441–475)

Snakefile                    # Morris rules block (lines 960–1103)
```

---

## Configuration

The `morris:` block in any config YAML controls the entire workflow.
The defaults live in `config.default.yaml` and are overridden per-scenario.

```yaml
morris:
  add_to_snakefile: false       # Set true to activate the Morris rules
  options:
    N: 4                        # Number of Morris trajectories
    num_levels: 4               # Grid levels for the Morris design
    seed: 42                    # RNG seed for reproducibility
  parameters:
    # Each entry is a PyPSA attribute to screen.
    # 'attr'   → PyPSA accessor string: n.{attr} is used via exec()
    # 'bounds' → multiplicative scaling factor [lower, upper]
    - name: load_scaling
      attr: loads_t.p_set
      bounds: [0.8, 1.2]
    - name: onwind_cf
      attr: "generators_t.p_max_pu.loc[:, n.generators.carrier == 'onwind']"
      bounds: [0.85, 1.15]
    - name: solar_cf
      attr: "generators_t.p_max_pu.loc[:, n.generators.carrier == 'solar']"
      bounds: [0.85, 1.15]
  results:
    # Metrics extracted from each solved network using n.stats.* (PyPSA ≥ 1.1).
    # 'stat'       → 'capex+opex', 'optimal_capacity', or 'energy_balance'
    # 'bus_carrier'→ optional filter for optimal_capacity / energy_balance
    - name: system_cost
      stat: capex+opex
    - name: H2_optimal_capacity
      stat: optimal_capacity
      bus_carrier: H2
    - name: AC_optimal_capacity
      stat: optimal_capacity
      bus_carrier: AC
```

**To enable the workflow** in a scenario config, override:

```yaml
morris:
  add_to_snakefile: true
```

---

## Workflow DAG

```
                          ┌────────────────┐
                          │  morris_sample │  (SALib sample matrix, once per scenario)
                          └───────┬────────┘
                                  │  sample_matrix.npy, problem.json
                  ┌───────────────┼──────────────────┐
                  │               │                  │
           ┌──────▼──────┐ ┌──────▼──────┐  ...  (K runs)
           │morris_perturb│ │morris_perturb│
           │   mr0        │ │   mr1        │
           └──────┬────────┘ └──────┬──────┘
                  │                 │   perturbed prenetworks
           ┌──────▼────────┐ ┌──────▼──────┐
           │solve_morris_  │ │solve_morris_ │
           │network  mr0   │ │network  mr1  │
           └──────┬────────┘ └──────┬──────┘
                  │                 │   solved networks
           ┌──────▼────────┐ ┌──────▼──────┐
           │morris_extract │ │morris_extract│
           │   mr0         │ │   mr1        │
           └──────┬────────┘ └──────┬──────┘
                  │                 │   per-run metric CSVs
                  └────────┬────────┘
                    ┌──────▼──────┐
                    │morris_analyze│  (collect all K CSVs → SALib analyze)
                    └──────┬──────┘
                           │  sensitivity_indices.csv + plots/
                    ┌──────▼──────┐
                    │  morris_all │  (aggregator target)
                    └─────────────┘
```

The `{morris_run}` wildcard (constrained to `mr[0-9]+`) parameterizes the
`morris_perturb`, `solve_morris_network`, and `morris_extract` rules across all
$K$ runs in parallel.

---

## Script Reference

### `scripts/morris/morris_sample.py`

**Conda env:** `pypsa-earth-morris`

Generates the Morris sample matrix and problem definition from `config["morris"]`.

| I/O | Path |
|-----|------|
| Output | `resources/{RDIR}morris/sample_matrix.npy` — shape $(K \times D)$ float64 |
| Output | `resources/{RDIR}morris/problem.json` — SALib problem dict |

Key logic:
```python
problem = {"num_vars": D, "names": [...], "bounds": [...]}
sample_matrix = SALib.sample.morris.sample(problem, N=N, num_levels=num_levels, seed=seed)
# shape: (N*(D+1), D) — each row is a vector of multiplicative factors
```

---

### `scripts/morris/morris_perturb.py`

**Conda env:** main `pypsa-earth` (needs full solver stack)

Loads row `i` of the sample matrix (from wildcard `mr{i}`) and applies
multiplicative perturbations to the prenetwork.

| I/O | Path |
|-----|------|
| Input | Prenetwork `.nc` (sector-coupled, all wildcards) |
| Input | `sample_matrix.npy`, `problem.json` |
| Output | Perturbed network `.nc` with `{morris_run}` wildcard |

Key logic:
```python
i = int(snakemake.wildcards.morris_run[2:])   # "mr5" → 5
row = sample_matrix[i, :]
for j, param in enumerate(parameters):
    exec(f"n.{param['attr']} = n.{param['attr']} * {row[j]}")
n.meta.update({"morris_run": ..., "morris_scaling_factors": {...}})
```

Morris metadata (run index, all scaling factors) is stored in `n.meta` for
traceability.

---

### `scripts/morris/morris_extract.py`

**Conda env:** `pypsa-earth-morris`

Extracts scalar metrics from a solved network using the `n.stats.*` API (PyPSA ≥ 1.1),
mirroring the approach in `post_processing/make_stats_dicts.py`.

| I/O | Path |
|-----|------|
| Input | Solved network `.nc` |
| Input | `problem.json` |
| Output | Single-row CSV: `morris_run, metric1, metric2, ...` |

Supported `stat` types:

| `stat` value | PyPSA call |
|---|---|
| `capex+opex` | `n.stats.capex().dropna().sum() + n.stats.opex().dropna().sum()` |
| `optimal_capacity` | `n.stats.optimal_capacity(bus_carrier=..., groupby="carrier", aggregate_across_components=True).sum()` |
| `energy_balance` | `n.stats.energy_balance(bus_carrier=..., groupby="carrier", aggregate_across_components=True).sum()` |

---

### `scripts/morris/morris_analyze.py`

**Conda env:** `pypsa-earth-morris`

Collects all $K$ per-run metric CSVs, runs `SALib.analyze.morris.analyze()` for
each output metric, and generates diagnostic plots.

| I/O | Path |
|-----|------|
| Input | All `morris/metrics/*.csv` (expanded over `morris_run`) |
| Input | `sample_matrix.npy`, `problem.json` |
| Output | `sensitivity_indices.csv` |
| Output | `plots/` directory (bar charts + scatter plots per metric) |

Output CSV columns: `metric, parameter, mu, mu_star, sigma, mu_star_conf`

Plots per metric:
- `bar_mu_star_{metric}.png` — horizontal bar chart of $\mu^*$ with confidence intervals
- `scatter_mu_star_sigma_{metric}.png` — $\mu^*$ vs $\sigma$ scatter with parameter labels

---

## Environment Setup

A dedicated conda environment `pypsa-earth-morris` is used for the sampling,
extraction, and analysis steps to avoid conflicts with the main `pypsa-earth` env:

```yaml
# envs/pypsa-earth-morris.yaml
name: pypsa-earth-morris
channels: [conda-forge, bioconda]
dependencies:
  - python>=3.10
  - pypsa>=1.1        # required for n.stats.* API
  - snakemake-minimal<8
  - salib
  - numpy
  - pandas
  - matplotlib
  - pip
```

**Install once:**
```bash
mamba env create -f envs/pypsa-earth-morris.yaml
```

The `solve_morris_network` rule intentionally uses the **main `pypsa-earth` env**
(no `conda:` directive) because it reuses `scripts/solve_network.py` and needs the
full Gurobi/HiGHS solver stack.

---

## Running the Analysis

### 1. Enable in config

In your scenario config (e.g. `configs/GSA/config.GSA_ZA_morris.yaml`), add:

```yaml
morris:
  add_to_snakefile: true
  options:
    N: 4
    num_levels: 4
    seed: 42
  parameters:
    - name: load_scaling
      attr: loads_t.p_set
      bounds: [0.8, 1.2]
    # ... more parameters
  results:
    - name: system_cost
      stat: capex+opex
    # ... more metrics
```

### 2. Check the run count

With $N$ trajectories and $D$ parameters, the total number of solve jobs is:

$$K = N \times (D + 1)$$

For $N=4$, $D=3$: $K=16$ independent optimisations. Each uses 16 threads and
the memory configured in `config["solving"]["mem"]`.

### 3. Launch via Snakemake

Dry-run to verify the DAG:
```bash
snakemake --configfile configs/GSA/config.GSA_ZA_morris.yaml \
          morris_all --dry-run --quiet
```

Local run:
```bash
snakemake --configfile configs/GSA/config.GSA_ZA_morris.yaml \
          morris_all --cores all --use-conda
```

SLURM run (via `run_slurm.sh`, change target to `morris_all`):
```bash
bash run_slurm.sh
```

---

## Output Files

All outputs are placed under `results/{SECDIR}/morris/`:

```
results/{SECDIR}/morris/
├── perturbed/                       # K perturbed prenetworks
│   └── elec_s..._exp{eopts}_{mr*}.nc
├── solved/                          # K solved networks
│   └── elec_s..._exp{eopts}_{mr*}.nc
├── metrics/                         # K single-row metric CSVs
│   └── elec_s..._exp{eopts}_{mr*}.csv
├── sensitivity_indices_s..._{eopts}.csv   # Final sensitivity table
└── plots_s..._{eopts}/
    ├── bar_mu_star_{metric}.png
    └── scatter_mu_star_sigma_{metric}.png

resources/{RDIR}/morris/
├── sample_matrix.npy               # Shape (K, D)
└── problem.json                    # SALib problem definition
```

---

## Interpreting Results

The key columns in `sensitivity_indices.csv` are:

| Column | Interpretation |
|--------|----------------|
| `mu_star` | **Primary screening metric.** Parameters with high $\mu^*$ have large effects on the output. |
| `sigma` | High $\sigma$ relative to $\mu^*$ indicates non-linearity or interaction with other parameters. |
| `mu` | Signed mean; positive/negative indicates direction of effect. |
| `mu_star_conf` | Bootstrap 95% confidence interval on $\mu^*$. |

**Decision rule (Campolongo et al., 2007):**
- $\mu^*$ small, $\sigma$ small → parameter is **non-influential** (can be fixed)
- $\mu^*$ large, $\sigma$ small → parameter has **linear, additive** effect
- $\mu^*$ large, $\sigma$ large → parameter has **non-linear or interaction** effects

The $\mu^*$ vs $\sigma$ scatter plot directly visualises these three regions.

---

## Example: ZA 2050 Test Run

Config file: `configs/GSA/config.GSA_ZA_morris.yaml`

| Setting | Value |
|---------|-------|
| Country | ZA (South Africa) |
| Clusters | 10 |
| Time resolution | 1H |
| Planning horizon | 2050 |
| Demand scenario | RF |
| Export options | `NH3v0.5+FTv0.5+MEOHv0.5+HBIv0.5` |
| Solver | Gurobi, 16 threads |
| Morris N | 4 trajectories |
| Morris D | 3 parameters (`load_scaling`, `onwind_cf`, `solar_cf`) |
| Total runs K | $4 \times (3+1) = 16$ |

**Screened parameters:**

| Parameter | `attr` | Bounds |
|-----------|--------|--------|
| `load_scaling` | `loads_t.p_set` | [0.80, 1.20] |
| `onwind_cf` | `generators_t.p_max_pu.loc[:, n.generators.carrier == 'onwind']` | [0.85, 1.15] |
| `solar_cf` | `generators_t.p_max_pu.loc[:, n.generators.carrier == 'solar']` | [0.85, 1.15] |

**Screened metrics:**

| Metric | `stat` | `bus_carrier` |
|--------|--------|---------------|
| `system_cost` | `capex+opex` | — |
| `H2_optimal_capacity` | `optimal_capacity` | H2 |
| `AC_optimal_capacity` | `optimal_capacity` | AC |

---

## References

- Morris, M.D. (1991). *Factorial sampling plans for preliminary computational experiments.*
  Technometrics, 33(2), 161–174.
- Campolongo, F., Cariboni, J., Saltelli, A. (2007). *An effective screening design for
  sensitivity analysis of large models.* Environmental Modelling & Software, 22, 1509–1518.
- SALib documentation: https://salib.readthedocs.io/en/latest/api/SALib.analyze.html#SALib.analyze.morris.analyze
- PyPSA `statistics` API: https://pypsa.readthedocs.io/en/latest/api_reference/networks.html#pypsa.Network.statistics
