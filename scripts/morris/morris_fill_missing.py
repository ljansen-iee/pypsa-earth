# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Creates NaN-filled placeholder metric CSVs for any missing Morris runs.

Run this after a SLURM batch in which some solve jobs failed. The placeholder
CSVs let Snakemake proceed to ``morris_analyze``, which already skips NaN
rows and drops incomplete trajectories automatically.

Usage (from repo root, with the morris conda env active):
    python scripts/morris/morris_fill_missing.py \\
        --configfile configs/GSA/config.GSA_ZA_morris.yaml

Or just run without arguments — it will load config.yaml from the repo root.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(configfile: str) -> dict:
    """Load and merge config.yaml + override configfile (like Snakemake does)."""
    repo_root = Path(__file__).parent.parent.parent

    with open(repo_root / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    if configfile:
        with open(configfile) as f:
            override = yaml.safe_load(f)
        # Deep-merge override into cfg (top-level keys only needed here)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v

    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configfile",
        default="configs/GSA/config.GSA_ZA_morris.yaml",
        help="Path to the scenario config YAML (default: configs/GSA/config.GSA_ZA_morris.yaml)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.parent
    cfg = load_config(args.configfile)

    run_name = cfg["run"]["name"]
    morris_cfg = cfg["morris"]

    D = len(morris_cfg["parameters"])
    N = morris_cfg["options"]["N"]
    K = N * (D + 1)

    # Reconstruct the metric file path pattern (mirrors the Snakefile)
    scenario = cfg["scenario"]
    simpl    = scenario["simpl"][0]
    clusters = scenario["clusters"][0]
    ll       = scenario["ll"][0]
    opts     = scenario["opts"][0]
    sopts    = scenario["sopts"][0]
    horizon  = scenario["planning_horizons"][0]
    dr       = cfg["costs"]["discountrate"][0]
    demand   = scenario["demand"][0]
    eopts    = scenario["eopts"][0]

    # Format discount rate the same way Snakemake does (strips trailing zeros)
    dr_str = str(dr).rstrip("0").rstrip(".")

    stem = (
        f"elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{horizon}"
        f"_{dr_str}_{demand}_exp{eopts}"
    )
    metrics_dir = repo_root / "results" / run_name / "morris" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Column names: morris_run + one column per result metric
    result_names = [r["name"] for r in morris_cfg["results"]]
    columns = ["morris_run"] + result_names

    solved_dir = repo_root / "results" / run_name / "morris" / "solved"

    created = 0
    skipped_has_network = 0
    for i in range(K):
        csv_path = metrics_dir / f"{stem}_mr{i}.csv"
        if not csv_path.exists():
            solved_path = solved_dir / f"{stem}_mr{i}.nc"
            if solved_path.exists():
                # Solved network exists but metrics not yet extracted — do NOT fill.
                # Run: snakemake --allowed-rules morris_extract first.
                skipped_has_network += 1
            else:
                # Solve failed / network missing — fill with NaN placeholder.
                nan_row = {"morris_run": f"mr{i}"}
                nan_row.update({col: float("nan") for col in result_names})
                pd.DataFrame([nan_row]).to_csv(csv_path, index=False)
                logger.info(f"  Created NaN placeholder: {csv_path.name}")
                created += 1

    if skipped_has_network > 0:
        logger.warning(
            f"\n{skipped_has_network} runs have solved networks but no metric CSV yet. "
            f"Run morris_extract first:\n"
            f"  snakemake --allowed-rules morris_extract --configfile {args.configfile} "
            f"-j8 --keep-going --rerun-trigger mtime\n"
            f"Then re-run this script."
        )
    if created == 0 and skipped_has_network == 0:
        logger.info("All metric CSVs already present — nothing to do.")
    elif created > 0:
        logger.info(
            f"\nCreated {created} NaN placeholder CSVs (failed/missing solve jobs)."
        )
        logger.info(
            "morris_analyze will drop the corresponding incomplete trajectories."
        )

    # Quick sanity check: how many complete trajectories remain?
    problem_path = repo_root / "resources" / run_name / "morris" / "problem.json"
    if problem_path.exists():
        with open(problem_path) as f:
            problem = json.load(f)
        D_check = problem["num_vars"]

        existing = set()
        for i in range(K):
            csv_path = metrics_dir / f"{stem}_mr{i}.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    first_metric = result_names[0] if result_names else None
                    if first_metric and not np.isnan(df[first_metric].values[0]):
                        existing.add(i)
                except Exception:
                    pass

        complete = 0
        for t in range(N):
            traj = list(range(t * (D_check + 1), t * (D_check + 1) + (D_check + 1)))
            if all(r in existing for r in traj):
                complete += 1

        logger.info(
            f"\nSanity check: {complete}/{N} complete trajectories "
            f"(from {len(existing)}/{K} successful runs)"
        )


if __name__ == "__main__":
    main()
