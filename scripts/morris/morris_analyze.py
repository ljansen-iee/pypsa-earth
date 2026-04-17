# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Performs Morris Method sensitivity analysis on extracted metrics.

Collects per-run metric CSVs, loads the sample matrix and problem definition,
and runs ``SALib.analyze.morris.analyze`` for each result metric.

Outputs
-------
- ``sensitivity_indices.csv``: μ, μ*, σ, μ*_conf per parameter per metric
- ``plots/``: bar charts (μ*) and scatter plots (μ* vs σ) per metric
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.analyze.morris import analyze as morris_analyze

logger = logging.getLogger(__name__)

COLORS = [
    "#005b7f", "#008598", "#179c7d", "#39c1cd", "#4CC2A6",
    "#669db2", "#a6bbc8", "#b2d235", "#C2D05C", "#face61",
    "#fce356", "#fdb913", "#f58220", "#d6a67c", "#f08591",
    "#a8508c", "#7c154d", "#bb0056", "#836bad", "#1c3f52",
    "#454545", "#C0C0C0", "#d3c7ae",
]

DIVERGING = [
    [0.0, "#179c7d"],
    [0.15, "#4CC2A6"],
    [0.3, "#39c1cd"],
    [0.45, "#a6bbc8"],
    [0.55, "#d3c7ae"],
    [0.7, "#f08591"],
    [0.85, "#a8508c"],
    [1.0, "#7c154d"],
]

SEQUENTIAL = [
    [0.0, "#005b7f"],
    [0.25, "#008598"],
    [0.5, "#39c1cd"],
    [0.75, "#a6bbc8"],
    [1.0, "#d3c7ae"],
]


if __name__ == "__main__":
    if "snakemake" not in globals():
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "morris_analyze",
            simpl="",
            clusters="10",
            ll="v1.0",
            opts="Co2L0.25",
            sopts="1H",
            planning_horizons="2050",
            discountrate="0.09",
            demand="RF",
            eopts="NH3v0.5+FTv0.5+MEOHv0.5+HBIv0.5",
            configfile="configs/GSA/config.GSA_ZA_morris.yaml"
        )

    morris_config = snakemake.params.morris
    results_config = morris_config["results"]
    metric_names = [r["name"] for r in results_config]

    # Load problem definition and sample matrix
    with open(snakemake.input.problem, "r") as f:
        problem = json.load(f)
    sample_matrix = np.load(snakemake.input.sample_matrix)

    logger.info(
        f"Problem: {problem['num_vars']} parameters, "
        f"sample matrix shape: {sample_matrix.shape}"
    )

    # Collect all per-run metric CSVs into a single DataFrame
    metric_files = snakemake.input.metrics
    if isinstance(metric_files, str):
        metric_files = [metric_files]

    dfs = []
    for f in sorted(metric_files):
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            logger.warning(f"Could not read {f}: {e}")

    if not dfs:
        raise ValueError("No metric files could be loaded.")

    all_metrics = pd.concat(dfs, ignore_index=True)

    # Sort by morris_run index to match sample matrix row order
    all_metrics["_sort_key"] = all_metrics["morris_run"].str[2:].astype(int)
    all_metrics = all_metrics.sort_values("_sort_key").reset_index(drop=True)
    all_metrics = all_metrics.drop(columns=["_sort_key"])

    logger.info(f"Collected {len(all_metrics)} metric rows")

    # Filter to complete trajectories only — handles missing/failed runs.
    # Each trajectory spans D+1 consecutive rows; incomplete ones are dropped
    # so the remaining sample matrix and Y stay aligned.
    D = problem["num_vars"]
    total_runs = sample_matrix.shape[0]
    N_original = total_runs // (D + 1)

    all_metrics["_run_idx"] = all_metrics["morris_run"].str[2:].astype(int)
    metrics_indexed = all_metrics.set_index("_run_idx")
    available_run_indices = set(metrics_indexed.index.tolist())

    complete_rows = []
    n_complete = 0
    for t in range(N_original):
        traj_rows = list(range(t * (D + 1), t * (D + 1) + (D + 1)))
        if all(r in available_run_indices for r in traj_rows):
            complete_rows.extend(traj_rows)
            n_complete += 1

    n_missing = total_runs - len(available_run_indices)
    logger.info(
        f"Complete trajectories: {n_complete} / {N_original} "
        f"({n_missing} individual runs missing, "
        f"{total_runs - len(complete_rows)} runs dropped)"
    )

    if n_complete == 0:
        raise ValueError(
            "No complete trajectories found. Cannot perform Morris analysis. "
            "Each trajectory requires all D+1 runs to be present."
        )

    if n_complete < N_original:
        logger.warning(
            f"Proceeding with {n_complete} complete trajectories out of {N_original}. "
            f"Results remain statistically valid but with reduced power."
        )

    sample_matrix_filtered = sample_matrix[complete_rows, :]

    # Run Morris analysis for each metric
    all_indices = []

    for metric_name in metric_names:
        if metric_name not in all_metrics.columns:
            logger.warning(f"Metric '{metric_name}' not found in collected data, skipping.")
            continue

        Y_candidate = metrics_indexed.loc[complete_rows, metric_name].values.astype(float)

        if np.any(np.isnan(Y_candidate)):
            # Drop trajectories that contain NaN for this specific metric rather
            # than skipping the metric entirely.  A single failed solve only
            # invalidates its own trajectory; the remaining ones are still valid.
            nan_run_set = {
                r for r in complete_rows
                if np.isnan(float(metrics_indexed.loc[r, metric_name]))
            }
            metric_rows = []
            metric_n_complete = 0
            for t in range(N_original):
                traj_rows = list(range(t * (D + 1), t * (D + 1) + (D + 1)))
                if (
                    all(r in available_run_indices for r in traj_rows)
                    and not any(r in nan_run_set for r in traj_rows)
                ):
                    metric_rows.extend(traj_rows)
                    metric_n_complete += 1

            n_nan_traj = n_complete - metric_n_complete
            if metric_n_complete == 0:
                logger.warning(
                    f"Metric '{metric_name}' has NaN values in all complete trajectories. "
                    f"Skipping Morris analysis for this metric."
                )
                continue

            logger.warning(
                f"Metric '{metric_name}': dropped {n_nan_traj} trajectory/trajectories "
                f"containing NaN. Proceeding with {metric_n_complete} complete trajectories."
            )
            metric_sample_matrix = sample_matrix[metric_rows, :]
            Y = metrics_indexed.loc[metric_rows, metric_name].values.astype(float)
        else:
            metric_sample_matrix = sample_matrix_filtered
            Y = Y_candidate

        Si = morris_analyze(problem, metric_sample_matrix, Y)

        for k, param_name in enumerate(problem["names"]):
            all_indices.append({
                "metric": metric_name,
                "parameter": param_name,
                "mu": Si["mu"][k],
                "mu_star": Si["mu_star"][k],
                "sigma": Si["sigma"][k],
                "mu_star_conf": Si["mu_star_conf"][k],
            })

        logger.info(f"Morris analysis for '{metric_name}' complete.")

    # Save sensitivity indices
    indices_df = pd.DataFrame(
        all_indices,
        columns=["metric", "parameter", "mu", "mu_star", "sigma", "mu_star_conf"],
    ) if all_indices else pd.DataFrame(
        columns=["metric", "parameter", "mu", "mu_star", "sigma", "mu_star_conf"]
    )
    indices_df.to_csv(snakemake.output.indices, index=False)
    logger.info(f"Saved sensitivity indices to {snakemake.output.indices}")

    if indices_df.empty:
        logger.error(
            "No sensitivity indices were computed — all metrics were skipped (all NaN). "
            "Make sure morris_extract has been run for solved networks before morris_analyze. "
            "Run: snakemake --allowed-rules morris_extract --configfile <cfg> -j8 --keep-going"
        )
        raise ValueError("No sensitivity indices computed — indices_df is empty.")

    # Create plots directory
    plot_dir = Path(snakemake.output.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    for metric_name in metric_names:
        metric_df = indices_df[indices_df["metric"] == metric_name]
        if metric_df.empty:
            continue

        # --- Bar chart: μ* (screening importance) ---
        fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(metric_df))))
        metric_df_sorted = metric_df.sort_values("mu_star", ascending=True)
        bar_colors = [
            COLORS[i % len(COLORS)] for i in range(len(metric_df_sorted))
        ]
        ax.barh(
            metric_df_sorted["parameter"],
            metric_df_sorted["mu_star"],
            xerr=metric_df_sorted["mu_star_conf"],
            capsize=3,
            color=bar_colors,
        )
        ax.set_xlabel("μ* (mean absolute elementary effect)")
        ax.set_title(f"Morris Screening: {metric_name}")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"bar_mu_star_{metric_name}.png", dpi=250)
        plt.close(fig)

        # --- Scatter plot: μ* vs σ ---
        fig, ax = plt.subplots(figsize=(7, 6))
        scatter_colors = [
            COLORS[i % len(COLORS)] for i in range(len(metric_df))
        ]
        ax.scatter(
            metric_df["mu_star"],
            metric_df["sigma"],
            s=80,
            color=scatter_colors,
            edgecolors="black",
            linewidths=0.5,
        )
        for _, row in metric_df.iterrows():
            ax.annotate(
                row["parameter"],
                (row["mu_star"], row["sigma"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
        ax.set_xlabel("μ* (mean absolute elementary effect)")
        ax.set_ylabel("σ (standard deviation of elementary effects)")
        ax.set_title(f"Morris Screening: {metric_name}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"scatter_mu_star_sigma_{metric_name}.png", dpi=250)
        plt.close(fig)

        logger.info(f"Saved plots for '{metric_name}'")

    logger.info("Morris analysis complete.")
