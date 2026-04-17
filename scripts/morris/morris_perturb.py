# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Perturbs a prenetwork for a single Morris run.

Reads the ``{morris_run}`` wildcard (e.g. ``mr5``), extracts row ``i=5``
from the sample matrix, and applies multiplicative scaling to each parameter
defined in the ``morris.parameters`` config section.

The perturbation logic mirrors ``monte_carlo.py``: each parameter's ``attr``
is a PyPSA accessor string used via ``exec(f"n.{attr} = n.{attr} * factor")``.

Inputs
------
- Prenetwork ``.nc`` file (sector-coupled, with export wildcards)
- ``sample_matrix.npy`` from ``morris_sample``
- ``problem.json`` from ``morris_sample``

Outputs
-------
- Perturbed prenetwork ``.nc`` file with ``{morris_run}`` wildcard
"""

import json
import logging

import numpy as np
import pypsa
from pypsa.costs import annuity

logger = logging.getLogger(__name__)


def apply_wacc_perturbation(n, r_old, r_new):
    """
    Rescale capital_cost of all extendable components by the annuity ratio.

    For each component with a positive lifetime and non-zero capital_cost,
    multiply capital_cost by annuity(r_new, lifetime) / annuity(r_old, lifetime).
    This correctly accounts for the nonlinear dependence of annualised costs
    on the discount rate, and for per-component lifetime differences.

    Uses ``pypsa.costs.annuity(discount_rate, lifetime)`` (PyPSA >= 1.1).
    """
    components = ["generators", "links", "stores", "storage_units", "lines"]
    total_scaled = 0

    for comp_name in components:
        df = getattr(n, comp_name)
        if df.empty or "capital_cost" not in df.columns or "lifetime" not in df.columns:
            continue

        mask = (df["capital_cost"] > 0) & (df["lifetime"] > 0)
        if not mask.any():
            continue

        lifetimes = df.loc[mask, "lifetime"]
        ratio = annuity(r_new, lifetimes) / annuity(r_old, lifetimes)
        df.loc[mask, "capital_cost"] *= ratio
        total_scaled += mask.sum()

        logger.info(
            f"  WACC: scaled {mask.sum()} {comp_name} capital_costs "
            f"(r: {r_old:.4f} → {r_new:.4f}, "
            f"ratio range: [{ratio.min():.4f}, {ratio.max():.4f}])"
        )

    logger.info(f"  WACC: {total_scaled} components rescaled in total")
    return total_scaled


if __name__ == "__main__":
    if "snakemake" not in globals():
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "morris_perturb",
            simpl="",
            clusters="10",
            ll="v1.2",
            opts="Co2L0.97",
            sopts="1H",
            planning_horizons="2030",
            discountrate="0.117",
            demand="EL",
            eopts="H2v1.0+NH3v1.0+MEOHv1.0",
            morris_run="mr0",
        )

    morris_config = snakemake.params.morris
    parameters = morris_config["parameters"]

    # Extract run index from wildcard, e.g. "mr5" -> 5
    morris_run_wildcard = snakemake.wildcards.morris_run
    i = int(morris_run_wildcard[2:])  # strip "mr" prefix

    # Load sample matrix and extract row i
    sample_matrix = np.load(snakemake.input.sample_matrix)
    row = sample_matrix[i, :]

    logger.info(
        f"Morris run {morris_run_wildcard} (row {i}): "
        f"scaling factors = {row.tolist()}"
    )

    # Load prenetwork
    n = pypsa.Network(snakemake.input.prenetwork)

    # Build a lookup of parameter name -> sampled factor so that
    # capex_deviation params can reference their linked macro parameter.
    param_factors = {param["name"]: float(row[j]) for j, param in enumerate(parameters)}

    # Apply perturbation for each parameter
    for j, param in enumerate(parameters):
        factor = row[j]
        param_type = param.get("type", "attr")

        if param_type == "macro":
            # Virtual parameter: no direct network modification.
            # Its factor is consumed by capex_deviation params that reference it.
            logger.info(
                f"  Macro param '{param['name']}' = {factor:.6f} "
                f"(applied via linked capex_deviation params)"
            )

        elif param_type == "capex_deviation":
            # Multiplier trick: effective_factor = macro_factor * deviation_factor
            macro_name = param["macro"]
            macro_factor = param_factors[macro_name]
            effective_factor = macro_factor * factor
            attr_list = param.get("attrs", [param["attr"]] if "attr" in param else [])
            for attr in attr_list:
                exec(f"n.{attr} = n.{attr} * {effective_factor}")
                logger.info(
                    f"  Scaled n.{attr} by effective factor {effective_factor:.6f} "
                    f"(macro={macro_factor:.6f} * dev={factor:.6f}, "
                    f"param '{param['name']}')"
                )

        elif param_type == "discount_rate":
            # Special handling: rescale all capital_costs via annuity ratio
            r_old = param["base_value"]
            r_new = r_old * factor
            apply_wacc_perturbation(n, r_old, r_new)
            logger.info(
                f"  discount_rate perturbation: r_old={r_old:.4f}, factor={factor:.6f}, "
                f"r_new={r_new:.4f} (param '{param['name']}')"
            )
        elif param_type == "co2_sequestration":
            # Store perturbed value in network metadata so that
            # solve_network.py can pick it up in add_co2_sequestration_limit
            base_val = param["base_value"]
            new_val = base_val * factor
            n.meta["co2_sequestration_potential_override"] = float(new_val)
            logger.info(
                f"  CO2 sequestration: {base_val} -> {new_val:.2f} MtCO2/a "
                f"(factor={factor:.6f}, param '{param['name']}')"
            )
        else:
            # Default: multiplicative scaling via attr accessor.
            # Supports both "attr" (single string) and "attrs" (list of strings)
            # so that coupled attributes (e.g. e_nom + e_initial) scale together.
            attr_list = param.get("attrs", [param["attr"]] if "attr" in param else [])
            for attr in attr_list:
                exec(f"n.{attr} = n.{attr} * {factor}")
                logger.info(
                    f"  Scaled n.{attr} by factor {factor:.6f} "
                    f"(param '{param['name']}')"
                )

    # Store Morris metadata in network
    n.meta.update({
        "morris_run": morris_run_wildcard,
        "morris_row_index": i,
        "morris_scaling_factors": {
            param["name"]: float(row[j])
            for j, param in enumerate(parameters)
        },
    })

    n.export_to_netcdf(snakemake.output[0])
    logger.info(f"Exported perturbed network to {snakemake.output[0]}")
