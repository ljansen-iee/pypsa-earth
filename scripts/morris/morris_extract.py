# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Extracts scalar result metrics from a solved Morris network.

Uses the ``n.stats.*`` API (PyPSA >= 1.1) to compute metrics defined in the
``morris.results`` config section. The extraction approach mirrors
``post_processing/make_stats_dicts.py``.

Supported ``stat`` types:
- ``capex+opex``: sum of ``n.stats.capex()`` + ``n.stats.opex()``
- ``optimal_capacity``: ``n.stats.optimal_capacity(bus_carrier=...)``
- ``energy_balance``: ``n.stats.energy_balance(bus_carrier=...)``
- ``export_marginal_price``: load-weighted mean price via ``n.stats.prices(groupby_time=True, weighting="load")[bus_carrier + ' export']``

Inputs
------
- Solved network ``.nc`` file
- ``problem.json`` from ``morris_sample``

Outputs
-------
- Single-row CSV with columns: ``morris_run, metric1, metric2, ...``
"""

import json
import logging

import pandas as pd
import pypsa

logger = logging.getLogger(__name__)


def extract_metric(n, result_config):
    """
    Extract a single scalar metric from the solved network.

    Parameters
    ----------
    n : pypsa.Network
        Solved network with optimization results.
    result_config : dict
        Single entry from config["morris"]["results"], e.g.
        {"name": "system_cost", "stat": "capex+opex"}

    Returns
    -------
    float
        Scalar metric value.
    """
    stat = result_config["stat"]
    bus_carrier = result_config.get("bus_carrier", None)

    if stat == "capex+opex":
        capex = n.stats.capex().dropna().sum()
        opex = n.stats.opex().dropna().sum()
        return float(capex + opex)

    elif stat == "optimal_capacity":
        if bus_carrier is None:
            raise ValueError(
                f"Result '{result_config['name']}' with stat='optimal_capacity' "
                f"requires 'bus_carrier' to be specified."
            )
        ds = n.stats.optimal_capacity(
            bus_carrier=bus_carrier,
            groupby="carrier",
            aggregate_across_components=True,
        )
        return float(ds.sum())

    elif stat == "energy_balance":
        if bus_carrier is None:
            raise ValueError(
                f"Result '{result_config['name']}' with stat='energy_balance' "
                f"requires 'bus_carrier' to be specified."
            )
        ds = n.stats.energy_balance(
            bus_carrier=bus_carrier,
            groupby="carrier",
            aggregate_across_components=True,
        )
        return float(ds.sum())

    elif stat == "export_marginal_price":
        if bus_carrier is None:
            raise ValueError(
                f"Result '{result_config['name']}' with stat='export_marginal_price' "
                f"requires 'bus_carrier' to be specified (e.g. 'NH3', 'FT', 'H2')."
            )
        bus_name = bus_carrier + " export"
        # n.stats.prices(groupby_time=True, weighting="load") returns a load-weighted
        # mean price per bus, matching the approach in post_processing/make_stats_dicts.py.
        prices = n.stats.prices(groupby_time=True, weighting="load")
        if bus_name not in prices.index:
            raise ValueError(
                f"Export bus '{bus_name}' not found in n.stats.prices() result. "
                f"Buses with 'export' present: "
                f"{[b for b in prices.index if 'export' in b.lower()]}"
            )
        return float(prices[bus_name])

    else:
        raise ValueError(
            f"Unknown stat type '{stat}' for result '{result_config['name']}'. "
            f"Supported: 'capex+opex', 'optimal_capacity', 'energy_balance', "
            f"'export_marginal_price'."
        )


if __name__ == "__main__":
    if "snakemake" not in globals():
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "morris_extract",
            simpl="",
            clusters="10",
            ll="v1.0",
            opts="Co2L0.25",
            sopts="1H",
            planning_horizons="2050",
            discountrate="0.09",
            demand="RF",
            eopts="NH3v0.5+FTv0.5+MEOHv0.5+HBIv0.5",
            morris_run="mr15",
            configfile="configs/GSA/config.GSA_ZA_morris.yaml"
        )

    morris_config = snakemake.params.morris
    results_config = morris_config["results"]

    # Extract morris_run wildcard
    morris_run = snakemake.wildcards.morris_run

    # Configure PyPSA statistics options (matching make_stats_dicts.py)
    pypsa.options.params.statistics.nice_names = False
    pypsa.options.params.statistics.drop_zero = False
    pypsa.options.params.statistics.round = 6

    # Load solved network
    n = pypsa.Network(snakemake.input.network)
    logger.info(f"Loaded solved network for {morris_run}")

    # Extract all metrics
    metrics = {"morris_run": morris_run}
    for result in results_config:
        name = result["name"]
        try:
            value = extract_metric(n, result)
            metrics[name] = value
            logger.info(f"  {name} = {value:.6f}")
        except Exception as e:
            logger.warning(f"  Failed to extract '{name}': {e}")
            metrics[name] = float("nan")

    # Write single-row CSV
    df = pd.DataFrame([metrics])
    df.to_csv(snakemake.output[0], index=False)
    logger.info(f"Saved metrics to {snakemake.output[0]}")
