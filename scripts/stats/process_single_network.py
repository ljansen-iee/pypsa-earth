"""
Process a single postnetwork file and write per-network stats to a directory of parquet files.

Called via Snakemake rule ``make_stats_single``.

Each output parquet is a single-row (or multi-row for gwkm) DataFrame with the
network's MultiIndex values prepended as plain columns, so that the aggregator
script can simply ``pd.concat`` them across all networks.

Inputs
------
snakemake.input.network : str
    Path to the solved postnetwork ``.nc`` file.

Outputs
-------
snakemake.output.stats_dir : str
    Directory populated with per-metric parquet files.

Params
------
snakemake.params.run_name_prefix : str
    Experiment name prefix (e.g. ``"WSA"``).  Used to infer country from run name.
"""

import logging
from pathlib import Path

import pandas as pd
import pypsa

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Setup from Snakemake
# ---------------------------------------------------------------------------

network_path = snakemake.input.network
out_dir = Path(snakemake.output.stats_dir)
out_dir.mkdir(parents=True, exist_ok=True)

wc = snakemake.wildcards
run_name = wc.run_name
run_name_prefix = snakemake.params.run_name_prefix

# Infer country as the first segment of run_name after the prefix
# e.g. prefix="WSA", run_name="WSA_MA_2050_low50" -> country="MA"
country = run_name.removeprefix(run_name_prefix + "_").split("_")[0]

# MultiIndex definition (same as in aggregate and original make_stats_dicts)
IDX_NAMES = [
    "run_name_prefix", "run_name", "country", "year",
    "simpl", "clusters", "ll", "opts", "sopts",
    "discountrate", "demand", "eopts",
]
idx_values = (
    run_name_prefix,
    run_name,
    country,
    int(wc.planning_horizons),
    wc.simpl,
    wc.clusters,
    wc.ll,
    wc.opts,
    wc.sopts,
    float(wc.discountrate),
    wc.demand,
    wc.eopts,
)
idx = pd.MultiIndex.from_tuples([idx_values], names=IDX_NAMES)


def save_series(ds: pd.Series, filename: str) -> None:
    """Save a Series as a single-row parquet with MultiIndex values prepended as columns."""
    if ds is None or ds.empty:
        pd.DataFrame(columns=IDX_NAMES).to_parquet(out_dir / filename, index=False)
        return
    df = ds.to_frame().T
    df.index = idx
    df.reset_index().to_parquet(out_dir / filename, index=False)


# ---------------------------------------------------------------------------
# Load network
# ---------------------------------------------------------------------------

pypsa.options.params.statistics.nice_names = False
pypsa.options.params.statistics.drop_zero = False
pypsa.options.params.statistics.round = 6

n = pypsa.Network(network_path)
logger.info(f"Loaded network: {network_path}")

# ---------------------------------------------------------------------------
# Energy balance per bus_carrier in TWh
# ---------------------------------------------------------------------------

BALANCE_CARRIERS = [
    "AC", "H2", "oil", "gas", "co2 stored", "co2", "methanol", "NH3",
    "steel", "HBI", "solid biomass", "biogas", "H2O", "purewater", "seawater",
]

for bus_carrier in BALANCE_CARRIERS:
    if bus_carrier not in n.buses.carrier.unique():
        continue

    ds = (
        n.stats.energy_balance(
            bus_carrier=bus_carrier,
            groupby="carrier",
            aggregate_across_components=True,
        )
        .div(1e6)
        .round(1)
    )

    if bus_carrier == "AC" and "low voltage" in n.buses.carrier.unique():
        ds_lv = (
            n.stats.energy_balance(
                bus_carrier="low voltage",
                groupby="carrier",
                aggregate_across_components=True,
            )
            .div(1e6)
            .round(1)
        )
        ds = pd.concat([ds, ds_lv])
        ds = ds[ds.index != "electricity distribution grid"]

    save_series(ds, f"balance_{bus_carrier.replace(' ', '_')}.parquet")

# ---------------------------------------------------------------------------
# Optimal capacities per bus_carrier in GW (output units)
# ---------------------------------------------------------------------------

OPT_CAP_CARRIERS = ["AC", "H2", "methanol", "NH3", "steel", "HBI"]

for bus_carrier in OPT_CAP_CARRIERS:
    ds = (
        n.stats.optimal_capacity(
            components=["Generator", "Link", "StorageUnit"],
            bus_carrier=bus_carrier,
            groupby="carrier",
            aggregate_across_components=True,
        )
        .loc[lambda x: x > 0]
        .div(1e3)
        .round(1)
    )

    if bus_carrier == "AC" and "low voltage" in n.buses.carrier.unique():
        ds_lv = (
            n.stats.optimal_capacity(
                components=["Generator", "Link", "StorageUnit"],
                bus_carrier="low voltage",
                groupby="carrier",
                aggregate_across_components=True,
            )
            .loc[lambda x: x > 0]
            .div(1e3)
            .round(1)
        )
        if not ds_lv.empty:
            ds_lv = ds_lv[ds_lv.index != "electricity distribution grid"]
            ds = pd.concat([ds, ds_lv])

    save_series(ds, f"optimal_capacity_{bus_carrier.replace(' ', '_')}.parquet")

# ---------------------------------------------------------------------------
# Capex / opex in billion currency units
# ---------------------------------------------------------------------------

for cost_type in ["capex", "opex"]:
    fn = getattr(n.stats, cost_type)
    ds = fn().dropna().groupby("carrier").sum().div(1e9).round(4)
    save_series(ds, f"costs_{cost_type}.parquet")

# ---------------------------------------------------------------------------
# Expanded storage capacity in GWh
# ---------------------------------------------------------------------------

ec_stores = n.stats.expanded_capacity(
    components=["Store"],
    groupby="carrier",
    aggregate_across_components=True,
)
STORE_COLS = ["H2 Store Tank", "H2 UHS", "battery", "home battery"]
ds_stores = ec_stores.reindex(STORE_COLS).div(1e3).round(1)
save_series(ds_stores, "stores.parquet")

# ---------------------------------------------------------------------------
# Load-weighted marginal prices in currency/MWh
# ---------------------------------------------------------------------------

BUS_CARRIERS_TO_PRICE = [
    "H2", "NH3", "FT", "HBI", "steel", "STEEL",
    "industry methanol", "shipping methanol", "MEOH",
    "H2O", "purewater", "seawater",
]
prices_lw = n.stats.prices(groupby_time=True, weighting="load")
price_data = {}
for bc in BUS_CARRIERS_TO_PRICE:
    for bus in n.buses.index[n.buses.index.str.contains(bc)]:
        if bus in prices_lw.index:
            price_data[bus] = prices_lw[bus]
save_series(pd.Series(price_data), "marginal_prices.parquet")

# ---------------------------------------------------------------------------
# GW·km (AC lines and H2 pipelines)
# Multi-row parquet: one row per carrier, with IDX_NAMES columns prepended
# ---------------------------------------------------------------------------

def calculate_gwkm(n, which="optimal"):
    """Calculate GW·km for transmission assets."""
    selection = [
        "H2 pipeline", "H2 pipeline retrofitted",
        "gas pipeline", "gas pipeline new", "DC",
    ]
    gwkm = n.links.loc[n.links.carrier.isin(selection)]
    if which == "optimal":
        lk, ln = "p_nom_opt", "s_nom_opt"
    elif which == "added":
        lk, ln = "(p_nom_opt - p_nom)", "(s_nom_opt - s_nom)"
    elif which == "existing":
        lk, ln = "p_nom", "s_nom"
    gwkm = gwkm.eval(f"length*{lk}").groupby(gwkm.carrier).sum() / 1e3  # GWkm
    gwkm["AC"] = n.lines.eval(f"length*{ln}").sum() / 1e3  # GWkm
    gwkm.index.name = None
    return gwkm.round(1)


try:
    df_gwkm = pd.DataFrame(
        {k: calculate_gwkm(n, which=k) for k in ["optimal", "added", "existing"]}
    )
    df_gwkm["ratio"] = df_gwkm["added"] / df_gwkm["existing"].replace(0, pd.NA)
    df_gwkm.index.name = "carrier"
    df_gwkm = df_gwkm.reset_index()
    # Prepend network index columns so the aggregator can reconstruct the MultiIndex
    for col, val in zip(reversed(IDX_NAMES), reversed(idx_values)):
        df_gwkm.insert(0, col, val)
    df_gwkm.to_parquet(out_dir / "gwkm.parquet", index=False)
except Exception as e:
    logger.warning(f"Could not calculate gwkm: {e}")
    pd.DataFrame(
        columns=IDX_NAMES + ["carrier", "optimal", "added", "existing", "ratio"]
    ).to_parquet(out_dir / "gwkm.parquet", index=False)

logger.info(f"Done. Stats written to {out_dir}")
