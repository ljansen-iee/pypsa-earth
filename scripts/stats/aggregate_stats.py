"""
Aggregate per-network stats parquet files into summary CSVs.

Called via Snakemake rule ``make_stats_aggregate``.

For each metric, reads all per-network parquet files produced by
``make_stats_single``, concatenates them, and writes the final summary
CSVs matching the structure of the original ``make_stats_dicts.py``.

Inputs
------
snakemake.input.stats_dirs : list of str
    All per-network stats directories produced by ``make_stats_single``.

Outputs
-------
snakemake.output.summary_dir : str
    Directory containing the final summary CSVs.

Params
------
snakemake.params.run_name_prefix : str
snakemake.params.run_names : list of str
snakemake.params.summary_version : str
snakemake.params.plot_summary_config : str
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# plot_helpers lives in post_processing/, not scripts/
sys.path.insert(0, "post_processing")
from plot_helpers import to_csv_nafix, prepare_dataframe, get_scen_col_function

idx_slice = pd.IndexSlice

# ---------------------------------------------------------------------------
# Setup from Snakemake
# ---------------------------------------------------------------------------

stats_dirs = [Path(d) for d in snakemake.input.stats_dirs]
sdir = Path(snakemake.output.summary_dir)
sdir.mkdir(parents=True, exist_ok=True)

run_name_prefix = snakemake.params.run_name_prefix
run_names = snakemake.params.run_names
plot_summary_config_path = snakemake.params.plot_summary_config

IDX_NAMES = [
    "run_name_prefix", "run_name", "country", "year",
    "simpl", "clusters", "ll", "opts", "sopts",
    "discountrate", "demand", "eopts",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_and_concat(metric_filename: str) -> pd.DataFrame:
    """Read a per-network parquet from all stats dirs and concatenate into one DataFrame.

    Returns a DataFrame with IDX_NAMES as its index and carriers as columns.
    """
    dfs = []
    for d in stats_dirs:
        p = d / metric_filename
        if p.exists():
            dfs.append(pd.read_parquet(p))
    if not dfs:
        return pd.DataFrame(columns=IDX_NAMES)
    combined = pd.concat(dfs, ignore_index=True)
    idx_cols = [c for c in IDX_NAMES if c in combined.columns]
    return combined.set_index(idx_cols)


def save_dict(d: dict, name: str) -> None:
    for key, df in d.items():
        safe_key = key.replace(" ", "_").replace("/", "_")
        path = sdir / f"{name}_{safe_key}.csv"
        to_csv_nafix(df, path)
        logger.info(f"Saved {path}")


# ---------------------------------------------------------------------------
# Energy balance per bus_carrier in TWh
# ---------------------------------------------------------------------------

BALANCE_CARRIERS = [
    "AC", "H2", "oil", "gas", "co2 stored", "co2", "methanol", "NH3",
    "steel", "HBI", "solid biomass", "biogas", "H2O", "purewater", "seawater",
]

balance_dict = {}
for bc in BALANCE_CARRIERS:
    df = read_and_concat(f"balance_{bc.replace(' ', '_')}.parquet")
    if not df.empty:
        balance_dict[bc] = df

# ---------------------------------------------------------------------------
# Optimal capacities per bus_carrier in GW
# ---------------------------------------------------------------------------

OPT_CAP_CARRIERS = ["AC", "H2", "methanol", "NH3", "steel", "HBI"]

optimal_capacity_dict = {}
for bc in OPT_CAP_CARRIERS:
    df = read_and_concat(f"optimal_capacity_{bc.replace(' ', '_')}.parquet")
    if not df.empty:
        optimal_capacity_dict[bc] = df

# ---------------------------------------------------------------------------
# Capex / opex in billion currency units
# ---------------------------------------------------------------------------

costs_dict = {
    "capex": read_and_concat("costs_capex.parquet"),
    "opex": read_and_concat("costs_opex.parquet"),
}

# ---------------------------------------------------------------------------
# Expanded storage capacity in GWh
# ---------------------------------------------------------------------------

stores = read_and_concat("stores.parquet")

# ---------------------------------------------------------------------------
# Load-weighted marginal prices in currency/MWh
# ---------------------------------------------------------------------------

load_avg_marginal_price = read_and_concat("marginal_prices.parquet")

# ---------------------------------------------------------------------------
# GW·km (multi-row parquet: one row per carrier per network)
# ---------------------------------------------------------------------------

gwkm_raw_dfs = []
for d in stats_dirs:
    p = d / "gwkm.parquet"
    if p.exists():
        gwkm_raw_dfs.append(pd.read_parquet(p))

gwkm_dict = {}
if gwkm_raw_dfs:
    gwkm_all = pd.concat(gwkm_raw_dfs, ignore_index=True)
    for carrier in ["AC", "H2 pipeline"]:
        subset = gwkm_all[gwkm_all["carrier"] == carrier].drop("carrier", axis=1)
        if not subset.empty:
            gwkm_dict[carrier] = subset.set_index(IDX_NAMES)

# ---------------------------------------------------------------------------
# Build nc_files index (union of all network indices seen)
# ---------------------------------------------------------------------------

all_tuples = set()
for df in list(balance_dict.values()) + list(optimal_capacity_dict.values()):
    if not df.empty:
        for tup in df.index:
            all_tuples.add(tup)

nc_files = pd.DataFrame(
    index=pd.MultiIndex.from_tuples(sorted(all_tuples), names=IDX_NAMES)
)

# ---------------------------------------------------------------------------
# Save all outputs
# ---------------------------------------------------------------------------

to_csv_nafix(nc_files, sdir / "nc_files.csv")
logger.info(f"Saved nc_files to {sdir / 'nc_files.csv'}")

save_dict(balance_dict, "balance_dict")
save_dict(optimal_capacity_dict, "optimal_capacity_dict")
save_dict(costs_dict, "costs_dict")

to_csv_nafix(stores, sdir / "stores.csv")
logger.info(f"Saved stores to {sdir / 'stores.csv'}")

to_csv_nafix(load_avg_marginal_price, sdir / "load_avg_marginal_price.csv")
logger.info(f"Saved load_avg_marginal_price to {sdir / 'load_avg_marginal_price.csv'}")

save_dict(gwkm_dict, "gwkm_dict")

# ---------------------------------------------------------------------------
# Marginal prices prepared (long-form with scen column, for visualization)
# ---------------------------------------------------------------------------

try:
    plot_summary_config = yaml.safe_load(Path(plot_summary_config_path).read_text())
    index_levels_to_drop = plot_summary_config["data"]["index_levels_to_drop"]
    scen_filter = plot_summary_config["data"]["scen_filter"]
    scen_col_func_name = plot_summary_config["data"]["scen_col_function"]

    set_scen_col = get_scen_col_function(scen_col_func_name)

    available_countries = nc_files.index.get_level_values("country").unique().tolist()
    available_years = nc_files.index.get_level_values("year").unique().tolist()
    idx_group_all = idx_slice[[run_name_prefix], :, available_countries, available_years]

    marginal_prices_prepared = prepare_dataframe(
        load_avg_marginal_price, idx_group_all, index_levels_to_drop, set_scen_col, drop_zero=False
    )
    if scen_filter:
        marginal_prices_prepared = marginal_prices_prepared.query("scen in @scen_filter")

    to_csv_nafix(marginal_prices_prepared, sdir / "marginal_prices_prepared.csv")
    logger.info(f"Saved marginal_prices_prepared to {sdir / 'marginal_prices_prepared.csv'}")
except Exception as e:
    logger.warning(f"Could not prepare marginal prices: {e}")

logger.info(f"Aggregation complete. All outputs written to {sdir}")
