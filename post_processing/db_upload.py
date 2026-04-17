"""
Upload consolidated and renamed summary tables to postgres database.

Mirrors every chart produced by plot_summary.ipynb, using the same
plot_helpers functions and plot_summary.yaml configuration.

Usage
-----
1. Configure ``plot_summary.yaml`` (countries, years, scen_filter, etc.)
2. Optionally set UPLOAD_RUN_NAME_PREFIX below to override the experiment
   name stored in the DB (useful for splitting experiments into chunks).
3. Run: ``python post_processing/db_upload.py``
"""
# %%
from pathlib import Path
import re
import pandas as pd
import country_converter as coco

idx_slice = pd.IndexSlice

from energyants.core.db import db_connector, run_sql, write_to_db
from plot_helpers import (
    chdir_to_root_dir,
    read_stats_dict,
    read_csv_nafix,
    prepare_dataframe,
    get_supply_demand_from_balance,
    rename_electricity,
    rename_h2,
    rename_gas,
    rename_oil,
    rename_co2,
    rename_co2_stored,
    rename_costs,
    rename_stores,
    rename_h2o,
    colors,
    get_scen_col_function,
    load_plot_config,
    build_ptx_exports_df,
    build_ptx_exports_h2_equiv_df,
    build_maritime_df,
    build_aviation_df,
    INDEX_COLS,
)


# ──────────────────────────────────────────────────────────────────────────────
# Override: set a different experiment name for the DB upload.
# If None, the name_prefix from plot_summary.yaml is used.
# ──────────────────────────────────────────────────────────────────────────────
UPLOAD_RUN_NAME_PREFIX = "AB_and_AT_compare_two_transition_scenarios"   # e.g. "DKS_ZA_2030_AB"
# UPLOAD_RUN_NAME_PREFIX = "AT_Test_export_volume" 
# DB target
DB_SCHEMA = "ptx_pathways"
DB_OWNER = "ptx_pathways_rolle"


# ──────────────────────────────────────────────────────────────────────────────
# Dataset meta-info: short descriptions for every table written to the DB.
# Keys are the table names used in upload(); values are human-readable strings.
# ──────────────────────────────────────────────────────────────────────────────
DATASET_META = {
    "electricity_supply_twh": (
        "Annual electricity supply by generation technology (solar, wind, hydro, …) per country "
        "and scenario in TWh. Derived from the AC carrier energy balance."
    ),
    "electricity_demand_twh": (
        "Annual electricity demand broken down by end-use sector per country and scenario in TWh. "
        "Derived from the AC carrier energy balance."
    ),
    "electricity_installed_capacity_gw": (
        "Optimal installed electricity generation and storage capacity by technology per country "
        "and scenario in GW. Load-shedding variables are excluded."
    ),
    "hydrogen_balance_twh": (
        "Annual hydrogen production, consumption, import and export flows per country and scenario "
        "in TWh, covering electrolysis, fuel cells and all H₂-related processes."
    ),
    "hydrogen_installed_capacity_gw": (
        "Optimal installed hydrogen infrastructure capacity (electrolysers, fuel cells, compressors, "
        "etc.) per country and scenario in GW."
    ),
    "liquid_fuel_balance_twh": (
        "Annual liquid-fuel (oil/e-fuel) supply and demand balance per country and scenario in TWh, "
        "including imports, refining and final consumption. Zero-value rows are excluded."
    ),
    "ch4_balance_twh": (
        "Annual methane/gas carrier supply and demand balance per country and scenario in TWh, "
        "covering natural gas, bio-methane and synthetic methane. Zero-value rows are excluded."
    ),
    "co2_capture_and_usage_mt": (
        "Annual CO₂ capture, geological storage and utilisation flows per country and scenario in "
        "Mt CO₂. Derived from the 'co2 stored' carrier balance."
    ),
    "co2_emissions_mt": (
        "Annual gross CO₂ emissions by sector and technology per country and scenario in Mt CO₂. "
        "Derived from the 'co2' carrier balance."
    ),
    "water_balance_twh": (
        "Annual general water (H₂O) supply and demand balance per country and scenario in TWh, "
        "covering all water flows in the energy system."
    ),
    "purewater_balance_twh": (
        "Annual pure/desalinated water supply and demand balance per country and scenario in TWh, "
        "tracking freshwater used for electrolysis and other processes."
    ),
    "seawater_balance_twh": (
        "Annual seawater intake and usage balance per country and scenario in TWh, representing "
        "the raw seawater feed to desalination and cooling processes."
    ),
    "storage_capacity_gwh": (
        "Optimal energy storage volume capacity (stores) by technology per country and scenario "
        "in GWh, covering batteries, hydrogen caverns, and other storage assets."
    ),
    "system_costs_billion_eur": (
        "Total annualised system costs decomposed into capital expenditure (CAPEX) and operational "
        "expenditure (OPEX) by technology per country and scenario in billion EUR."
    ),
    "ptx_export_twh": (
        "Power-to-X (PtX) export volumes by product type (H₂, ammonia, methanol, …) and exporting "
        "country per scenario in TWh."
    ),
    "ptx_export_h2eq_twh": (
        "PtX export volumes expressed in hydrogen-equivalent energy content per product and "
        "exporting country per scenario in TWh_H₂eq, enabling cross-product comparisons."
    ),
    "maritime_demand_twh": (
        "Annual maritime sector fuel demand by fuel type (ammonia, methanol, LNG, …) per country "
        "and scenario in TWh."
    ),
    "aviation_demand_twh": (
        "Annual aviation sector fuel demand by fuel type (e-kerosene, hydrogen, …) per country "
        "and scenario in TWh."
    ),
    "grid_km_ac_gwkm": (
        "AC transmission grid buildout showing existing and newly added line capacity per country "
        "and scenario in GW·km."
    ),
    "grid_km_h2_pipeline_gwkm": (
        "Hydrogen pipeline grid buildout showing existing and newly added pipeline capacity per "
        "country and scenario in GW·km."
    ),
    "water_cost_breakdown_eur_per_mwh_h2": (
        "Water-related cost components (desalination, seawater intake, H₂O pipeline, storage) "
        "normalised to H₂ output per country and scenario in €/MWh_H₂."
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def convert_country_to_iso3(df):
    """Convert country column to ISO-3 codes; drop unresolved rows."""
    df = df.copy()
    cc = coco.CountryConverter()
    df["country"] = cc.pandas_convert(
        series=pd.Series(df["country"]), to="ISO3", not_found="not found"
    )
    df = df.loc[df["country"] != "not found"]
    df["country"] = df["country"].astype(str)
    df = df[~df["country"].str.contains(",", na=False)].reset_index(drop=True)
    return df


def to_snake_case(name):
    """Convert a pretty variable name to snake_case for postgres."""
    s = name.strip()
    s = re.sub(r"[^A-Za-z0-9]+", " ", s).strip()
    s = re.sub(r"\s+", "_", s).lower()
    return s


def consistency_check(df):
    """Raise if duplicate (run_name_prefix, scen, year, country, variable) rows exist."""
    check_cols = [
        c for c in ["run_name_prefix", "run_name", "scen", "year", "country", "variable"]
        if c in df.columns
    ]
    duplicates = df.duplicated(subset=check_cols, keep=False)
    if duplicates.any():
        raise ValueError(
            f"Duplicate rows found for columns {check_cols}.\n"
            f"First duplicates:\n{df[duplicates].head(10)}"
        )


def get_missing_colors(df, colors_dict):
    """Print & return variables that have no colour specification."""
    missing = set(df["variable"]) - set(colors_dict.keys())
    if missing:
        print(f"  ⚠️  Missing colour specs: {missing}")
    return missing


# ──────────────────────────────────────────────────────────────────────────────
# Variable-name mapping registry
# ──────────────────────────────────────────────────────────────────────────────

_VARIABLE_MAP = {}   # snake_case → pretty name


def snake_case_variables(df):
    """Replace pretty variable names with snake_case; register mapping."""
    df = df.copy()
    for pretty in df["variable"].unique():
        sc = to_snake_case(pretty)
        _VARIABLE_MAP[sc] = pretty
    df["variable"] = df["variable"].map(to_snake_case)
    return df


def save_variable_mapping(sdir):
    """Save the accumulated snake_case → pretty mapping to CSV."""
    mapping_df = pd.DataFrame(
        sorted(_VARIABLE_MAP.items()),
        columns=["variable_snake_case", "variable_pretty"],
    )
    out = sdir / "variable_name_mapping.csv"
    mapping_df.to_csv(out, index=False)
    print(f"\n✅  Variable mapping saved to {out}  ({len(mapping_df)} entries)")


# ──────────────────────────────────────────────────────────────────────────────
# Upload helper
# ──────────────────────────────────────────────────────────────────────────────

def upload(df, table_name, color_key, run_name_prefix_override=None):
    """Validate, convert countries, snake-case variables, and write to DB."""
    if df.empty:
        print(f"  ⏭️  {table_name}: empty DataFrame — skipping")
        return

    df = convert_country_to_iso3(df)
    if df.empty:
        print(f"  ⏭️  {table_name}: no valid countries after ISO3 conversion — skipping")
        return

    # Ensure run_name_prefix exists in every table; override if requested
    if run_name_prefix_override:
        df["run_name_prefix"] = run_name_prefix_override
    elif "run_name_prefix" not in df.columns:
        df["run_name_prefix"] = RUN_NAME_PREFIX

    consistency_check(df)
    if color_key and color_key in colors:
        get_missing_colors(df, colors[color_key])

    df = snake_case_variables(df)

    write_to_db(
        df=df,
        table=table_name,
        schema=DB_SCHEMA,
        connector=db_connector('work_db'),
        owner=DB_OWNER,
        if_exists="append",
    )
    print(f"  ✅  Uploaded → {DB_SCHEMA}.{table_name}  ({len(df)} rows)")


# %%
# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

chdir_to_root_dir()

# ── Load YAML configuration (same as plot_summary.ipynb) ─────────────────
yaml_config = load_plot_config()

RUN_NAME_PREFIX = yaml_config["run"]["name_prefix"]
SUMMARY_VERSION = yaml_config["run"]["summary_version"]
SDIR = Path.cwd() / "results" / f"{RUN_NAME_PREFIX}_summary_{SUMMARY_VERSION}"

COUNTRIES = yaml_config["data"]["countries"]
YEARS = yaml_config["data"]["years"]
INDEX_LEVELS_TO_DROP = yaml_config["data"]["index_levels_to_drop"]
SCEN_FILTER = yaml_config["data"]["scen_filter"]
scen_order = yaml_config["data"]["scen_order"]

scen_col_func_name = yaml_config["data"]["scen_col_function"]
set_scen_col = get_scen_col_function(scen_col_func_name)

IDX_GROUP = idx_slice[[RUN_NAME_PREFIX], :, COUNTRIES, YEARS]

# Effective run name prefix for the DB
_upload_prefix = UPLOAD_RUN_NAME_PREFIX or RUN_NAME_PREFIX

print(f"Source dir : {SDIR}")
print(f"Countries  : {COUNTRIES}")
print(f"Years      : {YEARS}")
print(f"Scen filter: {SCEN_FILTER or 'None (all)'}")
print(f"DB prefix  : {_upload_prefix}")

# ── Load data (same as plot_summary.ipynb) ────────────────────────────────
balance_keys = yaml_config["carriers"]["balance"]
capacity_keys = yaml_config["carriers"]["capacity"]
cost_keys = yaml_config["carriers"]["costs"]

balance_dict = read_stats_dict("balance_dict", SDIR, keys=balance_keys)
optimal_capacity_dict = read_stats_dict("optimal_capacity_dict", SDIR, keys=capacity_keys)
costs_dict = read_stats_dict("costs_dict", SDIR, keys=cost_keys)

# Stores
try:
    stores = read_csv_nafix(SDIR / "stores.csv")
except FileNotFoundError:
    stores = None
    print("⚠️  stores.csv not found — stores upload will be skipped")

# Grid km
try:
    gwkm_ac = read_csv_nafix(SDIR / "gwkm_dict_AC.csv", index_col=INDEX_COLS)
except FileNotFoundError:
    gwkm_ac = None
try:
    gwkm_h2 = read_csv_nafix(SDIR / "gwkm_dict_H2 pipeline.csv", index_col=INDEX_COLS)
except FileNotFoundError:
    gwkm_h2 = None


# %%
# ══════════════════════════════════════════════════════════════════════════════
# UPLOADS — one section per chart in plot_summary.ipynb
# ══════════════════════════════════════════════════════════════════════════════

# --------------------------------------------------------------------------
# 1. Electricity supply (TWh)
# --------------------------------------------------------------------------
print("\n── Electricity Supply ──")
df = prepare_dataframe(
    balance_dict["AC"], IDX_GROUP, INDEX_LEVELS_TO_DROP, set_scen_col,
    rename_function=rename_electricity,
)
if SCEN_FILTER:
    df = df.query("scen in @SCEN_FILTER")
supply_df, _, demand_df, _ = get_supply_demand_from_balance(df)
upload(supply_df, "electricity_supply_twh", "electricity",
       run_name_prefix_override=_upload_prefix)

# --------------------------------------------------------------------------
# 2. Electricity demand (TWh)
# --------------------------------------------------------------------------
print("\n── Electricity Demand ──")
upload(demand_df, "electricity_demand_twh", "electricity",
       run_name_prefix_override=_upload_prefix)

# --------------------------------------------------------------------------
# 3. Electricity installed capacity (GW)
# --------------------------------------------------------------------------
print("\n── Electricity Capacity ──")
df = prepare_dataframe(
    optimal_capacity_dict["AC"], IDX_GROUP, INDEX_LEVELS_TO_DROP, set_scen_col,
    rename_function=rename_electricity,
)
df = df[~df["variable"].isin(["Load shedding", "load shedding"])]
if SCEN_FILTER:
    df = df.query("scen in @SCEN_FILTER")
upload(df, "electricity_installed_capacity_gw", "electricity",
       run_name_prefix_override=_upload_prefix)

# --------------------------------------------------------------------------
# 4. Hydrogen balance (TWh)
# --------------------------------------------------------------------------
print("\n── Hydrogen Balance ──")
df = prepare_dataframe(
    balance_dict["H2"], IDX_GROUP, INDEX_LEVELS_TO_DROP, set_scen_col,
    rename_function=rename_h2,
)
if SCEN_FILTER:
    df = df.query("scen in @SCEN_FILTER")
upload(df, "hydrogen_balance_twh", "hydrogen",
       run_name_prefix_override=_upload_prefix)

# --------------------------------------------------------------------------
# 5. Hydrogen installed capacity (GW)
# --------------------------------------------------------------------------
print("\n── Hydrogen Capacity ──")
df = prepare_dataframe(
    optimal_capacity_dict["H2"], IDX_GROUP, INDEX_LEVELS_TO_DROP, set_scen_col,
    rename_function=rename_h2,
)
df = df[~df["variable"].isin(["Load shedding", "load shedding"])]
if SCEN_FILTER:
    df = df.query("scen in @SCEN_FILTER")
upload(df, "hydrogen_installed_capacity_gw", "hydrogen",
       run_name_prefix_override=_upload_prefix)

# --------------------------------------------------------------------------
# 6. Liquid fuel balance (TWh)
# --------------------------------------------------------------------------
print("\n── Liquid Fuel Balance ──")
df = prepare_dataframe(
    balance_dict["oil"], IDX_GROUP, INDEX_LEVELS_TO_DROP, set_scen_col,
    rename_function=rename_oil,
)
df = df[df["value"] != 0]
if SCEN_FILTER:
    df = df.query("scen in @SCEN_FILTER")
upload(df, "liquid_fuel_balance_twh", "oil",
       run_name_prefix_override=_upload_prefix)

# --------------------------------------------------------------------------
# 7. CH4 balance (TWh)
# --------------------------------------------------------------------------
print("\n── CH4 Balance ──")
df = prepare_dataframe(
    balance_dict["gas"], IDX_GROUP, INDEX_LEVELS_TO_DROP, set_scen_col,
    rename_function=rename_gas,
)
df = df[df["value"] != 0]
if SCEN_FILTER:
    df = df.query("scen in @SCEN_FILTER")
upload(df, "ch4_balance_twh", "gas",
       run_name_prefix_override=_upload_prefix)

# --------------------------------------------------------------------------
# 8. CO2 Capture and Usage (Mt)
# --------------------------------------------------------------------------
print("\n── CO2 Capture & Usage ──")
if "co2 stored" in balance_dict:
    df = prepare_dataframe(
        balance_dict["co2 stored"], IDX_GROUP, INDEX_LEVELS_TO_DROP, set_scen_col,
        rename_function=rename_co2_stored,
    )
    df = df[df["value"] != 0]
    if SCEN_FILTER:
        df = df.query("scen in @SCEN_FILTER")
    upload(df, "co2_capture_and_usage_mt", "co2 stored",
           run_name_prefix_override=_upload_prefix)
else:
    print("  ⏭️  'co2 stored' carrier not available — skipping")

# --------------------------------------------------------------------------
# 9. CO2 Emissions (Mt)
# --------------------------------------------------------------------------
print("\n── CO2 Emissions ──")
if "co2" in balance_dict:
    df = prepare_dataframe(
        balance_dict["co2"], IDX_GROUP, INDEX_LEVELS_TO_DROP, set_scen_col,
        rename_function=rename_co2,
    )
    if SCEN_FILTER:
        df = df.query("scen in @SCEN_FILTER")
    upload(df, "co2_emissions_mt", "co2",
           run_name_prefix_override=_upload_prefix)
else:
    print("  ⏭️  'co2' carrier not available — skipping")

# --------------------------------------------------------------------------
# 10. Water balances (TWh) — H2O / purewater / seawater
# --------------------------------------------------------------------------
print("\n── Water Balances ──")
for water_carrier, table_name in [
    ("H2O",       "water_balance_twh"),
    ("purewater", "purewater_balance_twh"),
    ("seawater",  "seawater_balance_twh"),
]:
    if water_carrier in balance_dict:
        df = prepare_dataframe(
            balance_dict[water_carrier], IDX_GROUP, INDEX_LEVELS_TO_DROP, set_scen_col,
            rename_function=rename_h2o,
        )
        if SCEN_FILTER:
            df = df.query("scen in @SCEN_FILTER")
        upload(df, table_name, None,
               run_name_prefix_override=_upload_prefix)
    else:
        print(f"  ⏭️  '{water_carrier}' not available — skipping")

# --------------------------------------------------------------------------
# 11. Storage volume capacity (GWh)
# --------------------------------------------------------------------------
print("\n── Storage Volume Capacity ──")
if stores is not None:
    stores_indexed = (
        stores.set_index(INDEX_COLS)
        if not isinstance(stores.index, pd.MultiIndex)
        else stores
    )
    df = prepare_dataframe(
        stores_indexed, IDX_GROUP, INDEX_LEVELS_TO_DROP, set_scen_col,
        rename_function=rename_stores,
    )
    if SCEN_FILTER:
        df = df.query("scen in @SCEN_FILTER")
    upload(df, "storage_capacity_gwh", None,
           run_name_prefix_override=_upload_prefix)
else:
    print("  ⏭️  stores data not available — skipping")

# --------------------------------------------------------------------------
# 12. System costs (billion EUR)
# --------------------------------------------------------------------------
print("\n── System Costs ──")
stats_df = pd.concat([costs_dict["capex"], costs_dict["opex"]], axis=0)
df = prepare_dataframe(
    stats_df, IDX_GROUP, INDEX_LEVELS_TO_DROP, set_scen_col,
    rename_function=rename_costs,
)
df = df[df["value"] != 0]
if SCEN_FILTER:
    df = df.query("scen in @SCEN_FILTER")
upload(df, "system_costs_billion_eur", "costs",
       run_name_prefix_override=_upload_prefix)

# --------------------------------------------------------------------------
# 13. PtX export volumes (TWh)
# --------------------------------------------------------------------------
print("\n── PtX Export Volumes ──")
df_exports = build_ptx_exports_df(
    balance_dict, COUNTRIES, YEARS, set_scen_col, INDEX_LEVELS_TO_DROP,
    scen_filter=SCEN_FILTER, sdir=SDIR,
)
upload(df_exports, "ptx_export_twh", "export_ptx",
       run_name_prefix_override=_upload_prefix)

# --------------------------------------------------------------------------
# 14. PtX export volumes — H₂ equivalent (TWh)
# --------------------------------------------------------------------------
print("\n── PtX Export Volumes (H₂ equiv) ──")
df_exports_h2eq = build_ptx_exports_h2_equiv_df(
    balance_dict, COUNTRIES, YEARS, set_scen_col, INDEX_LEVELS_TO_DROP,
    scen_filter=SCEN_FILTER, sdir=SDIR,
)
upload(df_exports_h2eq, "ptx_export_h2eq_twh", "export_ptx_h2eq",
       run_name_prefix_override=_upload_prefix)

# --------------------------------------------------------------------------
# 15. Maritime fuel demand (TWh)
# --------------------------------------------------------------------------
print("\n── Maritime Fuel Demand ──")
df_maritime = build_maritime_df(
    balance_dict, COUNTRIES, YEARS, set_scen_col, INDEX_LEVELS_TO_DROP,
    scen_filter=SCEN_FILTER,
)
upload(df_maritime, "maritime_demand_twh", "maritime",
       run_name_prefix_override=_upload_prefix)

# --------------------------------------------------------------------------
# 16. Aviation fuel demand (TWh)
# --------------------------------------------------------------------------
print("\n── Aviation Fuel Demand ──")
df_aviation = build_aviation_df(
    balance_dict, COUNTRIES, YEARS, set_scen_col, INDEX_LEVELS_TO_DROP,
    scen_filter=SCEN_FILTER,
)
upload(df_aviation, "aviation_demand_twh", "aviation",
       run_name_prefix_override=_upload_prefix)

# --------------------------------------------------------------------------
# 17. Grid km — AC (GW·km)
# --------------------------------------------------------------------------
print("\n── Grid km (AC) ──")
if gwkm_ac is not None:
    df = gwkm_ac.copy().loc[IDX_GROUP].reset_index()
    df = set_scen_col(df, index_levels_to_drop=INDEX_LEVELS_TO_DROP)
    for col in ["existing", "added"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    agg_cols = [c for c in ["scen", "country", "year"] if c in df.columns]
    df = df.groupby(agg_cols, as_index=False)[["existing", "added"]].mean()
    if SCEN_FILTER:
        df = df[df["scen"].isin(SCEN_FILTER)]
    df_long = df.melt(
        id_vars=agg_cols, value_vars=["existing", "added"],
        var_name="variable", value_name="value",
    )
    df_long["variable"] = df_long["variable"].str.capitalize()
    upload(df_long, "grid_km_ac_gwkm", None,
           run_name_prefix_override=_upload_prefix)
else:
    print("  ⏭️  gwkm_dict_AC.csv not available — skipping")

# --------------------------------------------------------------------------
# 18. Grid km — H2 pipeline (GW·km)
# --------------------------------------------------------------------------
print("\n── Grid km (H2 Pipeline) ──")
if gwkm_h2 is not None:
    df = gwkm_h2.copy().loc[IDX_GROUP].reset_index()
    df = set_scen_col(df, index_levels_to_drop=INDEX_LEVELS_TO_DROP)
    for col in ["existing", "added"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    agg_cols = [c for c in ["scen", "country", "year"] if c in df.columns]
    df = df.groupby(agg_cols, as_index=False)[["existing", "added"]].mean()
    if SCEN_FILTER:
        df = df[df["scen"].isin(SCEN_FILTER)]
    df_long = df.melt(
        id_vars=agg_cols, value_vars=["existing", "added"],
        var_name="variable", value_name="value",
    )
    df_long["variable"] = df_long["variable"].str.capitalize()
    upload(df_long, "grid_km_h2_pipeline_gwkm", None,
           run_name_prefix_override=_upload_prefix)
else:
    print("  ⏭️  gwkm_dict_H2 pipeline.csv not available — skipping")

# --------------------------------------------------------------------------
# 19. Water cost breakdown (€/MWh_H2)
# --------------------------------------------------------------------------
print("\n── Water Cost Breakdown ──")
_H2O_COMPONENTS = [
    "desalination", "seawater", "H2O pipeline",
    "H2O store", "H2O store charger", "H2O store discharger",
    "H2O generator",
]
try:
    _capex = read_csv_nafix(SDIR / "costs_dict_capex.csv")
    _opex  = read_csv_nafix(SDIR / "costs_dict_opex.csv")
    _h2    = read_csv_nafix(SDIR / "balance_dict_H2.csv")

    def _filt(df_in):
        return df_in[df_in["country"].isin(COUNTRIES) & df_in["year"].isin(YEARS)].copy()

    capex_f, opex_f, h2_f = _filt(_capex), _filt(_opex), _filt(_h2)

    present_comps = [c for c in _H2O_COMPONENTS if c in capex_f.columns or c in opex_f.columns]
    if not present_comps:
        raise ValueError("No H2O cost components found")

    _IDX = list(INDEX_COLS)
    cap_sub = capex_f[_IDX + [c for c in present_comps if c in capex_f.columns]].rename(
        columns={c: f"{c}__cap" for c in present_comps if c in capex_f.columns}
    )
    opx_sub = opex_f[_IDX + [c for c in present_comps if c in opex_f.columns]].rename(
        columns={c: f"{c}__opx" for c in present_comps if c in opex_f.columns}
    )
    combined = cap_sub.merge(opx_sub, on=_IDX, how="inner")

    h2o_costs = combined[_IDX].copy()
    for comp in present_comps:
        cap_v = combined[f"{comp}__cap"].fillna(0) if f"{comp}__cap" in combined.columns else 0.0
        opx_v = combined[f"{comp}__opx"].fillna(0) if f"{comp}__opx" in combined.columns else 0.0
        h2o_costs[comp] = cap_v + opx_v

    h2_elec = h2_f[_IDX + ["H2 Electrolysis"]].copy()
    h2_elec["H2 Electrolysis"] = h2_elec["H2 Electrolysis"].abs()

    df_h2o = h2o_costs.merge(h2_elec, on=_IDX, how="inner")
    for comp in present_comps:
        df_h2o[comp] = df_h2o[comp] / df_h2o["H2 Electrolysis"] * 1000  # bn€/TWh → €/MWh_H2

    df_h2o = set_scen_col(df_h2o, index_levels_to_drop=INDEX_LEVELS_TO_DROP)
    df_scen = df_h2o[["country", "year", "scen"] + present_comps].copy()
    if SCEN_FILTER:
        df_scen = df_scen.query("scen in @SCEN_FILTER")
    df_scen["total"] = df_scen[present_comps].sum(axis=1)
    nonzero_comps = [c for c in present_comps if df_scen[c].abs().max() > 0.01]

    # Melt to long format matching the notebook's h2o_cost_bar_fig input
    id_cols_h2o = ["country", "year", "scen"]
    df_h2o_long = df_scen.melt(
        id_vars=id_cols_h2o,
        value_vars=nonzero_comps,
        var_name="variable",
        value_name="value",
    )
    df_h2o_long = df_h2o_long[df_h2o_long["value"].abs() > 0.01]
    upload(df_h2o_long, "water_cost_breakdown_eur_per_mwh_h2", None,
           run_name_prefix_override=_upload_prefix)

except FileNotFoundError as e:
    print(f"  ⏭️  H2O cost data not available: {e} — skipping")
except (ValueError, KeyError) as e:
    print(f"  ⏭️  H2O cost processing error: {e} — skipping")





# %%
# ══════════════════════════════════════════════════════════════════════════════
# Save variable name mapping
# ══════════════════════════════════════════════════════════════════════════════
save_variable_mapping(SDIR)

print("\n🎉  All uploads complete.")
