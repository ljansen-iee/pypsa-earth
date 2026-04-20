import pandas as pd
import pypsa
from pathlib import Path

# ── User configuration ─────────────────────────────────────────────────────────
PROJECT = "H2G"           # "DKS" or "H2G"
COUNTRIES = ["NAM"]       # one or more ISO-3 country codes; ISO-2 is looked up automatically from ISO3_TO_ISO2
# run_name_prefix = "MAPaper"  # Snakemake run.name prefix (without planning year)
run_name_prefix = "DKS_{cc2}"  # Snakemake run.name prefix; {cc2} is replaced by the ISO-2 country code
run_name_suffix = "AB"   # optional suffix appended as {prefix}_{yr}_{suffix}, e.g. "AB" → DKS_NA_2035_AB; set to "" to omit
PLANNING_HORIZONS = [2035]  # Snakemake run years → folder names and output file names
TECH_YEAR = {2035: 2030, 2050: 2040}  # maps planning horizon → tech year used for input data files
                                       # (H2G: selects t{tech_yr} file; DKS: ignored, paths are year-independent)
sdir = Path("../data/custom")  # Define saving directory
# ──────────────────────────────────────────────────────────────────────────────

# ISO-3 → ISO-2 country code mapping (extend as needed)
ISO3_TO_ISO2 = {
    "EGY": "EG", "MAR": "MA", "CHL": "CL", "ZAF": "ZA",
    "AGO": "AO", "COD": "CD", "ETH": "ET", "GHA": "GH",
    "KEN": "KE", "MRT": "MR", "NAM": "NA", "NGA": "NG",
    "TUN": "TN", "TZA": "TZ",
}

# ── Project-specific configurations ───────────────────────────────────────────
# ts_file / exp_lim_file:
#   - dict  → DKS-style: year-independent, keyed by ISO-3 country code
#   - str   → H2G-style: format string with {cc3} and {yr} placeholders
# exp_lim_index_col:
#   - 0     → DKS CSVs have an unnamed row-number column as first column
#   - None  → H2G CSVs start directly with gid_1 (no extra index column)
PROJECT_CONFIGS = {
    "DKS": {
        "DIR": "/mnt/data/gsa/energyants_results/DKS/",
        "ts_file": {
            "CHL": "DKS2-CL/150km_griddist/gadm_pv_wind_ts_CHL_2014_variant_1_mean_weather_year_150km_griddist.csv",
            "EGY": "DKS2-EG/timeseries/gadm_pv_wind_ts_EGY_2010_variant_1_mean_weather_year.csv",
            "MAR": "DKS2-MA/timeseries_per_gadm/gadm_pv_wind_ts_MAR_2016_variant_1_mean_weather_year.csv",
            "ZAF": "DKS2-ZA/timeseries/gadm_pv_wind_ts_ZAF_2013_variant_1_mean_weather_year.csv",
        },
        "exp_lim_file": {
            "CHL": "DKS2-CL/150km_griddist/gadm_max_expansion_limits_CHL_var_1_mean_weather_year_150km_griddist.csv",
            "EGY": "DKS2-EG/timeseries/gadm_max_expansion_limits_EGY_var_1_mean_weather_year.csv",
            "MAR": "DKS2-MA/timeseries_per_gadm/gadm_max_expansion_limits_MAR_var_1_mean_weather_year.csv",
            "ZAF": "DKS2-ZA/timeseries/gadm_max_expansion_limits_ZAF_var_1_mean_weather_year.csv",
        },
        "exp_lim_index_col": 0,
    },
    "H2G": {
        "DIR": "/mnt/data/gsa/energyants_results/H2G/",
        "ts_file": "timeseries/gadm_ts_{cc3}_my2013_t{yr}.csv",
        "exp_lim_file": "expansion_limits_meta_infos/gadm_max_expansion_limits_{cc3}_my2013_t{yr}.csv",
        "exp_lim_index_col": None,
    },
}


def resolve_path(template, cc3, yr):
    """Return the relative file path for a given country and year.

    If *template* is a dict (DKS-style, year-independent), look up by ISO-3
    country code.  If it is a format string (H2G-style, year-dependent),
    substitute ``{cc3}`` and ``{yr}``.
    """
    if isinstance(template, dict):
        return template[cc3]
    return template.format(cc3=cc3, yr=yr)


def run_folder(yr, cc3):
    """Build the Snakemake run directory name for a given planning year and country.

    Without suffix: ``{prefix}_{yr}``       (e.g. ``MAPaper_2035``)
    With suffix:    ``{prefix}_{yr}_{suffix}``  (e.g. ``DKS_NA_2035_AB``)

    ``{cc2}`` in *run_name_prefix* is substituted with the ISO-2 country code
    (e.g. ``"DKS_{cc2}"`` + ``"NAM"`` → ``"DKS_NA"``).  If the prefix contains
    no placeholder it is used as-is.
    """
    cc2 = ISO3_TO_ISO2[cc3]
    prefix = run_name_prefix.format(cc2=cc2)
    if run_name_suffix:
        return f"{prefix}_{yr}_{run_name_suffix}"
    return f"{prefix}_{yr}"


def split_column_name(col_name):
    """Parse a flat column name into (bus, carrier, resource_class).

    Expected format: ``CC3.X_Y_carrier_class_N``
    e.g. ``MAR.3_1_wind_class_2`` → ``("MA.3_1_AC", "onwind", "2")``
    """
    parts = col_name.split('_')
    bus = f"{parts[0]}_{parts[1]}_AC"
    for iso3, iso2 in ISO3_TO_ISO2.items():
        bus = bus.replace(iso3, iso2)
    carrier_map = {'wind': 'onwind', 'pv': 'solar'}
    carrier = carrier_map.get(parts[2], parts[2])
    res_class = parts[-1]
    return bus, carrier, res_class


if __name__ == "__main__":

    cfg = PROJECT_CONFIGS[PROJECT]
    DIR = cfg["DIR"]
    carriers = ["onwind", "solar"]

    relevant_attrs = [
        "bus", "carrier",
        "p_nom_extendable", "p_nom_min", "p_nom_max",
        "marginal_cost", "capital_cost",
        "p_max_pu", "lifetime"
    ]

    for cc3 in COUNTRIES:
        for yr in PLANNING_HORIZONS:
            tech_yr = TECH_YEAR[yr]

            # ── Load timeseries ────────────────────────────────────────────────────
            ts_path = DIR + resolve_path(cfg["ts_file"], cc3, tech_yr)
            ts = pd.read_csv(ts_path, index_col=0)
            ts.columns = pd.MultiIndex.from_tuples(
                [split_column_name(col) for col in ts.columns],
                names=['bus', 'carrier', 'class']
            )

            # ── Load expansion limits ──────────────────────────────────────────────
            exp_path = DIR + resolve_path(cfg["exp_lim_file"], cc3, tech_yr)
            max_expansion_limits = pd.read_csv(exp_path, index_col=cfg["exp_lim_index_col"])
            for iso3, iso2 in ISO3_TO_ISO2.items():
                max_expansion_limits["gid_1"] = max_expansion_limits["gid_1"].str.replace(iso3, iso2)
            max_expansion_limits.columns = ["node", "resource_class", "onwind", "solar"]
            max_expansion_limits["resource_class"] = max_expansion_limits["resource_class"].astype(str)

            # ── Load reference network (for costs and installed capacities) ────────
            n = pypsa.Network(f"../networks/{run_folder(yr, cc3)}/elec_s_10.nc")
            buses = n.buses[n.buses.carrier == "AC"].index

            # Write one reference generator table per carrier (class-independent)
            save_dir = sdir / run_folder(yr, cc3) / "renewables"
            save_dir.mkdir(exist_ok=True, parents=True)
            for car in carriers:
                gens_ref = (
                    n.generators[relevant_attrs]
                    .query("carrier == @car and bus in @buses")
                )
                gens_ref.to_csv(save_dir / f"{car}-{yr}.csv")

            # ── Loop over resource classes and carriers ────────────────────────────
            for resource_class in ts.columns.get_level_values('class').unique():
                for car in carriers:

                    # p_max_pu timeseries
                    p_max_pu = (
                        ts
                        .xs(resource_class, level='class', axis=1)
                        .xs(car, level='carrier', axis=1)
                    )
                    p_max_pu.columns = p_max_pu.columns + " " + resource_class + " " + car
                    p_max_pu.to_csv(save_dir / f"{resource_class} {car}-{yr}-p_max_pu.csv")

                    if resource_class == "1":
                        print(f"[{cc3}] Top 7 buses for {resource_class} {car} ({yr}):")
                        print(p_max_pu.sum().sort_values(ascending=False).head(7))

                    # Generator attributes table
                    gens = (
                        n.generators[relevant_attrs]
                        .query("carrier == @car and bus in @buses")
                    )
                    gens.index = buses + " " + resource_class + " " + car

                    max_exp_limits = max_expansion_limits.query("resource_class == @resource_class").copy()
                    max_exp_limits.index = max_exp_limits["node"] + "_AC" + " " + resource_class + " " + car
                    gens["p_nom_max"] = max_exp_limits.loc[gens.index, car].values

                    if resource_class == "1":
                        if any(gens["p_nom_min"] >= gens["p_nom_max"]):
                            gens_mask = gens["p_nom_min"] >= gens["p_nom_max"]
                            print(
                                f"[{cc3}] p_nom_min (installed capacity) exceeds p_nom_max for "
                                f"resource class {resource_class} and carrier {car}. "
                                "Setting p_nom_min to 0."
                            )
                            gens.loc[gens_mask, "p_nom_min"] = 0.
                    else:
                        # assume all installed capacity is in resource_class 1
                        gens["p_nom_min"] = 0.

                    gens.to_csv(save_dir / f"{resource_class} {car}-{yr}.csv")