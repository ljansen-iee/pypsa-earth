
import logging
logger = logging.getLogger(__name__)
from pathlib import Path
import re
import yaml
import pandas as pd
idx_slice = pd.IndexSlice
from itertools import product
import pypsa
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Comment out for debugging and development
warnings.simplefilter(action='ignore', category=DeprecationWarning) # Comment out for debugging and development

from plot_helpers import (
    chdir_to_root_dir,
    to_csv_nafix,
    prepare_dataframe,
    set_scen_col_MAPaper,
    get_scen_col_function,
)

chdir_to_root_dir()

#%%

run_name_prefix = "WSA"  # Experiment name. It can be freely chosen.

sdir = Path.cwd() / "results" / f"{run_name_prefix}_summary_v1"
sdir.mkdir(exist_ok=True, parents=True)

run_names = [
    # "DKS_CL_2030",
    # "DKS_CL_2050",
    # "DKS_EG_2030",
    # "DKS_EG_2050",
    # "DKS_MA_2030",
    # "DKS_MA_2050",
    # "DKS_ZA_2030",
    # "DKS_ZA_2050",
    # "DKS_CL_2030_AB",
    # "DKS_CL_2050_AB",
    # "DKS_EG_2030_AB",
    # "DKS_EG_2050_AB",
    # "DKS_MA_2030_AB",
    # "DKS_MA_2050_AB",
    # "DKS_ZA_2030_AB",
    # "DKS_ZA_2050_AB",
    # # "MAPaper_2030",
    # "MAPaper_2035",
    # "MAPaper_2035_best",
    # "MAPaper_2035_worst",
    
    "WSA_MA_2050_low50",
    "WSA_MA_2050_vestas3",
    # "WSA_MA_2050_sep50",
]

#%%

def extract_wildcards_from_filename(filename):
    """Extract scenario parameters from network filename using regex.
    
    Expected format: elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{year}_{discountrate}_{demand}_exp{eopts}.nc
    Example: elec_s_10_ec_lv1.1_Co2L1.09_1H_2035_0.078_Exp_expH2v1.0.nc
    
    Using the Snakemake wildcard constraints of PyPSA-Earth for the pattern matching.
    """
    pattern = r"elec_s([a-zA-Z0-9]*)_([0-9]+(?:m|flex)?)_ec_l([vc](?:[0-9.]+|opt|all)|all)_([^_]*)_([a-zA-Z0-9+\-.]+)_([0-9]+)_([0-9.]+)_([a-zA-Z0-9+\-.]+)_exp(.*?)\.nc"
    match = re.search(pattern, filename)
    if match:
        simpl, clusters, ll, opts, sopts, year, discountrate, demand, eopts = match.groups()
        return {
            "simpl": simpl if simpl else "",
            "clusters": clusters,
            "ll": ll,
            "opts": opts if opts else "",
            "sopts": sopts,
            "year": int(year),
            "discountrate": float(discountrate),
            "demand": demand,
            "eopts": eopts,
        }
    return None

def collect_files_from_directories(all_postnetworks_dir):
    """
    Collects all existing files from the directories specified in all_postnetworks_dir.
    """
    files_in_folder = {}
    for run_name, path in all_postnetworks_dir.items():
        if path.exists() and path.is_dir():
            files_in_folder[f"{run_name}"] = list(path.glob('*'))
        else:
            files_in_folder[f"{run_name}"] = []

    # Print the collected files for each country
    for key, files in files_in_folder.items():
        print(f"Files in {key}:")
        for file in files:
            print(f"  {file}")

    return files_in_folder

def init_stats_dict(network_files, keys, name):
    
    stats_dict = {
        key: pd.concat([pd.DataFrame(index=network_files.index)]) #, keys=[key], names=[name]
        for key in keys}

    return stats_dict

def save_stats_dict(stats_dict, stats_name, path_dir):
    for key, df in stats_dict.items():
        to_csv_nafix(df, path_dir / f"{stats_name}_{key}.csv")
        print(f"Saved {key} to {path_dir / f'{stats_name}_{key}.csv'}")

results_dir = "results"  # base results directory

postnetworks_dict = {
    run_name: Path.cwd() / results_dir / run_name / "postnetworks"
    for run_name in run_names
}

files_in_folder_dict = collect_files_from_directories(postnetworks_dict)

# Build nc_files DataFrame by parsing filenames directly
cols = ["run_name_prefix", "run_name", "country", "year", "simpl", "clusters", "ll", "opts", "sopts", "discountrate", "demand", "eopts"]
nc_files_data = []

for run_name in run_names:
    # NOTE: country is inferred as the segment immediately after the run_name_prefix
    # e.g. run_name_prefix="WSA", run_name="WSA_MA_2050_low50" -> country="MA"
    # Adjust manually if run names don't follow this convention.
    country = run_name.removeprefix(run_name_prefix + "_").split("_")[0]
    
    for file in files_in_folder_dict.get(run_name, []):
        wcs = extract_wildcards_from_filename(file.name)
        if wcs:
            nc_files_data.append({
                "run_name_prefix": run_name_prefix,
                "run_name": run_name,
                "country": country,
                "file": file,
                **wcs,
            })
            print(f"Adding nc file for {run_name}: {file.name}")

nc_files = pd.DataFrame(nc_files_data).set_index(cols) if nc_files_data else pd.DataFrame(columns=cols + ["file"]).set_index(cols)

if nc_files.empty:
    raise ValueError("No files found for the given run names and wildcards. Please check the configurations and file names.")

#%%

def calculate_gwkm(n, selection=None, decimals=1, which="optimal"):
    # from https://github.com/fneum/spatial-sector/blob/main/notebooks/single.py.ipynb
    if selection is None:
        selection = [
            "H2 pipeline",
            "H2 pipeline retrofitted",
            "gas pipeline",
            "gas pipeline new",
            "DC",
        ]

    gwkm = n.links.loc[n.links.carrier.isin(selection)]

    if which == "optimal":
        link_request = "p_nom_opt"
        line_request = "s_nom_opt"
    elif which == "added":
        link_request = "(p_nom_opt - p_nom)"
        line_request = "(s_nom_opt - s_nom)"
    elif which == "existing":
        link_request = "p_nom"
        line_request = "s_nom"

    gwkm = gwkm.eval(f"length*{link_request}").groupby(gwkm.carrier).sum() / 1e3  # GWkm
    gwkm["AC"] = n.lines.eval(f"length*{line_request}").sum() / 1e3  # GWkm

    gwkm.index.name = None

    return gwkm.round(decimals)


# initialise dicts per metric (market balance, optimal capacities, costs, marginal prices) with dataframes per bus_carrier or other groups

balance_dict = init_stats_dict(nc_files, keys=[
    "AC", "H2", "oil", "gas", "co2 stored", "co2", "methanol", "NH3", "steel", "HBI", "solid biomass", "biogas",
    "H2O", "purewater", "seawater",
    ], name="bus_carrier")

optimal_capacity_dict = init_stats_dict(nc_files, keys=["AC", "H2", "methanol", "NH3", "steel", "HBI"], name="bus_carrier")

costs_dict = init_stats_dict(nc_files, keys=["capex", "opex"], name="costs")

load_avg_marginal_price = pd.DataFrame(index=nc_files.index, columns=["H2 export", "FT export", "NH3 export"])
load_avg_marginal_price.columns.name = "bus" # NB: this is spatially resolved.

gwkm_dict = {
    "AC": pd.DataFrame(index=nc_files.index, columns=["optimal", "added", "existing", "ratio"]),
    "H2 pipeline": pd.DataFrame(index=nc_files.index, columns=["optimal", "added", "existing", "ratio"])
}

stores = pd.DataFrame(index=nc_files.index, columns=["H2 Store Tank", "H2 UHS", "battery", "home battery"])
stores.columns.name = "bus" # NB: this is spatially resolved.

pypsa.options.params.statistics.nice_names = False
pypsa.options.params.statistics.drop_zero = False
pypsa.options.params.statistics.round = 6

for nc_files_idx in nc_files.index:
    
    n = pypsa.Network(nc_files.at[nc_files_idx,"file"])

    # energy balance per bus_carrier in TWh
    for bus_carrier in balance_dict.keys():

        if bus_carrier not in n.buses.carrier.unique():
            continue

        ds = (
            n.stats.energy_balance(
                bus_carrier=bus_carrier, 
                groupby="carrier",
                aggregate_across_components=True
            )
            .div(1e6)
            .round(1)
        )

        balance_dict[bus_carrier].loc[nc_files_idx, ds.index] = ds.values

        if bus_carrier == "AC" and "low voltage" in n.buses.carrier.unique():

            ds = (
                n.stats.energy_balance(
                    bus_carrier="low voltage", 
                    groupby="carrier",                     
                    aggregate_across_components=True
                )
                .div(1e6)
                .round(1)
            )

            balance_dict[bus_carrier].loc[nc_files_idx, ds.index] = ds.values

            balance_dict[bus_carrier] = balance_dict[bus_carrier].drop("electricity distribution grid", axis=1)  

    # optimal capacities (installed + expanded) per bus_carrier in GW_output_unit
    for bus_carrier in optimal_capacity_dict.keys():

        ds = (
            n.stats.optimal_capacity(
                components=["Generator", "Link", "StorageUnit"], 
                bus_carrier=bus_carrier, 
                groupby="carrier",
                aggregate_across_components=True
            )
            .loc[lambda x: x > 0]
            .div(1e3)
            .round(1)
        )

        # if bus_carrier == "H2":
        #     ds = ds/n.links.groupby("carrier").mean(numeric_only=True).efficiency
          
        optimal_capacity_dict[bus_carrier].loc[nc_files_idx, ds.index] = ds.values

        if bus_carrier == "AC" and "low voltage" in n.buses.carrier.unique():

            ds = (
                n.stats.optimal_capacity(
                    components=["Generator", "Link", "StorageUnit"], 
                    bus_carrier="low voltage", 
                    groupby="carrier",
                    aggregate_across_components=True
                )
                .loc[lambda x: x > 0]
                .div(1e3)
                .round(1)
            )
            
            if not ds.empty:
                optimal_capacity_dict[bus_carrier].loc[nc_files_idx, ds.index] = ds.values

                if "electricity distribution grid" in optimal_capacity_dict[bus_carrier].columns:
                    optimal_capacity_dict[bus_carrier] = optimal_capacity_dict[bus_carrier].drop("electricity distribution grid", axis=1)

    # system capex per carrier in billion currency unit
    ds = n.stats.capex().dropna().groupby("carrier").sum().div(1e9).round(4)

    costs_dict["capex"].loc[nc_files_idx, ds.index] = ds.values

    # system opex per carrier in billion currency unit
    ds = n.stats.opex().dropna().groupby("carrier").sum().div(1e9).round(4)

    costs_dict["opex"].loc[nc_files_idx, ds.index] = ds.values

    # expanded storage capacity in GWh
    ec_stores = n.stats.expanded_capacity(
                components=["Store"], 
                groupby="carrier",
                aggregate_across_components=True
            )

    cols = stores.columns.intersection(ec_stores.index)
    stores.loc[nc_files_idx, cols] = ec_stores.div(1e3).round(1)[cols] # GWh
    
    # ASSUMPTIONS: assume marginal costs of last unit can be earned as export price for all export
    # adding those revenues for export as negative opex costs
    # TODO: decide how to integrate revenues from exports in the costs_dict
    # if bus_carrier + " export" in n.loads_t.p.columns.unique():
    #     H2_export_price = n.buses_t.marginal_price["H2 export"].mean() # NB: hourly pattern is interesting!
    #     costs_dict["opex"].at[nc_files_idx,"H2 export"] = -(
    #         n.loads_t.p_set["H2 export"].mul(n.buses_t.marginal_price["H2 export"]).sum()/1e9
    #     )

    # load averaged marginal prices per bus_carrier in currency/MWh

    bus_carriers_to_price = ["H2", "NH3", "FT", "HBI", "steel", "STEEL", "industry methanol", "shipping methanol", "MEOH", "H2O", "purewater", "seawater"]
    
    prices_load_weighted = n.stats.prices(groupby_time=True, weighting="load")
    
    # Filter and assign prices for buses matching carrier names (with or without " export" suffix)
    for bus_carrier in bus_carriers_to_price:
        matching_buses = n.buses.index[n.buses.index.str.contains(bus_carrier)]
        for bus in matching_buses:
            if bus in prices_load_weighted.index:
                load_avg_marginal_price.at[nc_files_idx, bus] = prices_load_weighted[bus]

    # Grid GW·km (AC lines and H2 pipelines)
    try:
        df_gwkm = pd.DataFrame({key: calculate_gwkm(n, which=key) for key in ["optimal", "added", "existing"]})
        df_gwkm["ratio"] = df_gwkm["added"] / df_gwkm["existing"].replace(0, pd.NA)
        if "AC" in df_gwkm.index:
            gwkm_dict["AC"].loc[nc_files_idx, :] = df_gwkm.loc["AC", ["optimal", "added", "existing", "ratio"]].values
        if "H2 pipeline" in df_gwkm.index:
            gwkm_dict["H2 pipeline"].loc[nc_files_idx, :] = df_gwkm.loc["H2 pipeline", ["optimal", "added", "existing", "ratio"]].values
    except Exception as e:
        print(f"Warning: Could not calculate gwkm for {nc_files_idx}: {e}")


# %%
to_csv_nafix(nc_files, sdir / "nc_files.csv")

save_stats_dict(balance_dict, "balance_dict", sdir)
save_stats_dict(optimal_capacity_dict, "optimal_capacity_dict", sdir)
save_stats_dict(costs_dict, "costs_dict", sdir)

to_csv_nafix(load_avg_marginal_price, sdir / "load_avg_marginal_price.csv")
print(f"Saved load_avg_marginal_price to {sdir / 'load_avg_marginal_price.csv'}")

to_csv_nafix(stores, sdir / "stores.csv")
print(f"Saved stores to {sdir / 'stores.csv'}")

save_stats_dict(gwkm_dict, "gwkm_dict", sdir)

# Prepare marginal prices for visualization: long-form with scen column, all bus carriers
plot_summary_config = yaml.safe_load(Path("post_processing/plot_summary.yaml").read_text())
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
print(f"Saved marginal_prices_prepared to {sdir / 'marginal_prices_prepared.csv'}")


# %%


# nh3_buses = n.buses[n.buses.index.str.contains("NH3")].index
# nh3_prices_per_t = n.buses_t.marginal_price[nh3_buses] * 5.17
# nh3_prices_per_t.round(0).plot(legend=False, title="NH3 €/t", drawstyle='steps')
# plt.show()

# h2_buses = n.buses[n.buses.index.str.contains("H2")].index
# h2_prices_per_t = n.buses_t.marginal_price[h2_buses] * 33.3333 / 1000
# h2_prices_per_t.round(0).plot(legend=False, title="H2 €/kg", drawstyle='steps')
# plt.show()


# h2_prices_per_t_weekly = h2_prices_per_t.resample('W').mean()
# h2_prices_per_t_weekly.round(0).plot(legend=False, title="H2 €/kg (Weekly Mean)", drawstyle='steps')
# plt.show()


# n.buses_t.marginal_price["Earth HBI"].round(0).plot(legend=False, title="HBI", drawstyle='steps')
# plt.show()


# n.buses_t.marginal_price["Earth steel"].round(0).plot(legend=False, title="Steel", drawstyle='steps')
# plt.show()




# %%
