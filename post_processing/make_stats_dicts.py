
import logging
logger = logging.getLogger(__name__)
from pathlib import Path
import yaml
import pandas as pd
idx_slice = pd.IndexSlice
from itertools import product
import pypsa
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Comment out for debugging and development
warnings.simplefilter(action='ignore', category=DeprecationWarning) # Comment out for debugging and development

from plot_helpers import (
    chdir_to_parent_dir,
    collect_files_from_directories,
    init_stats_dict, 
    save_stats_dict,
    to_csv_nafix
)

chdir_to_parent_dir()

#%%

run_name_prefix = "DKS" # Experiment name. It can be freely chosen.

sdir = Path.cwd() / "results"/ f"{run_name_prefix}_summary_CL"
sdir.mkdir(exist_ok=True, parents=True)

all_run_names = [
    "DKS_CL_2030",
    "DKS_CL_2050",
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
]   

#%%

# NB: the following code probably only works if only one country is selected per run.
all_wildcards = {
    run_name: 
    {
        "run_name_prefix": [], 
        "run_name": [],
        "countries": [], # ["EG", "NA", "MA", "ZA", "KE", "ET", "CG", "TZ", "GH", "TN", "NG"]
        "year": [], #2035, 2050
        "simpl": [],
        "clusters": [], # 10
        "ll": [], # copt
        "opts": [], # Co2L0.63
        "sopts": [], # 3H
        "discountrate": [], # 0.082
        "demand": [], # AB
        "eopts": [] # 3.33
    }
    for run_name in all_run_names
}

all_configs = {
    run_name:
        yaml.safe_load(
            Path(f"configs/DKS/config.{run_name}.yaml").read_text()) # TODO: config file names should be equal to the run name
        for run_name in all_run_names
}

all_postnetworks_dir = {
    run_name:
        Path.cwd() / all_configs[run_name]["results_dir"] / f"{run_name}" / "postnetworks"
        for run_name in all_run_names
}

for run_name, config in all_configs.items():
    all_wildcards[run_name]["run_name_prefix"].append(run_name_prefix)
    all_wildcards[run_name]["run_name"].append(run_name)
    all_wildcards[run_name]["countries"].extend(config["countries"])
    all_wildcards[run_name]["year"].extend(config["scenario"]["planning_horizons"])
    all_wildcards[run_name]["simpl"].extend(config["scenario"]["simpl"])
    all_wildcards[run_name]["clusters"].extend(config["scenario"]["clusters"])
    all_wildcards[run_name]["ll"].extend(config["scenario"]["ll"])
    all_wildcards[run_name]["opts"].extend(config["scenario"]["opts"])
    all_wildcards[run_name]["sopts"].extend(config["scenario"]["sopts"])
    all_wildcards[run_name]["discountrate"].extend(config["costs"]["discountrate"])
    all_wildcards[run_name]["demand"].extend(config["scenario"]["demand"])
    all_wildcards[run_name]["eopts"].extend(config["scenario"]["eopts"])


files_in_folder = collect_files_from_directories(all_postnetworks_dir)

cols = ["run_name_prefix", "run_name", "country", "year", "simpl", "clusters", "ll", "opts", "sopts", "discountrate", "demand", "eopts"]

nc_files = pd.DataFrame(columns=cols + ["file"]).set_index(cols)

for run_name in all_run_names:
    wc_keys = all_wildcards[run_name].keys()
    wc_values_combinations = list(product(*all_wildcards[run_name].values()))

    for combination in wc_values_combinations:
        wc = dict(zip(wc_keys, combination))
        for file in files_in_folder[f"{run_name}"]:
            if f"elec_s{wc['simpl']}_{wc['clusters']}_ec_l{wc['ll']}_{wc['opts']}_{wc['sopts']}_{wc['year']}_{wc['discountrate']}_{wc['demand']}_exp{wc['eopts']}.nc" in file.name:
                nc_files.at[combination, "file"] = file
                print("Adding nc file for", wc)
            else:
                #print("File not found:", file.name, "for:", f"{run_name_prefix}_{wc["countries"]}")
                continue

if nc_files.empty:
    raise ValueError("No files found for the given run names and wildcards. Please check the configurations and file names.")

#%%

# initialise dicts per metric (market balance, optimal capacities, costs, marginal prices) with dataframes per bus_carrier or other groups

balance_dict = init_stats_dict(nc_files, keys=[
    "AC", "H2", "oil", "gas", "co2 stored", "co2",
    #"freshwater"
    ], name="bus_carrier")

optimal_capacity_dict = init_stats_dict(nc_files, keys=["AC", "H2"], name="bus_carrier")

costs_dict = init_stats_dict(nc_files, keys=["capex", "opex"], name="costs")

time_avg_marginal_price = pd.DataFrame(index=nc_files.index, columns=["H2 export bus", "FT export bus", "NH3 export bus"])
time_avg_marginal_price.columns.name = "bus" # NB: this is spatially resolved.
load_avg_marginal_price = pd.DataFrame(index=nc_files.index, columns=["H2 export bus", "FT export bus", "NH3 export bus"])
load_avg_marginal_price.columns.name = "bus" # NB: this is spatially resolved.

for nc_files_idx in nc_files.index:
    
    n = pypsa.Network(nc_files.at[nc_files_idx,"file"])
    n.statistics.set_parameters(nice_names=False, drop_zero=False, round=6)

    ##### energy and mass market balance_dict per bus_carrier in TWh

    for bus_carrier in balance_dict.keys():

        ds = (
            n.statistics.energy_balance(bus_carrier=bus_carrier, drop_zero=True)
            .dropna()
            .groupby("carrier").sum()
            .div(1e6)
            .round(1)
        )

        balance_dict[bus_carrier].loc[nc_files_idx, ds.index] = ds.values

        if bus_carrier == "AC" and "low voltage" in n.buses.carrier.unique():

            ds = (
                n.statistics.energy_balance(bus_carrier="low voltage", drop_zero=True)
                .dropna()
                .groupby("carrier").sum()
                .div(1e6)
                .round(1)
            )

            balance_dict[bus_carrier].loc[nc_files_idx, ds.index] = ds.values
                
            balance_dict[bus_carrier].loc[nc_files_idx, ds.index] = ds.values

            # drop energy between AC and distribution grid 
            balance_dict[bus_carrier] = balance_dict[bus_carrier].drop("electricity distribution grid", axis=1)  

        # if bus_carrier == "H2" and ("H2 export" in n.loads.carrier.unique() or "H2 export load" in n.loads.carrier.unique()):

        #     ds = (
        #         n.statistics.energy_balance(bus_carrier="H2 export bus", drop_zero=True)
        #         .dropna()
        #         .groupby("carrier").sum()
        #         .div(1e6)
        #         .round(1)
        #     )

        #     balance_dict[bus_carrier].loc[nc_files_idx, ds.index] = ds.values

    ##### optimal production capacity per bus_carrier in GW

    for bus_carrier in optimal_capacity_dict.keys():

        ds = n.statistics.optimal_capacity(comps=["Generator", "Link", "StorageUnit"], bus_carrier=bus_carrier).dropna().groupby("carrier").sum().div(1e3).round(1)
        ds = ds[ds > 0]

        # if bus_carrier == "H2":
        #     ds = ds/n.links.groupby("carrier").mean(numeric_only=True).efficiency
          
        optimal_capacity_dict[bus_carrier].loc[nc_files_idx, ds.index] = ds.values

        if bus_carrier == "AC" and "low voltage" in n.buses.carrier.unique():

            ds = n.statistics.optimal_capacity(comps=["Generator", "Link", "StorageUnit"], bus_carrier=bus_carrier).dropna().groupby("carrier").sum().div(1e3).round(1)
            ds = ds[ds>0]

            optimal_capacity_dict[bus_carrier].loc[nc_files_idx, ds.index] = ds.values

    ##### costs per carrier

    ds = n.statistics.capex(drop_zero=True).dropna().groupby("carrier").sum().div(1e9).round(4)

    costs_dict["capex"].loc[nc_files_idx, ds.index] = ds.values
        
    ds = n.statistics.opex(drop_zero=True).dropna().groupby("carrier").sum().div(1e9).round(4)

    costs_dict["opex"].loc[nc_files_idx, ds.index] = ds.values
    
    # ASSUMPTIONS: assume marginal costs of last unit can be earned as export price for all export
    # adding those revenues for export as negative opex costs
    # TODO: decide how to integrate revenues from exports in the costs_dict
    # if bus_carrier + " export" in n.loads_t.p.columns.unique():
    #     H2_export_price = n.buses_t.marginal_price["H2 export bus"].mean() # NB: hourly pattern is interesting!
    #     costs_dict["opex"].at[nc_files_idx,"H2 export"] = -(
    #         n.loads_t.p_set["H2 export"].mul(n.buses_t.marginal_price["H2 export bus"]).sum()/1e9
    #     )

    ##### time and load averaged marginal prices per bus_carrier in EUR/MWh

    for bus_carrier in ["H2","FT","NH3"]:
        buses = n.buses.loc[n.buses.index.str.contains(bus_carrier)]
        for bus in buses.index:
            value = n.buses_t.marginal_price[bus].mean() # NB: hourly pattern is interesting! 
            time_avg_marginal_price.at[nc_files_idx, bus] = value

            if bus_carrier + " export" in n.loads_t.p.columns.unique():
                demand = n.loads_t.p[bus_carrier + " export"]
                value = ((demand*n.buses_t.marginal_price[bus_carrier + " export bus"]).sum())/(demand.sum())
                load_avg_marginal_price.at[nc_files_idx, bus] = value
            elif bus_carrier + " export" in n.generators_t.p.columns.unique():
                logger.warning(f"No exogenous load for {bus_carrier} export bus found")
                logger.warning(f"Marginal prices are impacted by endogenous export and export prices!")
                demand = -1*n.generators_t.p[bus_carrier + " export"]
                value = ((demand*n.buses_t.marginal_price[bus_carrier + " export bus"]).sum())/(demand.sum())
                load_avg_marginal_price.at[nc_files_idx, bus] = value
            else:
                logger.warning(f"No exogenous or endogenous load found for {bus_carrier} export bus in {nc_files_idx}")


# %%
save_stats_dict(balance_dict, "balance_dict", sdir)
save_stats_dict(optimal_capacity_dict, "optimal_capacity_dict", sdir)
save_stats_dict(costs_dict, "costs_dict", sdir)
to_csv_nafix(time_avg_marginal_price, sdir / "time_avg_marginal_price.csv")
print(f"Saved time_avg_marginal_price to {sdir / 'time_avg_marginal_price.csv'}")
to_csv_nafix(load_avg_marginal_price, sdir / "load_avg_marginal_price.csv")
print(f"Saved load_avg_marginal_price to {sdir / 'load_avg_marginal_price.csv'}")
# %%
