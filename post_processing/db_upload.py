"""
Upload consolidated and renamed summary tables to postgres database.
"""
#%%
from pathlib import Path
import pandas as pd
idx_slice = pd.IndexSlice
from energyants.core.db import db_connector, run_sql, write_to_db
from plot_helpers import (
    chdir_to_parent_dir,
    convert_country_to_iso3,
    consistency_check,
    read_stats_dict,
    drop_index_levels, 
    update_layout, 
    prepare_dataframe,
    get_supply_demand_from_balance,
    rename_electricity, 
    rename_h2,
    rename_gas,
    rename_oil,
    rename_co2,
    rename_costs,
    rename_to_upper_case,
    colors,
    get_missing_colors,
    nice_title,
    save_plotly_fig)


import country_converter as coco

def convert_country_to_iso3(df):
    """
    Convert country names to ISO3 codes and drop rows with "not found" codes.
    """

    df = df.copy()
    cc = coco.CountryConverter()
    Country = pd.Series(df["country"])
    
    df["country"] = cc.pandas_convert(series=Country, to="ISO3", not_found="not found")
    
    df = df.loc[df["country"] != "not found"]

    # Convert country column that contains lists for some country names that are identified with more than one country.
    df["country"] = df["country"].astype(str)

    # Remove all iso2 conversions for some country names that are identified with more than one country.
    df = df[~df.country.str.contains(",", na=False)].reset_index(drop=True)

    return df


def rename_to_pg_format(variable):
    return variable.replace(" ", "_").lower()

db_connection = db_connector(name="work_db_82")

#%%
chdir_to_parent_dir()

run_name_prefix = "Scenarios_AB_and_AT" # Experiment name

sdir = Path.cwd() / "results"/ f"{run_name_prefix}_summary_20250513"
sdir.mkdir(exist_ok=True, parents=True)

idx_group = idx_slice[["Scenarios_AB_and_AT"],:,:,:] 

#%%
balance_dict = read_stats_dict("balance_dict", sdir, keys=["AC", "H2", "oil", "gas", "co2 stored"])
optimal_capacity_dict = read_stats_dict("optimal_capacity_dict", sdir, keys=["AC", "H2"])
costs_dict = read_stats_dict("costs_dict", sdir, keys=["capex", "opex"])
mean_marginal_prices = read_stats_dict("mean_marginal_prices", sdir, keys=["H2 export bus"])

# simplify index for plotting: define a scen_str column and drop all other columns
#NB: the following is specific to the experiment

index_levels_to_drop=["simpl", "clusters", "ll", "opts", "sopts", "discountrate", "demand"]

for key in balance_dict.keys():
    balance_dict[key] = drop_index_levels(balance_dict[key], to_drop=index_levels_to_drop)

for key in optimal_capacity_dict.keys():
    optimal_capacity_dict[key] = drop_index_levels(optimal_capacity_dict[key], to_drop=index_levels_to_drop)
    
for key in costs_dict.keys():
    costs_dict[key] = drop_index_levels(costs_dict[key], to_drop=index_levels_to_drop)


#%%


df = prepare_dataframe(balance_dict["AC"], idx_group)
df = convert_country_to_iso3(df)
df["variable"] = df.variable.map(rename_electricity).map(rename_to_upper_case)
df["variable_lower"] = df.variable.map(rename_to_pg_format)
(supply_df, supply_sum_df, 
 demand_df, demand_sum_df) = get_supply_demand_from_balance(df)

fig_name = "electricity_supply_twh"

#supply_df = convert_country_to_iso3(supply_df)
consistency_check(supply_df)
get_missing_colors(supply_df, colors["electricity"])

write_to_db(
    df=supply_df,
    table=fig_name,
    schema="ptx_pathways",
    connector=db_connection,
    owner="ptx_pathways_rolle",
    if_exists="replace",
)


fig_name = "electricity_demand_twh"

consistency_check(demand_df)
get_missing_colors(demand_df, colors["electricity"])

write_to_db(
    df=demand_df,
    table=fig_name,
    schema="ptx_pathways",
    connector=db_connection,
    owner="ptx_pathways_rolle",
    if_exists="replace",
)


#%%

df = prepare_dataframe(balance_dict["H2"], idx_group)
df = convert_country_to_iso3(df)
df = df[df["value"]!=0]
df["variable"] = df.variable.map(rename_h2)
df["variable_lower"] = df.variable.map(rename_to_pg_format)

# (supply_df, supply_sum_df, 
#  demand_df, demand_sum_df) = get_supply_demand_from_balance(df)

fig_name = "h2_balance_twh"

consistency_check(df)
get_missing_colors(df, colors["hydrogen"])


write_to_db(
    df=df,
    table=fig_name,
    schema="ptx_pathways",
    connector=db_connection,
    owner="ptx_pathways_rolle",
    if_exists="replace",
)


#%%

df = prepare_dataframe(balance_dict["oil"], idx_group)
df = convert_country_to_iso3(df)
df = df[df["value"]!=0]
df["variable"] = df.variable.map(rename_oil)
df["variable_lower"] = df.variable.map(rename_to_pg_format)

# (supply_df, supply_sum_df, 
#  demand_df, demand_sum_df) = get_supply_demand_from_balance(df)

fig_name = "liquid_fuel_balance_twh"


consistency_check(df)
get_missing_colors(df, colors["oil"])

write_to_db(
    df=df,
    table=fig_name,
    schema="ptx_pathways",
    connector=db_connection,
    owner="ptx_pathways_rolle",
    if_exists="replace",
)

#%%

df = prepare_dataframe(balance_dict["gas"], idx_group)
df = convert_country_to_iso3(df)
df["variable"] = df.variable.map(rename_gas)
df["variable_lower"] = df.variable.map(rename_to_pg_format)

df = df.groupby(
    ["run_name_prefix","scen","year","country", "variable", "variable_lower"], 
    as_index=False).sum().round(1)
df = df[df["value"]!=0]
# (supply_df, supply_sum_df, 
#  demand_df, demand_sum_df) = get_supply_demand_from_balance(df)


fig_name = "ch4_balance_twh"
consistency_check(df)
get_missing_colors(df, colors["gas"])
write_to_db(
    df=df,
    table=fig_name,
    schema="ptx_pathways",
    connector=db_connection,
    owner="ptx_pathways_rolle",
    if_exists="replace",
)

#%%


df = prepare_dataframe(balance_dict["co2 stored"], idx_group)
df = convert_country_to_iso3(df)
df = df[df["value"]!=0]
df["variable"] = df.variable.map(rename_co2)
df["variable_lower"] = df.variable.map(rename_to_pg_format)

# (supply_df, supply_sum_df, 
#  demand_df, demand_sum_df) = get_supply_demand_from_balance(df)

fig_name = "co2_capture_and_usage_mt"
consistency_check(df)
get_missing_colors(df, colors["co2"])

write_to_db(
    df=df,
    table=fig_name,
    schema="ptx_pathways",
    connector=db_connection,
    owner="ptx_pathways_rolle",
    if_exists="replace",
)

#%%

df = prepare_dataframe(optimal_capacity_dict["AC"], idx_group)
df = convert_country_to_iso3(df)
df = df[df["value"]<1e6] #drop loadshedding capacity
df["variable"] = df.variable.map(rename_electricity).map(rename_to_upper_case)
df["variable_lower"] = df.variable.map(rename_to_pg_format)

df = df.groupby(
    ["run_name_prefix","scen","year","country", "variable", "variable_lower"], 
    as_index=False).sum().round(1)
df = df[df["value"]!=0]


fig_name = "electricity_installed_capacity_gw"
consistency_check(df)
get_missing_colors(df, colors["electricity"])

write_to_db(
    df=df,
    table=fig_name,
    schema="ptx_pathways",
    connector=db_connection,
    owner="ptx_pathways_rolle",
    if_exists="replace",
)


#%%

df = prepare_dataframe(optimal_capacity_dict["H2"], idx_group)
df = convert_country_to_iso3(df)
df = df[df["value"]<1e6] #drop loadshedding capacity
df["variable"] = df.variable.map(rename_h2)
df["variable_lower"] = df.variable.map(rename_to_pg_format)

fig_name = "h2_installed_capacity_gw"
consistency_check(df)
get_missing_colors(df, colors["hydrogen"])

write_to_db(
    df=df,
    table=fig_name,
    schema="ptx_pathways",
    connector=db_connection,
    owner="ptx_pathways_rolle",
    if_exists="replace",
)

#%%

stats_df = pd.concat([costs_dict["capex"],costs_dict["opex"]],axis=0)
df = prepare_dataframe(stats_df, idx_group)
df = convert_country_to_iso3(df)
df["variable"] = df.variable.map(rename_costs)
df["variable_lower"] = df.variable.map(rename_to_pg_format)

df = df.groupby(
    ["run_name_prefix","scen","year","country", "variable", "variable_lower"], 
    as_index=False).sum().round(1)
df = df[df["value"]!=0]


(supply_df, supply_sum_df, 
 demand_df, demand_sum_df) = get_supply_demand_from_balance(df, threshold=0., round=1)

fig_name = "system_costs_billion_eur"
consistency_check(supply_df)

get_missing_colors(supply_df, colors["costs"])



write_to_db(
    df=supply_df,
    table=fig_name,
    schema="ptx_pathways",
    connector=db_connection,
    owner="ptx_pathways_rolle",
    if_exists="replace",
)
