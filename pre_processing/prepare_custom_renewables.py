import pandas as pd
import pypsa

DIR = "C:/Users/ljansen/OneDrive - Fraunhofer/Globale ESA/04_methods_and_models/energyANTS/"
COUNTRY = ("CHL", "CL")

# Convert column index to multi-level index
# Assuming columns are in format like: "EGY.8.1_wind_class_1" or similar
def split_column_name(col_name):
    """
    Parse column name into GID_1, carrier (onwind/solar), and class components
    Expected format: GID_1_carrier_class_X
    """
    parts = col_name.split('_')
    
    # Extract GID_1 (should be like EGY.24_1 or EGY.8_1)
    # The GID_1 consists of the first two parts joined by underscore
    bus = f"{parts[0]}_{parts[1]}_AC"  # e.g., "EGY.24_1_AC"
    bus = bus.replace("EGY", "EG").replace("MAR", "MA").replace("CHL", "CL").replace("ZAF", "ZA")

    carrier_map = {
        'wind': 'onwind',
        'pv': 'solar'
    }

    carrier = carrier_map.get(parts[2], parts[2])
    # Extract class (class_1, class_2, etc.) - should be the remaining parts
    res_class = parts[-1]  # only use number

    return bus, carrier, res_class


if __name__ == "__main__":

    ts_files = {
        "CHL": f"{DIR}DKS2-CL/150km_griddist/gadm_pv_wind_ts_CHL_2014_variant_1_mean_weather_year_150km_griddist.csv",
        "EGY": f"{DIR}DKS2-EG/timeseries/gadm_pv_wind_ts_EGY_2010_variant_1_mean_weather_year.csv",
        "MAR": f"{DIR}DKS2-MA/timeseries_per_gadm/gadm_pv_wind_ts_MAR_2016_variant_1_mean_weather_year.csv",
        "ZAF": f"{DIR}DKS2-ZA/timeseries/gadm_pv_wind_ts_ZAF_2013_variant_1_mean_weather_year.csv"
    }

    ts = pd.read_csv(ts_files[COUNTRY[0]], index_col=0)

    # # convert columns from flat to multi-level index
    # for key, df in ts.items():

    # Apply the parsing to all columns
    column_tuples = [split_column_name(col) for col in ts.columns]

    # Create multi-level index
    multi_index = pd.MultiIndex.from_tuples(
        column_tuples, 
        names=['bus', 'carrier', 'class']# Apply the multi-index to the dataframe
    )

    ts.columns = multi_index

    run_name_prefix = f"DKS_{COUNTRY[1]}"  # Change this to the desired run name without planning year
    carriers = ["onwind", "solar"]

    for resource_class in ts.columns.get_level_values('class').unique():
        for car in carriers:
            for yr in [2030, 2050]:

                p_max_pu = (
                    ts
                    .xs(resource_class, level='class', axis=1)
                    .xs(car, level='carrier', axis=1))

                buses = p_max_pu.columns

                p_max_pu.columns = buses + " " + resource_class + " " + car

                p_max_pu.to_csv(
                    f"../data/custom/{run_name_prefix}_{yr}/renewables/{resource_class} {car}-{yr}-p_max_pu.csv"
                )

            if resource_class == "1":
                # print the top 7 buses with highest p_max_pu for this resource class and carrier
                print(f"Top 7 buses for {resource_class} {car}:")
                print(p_max_pu.sum().sort_values(ascending=False).head(7))


    # copy paste generators tables for now.
    # attributes could be modified here later, it costs should be modified via costs.csv files

    max_expansion_limits_files = {
        "CHL": f"{DIR}DKS2-CL/150km_griddist/gadm_max_expansion_limits_CHL_var_1_mean_weather_year_150km_griddist.csv",
        "EGY": f"{DIR}DKS2-EG/timeseries/gadm_max_expansion_limits_EGY_var_1_mean_weather_year.csv",
        "MAR": f"{DIR}DKS2-MA/timeseries_per_gadm/gadm_max_expansion_limits_MAR_var_1_mean_weather_year.csv",
        "ZAF": f"{DIR}DKS2-ZA/timeseries/gadm_max_expansion_limits_ZAF_var_1_mean_weather_year.csv"
    }

    max_expansion_limits = pd.read_csv(max_expansion_limits_files[COUNTRY[0]], index_col=0)
    max_expansion_limits["gid_1"] = (
        max_expansion_limits["gid_1"]
        .str.replace("EGY", "EG")
        .str.replace("MAR", "MA")
        .str.replace("CHL", "CL")
        .str.replace("ZAF", "ZA")
    )
    max_expansion_limits.columns = [
        "node", "resource_class", "onwind", "solar"
    ]
    max_expansion_limits["resource_class"] = max_expansion_limits["resource_class"].astype(str)

   
    relevant_attrs = [
        "bus", "carrier", 
        "p_nom_extendable", "p_nom_min", "p_nom_max", 
        "marginal_cost", "capital_cost", 
        "p_max_pu", "lifetime"
    ]

    for resource_class in ts.columns.get_level_values('class').unique():        
        for car in carriers:
            max_exp_limits = max_expansion_limits.copy()
            max_exp_limits = max_exp_limits.query("resource_class == @resource_class")
            max_exp_limits.index = max_exp_limits["node"] + "_AC" + " " + max_exp_limits["resource_class"] + " " + car
            for yr in [2030, 2050]: # use different network for other years because of different costs
                n = pypsa.Network(f"../networks/{run_name_prefix}_{yr}/elec_s_10.nc")

                buses = n.buses[n.buses.carrier == "AC"].index

                gens = (
                    n.generators[relevant_attrs]
                    .query("carrier == @car and bus in @buses"))

                gens.to_csv(
                    f"../data/custom/{run_name_prefix}_{yr}/renewables/{car}-{yr}.csv", # as reference
                )

                gens.index = buses + " " + resource_class + " " + car

                gens["p_nom_max"] = max_exp_limits.loc[gens.index, car].values

                if any(gens["p_nom_min"]>=gens["p_nom_max"]) and resource_class == "1":
                    gens_mask = gens["p_nom_min"] >= gens["p_nom_max"]
                    print(
                        f"p_nom_min (installed capacity) exceeds p_nom_max for "
                        f"resource class {resource_class} and carrier {car}. "
                        "setting p_nom_min to 0."
                    )
                    gens.loc[gens_mask, "p_nom_min"] = 0.
                
                if resource_class != "1":
                    # assume all installed capacity is in resource_class 1
                    gens["p_nom_min"] = 0. 

                gens.to_csv(
                    f"../data/custom/{run_name_prefix}_{yr}/renewables/{resource_class} {car}-{yr}.csv",
                )