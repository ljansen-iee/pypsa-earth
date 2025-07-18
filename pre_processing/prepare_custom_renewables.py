import pandas as pd
import pypsa

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
    bus = bus.replace("EGY", "EG")

    carrier_map = {
        'wind': 'onwind',
        'pv': 'solar'
    }

    carrier = carrier_map.get(parts[2], parts[2])
    # Extract class (class_1, class_2, etc.) - should be the remaining parts
    res_class = parts[-1]  # only use number

    return bus, carrier, res_class


if __name__ == "__main__":

    ts_EGY_2010_variant_1_path = "C:/Users/ljansen/OneDrive - Fraunhofer/Globale ESA/04_methods_and_models/energyANTS/DKS2-EG/timeseries/pv_wind_ts_EGY_2010_variant_1.csv"
    ts_EGY_2010_variant_2_path = "C:/Users/ljansen/OneDrive - Fraunhofer/Globale ESA/04_methods_and_models/energyANTS/DKS2-EG/timeseries/pv_wind_ts_EGY_2010_variant_2.csv"
    ts_EGY_2010_variant_4_path = "C:/Users/ljansen/OneDrive - Fraunhofer/Globale ESA/04_methods_and_models/energyANTS/DKS2-EG/timeseries/pv_wind_ts_EGY_2010_variant_4.csv"
    ts_EGY_2010_variant_5_path = "C:/Users/ljansen/OneDrive - Fraunhofer/Globale ESA/04_methods_and_models/energyANTS/DKS2-EG/timeseries/pv_wind_ts_EGY_2010_variant_5.csv"

    ts_EGY_2010 = {
        "v1": pd.read_csv(ts_EGY_2010_variant_1_path, index_col=0),
        "v2": pd.read_csv(ts_EGY_2010_variant_2_path, index_col=0),
        "v4": pd.read_csv(ts_EGY_2010_variant_4_path, index_col=0),
        "v5": pd.read_csv(ts_EGY_2010_variant_5_path, index_col=0)
    }

    # convert columns from flat to multi-level index
    for key, df in ts_EGY_2010.items():

        # Apply the parsing to all columns
        column_tuples = [split_column_name(col) for col in df.columns]

        # Create multi-level index
        multi_index = pd.MultiIndex.from_tuples(
            column_tuples, 
            names=['bus', 'carrier', 'class']# Apply the multi-index to the dataframe
        )

        df.columns = multi_index

    variant = "v1"  # Change this to the desired variant
    run_name = "DKS_EG_2050"  # Change this to the desired run name

    ts = ts_EGY_2010[variant]

    carriers = ["onwind", "solar"]

    for resource_class in ts.columns.get_level_values('class').unique():
        for car in carriers:
            for yr in [2030, 2035, 2050]:

                p_max_pu = (
                    ts
                    .xs(resource_class, level='class', axis=1)
                    .xs(car, level='carrier', axis=1))

                buses = p_max_pu.columns

                p_max_pu.columns = buses + " " + resource_class + " " + car

                p_max_pu.to_csv(
                    f"../data/custom/{run_name}/renewables/{resource_class} {car}-{yr}-p_max_pu.csv"
                )

            if resource_class == "1":
                # print the top 7 buses with highest p_max_pu for this resource class and carrier
                print(f"Top 7 buses for {resource_class} {car}:")
                print(p_max_pu.sum().sort_values(ascending=False).head(7))


    # copy paste generators tables for now.
    # attributes could be modified here later, it costs should be modified via costs.csv files

    max_expansion_limits_EGY = {
        "v1": pd.read_csv(
            "C:/Users/ljansen/OneDrive - Fraunhofer/Globale ESA/04_methods_and_models/energyANTS/DKS2-EG/pot_areas/max_expansion_limits_EGY_var_1.csv",
            index_col=0,)
    }

    max_expansion_limits_EGY[variant] = max_expansion_limits_EGY[variant]
    max_expansion_limits_EGY[variant]["GID_1"] = (
        max_expansion_limits_EGY[variant]["GID_1"]
        .str.replace("EGY", "EG")
        .str.replace("MAR", "MA")
        .str.replace("CHL", "CL")
        .str.replace("ZAF", "ZA")
    )
    max_expansion_limits_EGY[variant].columns = [
        "node", "resource_class", "onwind", "solar"
    ]
    max_expansion_limits_EGY[variant]["resource_class"] = max_expansion_limits_EGY[variant]["resource_class"].astype(str)

   
    relevant_attrs = [
        "bus", "carrier", 
        "p_nom_extendable", "p_nom_min", "p_nom_max", 
        "marginal_cost", "capital_cost", 
        "p_max_pu", "lifetime"
    ]

    for resource_class in ts.columns.get_level_values('class').unique():        
        for car in carriers:
            max_exp_limits = max_expansion_limits_EGY[variant].copy()
            max_exp_limits = max_exp_limits.query("resource_class == @resource_class")
            max_exp_limits.index = max_exp_limits["node"] + "_AC" + " " + max_exp_limits["resource_class"] + " " + car
            for yr in [2030, 2050]: # use different network for other years because of different costs
                n = pypsa.Network(f"EG_elec_{yr}/elec_s_10.nc")

                buses = n.buses[n.buses.carrier == "AC"].index

                gens = (
                    n.generators[relevant_attrs]
                    .query("carrier == @car and bus in @buses"))

                gens.to_csv(
                    f"../data/custom/{run_name}/renewables/{car}-{yr}.csv", # as reference
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
                    f"../data/custom/{run_name}/renewables/{resource_class} {car}-{yr}.csv",
                )