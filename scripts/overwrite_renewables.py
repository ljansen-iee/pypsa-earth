# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""

custom_data:
  renewables: 
    update_data: true # if true then custom custom_res_pot_{car}_{year}_{dr}.csv and custom_res_ins_{car}_{year}_{dr}.csv files must be placed in "data/custom/renewables"
    carrier: ["solar","onwind"] # ['csp', 'rooftop-solar', 'solar']
    resource_classes: 3

"""


import os
from itertools import dropwhile
from types import SimpleNamespace

from pathlib import Path
import numpy as np
import pandas as pd
import pypsa
import pytz
import xarray as xr
from _helpers import mock_snakemake, override_component_attrs, BASE_DIR
from prepare_sector_network import remove_carrier_related_components
from add_electricity import load_costs


def remove_leap_day(df):
    return df[~((df.index.month == 2) & (df.index.day == 29))]

if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "overwrite_renewables",
            simpl="",
            clusters="10",
            ll="v1.3",
            opts="Co2L0.24",
            planning_horizons=2050,
            sopts="1h",
            discountrate=0.090,
            demand="NZ",
            # configfile="config.RE_classes_EG_2050.yaml"
        )
        rootpath = ".."
    else:
        rootpath = "."



    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)
    m = n.copy()
    if snakemake.params.custom_data_renewables["update_data"]:

        year = snakemake.wildcards["planning_horizons"]

        dr = snakemake.wildcards["discountrate"]

        countries = snakemake.params.countries

        run_name = snakemake.params.run

        carriers = snakemake.params.custom_data_renewables["carrier"]

        resource_classes_n = snakemake.params.custom_data_renewables["resource_classes"]
        if resource_classes_n == 1:
            resource_classes = [""]
        elif resource_classes_n > 1:
            resource_classes = [str(i) + " " for i in range(1, resource_classes_n + 1)]
        else:
            raise ValueError(f"resource_classes_n must be greater than 0, got {resource_classes_n}")

    

        buses = list(n.buses[n.buses.carrier == "AC"].index)

        to_drop = n.generators.query("carrier in @carriers and bus in @buses").index
        n.mremove("Generator", to_drop)

        filepath = Path(rootpath) / Path(snakemake.params.custom_data_renewables_path)

        for car in carriers:                        
            for resource_class in resource_classes:
                gens = (
                    pd.read_csv(
                        filepath / f"{resource_class}{car}-{year}.csv",
                        index_col=0,
                    )
                )
                if not set(gens['bus']).issubset(set(buses)):
                    raise ValueError("Bus values in gens do not match values in buses")
                if not set(buses).issubset(set(gens['bus'])):
                    raise ValueError("Not all buses are present in gens['bus']")

                p_max_pu = (
                    pd.read_csv(
                        filepath / f"{resource_class}{car}-{year}-p_max_pu.csv",
                        index_col=0,
                        parse_dates=True,
                    )
                )

                p_max_pu = remove_leap_day(p_max_pu)

                p_max_pu.index = n.generators_t.p_max_pu.index
                missing_indices = pd.Index(gens.index).difference(p_max_pu.columns)
                if not missing_indices.empty:
                    raise ValueError(f"The following gens.index values are missing in p_max_pu.columns: {missing_indices.tolist()}")

                gen_buses = gens["bus"].values
                
                # if not snakemake.params.custom_data_renewables.get("overwrite_costs", False):
                #     # If overwrite_costs is False, use the costs used in the previous rules, saved in resources folder
                #     costs = load_costs(
                #         Path(rootpath) / Path(snakemake.input.costs),
                #         snakemake.params["cost_config"],
                #         snakemake.params["elec_config"],
                #         Nyears=1
                #         )
                #     gens["capital_cost"] = costs.loc[car, "capital_cost"]
                #     gens["marginal_cost"] = costs.loc[car, "marginal_cost"]
                
                if not snakemake.params.custom_data_renewables.get("overwrite_costs", False):
                    gens["capital_cost"] = m.generators.query("carrier == @car")["capital_cost"].unique()[0]
                    gens["marginal_cost"] = m.generators.query("carrier == @car")["marginal_cost"].unique()[0]

                n.madd(
                    "Generator",
                    gen_buses + " " + resource_class + car,
                    bus=gen_buses,
                    carrier=car,
                    p_nom_extendable=True,
                    p_nom_min=gens["p_nom_min"], #gens["p_nom_min"].where(gens["p_nom_min"] > 20, 0).round(0),
                    p_nom_max=gens["p_nom_max"],
                    capital_cost=gens["capital_cost"],
                    marginal_cost=gens["marginal_cost"],
                    p_max_pu=p_max_pu,
                    lifetime=gens["lifetime"]
                )

                # Check if all gens with p_nom_min > 0 have positive sum of p_max_pu
                gens_to_check = n.generators.query("carrier in @carriers and bus in @buses and p_nom_min > 0")
                if not gens_to_check.empty:
                    p_max_pu_sums = n.generators_t.p_max_pu[gens_to_check.index].sum()
                    invalid_gens = p_max_pu_sums[p_max_pu_sums <= 0].index.tolist()
                    if invalid_gens:
                        raise ValueError(f"Generators with p_nom_min > 0 but zero p_max_pu sum: {invalid_gens}")

        gens_wo_potential = n.generators.query("p_nom_max == 0").index
        n.mremove("Generator", gens_wo_potential)

        if snakemake.params.custom_data_renewables.get("carriers_to_drop"):
            remove_carrier_related_components(
                n, snakemake.params.custom_data_renewables["carriers_to_drop"]
            )
    else:
        print("Skipping overwrite renewables")

    # Check if any values in n.generators_t.p_max_pu are lower than 0.01
    mask = (n.generators_t.p_max_pu != 0) & (n.generators_t.p_max_pu < 0.01)
    if (mask).any().any():
        print("Warning: Some non-zero values in n.generators_t.p_max_pu are lower than 0.01." \
        " It is recommended to activate solve_opts['clip_p_max_pu']")
 
    n.export_to_netcdf(snakemake.output[0])