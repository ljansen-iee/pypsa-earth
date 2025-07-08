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



if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "overwrite_renewables",
            simpl="",
            clusters="10",
            ll="copt",
            opts="Co2L0.24",
            planning_horizons=2050,
            sopts="1h",
            discountrate=0.082,
            demand="NZ",
        )



    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)
    m = n.copy()
    if snakemake.params.custom_data_renewables["enable"]:

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



        for car in carriers:
            
            to_drop = n.generators.query("carrier == @car and bus in @buses").index
            n.mremove("Generator", to_drop)
            
            for resource_class in resource_classes:

                gens = (
                    pd.read_csv(
                        f"data/custom/renewables/{run_name}/{resource_class}{car}-{year}.csv",
                        index_col=0,
                        parse_dates=True,
                    )
                )
                if not set(gens['bus']).issubset(set(buses)):
                    raise ValueError("Bus values in gens do not match values in buses")

                p_max_pu = (
                    pd.read_csv(
                        f"data/custom/renewables/{run_name}/{resource_class}{car}-{year}-p_max_pu.csv",
                        index_col=0,
                        parse_dates=True,
                    )
                ).round(3)
                p_max_pu.index = n.generators_t.p_max_pu.index
                missing_indices = pd.Index(gens.index).difference(p_max_pu.columns)
                if not missing_indices.empty:
                    raise ValueError(f"The following gens.index values are missing in p_max_pu.columns: {missing_indices.tolist()}")

                buses = gens["bus"].values
                n.madd(
                    "Generator",
                    buses + " " + resource_class + car,
                    #suffix=" " + resource_class + car,
                    bus=buses,
                    carrier=car,
                    p_nom_extendable=True,
                    p_nom_min=gens["p_nom_min"],
                    p_nom_max=gens["p_nom_max"],
                    capital_cost=gens["capital_cost"],
                    marginal_cost=gens["marginal_cost"],
                    p_max_pu=p_max_pu,
                    lifetime=gens["lifetime"],
                    overwrite=False,
                )


    else:
        print("No RES potential carriers to override...")

    n.export_to_netcdf(snakemake.output[0])


