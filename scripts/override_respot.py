# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

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


def override_values(tech, year, dr):

    custom_res_t = pd.read_csv(
            custom_res_pot_paths["custom_res_pot_{0}_{1}_{2}".format(tech, year, dr)],
            index_col=0,
            parse_dates=True,
        ).filter(buses, axis=1)

    custom_res = (
        pd.read_csv(
            custom_res_ins_paths["custom_res_ins_{0}_{1}_{2}".format(tech, year, dr)],
            #index_col=0,
        )
        #.filter(buses, axis=0)
        .reset_index()
    )
    custom_res["Generator"] = custom_res["Generator"].apply(lambda x: x + " " + tech)
    custom_res = custom_res.set_index("Generator")


    if tech.replace("-", " ") in n.generators.carrier.unique():
        to_drop = n.generators[n.generators.carrier == tech].index
        n.mremove("Generator", to_drop)

    if snakemake.wildcards["planning_horizons"] == 2050:
        directory = "results/" + snakemake.params.run.replace("2050", "2030")
        n_name = snakemake.input.network.split("/")[-1].replace(
            n.config["scenario"]["clusters"], ""
        )
        df = pd.read_csv(directory + "/res_caps_" + n_name, index_col=0)
        # df = pd.read_csv(snakemake.config["custom_data"]["existing_renewables"], index_col=0)
        existing_res = df.loc[tech]
        existing_res.index = existing_res.index.str.apply(lambda x: x + tech)
    else:
        existing_res = custom_res["installedcapacity"].values

    n.madd(
        "Generator",
        buses,
        " " + tech,
        bus=buses,
        carrier=tech,
        p_nom_extendable=True,
        p_nom_max=custom_res["p_nom_max"].values,
        # weight=ds["weight"].to_pandas(),
        # marginal_cost=custom_res["fixedomEuroPKW"].values * 1000,
        capital_cost=custom_res["annualcostEuroPMW"].values,
        efficiency=1.0,
        p_max_pu=custom_res_t,
        lifetime=custom_res["lifetime"][0],
        p_nom_min=existing_res,
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "override_respot",
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
    if snakemake.params.custom_data_renewables["update_data"]:
        buses = list(n.buses[n.buses.carrier == "AC"].index)
        countries = snakemake.params.countries
        techs = snakemake.params.custom_data_renewables["res_techs"]
        year = snakemake.wildcards["planning_horizons"]
        dr = snakemake.wildcards["discountrate"]


        custom_res_pot_paths = {
            f"custom_res_pot_{tech}_{year}_{dr}": 
                f"data/custom/renewables/{tech}_{year}_{dr}_potential.csv"
            for tech in techs
        }
        custom_res_ins_paths = {
            f"custom_res_ins_{tech}_{year}_{dr}": 
                f"data/custom/renewables/{tech}_{year}_{dr}_installable.csv"
            for tech in techs
        }

        for tech in techs:
            override_values(tech, year, dr)

    else:
        print("No RES potential techs to override...")

    n.export_to_netcdf(snakemake.output[0])