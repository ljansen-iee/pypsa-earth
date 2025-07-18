# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Created on Thu Jul 14 21:18:06 2022.

@author: user
"""

import logging
import os
from itertools import product
from pathlib import Path

import pandas as pd
from _helpers import BASE_DIR, mock_snakemake, read_csv_nafix

_logger = logging.getLogger(__name__)


def calculate_end_values(df):
    return (1 + df) ** no_years


def country_to_nodal(industrial_production, keys):
    # keys["country"] = keys.index.str[:2]  # TODO 2digit_3_digit adaptation needed

    nodal_production = pd.DataFrame(
        index=keys.index, columns=industrial_production.columns, dtype=float
    )

    countries = keys.country.unique()
    sectors = industrial_production.columns

    for country, sector in product(countries, sectors):
        buses = keys.index[keys.country == country]

        if sector not in keys.columns or keys[sector].sum() == 0:
            mapping = "gdp"
        else:
            mapping = sector

        key = keys.loc[buses, mapping]
        # print(sector)
        nodal_production.loc[buses, sector] = (
            industrial_production.at[country, sector] * key
        )

    return nodal_production


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "build_industry_demand",
            simpl="",
            clusters="10",
            planning_horizons=2050,
            demand="NZ",
        )

    countries = snakemake.params.countries

    clean_industry_list = [
        "iron and steel",
        "chemical and petrochemical",
        "non-ferrous metals",
        "non-metallic minerals",
        "transport equipment",
        "machinery",
        "mining and quarrying",
        "food and tobacco",
        "paper pulp and print",
        "wood and wood products",
        "textile and leather",
        "construction",
        "other",
    ]

    all_carriers = [
        "electricity",
        "gas",
        "coal",
        "oil",
        "hydrogen",
        "biomass",
        "low-temperature heat",
        "ammonia",
        "methanol",
        "process emissions",
    ]

    def fill_and_merge_other_industries(industry_totals_df):
        # Fill missing carriers with 0s
        for country in countries:
            carriers_present = industry_totals_df.xs(country, level=0).index
            missing_carriers = set(all_carriers) - set(carriers_present)
            for carrier in missing_carriers:
                # Add the missing carrier with a value of 0
                if missing_carriers == "process emissions":
                    raise ValueError(
                        "Process emissions should not be missing. Check the custom data file."
                    )
                else:
                    industry_totals_df.loc[(country, carrier), :] = 0

        # temporary fix: merge other manufacturing, construction and non-fuel into other and drop the column
        other_cols = list(set(industry_totals_df.columns) - set(clean_industry_list))
        if len(other_cols) > 0:
            industry_totals_df["other"] += industry_totals_df[other_cols].sum(axis=1)
            industry_totals_df.drop(columns=other_cols, inplace=True)
        return industry_totals_df

    if snakemake.params.custom_industry_totals:
        _logger.info(
            "Fetching custom industry demand data.. expecting file at 'data/custom/industry_totals_{0}_{1}.csv'".format(
                snakemake.wildcards["planning_horizons"], snakemake.wildcards["demand"]
            )
        )

        industry_totals = pd.read_csv(
            os.path.join(
                BASE_DIR,
                "data/custom/industry_totals_{0}_{1}.csv".format(
                    snakemake.wildcards["planning_horizons"],
                    snakemake.wildcards["demand"]
                ),
            ),
            index_col=[0, 1],
        )

        industry_totals = fill_and_merge_other_industries(industry_totals)

        keys_path = snakemake.input.industrial_distribution_key

        dist_keys = pd.read_csv(
            keys_path, index_col=0, keep_default_na=False, na_values=[""]
        )
        production_base = pd.DataFrame(
            1, columns=industry_totals.columns, index=countries
        )
        nodal_keys = country_to_nodal(production_base, dist_keys)

        nodal_df = pd.DataFrame()

        for country in countries:
            nodal_production_tom_co = nodal_keys[
                nodal_keys.index.to_series().str.startswith(country)
            ]
            industry_base_totals_co = industry_totals.loc[country]
            # final energy consumption per node and industry (TWh/a)
            nodal_df_co = nodal_production_tom_co.dot(industry_base_totals_co.T)
            nodal_df = pd.concat([nodal_df, nodal_df_co])

    else:
        no_years = int(snakemake.wildcards.planning_horizons) - int(
            snakemake.params.base_year
        )

        cagr = read_csv_nafix(snakemake.input.industry_growth_cagr, index_col=0)

        # Building nodal industry production growth
        for country in countries:
            if country not in cagr.index:
                cagr.loc[country] = cagr.loc["DEFAULT"]
                _logger.warning(
                    "No industry growth data for "
                    + country
                    + " using default data instead."
                )
            else:
                cagr.loc[country] = cagr.loc[country].fillna(cagr.loc["DEFAULT"])

        cagr = cagr[cagr.index.isin(countries)]

        growth_factors = calculate_end_values(cagr)

        industry_base_totals = read_csv_nafix(
            snakemake.input["base_industry_totals"], index_col=[0, 1]
        )

        production_base = cagr.map(lambda x: 1)
        production_tom = production_base * growth_factors

        # non-used line; commented out
        # industry_totals = (production_tom * industry_base_totals).fillna(0)

        industry_util_factor = snakemake.params.industry_util_factor

        # Load distribution keys
        keys_path = snakemake.input.industrial_distribution_key

        dist_keys = pd.read_csv(
            keys_path, index_col=0, keep_default_na=False, na_values=[""]
        )

        # production of industries per node compared to current
        nodal_production_tom = country_to_nodal(production_tom, dist_keys)

        emission_factors = {  # Based on JR data following PyPSA-EUR
            "iron and steel": 0.025,
            "chemical and petrochemical": 0.51,  # taken from HVC including process and feedstock
            "non-ferrous metals": 1.5,  # taken from Aluminum primary
            "non-metallic minerals": 0.542,  # taken for cement
            "transport equipment": 0,
            "machinery": 0,
            "mining and quarrying": 0,  # assumed
            "food and tobacco": 0,
            "paper pulp and print": 0,
            "wood and wood products": 0,
            "textile and leather": 0,
            "construction": 0,  # assumed
            "other": 0,
        }

        # fill industry_base_totals
        level_2nd = industry_base_totals.index.get_level_values(1).unique()
        mlv_index = pd.MultiIndex.from_product([countries, level_2nd])
        industry_base_totals = industry_base_totals.reindex(mlv_index, fill_value=0)

        geo_locs = pd.read_csv(
            snakemake.input.industrial_database,
            sep=",",
            header=0,
            keep_default_na=False,
            index_col=0,
        )
        geo_locs["capacity"] = pd.to_numeric(geo_locs.capacity)

        def match_technology(df):
            industry_mapping = {
                "Integrated steelworks": "iron and steel",
                "DRI + Electric arc": "iron and steel",
                "Electric arc": "iron and steel",
                "Cement": "non-metallic minerals",
                "HVC": "chemical and petrochemical",
                "Paper": "paper pulp and print",
            }

            df["industry"] = df["technology"].map(industry_mapping)
            return df

        # Calculating emissions

        # get the subset of countries that al
        countries_geo = geo_locs.index.unique().intersection(countries)
        geo_locs = match_technology(geo_locs).loc[countries_geo]

        aluminium_year = snakemake.params.aluminium_year
        AL = read_csv_nafix(
            os.path.join(BASE_DIR, "data/AL_production.csv"), index_col=0
        )
        # Filter data for the given year and countries
        AL_prod_tom = AL.query("Year == @aluminium_year and index in @countries_geo")[
            "production[ktons/a]"
        ]

        # Check if aluminum data is missing for any countries
        for country in countries_geo:
            if country not in AL_prod_tom.index:
                _logger.warning(
                    f"No aluminum production data found for {country}. Filled with 0.0."
                )

        # Reindex and fill missing values with 0.0
        AL_prod_tom = AL_prod_tom.reindex(countries_geo, fill_value=0.0)

        # Estimate emissions for aluminum production and converting from ktons to tons
        AL_emissions = AL_prod_tom * emission_factors["non-ferrous metals"] * 1000

        Steel_emissions = (
            geo_locs[geo_locs.industry == "iron and steel"]
            .groupby("country")
            .sum()
            .capacity
            * 1000
            * emission_factors["iron and steel"]
            * industry_util_factor
        )
        NMM_emissions = (
            geo_locs[geo_locs.industry == "non-metallic minerals"]
            .groupby("country")
            .sum()
            .capacity
            * 1000
            * emission_factors["non-metallic minerals"]
            * industry_util_factor
        )
        refinery_emissons = (
            geo_locs[geo_locs.industry == "chemical and petrochemical"]
            .groupby("country")
            .sum()
            .capacity
            * emission_factors["chemical and petrochemical"]
            * 0.136
            * 365
            * industry_util_factor
        )

        for country in countries:
            industry_base_totals.loc[(country, "process emissions"), :] = 0
            try:
                industry_base_totals.loc[
                    (country, "process emissions"), "non-metallic minerals"
                ] = NMM_emissions.loc[country]
            except KeyError:
                pass

            try:
                industry_base_totals.loc[
                    (country, "process emissions"), "iron and steel"
                ] = Steel_emissions.loc[country]
            except KeyError:
                pass
            try:
                industry_base_totals.loc[
                    (country, "process emissions"), "non-ferrous metals"
                ] = AL_emissions.loc[country]
            except KeyError:
                pass
            try:
                industry_base_totals.loc[
                    (country, "process emissions"), "chemical and petrochemical"
                ] = refinery_emissons.loc[country]
            except KeyError:
                pass
        industry_base_totals = industry_base_totals.sort_index()


        industry_base_totals = fill_and_merge_other_industries(
            industry_base_totals
        )

        nodal_df = pd.DataFrame()

        for country in countries:
            nodal_production_tom_co = nodal_production_tom[
                nodal_production_tom.index.to_series().str.startswith(country)
            ]
            industry_base_totals_co = industry_base_totals.loc[country]
            # final energy consumption per node and industry (TWh/a)
            nodal_df_co = nodal_production_tom_co.dot(industry_base_totals_co.T)
            nodal_df = pd.concat([nodal_df, nodal_df_co])

        # save industry_totals per country and carrier 
        # with tomorrows production level of default (non-custom) workflow
        industry_totals = industry_base_totals.mul(production_tom,level=0)

        
        # Compare sums with precision threshold using numpy.allclose # TODO can be removed after debugging
        import numpy as np
        for country in countries:
            country_sum = industry_totals.sum(axis=1).xs(country)
            nodal_sum = nodal_df.sum()
            if not np.allclose(country_sum, nodal_sum, atol=0.001):
                diff = (country_sum - nodal_sum).abs()
                raise ValueError(
                    f"Sum mismatch for {country}: max diff {diff.max():.4f} exceeds threshold 0.001"
                )
        
        path = Path(BASE_DIR) / Path(snakemake.params.resource_demand_path)
        fn = "industrial_totals_{0}_{1}.csv".format(
            snakemake.wildcards["planning_horizons"], snakemake.wildcards["demand"]
        )
        print(f"Saving industrial totals to {path / fn} for validation purposes.")
        industry_totals.to_csv(path / fn,float_format="%.2f")

    rename_sectors = {
        "elec": "electricity",
        "biomass": "solid biomass",
        "heat": "low-temperature heat",
    }
    nodal_df.rename(columns=rename_sectors, inplace=True)

    nodal_df.index.name = "MWh/a (tCO2/a)"

    nodal_df.to_csv(
        snakemake.output.industrial_energy_demand_per_node, float_format="%.2f"
    )
