# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Solves linear optimal power flow for a network iteratively while updating
reactances.

Relevant Settings
-----------------

.. code:: yaml

    solving:
        tmpdir:
        options:
            formulation:
            clip_p_max_pu:
            load_shedding:
            noisy_costs:
            nhours:
            min_iterations:
            max_iterations:
            skip_iterations:
            track_iterations:
        solver:
            name:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`electricity_cf`, :ref:`solving_cf`, :ref:`plotting_cf`

Inputs
------

- ``networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: confer :ref:`prepare`

Outputs
-------

- ``results/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: Solved PyPSA network including optimisation results

    .. image:: /img/results.png
        :width: 40 %

Description
-----------

Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning)
is provided in the
`documentation of PyPSA <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#linear-optimal-power-flow>`_.
The optimization is based on the :func:`network.optimize` function.
Additionally, some extra constraints specified in :mod:`prepare_network` and :mod:`solve_network` are added.

Solving the network in multiple iterations is motivated through the dependence of transmission line capacities and impedances on values of corresponding flows.
As lines are expanded their electrical parameters change, which renders the optimisation bilinear even if the power flow
equations are linearized.
To retain the computational advantage of continuous linear programming, a sequential linear programming technique
is used, where in between iterations the line impedances are updated.
Details (and errors introduced through this heuristic) are discussed in the paper

- Fabian Neumann and Tom Brown. `Heuristics for Transmission Expansion Planning in Low-Carbon Energy System Models <https://arxiv.org/abs/1907.10548>`_), *16th International Conference on the European Energy Market*, 2019. `arXiv:1907.10548 <https://arxiv.org/abs/1907.10548>`_.

.. warning::
    Capital costs of existing network components are not included in the objective function,
    since for the optimisation problem they are just a constant term (no influence on optimal result).

    Therefore, these capital costs are not included in ``network.objective``!

    If you want to calculate the full total annual system costs add these to the objective value.

.. tip::
    The rule :mod:`solve_all_networks` runs
    for all ``scenario`` s in the configuration file
    the rule :mod:`solve_network`.
"""
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from _helpers import configure_logging, create_logger, override_component_attrs
from linopy import merge
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.optimization.abstract import optimize_transmission_expansion_iteratively
from pypsa.optimization.optimize import optimize

logger = create_logger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)


def get_load_shedding_capacity(n, safety_margin=1.2):
    """
    Calculate required load shedding p_nom per bus based on the
    maximum aggregated load observed in any snapshot.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network
    safety_margin : float, default 1.2
        Safety factor to apply to the maximum load

    Returns
    -------
    pd.Series
        Required p_nom per bus for load shedding.
    """

    load_shedding_p_nom = pd.Series(0.0, index=n.buses.index)

    for bus_name, bus_loads in n.loads.groupby("bus"):

        if not n.loads_t.p_set.empty:
            bus_load_timeseries = n.loads_t.p_set[
                bus_loads.index.intersection(n.loads_t.p_set.columns)
            ]
            # Sum loads across all components at this bus for each snapshot
            total_load_per_snapshot = bus_load_timeseries.sum(axis=1)
            max_total_load = total_load_per_snapshot.max()
        else:
            max_total_load = bus_loads["p_set"].sum()

        required_p_nom = max_total_load * safety_margin

        load_shedding_p_nom[bus_name] = required_p_nom

    return load_shedding_p_nom


def prepare_network(n, solve_opts, config):
    if "clip_p_max_pu" in solve_opts:
        for df in (
            n.generators_t.p_max_pu,
            n.generators_t.p_min_pu,
            n.storage_units_t.inflow,
        ):
            df.where(df > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

    if "lv_limit" in n.global_constraints.index:
        n.line_volume_limit = n.global_constraints.at["lv_limit", "constant"]
        n.line_volume_limit_dual = n.global_constraints.at["lv_limit", "mu"]

    if solve_opts.get("load_shedding"):
        required_p_nom = get_load_shedding_capacity(n, safety_margin=1.2)
        n.add("Carrier", "load shedding", color="#dd2e23", nice_name="Load shedding")
        n.madd(
            "Generator",
            n.buses.index,
            " load shedding",
            bus=n.buses.index,
            carrier="load shedding",
            sign=1,
            marginal_cost=solve_opts.get("load_shedding") * 1000,  # convert to Eur/MWh
            p_nom=required_p_nom.reindex(n.buses.index, fill_value=0.5e6),
        )

    if solve_opts.get("noisy_costs"):
        for t in n.iterate_components():
            # if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if "marginal_cost" in t.df:
                np.random.seed(174)
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (
                    np.random.random(len(t.df)) - 0.5
                )

        for t in n.iterate_components(["Line", "Link"]):
            np.random.seed(123)
            t.df["capital_cost"] += (
                1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)
            ) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

    if snakemake.config["foresight"] == "myopic":
        add_land_use_constraint(n)

    return n


def add_CCL_constraints(n, config):
    """
    Add CCL (country & carrier limit) constraint to the network.

    Add minimum and maximum levels of generator nominal capacity per carrier
    for individual countries. Opts and path for agg_p_nom_minmax.csv must be defined
    in config.yaml. Default file is available at data/agg_p_nom_minmax.csv.

    Parameters
    ----------
    n : pypsa.Network
    config : dict

    Example
    -------
    scenario:
        opts: [Co2L-CCL-24H]
    electricity:
        agg_p_nom_limits: data/agg_p_nom_minmax.csv
    """
    agg_p_nom_limits = config["electricity"].get("agg_p_nom_limits")

    try:
        agg_p_nom_minmax = pd.read_csv(agg_p_nom_limits, index_col=list(range(2)))
    except IOError:
        logger.exception(
            "Need to specify the path to a .csv file containing "
            "aggregate capacity limits per country in "
            "config['electricity']['agg_p_nom_limit']."
        )
    logger.info(
        "Adding per carrier generation capacity constraints for " "individual countries"
    )

    gen_country = n.generators.bus.map(n.buses.country)
    capacity_variable = n.model["Generator-p_nom"]

    lhs = []
    ext_carriers = n.generators.query("p_nom_extendable").carrier.unique()
    for c in ext_carriers:
        ext_carrier = n.generators.query("p_nom_extendable and carrier == @c")
        country_grouper = (
            ext_carrier.bus.map(n.buses.country)
            .rename_axis("Generator-ext")
            .rename("country")
        )
        ext_carrier_per_country = capacity_variable.loc[
            country_grouper.index
        ].groupby_sum(country_grouper)
        lhs.append(ext_carrier_per_country)
    lhs = merge(lhs, dim=pd.Index(ext_carriers, name="carrier"))

    min_matrix = agg_p_nom_minmax["min"].to_xarray().unstack().reindex_like(lhs)
    max_matrix = agg_p_nom_minmax["max"].to_xarray().unstack().reindex_like(lhs)

    n.model.add_constraints(
        lhs >= min_matrix, name="agg_p_nom_min", mask=min_matrix.notnull()
    )
    n.model.add_constraints(
        lhs <= max_matrix, name="agg_p_nom_max", mask=max_matrix.notnull()
    )


def add_EQ_constraints(n, o, scaling=1e-1):
    """
    Add equity constraints to the network.

    Currently this is only implemented for the electricity sector only.

    Opts must be specified in the config.yaml.

    Parameters
    ----------
    n : pypsa.Network
    o : str

    Example
    -------
    scenario:
        opts: [Co2L-EQ0.7-24h]

    Require each country or node to on average produce a minimal share
    of its total electricity consumption itself. Example: EQ0.7c demands each country
    to produce on average at least 70% of its consumption; EQ0.7 demands
    each node to produce on average at least 70% of its consumption.
    """
    float_regex = "[0-9]*\.?[0-9]+"
    level = float(re.findall(float_regex, o)[0])
    if o[-1] == "c":
        ggrouper = n.generators.bus.map(n.buses.country)
        lgrouper = n.loads.bus.map(n.buses.country)
        sgrouper = n.storage_units.bus.map(n.buses.country)
    else:
        ggrouper = n.generators.bus
        lgrouper = n.loads.bus
        sgrouper = n.storage_units.bus
    load = (
        n.snapshot_weightings.generators
        @ n.loads_t.p_set.groupby(lgrouper, axis=1).sum()
    )
    inflow = (
        n.snapshot_weightings.stores
        @ n.storage_units_t.inflow.groupby(sgrouper, axis=1).sum()
    )
    inflow = inflow.reindex(load.index).fillna(0.0)
    rhs = scaling * (level * load - inflow)
    dispatch_variable = n.model["Generator-p"]
    lhs_gen = (
        (dispatch_variable * (n.snapshot_weightings.generators * scaling))
        .groupby(ggrouper.to_xarray())
        .sum()
        .sum("snapshot")
    )
    # the current formulation implies that the available hydro power is (inflow - spillage)
    # it implies efficiency_dispatch is 1 which is not quite general
    # see https://github.com/pypsa-meets-earth/pypsa-earth/issues/1245 for possible improvements
    if not n.storage_units_t.inflow.empty:
        spillage_variable = n.model["StorageUnit-spill"]
        lhs_spill = (
            (spillage_variable * (-n.snapshot_weightings.stores * scaling))
            .groupby_sum(sgrouper)
            .groupby(sgrouper.to_xarray())
            .sum()
            .sum("snapshot")
        )
        lhs = lhs_gen + lhs_spill
    else:
        lhs = lhs_gen
    n.model.add_constraints(lhs >= rhs, name="equity_min")


def add_BAU_constraints(n, config):
    """
    Add a per-carrier minimal overall capacity.

    BAU_mincapacities and opts must be adjusted in the config.yaml.

    Parameters
    ----------
    n : pypsa.Network
    config : dict

    Example
    -------
    scenario:
        opts: [Co2L-BAU-24h]
    electricity:
        BAU_mincapacities:
            solar: 0
            onwind: 0
            OCGT: 100000
            offwind-ac: 0
            offwind-dc: 0
    Which sets minimum expansion across all nodes e.g. in Europe to 100GW.
    OCGT bus 1 + OCGT bus 2 + ... > 100000
    """
    mincaps = pd.Series(config["electricity"]["BAU_mincapacities"])
    p_nom = n.model["Generator-p_nom"]
    ext_i = n.generators.query("p_nom_extendable")
    ext_carrier_i = xr.DataArray(ext_i.carrier.rename_axis("Generator-ext"))
    lhs = p_nom.groupby(ext_carrier_i).sum()
    rhs = mincaps[lhs.indexes["carrier"]].rename_axis("carrier")
    n.model.add_constraints(lhs >= rhs, name="bau_mincaps")


def add_SAFE_constraints(n, config):
    """
    Add a capacity reserve margin of a certain fraction above the peak demand.
    Renewable generators and storage do not contribute. Ignores network.

    Parameters
    ----------
        n : pypsa.Network
        config : dict

    Example
    -------
    config.yaml requires to specify opts:

    scenario:
        opts: [Co2L-SAFE-24h]
    electricity:
        SAFE_reservemargin: 0.1
    Which sets a reserve margin of 10% above the peak demand.
    """
    peakdemand = n.loads_t.p_set.sum(axis=1).max()
    margin = 1.0 + config["electricity"]["SAFE_reservemargin"]
    reserve_margin = peakdemand * margin
    conventional_carriers = config["electricity"]["conventional_carriers"]
    ext_gens_i = n.generators.query(
        "carrier in @conventional_carriers & p_nom_extendable"
    ).index
    capacity_variable = n.model["Generator-p_nom"]
    p_nom = n.model["Generator-p_nom"].loc[ext_gens_i]
    lhs = p_nom.sum()
    exist_conv_caps = n.generators.query(
        "~p_nom_extendable & carrier in @conventional_carriers"
    ).p_nom.sum()
    rhs = reserve_margin - exist_conv_caps
    n.model.add_constraints(lhs >= rhs, name="safe_mintotalcap")


def add_operational_reserve_margin_constraint(n, sns, config):
    """
    Build reserve margin constraints based on the formulation
    as suggested in GenX
    https://energy.mit.edu/wp-content/uploads/2017/10/Enhanced-Decision-Support-for-a-Changing-Electricity-Landscape.pdf
    It implies that the reserve margin also accounts for optimal
    dispatch of distributed energy resources (DERs) and demand response
    which is a novel feature of GenX.
    """
    reserve_config = config["electricity"]["operational_reserve"]
    EPSILON_LOAD = reserve_config["epsilon_load"]
    EPSILON_VRES = reserve_config["epsilon_vres"]
    CONTINGENCY = reserve_config["contingency"]

    # Reserve Variables
    n.model.add_variables(
        0, np.inf, coords=[sns, n.generators.index], name="Generator-r"
    )
    reserve = n.model["Generator-r"]
    summed_reserve = reserve.sum("Generator")

    # Share of extendable renewable capacities
    ext_i = n.generators.query("p_nom_extendable").index
    vres_i = n.generators_t.p_max_pu.columns
    if not ext_i.empty and not vres_i.empty:
        capacity_factor = n.generators_t.p_max_pu[vres_i.intersection(ext_i)]
        p_nom_vres = (
            n.model["Generator-p_nom"]
            .loc[vres_i.intersection(ext_i)]
            .rename({"Generator-ext": "Generator"})
        )
        lhs = summed_reserve + (
            p_nom_vres * (-EPSILON_VRES * xr.DataArray(capacity_factor))
        ).sum("Generator")

    # Total demand per t
    demand = get_as_dense(n, "Load", "p_set").sum(axis=1)

    # VRES potential of non extendable generators
    capacity_factor = n.generators_t.p_max_pu[vres_i.difference(ext_i)]
    renewable_capacity = n.generators.p_nom[vres_i.difference(ext_i)]
    potential = (capacity_factor * renewable_capacity).sum(axis=1)

    # Right-hand-side
    rhs = EPSILON_LOAD * demand + EPSILON_VRES * potential + CONTINGENCY

    n.model.add_constraints(lhs >= rhs, name="reserve_margin")


def update_capacity_constraint(n):
    gen_i = n.generators.index
    ext_i = n.generators.query("p_nom_extendable").index
    fix_i = n.generators.query("not p_nom_extendable").index

    dispatch = n.model["Generator-p"]
    reserve = n.model["Generator-r"]

    capacity_fixed = n.generators.p_nom[fix_i]

    p_max_pu = get_as_dense(n, "Generator", "p_max_pu")

    lhs = dispatch + reserve

    # TODO check if `p_max_pu[ext_i]` is safe for empty `ext_i` and drop if cause in case
    if not ext_i.empty:
        capacity_variable = n.model["Generator-p_nom"].rename(
            {"Generator-ext": "Generator"}
        )
        lhs = dispatch + reserve - capacity_variable * xr.DataArray(p_max_pu[ext_i])

    rhs = (p_max_pu[fix_i] * capacity_fixed).reindex(columns=gen_i, fill_value=0)

    n.model.add_constraints(lhs <= rhs, name="gen_updated_capacity_constraint")


def add_operational_reserve_margin(n, sns, config):
    """
    Parameters
    ----------
        n : pypsa.Network
        sns: pd.DatetimeIndex
        config : dict

    Example:
    --------
    config.yaml requires to specify operational_reserve:
    operational_reserve: # like https://genxproject.github.io/GenX/dev/core/#Reserves
        activate: true
        epsilon_load: 0.02 # percentage of load at each snapshot
        epsilon_vres: 0.02 # percentage of VRES at each snapshot
        contingency: 400000 # MW
    """

    add_operational_reserve_margin_constraint(n, sns, config)

    update_capacity_constraint(n)


def add_battery_constraints(n):
    """
    Add constraint ensuring that charger = discharger, i.e.
    1 * charger_size - efficiency * discharger_size = 0
    """
    if not n.links.p_nom_extendable.any():
        return

    discharger_bool = n.links.index.str.contains("battery discharger")
    charger_bool = n.links.index.str.contains("battery charger")

    dischargers_ext = n.links[discharger_bool].query("p_nom_extendable").index
    chargers_ext = n.links[charger_bool].query("p_nom_extendable").index

    eff = n.links.efficiency[dischargers_ext].values
    lhs = (
        n.model["Link-p_nom"].loc[chargers_ext]
        - n.model["Link-p_nom"].loc[dischargers_ext] * eff
    )

    n.model.add_constraints(lhs == 0, name="Link-charger_ratio")


def add_RES_constraints(n, res_share, config):
    """
    The constraint ensures that a predefined share of power is generated
    by renewable sources

    Parameters
    ----------
        n : pypsa.Network
        res_share: float
        config : dict
    """

    logger.warning(
        "The add_RES_constraints() is still work in progress. "
        "Unexpected results might be incurred, particularly if "
        "temporal clustering is applied or if an unexpected change of technologies "
        "is subject to future improvements."
    )

    renew_techs = config["electricity"]["renewable_carriers"]

    charger = ["H2 electrolysis", "battery charger"]
    discharger = ["H2 fuel cell", "battery discharger"]

    ren_gen = n.generators.query("carrier in @renew_techs")
    ren_stores = n.storage_units.query("carrier in @renew_techs")
    ren_charger = n.links.query("carrier in @charger")
    ren_discharger = n.links.query("carrier in @discharger")

    gens_i = ren_gen.index
    stores_i = ren_stores.index
    charger_i = ren_charger.index
    discharger_i = ren_discharger.index

    stores_t_weights = n.snapshot_weightings.stores

    lgrouper = n.loads.bus.map(n.buses.country)
    ggrouper = ren_gen.bus.map(n.buses.country)
    sgrouper = ren_stores.bus.map(n.buses.country)
    cgrouper = ren_charger.bus0.map(n.buses.country)
    dgrouper = ren_discharger.bus0.map(n.buses.country)

    load = (
        n.snapshot_weightings.generators
        @ n.loads_t.p_set.groupby(lgrouper, axis=1).sum()
    )
    rhs = res_share * load

    # Generators
    lhs_gen = (
        (n.model["Generator-p"].loc[:, gens_i] * n.snapshot_weightings.generators)
        .groupby(ggrouper.to_xarray())
        .sum()
    )

    # StorageUnits
    store_disp_expr = (
        n.model["StorageUnit-p_dispatch"].loc[:, stores_i] * stores_t_weights
    )
    store_expr = n.model["StorageUnit-p_store"].loc[:, stores_i] * stores_t_weights
    charge_expr = n.model["Link-p"].loc[:, charger_i] * stores_t_weights.apply(
        lambda r: r * n.links.loc[charger_i].efficiency
    )
    discharge_expr = n.model["Link-p"].loc[:, discharger_i] * stores_t_weights.apply(
        lambda r: r * n.links.loc[discharger_i].efficiency
    )

    lhs_dispatch = store_disp_expr.groupby(sgrouper).sum()
    lhs_store = store_expr.groupby(sgrouper).sum()

    # Stores (or their resp. Link components)
    # Note that the variables "p0" and "p1" currently do not exist.
    # Thus, p0 and p1 must be derived from "p" (which exists), taking into account the link efficiency.
    lhs_charge = charge_expr.groupby(cgrouper).sum()

    lhs_discharge = discharge_expr.groupby(cgrouper).sum()

    lhs = lhs_gen + lhs_dispatch - lhs_store - lhs_charge + lhs_discharge

    n.model.add_constraints(lhs == rhs, name="res_share")


def add_land_use_constraint(n):
    if "m" in snakemake.wildcards.clusters:
        _add_land_use_constraint_m(n)
    else:
        _add_land_use_constraint(n)


def _add_land_use_constraint(n):
    # warning: this will miss existing offwind which is not classed AC-DC and has carrier 'offwind'

    for carrier in ["solar", "onwind", "offwind-ac", "offwind-dc"]:
        existing = (
            n.generators.loc[n.generators.carrier == carrier, "p_nom"]
            .groupby(n.generators.bus.map(n.buses.location))
            .sum()
        )
        existing.index += " " + carrier + "-" + snakemake.wildcards.planning_horizons
        n.generators.loc[existing.index, "p_nom_max"] -= existing

    n.generators.p_nom_max.clip(lower=0, inplace=True)


def _add_land_use_constraint_m(n):
    # if generators clustering is lower than network clustering, land_use accounting is at generators clusters

    planning_horizons = snakemake.config["scenario"]["planning_horizons"]
    grouping_years = snakemake.config["existing_capacities"]["grouping_years"]
    current_horizon = snakemake.wildcards.planning_horizons

    for carrier in ["solar", "onwind", "offwind-ac", "offwind-dc"]:
        existing = n.generators.loc[n.generators.carrier == carrier, "p_nom"]
        ind = list(
            set(
                [
                    i.split(sep=" ")[0] + " " + i.split(sep=" ")[1]
                    for i in existing.index
                ]
            )
        )

        previous_years = [
            str(y)
            for y in planning_horizons + grouping_years
            if y < int(snakemake.wildcards.planning_horizons)
        ]

        for p_year in previous_years:
            ind2 = [
                i for i in ind if i + " " + carrier + "-" + p_year in existing.index
            ]
            sel_current = [i + " " + carrier + "-" + current_horizon for i in ind2]
            sel_p_year = [i + " " + carrier + "-" + p_year for i in ind2]
            n.generators.loc[sel_current, "p_nom_max"] -= existing.loc[
                sel_p_year
            ].rename(lambda x: x[:-4] + current_horizon)

    n.generators.p_nom_max.clip(lower=0, inplace=True)


def add_h2_network_cap(n, cap):
    h2_network = n.links.loc[n.links.carrier == "H2 pipeline"]
    if h2_network.index.empty:
        return
    h2_network_cap = n.model["Link-p_nom"]
    h2_network_cap_index = h2_network_cap.indexes["Link-ext"]
    subset_index = h2_network.index.intersection(h2_network_cap_index)
    diff_index = h2_network_cap_index.difference(subset_index)
    if len(diff_index) > 0:
        logger.warning(
            f"Impossible to set a limit for H2 pipelines extension for the following links: {diff_index}"
        )
    lhs = (
        h2_network_cap.loc[subset_index] * h2_network.loc[subset_index, "length"]
    ).sum()
    rhs = cap * 1000
    n.model.add_constraints(lhs <= rhs, name="h2_network_cap")


def hydrogen_temporal_constraint(n, n_ref, time_period):

    res_techs = [
        "csp",
        "solar",
        "onwind",
        "offwind-ac",
        "offwind-dc",
        "ror",
    ]

    res_stor_techs = ["hydro"]

    allowed_excess = snakemake.params.policy_config["hydrogen"]["allowed_excess"]

    # Generation
    res_gen_index = n.generators.loc[n.generators.carrier.isin(res_techs)].index
    res_stor_index = n.storage_units.loc[
        n.storage_units.carrier.isin(res_stor_techs)
    ].index

    weightings_gen = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [1.0] * len(res_gen_index)),
        index=n.snapshots,
        columns=res_gen_index,
    )

    p_gen_var = n.model["Generator-p"].loc[:, res_gen_index]

    res = (weightings_gen * p_gen_var).sum(dim="Generator")

    # Store
    if not res_stor_index.empty:
        weightings_stor = pd.DataFrame(
            np.outer(n.snapshot_weightings["generators"], [1.0] * len(res_stor_index)),
            index=n.snapshots,
            columns=res_stor_index,
        )

        p_dispatch_var = n.model["StorageUnit-p_dispatch"].loc[:, res_stor_index]

        store = (weightings_stor * p_dispatch_var).sum(dim="StorageUnit")

        res = res + store

    # Electrolysis
    electrolysis_carriers = [
        "H2 Electrolysis",
        "Alkaline electrolyzer large",
        "Alkaline electrolyzer medium",
        "Alkaline electrolyzer small",
        "PEM electrolyzer",
        "SOEC",
    ]
    electrolysis_index = n.links.index[n.links.carrier.isin(electrolysis_carriers)]

    link_p = n.model["Link-p"]
    electrolysis = link_p.loc[:, electrolysis_index]

    weightings_electrolysis = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [1.0] * len(electrolysis_index)),
        index=n.snapshots,
        columns=electrolysis_index,
    )

    elec_input = (-allowed_excess * weightings_electrolysis * electrolysis).sum(
        dim="Link"
    )

    # Grouping
    if time_period == "hour":
        res = res.groupby("snapshot").sum().rename({"snapshot": "hour"})
        elec_input = elec_input.groupby("snapshot").sum().rename({"snapshot": "hour"})
    elif time_period == "month":
        res = res.groupby("snapshot.month").sum()
        elec_input = elec_input.groupby("snapshot.month").sum()
    elif time_period == "year":
        res = res.groupby("snapshot.year").sum()
        elec_input = elec_input.groupby("snapshot.year").sum()

    # Defining the constraints
    for label in res.coords[time_period].values:
        lhs = res.loc[label] + elec_input.loc[label]
        n.model.add_constraints(lhs >= 0.0, name=f"RESconstraints_{label}")


def add_chp_constraints(n):
    electric_bool = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("electric")
    )
    heat_bool = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("heat")
    )

    electric = n.links.index[electric_bool]
    heat = n.links.index[heat_bool]

    electric_ext = n.links[electric_bool].query("p_nom_extendable").index
    heat_ext = n.links[heat_bool].query("p_nom_extendable").index

    electric_fix = n.links[electric_bool].query("~p_nom_extendable").index
    heat_fix = n.links[heat_bool].query("~p_nom_extendable").index

    p = n.model["Link-p"]  # dimension: [time, link]

    # output ratio between heat and electricity and top_iso_fuel_line for extendable
    if not electric_ext.empty:
        p_nom = n.model["Link-p_nom"]

        lhs = (
            p_nom.loc[electric_ext]
            * (n.links.p_nom_ratio * n.links.efficiency)[electric_ext].values
            - p_nom.loc[heat_ext] * n.links.efficiency[heat_ext].values
        )
        n.model.add_constraints(lhs == 0, name="chplink-fix_p_nom_ratio")

        rename = {"Link-ext": "Link"}
        lhs = (
            p.loc[:, electric_ext]
            + p.loc[:, heat_ext]
            - p_nom.rename(rename).loc[electric_ext]
        )
        n.model.add_constraints(lhs <= 0, name="chplink-top_iso_fuel_line_ext")

    # top_iso_fuel_line for fixed
    if not electric_fix.empty:
        lhs = p.loc[:, electric_fix] + p.loc[:, heat_fix]
        rhs = n.links.p_nom[electric_fix]
        n.model.add_constraints(lhs <= rhs, name="chplink-top_iso_fuel_line_fix")

    # back-pressure
    if not electric.empty:
        lhs = (
            p.loc[:, heat] * (n.links.efficiency[heat] * n.links.c_b[electric].values)
            - p.loc[:, electric] * n.links.efficiency[electric]
        )
        n.model.add_constraints(lhs <= rhs, name="chplink-backpressure")


def add_co2_sequestration_limit(n, sns):
    co2_stores = n.stores.loc[n.stores.carrier == "co2 stored"].index

    if co2_stores.empty:
        return

    vars_final_co2_stored = n.model["Store-e"].loc[sns[-1], co2_stores]

    lhs = (1 * vars_final_co2_stored).sum()
    rhs = (
        n.config["sector"].get("co2_sequestration_potential", 5) * 1e6
    )  # TODO change 200 limit (Europe)

    name = "co2_sequestration_limit"

    n.model.add_constraints(lhs <= rhs, name=f"GlobalConstraint-{name}")


def set_h2_colors(n):
    blue_h2 = n.model["Link-p"].loc[
        n.links.index[n.links.index.str.contains("blue H2")]
    ]

    pink_h2 = n.model["Link-p"].loc[
        n.links.index[n.links.index.str.contains("pink H2")]
    ]

    fuelcell_ind = n.loads[n.loads.carrier == "land transport fuel cell"].index

    other_ind = n.loads[
        (n.loads.carrier == "H2 for industry")
        | (n.loads.carrier == "H2 for shipping")
        | (n.loads.carrier == "H2")
    ].index

    load_fuelcell = (
        n.loads_t.p_set[fuelcell_ind].sum(axis=1) * n.snapshot_weightings["generators"]
    ).sum()

    load_other_h2 = n.loads.loc[other_ind].p_set.sum() * 8760

    load_h2 = load_fuelcell + load_other_h2

    weightings_blue = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [1.0] * len(blue_h2.columns)),
        index=n.snapshots,
        columns=blue_h2.columns,
    )

    weightings_pink = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [1.0] * len(pink_h2.columns)),
        index=n.snapshots,
        columns=pink_h2.columns,
    )

    total_blue = (weightings_blue * blue_h2).sum().sum()

    total_pink = (weightings_pink * pink_h2).sum().sum()

    rhs_blue = load_h2 * snakemake.config["sector"]["hydrogen"]["blue_share"]
    rhs_pink = load_h2 * snakemake.config["sector"]["hydrogen"]["pink_share"]

    n.model.add_constraints(total_blue == rhs_blue, name="blue_h2_share")

    n.model.add_constraints(total_pink == rhs_pink, name="pink_h2_share")


def add_specific_tech_for_export_constraint(n, link_tech, export_carrier):
    """
    Add a constraint to ensure that the production of a specified carrier (e.g., via a specific process/link)
    is greater than or equal to the exported quantity of a corresponding export commodity.
    This is useful for linking an export flow to its specific production route.

    This creates an inequality constraint with sense ">=":
    LHS: Sum of (Link-p * efficiency) for all links with the specified link_tech
    RHS: Sum of Link-p for all links with the specified export_carrier

    Parameters
    ----------
    n : pypsa.Network
        Network to add constraint to
    link_tech : str
        Carrier name for production links (e.g., "Fischer-Tropsch")
    export_carrier : str
        Carrier name for export links (e.g., "FT export")

    Returns
    -------
    None

    Example
    -------
    >>> n.optimize.create_model(snapshots = n.snapshots)
    >>> add_specific_tech_for_export_constraint(n, "Fischer-Tropsch", "FT export")
    # Adds constraint: Sum(link_p * efficiency) >= Sum(export_link_p)
    """
    from xarray import DataArray

    logger.info(f"Adding export link_tech constraint for: {link_tech} -> {export_carrier}")

    m = n.model
    sns = n.snapshots

    weightings = n.snapshot_weightings.loc[sns]

    if n._multi_invest:
        period_weighting = n.investment_period_weightings.years[sns.unique("period")]
        weightings = weightings.mul(period_weighting, level=0, axis=0)

    links = n.links.query(f"carrier == '{link_tech}'")

    export_links = n.links.query(f"carrier == '{export_carrier}'")

    if links.empty:
        logger.warning(f"No production links found for carrier '{link_tech}'. Skipping constraint.")
        return

    if export_links.empty:
        logger.warning(f"No export links found for carrier '{export_carrier}'. Skipping constraint.")
        return

    # Define weighting variable once for both generators and links
    w = DataArray(weightings.generators[sns])
    if "dim_0" in w.dims:
        w = w.rename({"dim_0": "snapshot"})

    # LHS: Sum of (Link-p * efficiency) for production links
    link_p = m["Link-p"].loc[sns, links.index]
    efficiency = links["efficiency"]
    lhs_expr = (link_p * efficiency * w).sum()

    # RHS: Sum of Link-p for export links
    export_link_p = m["Link-p"].loc[sns, export_links.index]
    rhs_expr = (export_link_p * w).sum()

    constraint_name = f"specific_tech_for_export_constraint-{export_carrier}"
    m.add_constraints(lhs_expr >= rhs_expr, name=constraint_name)

    logger.info(f"Added constraint '{constraint_name}': "
                f"{len(links)} production links of type {link_tech} >= "
                f"consumption of {len(export_links)} export links {export_carrier}")

def add_specific_h2_techs_for_h2_export_and_conversion_constraint(n, h2_production_techs):
    """
    Add a constraint to ensure that H2 production previously defined as green H2 production is >= to
    the hydrogen consumption by H2 conversion (Haber-Bosch, Fischer-Tropsch) or direct H2 export.

    This creates an inequality constraint with sense ">=":
    LHS: Sum of (Link-p * efficiency) for all H2 production technology links
    RHS: Sum of hydrogen consumption by all H2 conversion carriers (Haber-Bosch, Fischer-Tropsch)
         plus direct H2 export links

    Parameters
    ----------
    n : pypsa.Network
        Network to add constraint to

    Returns
    -------
    None

    Example
    -------
    >>> n.optimize.create_model(snapshots = n.snapshots)
    >>> add_specific_h2_techs_for_h2_export_and_conversion_constraint(n, ["H2 Electrolysis"])
    # Adds constraint: Sum(h2_electrolysis_p * efficiency) >= Sum(h2_conversion_consumption+h2_ex)
    """
    from xarray import DataArray

    logger.info(
        "Adding H2 production for direct export or conversion constraint: " \
        "Sum(H2 production) >= Sum(H2 consumption for direct H2 export, Haber-Bosch and Fischer-Tropsch)")

    m = n.model
    sns = n.snapshots

    weightings = n.snapshot_weightings.loc[sns]

    if n._multi_invest:
        period_weighting = n.investment_period_weightings.years[sns.unique("period")]
        weightings = weightings.mul(period_weighting, level=0, axis=0)

    h2_production_links = n.links.query(f"carrier in {h2_production_techs}")

    h2_conversion_carriers = ["Haber-Bosch", "Fischer-Tropsch"]
    h2_export_carriers = ["H2 export"]  # Direct H2 export
    
    h2_conversion_links = n.links.query(f"carrier in {h2_conversion_carriers}")
    
    h2_direct_export_links = n.links.query(f"carrier in {h2_export_carriers}")

    if h2_production_links.empty:
        logger.warning("No H2 production technology links found. Skipping constraint.")
        return

    if h2_conversion_links.empty and h2_direct_export_links.empty:
        logger.warning("No H2 conversion or export links found. Skipping constraint.")
        return

    w = DataArray(weightings.generators[sns])
    if "dim_0" in w.dims:
        w = w.rename({"dim_0": "snapshot"})

    # LHS: Sum of (Link-p * efficiency) for H2 production technology links
    link_p_h2_production = m["Link-p"].loc[sns, h2_production_links.index]
    # Convert efficiency to DataArray for proper broadcasting
    efficiency_h2_production_da = DataArray(h2_production_links["efficiency"], dims=["Link"], coords={"Link": h2_production_links.index})
    lhs_expr = (link_p_h2_production * efficiency_h2_production_da * w).sum()

    # RHS: Sum of hydrogen consumption by H2 conversion processes
    rhs_expressions = []
    
    # Add Haber-Bosch H2 consumption (efficiency2 represents hydrogen input as negative)
    haber_bosch_links = n.links.query("carrier == 'Haber-Bosch'")
    if not haber_bosch_links.empty:
        link_p_hb = m["Link-p"].loc[sns, haber_bosch_links.index]
        # efficiency2 is negative hydrogen input, so we negate to get positive consumption
        # Convert efficiency2 to DataArray for proper broadcasting
        efficiency2_da = DataArray(-haber_bosch_links["efficiency2"], dims=["Link"], coords={"Link": haber_bosch_links.index})
        h2_consumption_hb = (efficiency2_da * link_p_hb * w).sum()
        rhs_expressions.append(h2_consumption_hb)
        logger.info(f"Added Haber-Bosch H2 consumption: {len(haber_bosch_links)} links")

    # Add Fischer-Tropsch H2 consumption (from bus0 which is H2 bus)
    fischer_tropsch_links = n.links.query("carrier == 'Fischer-Tropsch'")
    if not fischer_tropsch_links.empty:
        link_p_ft = m["Link-p"].loc[sns, fischer_tropsch_links.index]
        # For Fischer-Tropsch, H2 is consumed from bus0, so we use the link power directly
        h2_consumption_ft = (link_p_ft * w).sum()
        rhs_expressions.append(h2_consumption_ft)
        logger.info(f"Added Fischer-Tropsch H2 consumption: {len(fischer_tropsch_links)} links")

    # Add direct H2 export consumption
    if not h2_direct_export_links.empty:
        link_p_export = m["Link-p"].loc[sns, h2_direct_export_links.index]
        h2_export_consumption = (link_p_export * w).sum()
        rhs_expressions.append(h2_export_consumption)
        logger.info(f"Added direct H2 export consumption: {len(h2_direct_export_links)} links")

    if not rhs_expressions:
        logger.warning("No H2 consumption processes found. Skipping constraint.")
        return

    # Sum all RHS expressions
    rhs_expr = sum(rhs_expressions)

    constraint_name = "h2_techs_for_h2_export_and_conversion_constraint"
    m.add_constraints(lhs_expr >= rhs_expr, name=constraint_name)

    total_conversion_links = len(h2_conversion_links) + len(h2_direct_export_links)
    logger.info(f"Added constraint '{constraint_name}': "
                f"{len(h2_production_links)} H2 production technology links >= "
                f"H2 consumption by {total_conversion_links} export-related links")


def add_hourly_green_h2_constraint(n, h2_production_techs, green_h2_fraction=1.0):
    """
    Add a constraint to ensure that the power generation from renewable sources (solar and onwind)
    plus hydro storage unit dispatch plus battery discharger output is greater than or equal to 
    the power consumption of H2 production technology links. This constraint is applied hourly (per snapshot).

    This creates an inequality constraint with sense ">=":
    LHS: Sum of Generator-p for carriers ["solar", "onwind"] + Sum of StorageUnit-p_dispatch for carrier "hydro"
         + Sum of (Link-p * efficiency) for battery discharger links
    RHS: Sum of Link-p for H2 production technology links * green_h2_fraction

    Parameters
    ----------
    n : pypsa.Network
        Network to add constraint to
    h2_production_techs : list
        List of H2 production technology carriers
    green_h2_fraction : float, default 1.0
        Fraction of H2 production that must be green (between 0 and 1)

    Returns
    -------
    None

    Example
    -------
    >>> n.optimize.create_model(snapshots = n.snapshots)
    >>> add_hourly_green_h2_constraint(n, ["H2 Electrolysis"])
    # Adds constraint: Sum(renewable_gen_p + hydro_storage_p + battery_discharge_p) >= Sum(h2_electrolysis_p)

    Note
    ----
    This function should be called after the network model has been built but before
    optimization (e.g., in the extra_functionality function of solve_network.py).
    """
    from xarray import DataArray

    logger.info("Adding hourly green H2 constraint for all H2 production: renewable generation + hydro storage + battery discharge >= H2 production consumption")

    m = n.model
    sns = n.snapshots

    weightings = n.snapshot_weightings.loc[sns]

    if n._multi_invest:
        period_weighting = n.investment_period_weightings.years[sns.unique("period")]
        weightings = weightings.mul(period_weighting, level=0, axis=0)

    renewable_techs = ["solar", "onwind", "offwind-ac", "offwind-dc", "csp", "ror"]
    renewable_gens = n.generators.query(f"carrier in {renewable_techs}")

    hydro_storage_units = n.storage_units.query("carrier == 'hydro'")

    battery_discharger_links = n.links[n.links.index.str.contains("battery discharger")]

    h2_production_links = n.links.query(f"carrier in {h2_production_techs}")

    if renewable_gens.empty and hydro_storage_units.empty and battery_discharger_links.empty:
        logger.warning("No renewable generators, hydro storage units, or battery dischargers found. Skipping constraint.")
        return

    if h2_production_links.empty:
        logger.warning("No H2 production technology links found. Skipping constraint.")
        return

    w = DataArray(weightings.generators[sns]) # Use generators weighting for both generators and links
    if "dim_0" in w.dims:
        w = w.rename({"dim_0": "snapshot"})
    
    w_storage = DataArray(weightings.stores[sns])
    if "dim_0" in w_storage.dims:
        w_storage = w_storage.rename({"dim_0": "snapshot"})

    # LHS: Sum of Generator-p for renewable generators
    lhs_expressions = []
    
    if not renewable_gens.empty:
        gen_p = m["Generator-p"].loc[sns, renewable_gens.index]
        lhs_expressions.append((gen_p * w).sum())

    if not hydro_storage_units.empty:
        storage_p = m["StorageUnit-p_dispatch"].loc[sns, hydro_storage_units.index]
        lhs_expressions.append((storage_p * w_storage).sum())

    if not battery_discharger_links.empty:
        battery_p = m["Link-p"].loc[sns, battery_discharger_links.index]
        # Convert efficiency to DataArray for proper broadcasting
        efficiency_battery_da = DataArray(battery_discharger_links["efficiency"], dims=["Link"], coords={"Link": battery_discharger_links.index})
        lhs_expressions.append((battery_p * efficiency_battery_da * w).sum())

    # Combine all LHS expressions
    lhs_expr = sum(lhs_expressions)

    # RHS: Sum of (Link-p) for H2 production technology links
    link_p = m["Link-p"].loc[sns, h2_production_links.index]
    # efficiency = h2_production_links["efficiency"] # p is electricity input, no need to multiply by efficiency
    rhs_expr = (link_p * w * green_h2_fraction).sum()

    constraint_name = "hourly_green_h2_constraint"
    m.add_constraints(lhs_expr >= rhs_expr, name=constraint_name)

    logger.info(f"Added constraint '{constraint_name}': "
                f"{len(renewable_gens)} renewable generators + "
                f"{len(hydro_storage_units)} hydro storage units + "
                f"{len(battery_discharger_links)} battery dischargers >= "
                f"electricity consumption of {len(h2_production_links)} H2 production technology links")

def add_total_green_h2_constraint(n, h2_production_techs, green_h2_fraction=1.0):
    """
    Add a constraint to ensure that the total power generation from renewable sources (solar and onwind)
    plus hydro storage unit dispatch is greater than or equal to the total power consumption of H2 production technology links.
    This constraint sums across all snapshots (annual/total constraint).

    This creates an inequality constraint with sense ">=":
    LHS: Sum across all snapshots of (Generator-p for renewable carriers + StorageUnit-p_dispatch for hydro)
    RHS: Sum across all snapshots of Link-p for H2 production technology links * green_h2_fraction

    Parameters
    ----------
    n : pypsa.Network
        Network to add constraint to
    h2_production_techs : list
        List of H2 production technology carriers
    green_h2_fraction : float, default 1.0
        Fraction of H2 production that must be green (between 0 and 1)

    Returns
    -------
    None

    Example
    -------
    >>> n.optimize.create_model(snapshots = n.snapshots)
    >>> add_total_green_h2_constraint(n, ["H2 Electrolysis"])
    # Adds constraint: Sum_all_snapshots(renewable_gen_p + hydro_storage_p) >= Sum_all_snapshots(h2_electrolysis_p * 1.0)
    >>> add_total_green_h2_constraint(n, ["H2 Electrolysis"], green_h2_fraction=0.8)
    # Adds constraint: Sum_all_snapshots(renewable_gen_p + hydro_storage_p) >= Sum_all_snapshots(h2_electrolysis_p * 0.8)

    Note
    ----
    This function should be called after the network model has been built but before
    optimization (e.g., in the extra_functionality function of solve_network.py).
    Battery dischargers are excluded from this constraint to focus on primary renewable sources.
    
    It makes sense to combine add_total_green_h2_constraint with add_hourly_green_h2_constraint 
    to ensure that battery use in the hourly case is sourced from green energy in total:
    - The hourly constraint allows batteries to discharge for H2 production
    - The total constraint ensures that over the entire period, primary renewable sources 
      (excluding batteries) generate enough energy to cover all H2 production
    - Together, they guarantee that any battery energy used for H2 production was originally 
      stored from renewable sources, maintaining true "green" hydrogen certification
    - This dual approach provides operational flexibility while ensuring renewable origin.
    """
    from xarray import DataArray

    logger.info("Adding total green H2 constraint: total renewable generation + hydro storage >= total H2 production consumption")

    m = n.model
    sns = n.snapshots

    weightings = n.snapshot_weightings.loc[sns]

    if n._multi_invest:
        period_weighting = n.investment_period_weightings.years[sns.unique("period")]
        weightings = weightings.mul(period_weighting, level=0, axis=0)

    renewable_techs = ["solar", "onwind", "offwind-ac", "offwind-dc", "csp", "ror"]
    renewable_gens = n.generators.query(f"carrier in {renewable_techs}")

    hydro_storage_units = n.storage_units.query("carrier == 'hydro'")

    h2_production_links = n.links.query(f"carrier in {h2_production_techs}")

    if renewable_gens.empty and hydro_storage_units.empty:
        logger.warning("No renewable generators or hydro storage units found. Skipping constraint.")
        return

    if h2_production_links.empty:
        logger.warning("No H2 production technology links found. Skipping constraint.")
        return

    w = DataArray(weightings.generators[sns]) # Use generators weighting for both generators and links
    if "dim_0" in w.dims:
        w = w.rename({"dim_0": "snapshot"})
    
    w_storage = DataArray(weightings.stores[sns])
    if "dim_0" in w_storage.dims:
        w_storage = w_storage.rename({"dim_0": "snapshot"})

    # LHS: Sum across all snapshots of renewable generation and hydro storage dispatch
    lhs_expressions = []
    
    if not renewable_gens.empty:
        gen_p = m["Generator-p"].loc[sns, renewable_gens.index]
        lhs_expressions.append((gen_p * w).sum())

    if not hydro_storage_units.empty:
        storage_p = m["StorageUnit-p_dispatch"].loc[sns, hydro_storage_units.index]
        lhs_expressions.append((storage_p * w_storage).sum())

    lhs_expr = sum(lhs_expressions)

    # RHS: Sum across all snapshots of H2 electrolysis consumption
    link_p = m["Link-p"].loc[sns, h2_production_links.index]
    rhs_expr = (link_p * w * green_h2_fraction).sum()

    constraint_name = "total_green_h2_constraint"
    m.add_constraints(lhs_expr >= rhs_expr, name=constraint_name)

    logger.info(f"Added constraint '{constraint_name}': "
                f"Total from {len(renewable_gens)} renewable generators + "
                f"{len(hydro_storage_units)} hydro storage units >= "
                f"total electricity consumption of {len(h2_production_links)} H2 production technology links")

def add_existing(n):
    if snakemake.wildcards["planning_horizons"] == "2050":
        directory = (
            "results/"
            + "Existing_capacities/"
            + snakemake.config["run"].replace("2050", "2030")
        )
        n_name = (
            snakemake.input.network.split("/")[-1]
            .replace(str(snakemake.config["scenario"]["clusters"][0]), "")
            .replace(str(snakemake.config["costs"]["discountrate"][0]), "")
            .replace("_presec", "")
            .replace(".nc", ".csv")
        )
        df = pd.read_csv(directory + "/electrolyzer_caps_" + n_name, index_col=0)
        existing_electrolyzers = df.p_nom_opt.values

        h2_index = n.links[n.links.carrier == "H2 Electrolysis"].index
        n.links.loc[h2_index, "p_nom_min"] = existing_electrolyzers

        # n_name = snakemake.input.network.split("/")[-1].replace(str(snakemake.config["scenario"]["clusters"][0]), "").\
        #     replace(".nc", ".csv").replace(str(snakemake.config["costs"]["discountrate"][0]), "")
        df = pd.read_csv(directory + "/res_caps_" + n_name, index_col=0)

        for tech in snakemake.config["custom_data"]["renewables"]:
            # df = pd.read_csv(snakemake.config["custom_data"]["existing_renewables"], index_col=0)
            existing_res = df.loc[tech]
            existing_res.index = existing_res.index.str.apply(lambda x: x + tech)
            tech_index = n.generators[n.generators.carrier == tech].index
            n.generators.loc[tech_index, tech] = existing_res


def add_lossy_bidirectional_link_constraints(n: pypsa.components.Network) -> None:
    """
    Ensures that the two links simulating a bidirectional_link are extended the same amount.
    """

    if not n.links.p_nom_extendable.any() or "reversed" not in n.links.columns:
        return

    # ensure that the 'reversed' column is boolean and identify all link carriers that have 'reversed' links
    n.links["reversed"] = n.links.reversed.fillna(0).astype(bool)
    carriers = n.links.loc[n.links.reversed, "carrier"].unique()  # noqa: F841

    # get the indices of all forward links (non-reversed), that have a reversed counterpart
    forward_i = n.links.query(
        "carrier in @carriers and ~reversed and p_nom_extendable"
    ).index

    # function to get backward (reversed) indices corresponding to forward links
    # this function is required to properly interact with the myopic naming scheme
    def get_backward_i(forward_i):
        return pd.Index(
            [
                (
                    re.sub(r"-(\d{4})$", r"-reversed-\1", s)
                    if re.search(r"-\d{4}$", s)
                    else s + "-reversed"
                )
                for s in forward_i
            ]
        )

    # get the indices of all backward links (reversed)
    backward_i = get_backward_i(forward_i)

    # get the p_nom optimization variables for the links using the get_var function
    links_p_nom = n.model["Link-p_nom"]

    # only consider forward and backward links that are present in the optimization variables
    subset_forward = forward_i.intersection(links_p_nom.indexes["Link-ext"])
    subset_backward = backward_i.intersection(links_p_nom.indexes["Link-ext"])

    # ensure we have a matching number of forward and backward links
    if len(subset_forward) != len(subset_backward):
        raise ValueError("Mismatch between forward and backward links.")

    # define the lefthand side of the constrain p_nom (forward) - p_nom (backward) = 0
    # this ensures that the forward links always have the same maximum nominal power as their backward counterpart
    lhs = links_p_nom.loc[backward_i] - links_p_nom.loc[forward_i]

    # add the constraint to the PySPA model
    n.model.add_constraints(lhs == 0, name="Link-bidirectional_sync")


def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to
    ``pypsa.linopf.network_lopf``.

    If you want to enforce additional custom constraints, this is a good location to add them.
    The arguments ``opts`` and ``snakemake.config`` are expected to be attached to the network.
    """
    opts = n.opts
    config = n.config
    if "BAU" in opts and n.generators.p_nom_extendable.any():
        add_BAU_constraints(n, config)
    if "SAFE" in opts and n.generators.p_nom_extendable.any():
        add_SAFE_constraints(n, config)
    if "CCL" in opts and n.generators.p_nom_extendable.any():
        add_CCL_constraints(n, config)
    reserve = config["electricity"].get("operational_reserve", {})
    if reserve.get("activate"):
        add_operational_reserve_margin(n, snapshots, config)
    for o in opts:
        if "RES" in o:
            res_share = float(re.findall("[0-9]*\.?[0-9]+$", o)[0])
            add_RES_constraints(n, res_share, config)
    for o in opts:
        if "EQ" in o:
            add_EQ_constraints(n, o)

    add_battery_constraints(n)
    add_lossy_bidirectional_link_constraints(n)

    if snakemake.config["sector"]["chp"]:
        logger.info("setting CHP constraints")
        add_chp_constraints(n)

    additionality = snakemake.params.policy_config["hydrogen"]["additionality"]
    ref_for_additionality = snakemake.params.policy_config["hydrogen"]["is_reference"]
    temportal_matching_period = snakemake.params.policy_config["hydrogen"][
        "temporal_matching"
    ]

    if temportal_matching_period == "no_temporal_matching":
        logger.info("no h2 temporal constraint set")

    elif additionality:
        if ref_for_additionality:
            logger.info("preparing reference case for additionality constraint")
        else:
            logger.info(
                "setting h2 export to {}ly matching constraint with additionality".format(
                    temportal_matching_period
                )
            )
            hydrogen_temporal_constraint(n, n_ref, temportal_matching_period)
    elif not additionality:
        logger.info(
            "setting h2 export to {}ly matching constraint without additionality".format(
                temportal_matching_period
            )
        )
        hydrogen_temporal_constraint(n, n_ref, temportal_matching_period)

    elif (temportal_matching_period == "hour"):
        green_h2_fraction = snakemake.config["policy_config"]["hydrogen"]["allowed_excess"]
        logger.info(f"setting total and hourly green H2 constraint with allowed excess (green H2 fraction): {green_h2_fraction}")
        h2_production_technologies = [
                "H2 Electrolysis",
                "Alkaline electrolyzer large",
                "Alkaline electrolyzer medium",
                "Alkaline electrolyzer small", 
                "PEM electrolyzer",
                "SOEC",
                "Solid biomass steam reforming",
                "Biomass gasification",
                "Biomass gasification CC",
            ]

        # NB: This constraints all H2 electrolysis, i.e. for h2 used domestically or for exports.
        add_hourly_green_h2_constraint(n, h2_production_technologies, green_h2_fraction) # includes battery discharge on lhs_expr
        add_total_green_h2_constraint(n, h2_production_technologies, green_h2_fraction) # excludes battery discharge on lhs_expr

    else:
        raise ValueError(
            'temporal_matching value is invalid, check config["policy_config"]'
        )
    
    if snakemake.config["policy_config"]["hydrogen"]["technology_matching_for_derivatives_and_export"]:
        logger.info("setting specific H2 production technologies for derivatives and export")
        add_specific_h2_techs_for_h2_export_and_conversion_constraint(n, h2_production_technologies)

    if n.links.carrier.str.contains("NH3 export").any():
        logger.info("adding specific_link_for_export_constraint to ensure that Haber-Bosch >= NH3 export link")
        add_specific_tech_for_export_constraint(n, "Haber-Bosch", "NH3 export") # Probably not required at the moment
    if n.links.carrier.str.contains("FT export").any():
        logger.info("adding specific_link_for_export_constraint to ensure that Fischer-Tropsch >= FT export link")
        add_specific_tech_for_export_constraint(n, "Fischer-Tropsch", "FT export") # otherwise fossil oil could be exported


    if snakemake.config["sector"]["hydrogen"]["network"]:
        if snakemake.config["sector"]["hydrogen"]["network_limit"]:
            add_h2_network_cap(
                n, snakemake.config["sector"]["hydrogen"]["network_limit"]
            )

    if snakemake.config["sector"]["hydrogen"]["set_color_shares"]:
        logger.info("setting H2 color mix")
        set_h2_colors(n)


def solve_network(n, config, solving, **kwargs):
    set_of_options = solving["solver"]["options"]
    cf_solving = solving["options"]

    kwargs["solver_options"] = (
        solving["solver_options"][set_of_options] if set_of_options else {}
    )
    kwargs["solver_name"] = solving["solver"]["name"]
    kwargs["extra_functionality"] = extra_functionality

    skip_iterations = cf_solving.get("skip_iterations", False)
    if not n.lines.s_nom_extendable.any():
        skip_iterations = True
        logger.info("No expandable lines found. Skipping iterative solving.")

    # add to network for extra_functionality
    n.config = config
    n.opts = opts

    if skip_iterations:
        status, condition = n.optimize(**kwargs)
    else:
        kwargs["track_iterations"] = (cf_solving.get("track_iterations", False),)
        kwargs["min_iterations"] = (cf_solving.get("min_iterations", 4),)
        kwargs["max_iterations"] = (cf_solving.get("max_iterations", 6),)
        status, condition = n.optimize.optimize_transmission_expansion_iteratively(
            **kwargs
        )

    if status != "ok":  # and not rolling_horizon:
        logger.warning(
            f"Solving status '{status}' with termination condition '{condition}'"
        )
    if "infeasible" in condition:
        labels = n.model.compute_infeasibilities()
        logger.info(f"Labels:\n{labels}")
        n.model.print_infeasibilities()
        raise RuntimeError("Solving status 'infeasible'")

    return n

##### temporary validation functions

def validate_green_h2_constraints(n, h2_production_techs):
    """
    Validate the impact and effectiveness of green hydrogen constraints.
    
    This function checks:
    1. Hourly green H2 constraint compliance
    2. Total green H2 constraint compliance  
    3. H2 electrolysis vs consumption balance
    4. Renewable energy allocation to H2 production
    
    Parameters
    ----------
    n : pypsa.Network
        Solved PyPSA network with optimization results
        
    Returns
    -------
    dict
        Dictionary containing validation results and metrics
    """
    from xarray import DataArray
    import pandas as pd
    
    logger.info("=== Validating Green H2 Constraints ===")
    
    validation_results = {}
    sns = n.snapshots
    
    # Get renewable generators and H2 production technology links
    renewable_techs = ["solar", "onwind", "offwind-ac", "offwind-dc", "csp", "ror"]
    renewable_gens = n.generators.query(f"carrier in {renewable_techs}")
    hydro_storage_units = n.storage_units.query("carrier == 'hydro'")
    battery_discharger_links = n.links[n.links.index.str.contains("battery discharger")]
    h2_production_links = n.links.query(f"carrier in {h2_production_techs}")
    
    if h2_production_links.empty:
        logger.warning("No H2 production technology links found. Skipping green H2 validation.")
        return {"error": "No H2 production technology links found"}
    
    # Calculate renewable generation
    renewable_generation = pd.Series(0.0, index=sns, name="renewable_generation")
    if not renewable_gens.empty:
        renewable_generation += n.generators_t.p[renewable_gens.index].sum(axis=1)
    
    # Calculate hydro storage dispatch
    hydro_dispatch = pd.Series(0.0, index=sns, name="hydro_dispatch")
    if not hydro_storage_units.empty:
        hydro_dispatch += n.storage_units_t.p_dispatch[hydro_storage_units.index].sum(axis=1)
    
    # Calculate battery discharge
    battery_discharge = pd.Series(0.0, index=sns, name="battery_discharge")
    if not battery_discharger_links.empty:
        battery_discharge_raw = n.links_t.p0[battery_discharger_links.index].sum(axis=1)
        battery_efficiency = battery_discharger_links["efficiency"].mean()
        battery_discharge = battery_discharge_raw * battery_efficiency
    
    # Calculate H2 electrolysis consumption
    h2_consumption = n.links_t.p0[h2_production_links.index].sum(axis=1)
    
    # 1. Hourly constraint validation
    hourly_green_supply = renewable_generation + hydro_dispatch + battery_discharge
    hourly_violations = (hourly_green_supply < h2_consumption).sum()
    hourly_violation_magnitude = (h2_consumption - hourly_green_supply).clip(lower=0).sum()
    
    validation_results["hourly_constraint"] = {
        "violations_count": int(hourly_violations),
        "total_snapshots": len(sns),
        "violation_percentage": float(hourly_violations / len(sns) * 100),
        "violation_magnitude_MWh": float(hourly_violation_magnitude),
        "max_hourly_deficit_MW": float((h2_consumption - hourly_green_supply).max()) if hourly_violations > 0 else 0.0
    }
    
    # 2. Total constraint validation (excluding batteries)
    total_renewable = (renewable_generation * n.snapshot_weightings.generators).sum()
    total_hydro = (hydro_dispatch * n.snapshot_weightings.stores).sum()
    total_h2_consumption = (h2_consumption * n.snapshot_weightings.generators).sum()
    total_green_supply_primary = total_renewable + total_hydro
    
    validation_results["total_constraint"] = {
        "total_renewable_MWh": float(total_renewable),
        "total_hydro_MWh": float(total_hydro),
        "total_primary_green_MWh": float(total_green_supply_primary),
        "total_h2_consumption_MWh": float(total_h2_consumption),
        "surplus_deficit_MWh": float(total_green_supply_primary - total_h2_consumption),
        "constraint_satisfied": bool(total_green_supply_primary >= total_h2_consumption - 1e-2)  # Small tolerance
    }
    
    # 3. H2 production vs consumption balance
    h2_production = n.links_t.p0[h2_production_links.index] * n.links.loc[h2_production_links.index, "efficiency"].values
    total_h2_production = (h2_production.sum(axis=1) * n.snapshot_weightings.generators).sum()
    
    validation_results["h2_balance"] = {
        "total_h2_production_MWh": float(total_h2_production),
        "total_electricity_consumption_MWh": float(total_h2_consumption),
        "average_efficiency": float(n.links.loc[h2_production_links.index, "efficiency"].mean())
    }
    
    # 4. Energy allocation analysis
    total_battery_discharge = (battery_discharge * n.snapshot_weightings.generators).sum()
    total_green_with_battery = total_green_supply_primary + total_battery_discharge
    
    validation_results["energy_allocation"] = {
        "renewable_share_of_h2": float(total_renewable / total_h2_consumption * 100) if total_h2_consumption > 0 else 0,
        "hydro_share_of_h2": float(total_hydro / total_h2_consumption * 100) if total_h2_consumption > 0 else 0,
        "battery_share_of_h2": float(total_battery_discharge / total_h2_consumption * 100) if total_h2_consumption > 0 else 0,
        "total_battery_discharge_MWh": float(total_battery_discharge),
        "primary_green_coverage_ratio": float(total_green_supply_primary / total_h2_consumption) if total_h2_consumption > 0 else 0
    }
    
    # Log summary
    logger.info(f"Hourly violations: {hourly_violations}/{len(sns)} snapshots ({hourly_violations/len(sns)*100:.1f}%)")
    logger.info(f"Total constraint satisfied: {validation_results['total_constraint']['constraint_satisfied']}")
    logger.info(f"Primary green coverage: {validation_results['energy_allocation']['primary_green_coverage_ratio']:.2f}")
    logger.info(f"Renewable share of H2: {validation_results['energy_allocation']['renewable_share_of_h2']:.1f}%")
    
    return validation_results


def validate_export_constraints(n, h2_production_techs):
    """
    Validate the impact and effectiveness of export-related constraints.
    
    This function checks:
    1. H2 electrolysis vs export consumption balance
    2. Individual export constraint compliance (NH3, FT)
    3. Export efficiency and utilization metrics
    
    Parameters
    ----------
    n : pypsa.Network
        Solved PyPSA network with optimization results
        
    Returns
    -------
    dict
        Dictionary containing export validation results and metrics
    """
    import pandas as pd
    
    logger.info("=== Validating Export Constraints ===")
    
    validation_results = {}
    sns = n.snapshots
    
    # Get relevant links
    h2_production_links = n.links.query(f"carrier in {h2_production_techs}")
    haber_bosch_links = n.links.query("carrier == 'Haber-Bosch'")
    fischer_tropsch_links = n.links.query("carrier == 'Fischer-Tropsch'")
    h2_export_links = n.links.query("carrier == 'H2 export'")
    nh3_export_links = n.links.query("carrier == 'NH3 export'")
    ft_export_links = n.links.query("carrier == 'FT export'")
    
    if h2_production_links.empty:
        logger.warning("No H2 production technology links found. Skipping export validation.")
        return {"error": "No H2 production technology links found"}
    
    # 1. H2 production vs export consumption
    h2_production_total = 0
    if not h2_production_links.empty:
        h2_production = n.links_t.p0[h2_production_links.index] * n.links.loc[h2_production_links.index, "efficiency"].values
        h2_production_total = (h2_production.sum(axis=1) * n.snapshot_weightings.generators).sum()
    
    # Calculate total H2 consumption for exports
    h2_consumption_exports = 0
    
    # Direct H2 export
    h2_direct_export = 0
    if not h2_export_links.empty:
        h2_direct_export = (n.links_t.p0[h2_export_links.index].sum(axis=1) * n.snapshot_weightings.generators).sum()
        h2_consumption_exports += h2_direct_export
    
    # H2 for Haber-Bosch (using efficiency2)
    h2_for_haber_bosch = 0
    if not haber_bosch_links.empty:
        hb_production = n.links_t.p0[haber_bosch_links.index].sum(axis=1)
        hb_h2_consumption_rate = -n.links.loc[haber_bosch_links.index, "efficiency2"].mean()  # efficiency2 is negative
        h2_for_haber_bosch = (hb_production * hb_h2_consumption_rate * n.snapshot_weightings.generators).sum()
        h2_consumption_exports += h2_for_haber_bosch
    
    # H2 for Fischer-Tropsch
    h2_for_fischer_tropsch = 0
    if not fischer_tropsch_links.empty:
        h2_for_fischer_tropsch = (n.links_t.p0[fischer_tropsch_links.index].sum(axis=1) * n.snapshot_weightings.generators).sum()
        h2_consumption_exports += h2_for_fischer_tropsch
    
    validation_results["h2_electrolysis_balance"] = {
        "total_h2_production_MWh": float(h2_production_total),
        "total_h2_for_exports_MWh": float(h2_consumption_exports),
        "h2_direct_export_MWh": float(h2_direct_export),
        "h2_for_haber_bosch_MWh": float(h2_for_haber_bosch),
        "h2_for_fischer_tropsch_MWh": float(h2_for_fischer_tropsch),
        "surplus_deficit_MWh": float(h2_production_total - h2_consumption_exports),
        "constraint_satisfied": bool(h2_production_total >= h2_consumption_exports - 1e-2)
    }
    
    # 2. Individual export constraints validation
    export_constraints = {}
    nh3_export_total = 0  # Initialize to avoid scope issues
    ft_export_total = 0   # Initialize to avoid scope issues
    
    # NH3 export constraint: Haber-Bosch production >= NH3 export
    if not haber_bosch_links.empty and not nh3_export_links.empty:
        hb_production_total = (n.links_t.p0[haber_bosch_links.index] * 
                              n.links.loc[haber_bosch_links.index, "efficiency"].values).sum(axis=1)
        hb_production_weighted = (hb_production_total * n.snapshot_weightings.generators).sum()
        
        nh3_export_total = (n.links_t.p0[nh3_export_links.index].sum(axis=1) * n.snapshot_weightings.generators).sum()
        
        export_constraints["nh3_export"] = {
            "haber_bosch_production_MWh": float(hb_production_weighted),
            "nh3_export_consumption_MWh": float(nh3_export_total),
            "surplus_deficit_MWh": float(hb_production_weighted - nh3_export_total),
            "constraint_satisfied": bool(hb_production_weighted >= nh3_export_total - 1e-2),
            "utilization_rate": float(nh3_export_total / hb_production_weighted * 100) if hb_production_weighted > 0 else 0
        }
    
    # FT export constraint: Fischer-Tropsch production >= FT export
    if not fischer_tropsch_links.empty and not ft_export_links.empty:
        ft_production_total = (n.links_t.p0[fischer_tropsch_links.index] * 
                              n.links.loc[fischer_tropsch_links.index, "efficiency"].values).sum(axis=1)
        ft_production_weighted = (ft_production_total * n.snapshot_weightings.generators).sum()
        
        ft_export_total = (n.links_t.p0[ft_export_links.index].sum(axis=1) * n.snapshot_weightings.generators).sum()
        
        export_constraints["ft_export"] = {
            "fischer_tropsch_production_MWh": float(ft_production_weighted),
            "ft_export_consumption_MWh": float(ft_export_total),
            "surplus_deficit_MWh": float(ft_production_weighted - ft_export_total),
            "constraint_satisfied": bool(ft_production_weighted >= ft_export_total - 1e-2),
            "utilization_rate": float(ft_export_total / ft_production_weighted * 100) if ft_production_weighted > 0 else 0
        }
    
    validation_results["export_constraints"] = export_constraints
    
    # 3. Overall export metrics
    total_exports = h2_direct_export + nh3_export_total + ft_export_total
    
    validation_results["export_summary"] = {
        "total_export_value_MWh": float(total_exports),
        "h2_export_share": float(h2_direct_export / total_exports * 100) if total_exports > 0 else 0,
        "nh3_export_share": float(nh3_export_total / total_exports * 100) if total_exports > 0 else 0,
        "ft_export_share": float(ft_export_total / total_exports * 100) if total_exports > 0 else 0,
        "h2_utilization_for_exports": float(h2_consumption_exports / h2_production_total * 100) if h2_production_total > 0 else 0
    }
    
    # Log summary
    logger.info(f"H2 electrolysis balance satisfied: {validation_results['h2_electrolysis_balance']['constraint_satisfied']}")
    logger.info(f"H2 utilization for exports: {validation_results['export_summary']['h2_utilization_for_exports']:.1f}%")
    if 'nh3_export' in export_constraints:
        logger.info(f"NH3 export constraint satisfied: {export_constraints['nh3_export']['constraint_satisfied']}")
    if 'ft_export' in export_constraints:
        logger.info(f"FT export constraint satisfied: {export_constraints['ft_export']['constraint_satisfied']}")
    
    return validation_results


def validate_all_export_constraints(n, h2_production_techs=None):
    
    if h2_production_techs is None:
        h2_production_techs = [
            "H2 Electrolysis",
            "Alkaline electrolyzer large",
            "Alkaline electrolyzer medium",
            "Alkaline electrolyzer small",
            "PEM electrolyzer",
            "SOEC",
            "Solid biomass steam reforming",
            "Biomass gasification",
            "Biomass gasification CC",
        ]
    
    logger.info("=== Comprehensive Constraint Validation ===")
    
    validation_results = {
        "green_h2": validate_green_h2_constraints(n, h2_production_techs),
        "exports": validate_export_constraints(n, h2_production_techs)
    }
    
    # Overall assessment
    green_h2_ok = validation_results["green_h2"].get("total_constraint", {}).get("constraint_satisfied", False)
    exports_ok = validation_results["exports"].get("h2_electrolysis_balance", {}).get("constraint_satisfied", False)
    
    validation_results["summary"] = {
        "all_constraints_satisfied": green_h2_ok and exports_ok,
        "green_h2_constraints_ok": green_h2_ok,
        "export_constraints_ok": exports_ok,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    logger.info(f"=== Validation Summary ===")
    logger.info(f"All constraints satisfied: {validation_results['summary']['all_constraints_satisfied']}")
    logger.info(f"Green H2 constraints OK: {green_h2_ok}")
    logger.info(f"Export constraints OK: {exports_ok}")
    
    return validation_results


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_sector_network",
            simpl="",
            clusters="10",
            ll="v1.1",
            opts="Co2L0.59",
            planning_horizons="2030",
            discountrate="0.082",
            demand="RF",
            sopts="144H",
            eopts="H2v1.0+NH3v1.0+FTv1.0",
            # configfile="config.tutorial.yaml",
        )

    configure_logging(snakemake)

    opts = snakemake.wildcards.opts.split("-")
    solve_opts = snakemake.config["solving"]["options"]

    is_sector_coupled = "sopts" in snakemake.wildcards.keys()

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)

    if snakemake.params.augmented_line_connection.get("add_to_snakefile"):
        n.lines.loc[n.lines.index.str.contains("new"), "s_nom_min"] = (
            snakemake.params.augmented_line_connection.get("min_expansion")
        )

    if (
        snakemake.config["custom_data"]["add_existing"]
        and snakemake.wildcards.planning_horizons == "2050"
        and is_sector_coupled
    ):
        add_existing(n)

    if (
        snakemake.params.policy_config["hydrogen"]["additionality"]
        and not snakemake.params.policy_config["hydrogen"]["is_reference"]
        and snakemake.params.policy_config["hydrogen"]["temporal_matching"]
        != "no_temporal_matching"
        and is_sector_coupled
    ):
        n_ref_path = snakemake.params.policy_config["hydrogen"]["path_to_ref"]
        n_ref = pypsa.Network(n_ref_path)
    else:
        n_ref = None

    n = prepare_network(n, solve_opts, config=solve_opts)

    n = solve_network(
        n,
        config=snakemake.config,
        solving=snakemake.params.solving,
        log_fn=snakemake.log.solver,
    )
    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
    logger.info(f"Objective function: {n.objective}")
    logger.info(f"Objective constant: {n.objective_constant}")


    results = validate_all_export_constraints(
        n, 
        getattr(snakemake.config["policy_config"], 'h2_production_technologies', None)
    )

    # Check if all constraints are satisfied
    if results["summary"]["all_constraints_satisfied"]:
        logger.info("All green H2 and export constraints satisfied!")
    else:
        logger.warning("Some constraints violated - check validation results")