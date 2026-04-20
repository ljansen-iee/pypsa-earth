import logging
logger = logging.getLogger(__name__)
from pathlib import Path
import os
import pandas as pd
idx_slice = pd.IndexSlice
import pypsa
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Comment out for debugging and development
warnings.simplefilter(action='ignore', category=DeprecationWarning) # Comment out for debugging and development

import cartopy.crs as ccrs
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

from plot_helpers import (
    mock_snakemake,
    rename_h2,
    rename_to_upper_case,
    colors,)

"""
Contextily Background Maps:
This script supports adding background map tiles using contextily. Configure the following options in contextily_opts:
- add_basemap: True/False to enable/disable background maps
- basemap_source: Choose from ctx.providers (e.g., OpenStreetMap.Mapnik, CartoDB.Positron, Stamen.Terrain)
- basemap_alpha: Transparency level (0.0-1.0)
- basemap_zoom: Zoom level ('auto' for automatic, or integer value)

Popular basemap sources:
- ctx.providers.OpenStreetMap.Mapnik (default)
- ctx.providers.CartoDB.Positron (light theme)
- ctx.providers.CartoDB.DarkMatter (dark theme)
- ctx.providers.Stamen.Terrain (topographic)
- ctx.providers.ESRI.WorldImagery (satellite)
"""

def assign_location(n):
    """Assign location to network components for plotting"""
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        ifind = pd.Series(c.df.index.str.find(" ", start=4), c.df.index)

        for i in ifind.value_counts().index:
            # these have already been assigned defaults
            if i == -1:
                continue

            names = ifind.index[ifind == i]
            c.df.loc[names, "location"] = names.str[:i]

def group_pipes(df, drop_direction=False):
    """
    Group pipes which connect same buses and return overall capacity.
    """
    df = df.copy()
    if drop_direction:
        positive_order = df.bus0 < df.bus1
        df_p = df[positive_order]
        swap_buses = {"bus0": "bus1", "bus1": "bus0"}
        df_n = df[~positive_order].rename(columns=swap_buses)
        df = pd.concat([df_p, df_n])

    # there are pipes for each investment period rename to AC buses name for plotting
    df["index_orig"] = df.index
    df.index = df.apply(
        lambda x: f"H2 pipeline {x.bus0.replace(' H2', '')} -> {x.bus1.replace(' H2', '')}",
        axis=1,
    )
    return df.groupby(level=0).agg(
        {"p_nom_opt": "sum", "bus0": "first", "bus1": "first", "index_orig": "first"}
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        # os.chdir(Path(__file__).parent.resolve().parent)
        snakemake = mock_snakemake(
            "plot_hydrogen_network",
            simpl="",
            clusters="10",
            ll="v1.1",
            opts="Co2L0.80",
            planning_horizons="2035",
            sopts="1H",
            discountrate=0.090,
            demand="RF",
            eopts="H2v1.0+HBIv0.5",
            )

    # scenario:

    n = pypsa.Network(snakemake.input.network)

    assign_location(n)
    pypsa.options.params.statistics.nice_names = False
    pypsa.options.params.statistics.drop_zero = False
    pypsa.options.params.statistics.round = 6
    regions = gpd.read_file(snakemake.input.regions_onshore).set_index("name")

    map_opts = {}
    map_opts["geomap_colors"] = False
    # map_opts["geomap_colors"] = {
    #                 #"ocean": "lightblue",
    #                 #"land": "whitesmoke",
    #                 "border": "darkgray",
    #                 "coastline": "black",
    #             }

    map_opts["boundaries"] = regions.total_bounds[[0, 2, 1, 3]] + [-1, 1, -1, 1]

    # Contextily background map options (separate from map_opts)
    contextily_opts = {
        "add_basemap": False,  # Set to False to disable background maps
        "basemap_source": ctx.providers.Esri.WorldShadedRelief,  # Default basemap
        "basemap_alpha": 0.6,  # Transparency of background map
        "basemap_zoom": "auto"  # Auto-determine zoom level
    }

    crs = ccrs.Mercator(n.buses.query("carrier == 'AC'").x.mean()) #ccrs.Mercator()#ccrs.Mercator(n.buses.query("carrier == 'AC'").x.mean())

    # bus_scale = {"2030":3e3, "2050":3e4}[yr]
    # link_scale = {"2030":10, "2050":5e2}[yr]
    bus_size_factor=6e4
    with_legend=True

    tech_colors = colors["hydrogen"]

    preferred_order = pd.Index(['H2 Electrolysis', "Methanol steam reforming", "H2 Fuel Cell"])


    h2_storage = n.stores.query("carrier == 'H2 Store'")
    regions["H2_storage"] = (
        h2_storage.rename(index=h2_storage.bus.map(n.buses.location)) # removes " H2 Store" from index
        .e_nom_opt.groupby(level=0)
        .sum()
        .div(1e3)
    ) # GWh
    regions["H2_storage"] = regions["H2_storage"].where(regions["H2_storage"] > 0.1)

    # Net total h2 consumption
    H2_consumption = -(
        n.statistics
        .energy_balance(
            groupby=["carrier","bus"],
            bus_carrier="H2",
            drop_zero=True)
        .drop("H2 pipeline", level="carrier"))
    H2_consumption = (
        H2_consumption[H2_consumption>1e-3]
        .groupby(["bus"])
        .sum()
        .div(1e6).round(2)) # TWh
    H2_consumption.index = H2_consumption.index.str.removesuffix(" H2")
    regions["H2_consumption"] = H2_consumption

    oc = (
        n.statistics
        .optimal_capacity(
            comps=["Link"], 
            bus_carrier="H2", 
            groupby=["bus","carrier"],
            aggregate_across_components=True,
            drop_zero=True)
        .drop("H2 pipeline", level="carrier")
        .round(2)
    )
    oc = oc[oc>5].reset_index()
    oc = oc[~oc["carrier"].str.contains("shedding")]
    oc["carrier"] = oc["carrier"].map(rename_h2).map(rename_to_upper_case)
    oc = oc.groupby(["bus","carrier"]).sum().round(2).squeeze().unstack()

    oc.rename(index=lambda x: x.replace(" H2", ""), level=0, inplace=True)
    oc = oc.reindex(n.buses.query("carrier == 'AC'").index, level=0, fill_value=0)
    oc = oc.stack()

    # Filter carriers by threshold to avoid plotting very small capacities
    threshold = 5 # MW  
    carriers_filtered = oc.groupby(level=1).sum()
    carriers_filtered = carriers_filtered.where(carriers_filtered > threshold).dropna()
    carriers_filtered = preferred_order.intersection(carriers_filtered.index).append(
            carriers_filtered.index.difference(preferred_order)
        )
    carriers_list = list(carriers_filtered)

    # Check for missing tech_colors
    for item in carriers_list:
        if item not in tech_colors:
            logger.warning(f"{item} not in tech_colors dict!")

    # Data validation - drop non-buses
    to_drop = oc.index.levels[0].symmetric_difference(n.buses.query("carrier == 'AC'").index)
    if len(to_drop) != 0:
        logger.info(f"Dropping non-buses {to_drop.tolist()}")
        oc.drop(to_drop, level=0, inplace=True, axis=0, errors="ignore")

    # make sure they are removed from index
    oc.index = pd.MultiIndex.from_tuples(oc.index.values)

    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    n.remove("Link", n.links[~n.links.carrier.str.contains("H2 pipeline")].index)

    foresight = "overnight"

    # PDF has minimum width, so set these to zero
    line_lower_threshold = 2
    line_upper_threshold = 1e4
    branch_width_factor = {"2035":2e2,"2040":3e2, "2050":5e2}["2040"]

    color_h2_pipe = "#4cc2a6"
    color_retrofit = "#499a9c"

    carriers = ["H2 Electrolysis", "Methanol steam reforming", "H2 Fuel Cell"] #, "H2 Fuel Cell"

    h2_new = n.links[n.links.carrier == "H2 pipeline"]
    h2_retro = n.links[n.links.carrier == "H2 pipeline retrofitted"]

    if foresight == "myopic":
        # sum capacitiy for pipelines from different investment periods
        h2_new = group_pipes(h2_new)

        if not h2_retro.empty:
            h2_retro = (
                group_pipes(h2_retro, drop_direction=True)
                .reindex(h2_new.index)
                .fillna(0)
            )

    if not h2_retro.empty:
        if foresight != "myopic":
            positive_order = h2_retro.bus0 < h2_retro.bus1
            h2_retro_p = h2_retro[positive_order]
            swap_buses = {"bus0": "bus1", "bus1": "bus0"}
            h2_retro_n = h2_retro[~positive_order].rename(columns=swap_buses)
            h2_retro = pd.concat([h2_retro_p, h2_retro_n])

            h2_retro["index_orig"] = h2_retro.index
            h2_retro.index = h2_retro.apply(
                lambda x: f"H2 pipeline {x.bus0.replace(' H2', '')} -> {x.bus1.replace(' H2', '')}",
                axis=1,
            )

        retro_w_new_i = h2_retro.index.intersection(h2_new.index)
        h2_retro_w_new = h2_retro.loc[retro_w_new_i]

        retro_wo_new_i = h2_retro.index.difference(h2_new.index)
        h2_retro_wo_new = h2_retro.loc[retro_wo_new_i]
        h2_retro_wo_new.index = h2_retro_wo_new.index_orig.apply(
            lambda x: x.split("-2")[0]
        )

        to_concat = [h2_new, h2_retro_w_new, h2_retro_wo_new]
        h2_total = pd.concat(to_concat).p_nom_opt.groupby(level=0).sum()

    else:
        h2_total = h2_new.p_nom_opt

    link_widths_total = h2_total / branch_width_factor

    n.links.rename(index=lambda x: x.split("-2")[0], inplace=True)

    # group links by summing up p_nom values and taking the first value of the rest of the columns
    other_cols = dict.fromkeys(n.links.columns.drop(["p_nom_opt", "p_nom"]), "first")
    n.links = n.links.groupby(level=0).agg(
        {"p_nom_opt": "sum", "p_nom": "sum", **other_cols}
    )

    link_widths_total = link_widths_total.reindex(n.links.index).fillna(0.0)
    link_widths_total[n.links.p_nom_opt < line_lower_threshold] = 0.0

    retro = n.links.p_nom_opt.where(
        n.links.carrier == "H2 pipeline retrofitted", other=0.0
    )
    link_widths_retro = retro / branch_width_factor
    link_widths_retro[n.links.p_nom_opt < line_lower_threshold] = 0.0

    n.links.bus0 = n.links.bus0.str.replace(" H2", "")
    n.links.bus1 = n.links.bus1.str.replace(" H2", "")

    # Transform regions to the projection CRS for plotting
    if hasattr(crs, 'proj4_init'):
        regions = regions.to_crs(crs.proj4_init)
    else:
        # For newer versions, use the CRS string representation
        regions = regions.to_crs(crs.to_string())

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": crs})

    color_h2_pipe = "#7c154d"
    color_retrofit = "#bb0056"

    with plt.rc_context({"patch.linewidth": 0.3}):
        n.plot(
            geomap=True,
            bus_alpha=1.0,
            bus_sizes=oc/bus_size_factor,
            bus_colors=tech_colors,
            link_colors=color_h2_pipe,
            link_widths=link_widths_total,
            branch_components=["Link"],
            ax=ax,
            geomap_resolution="10m",
            **map_opts,
        )

        n.plot(
            geomap=True,
            bus_alpha=1.0,
            bus_sizes=0,
            link_colors=color_retrofit,
            link_widths=link_widths_retro,
            branch_components=["Link"],
            ax=ax,
            geomap_resolution="10m",
            **map_opts,
        )

    regions.plot(
        ax=ax,
        column="H2_consumption",
        cmap="Blues",
        edgecolor="darkgray", #"darkgray"whitesmoke
        facecolor="white",
        aspect="auto",
        transform=crs,
        linewidth=1,
        vmax=regions["H2_consumption"].abs().max()*1 if not (regions["H2_consumption"] == 0).all() else 1,
        vmin=0,
        legend=True,
        legend_kwds={
            "label": "Hydrogen consumption [TWh]",
            "shrink": 0.35,
            "orientation":"vertical",
            "pad": -0.05,
            "extend": "max",
        },
    )

    # Add contextily basemap if enabled
    if contextily_opts.get("add_basemap", False):
        try:
            # Convert CRS to format that contextily can understand
            if hasattr(crs, 'proj4_init'):
                basemap_crs = crs.proj4_init
            else:
                # For newer versions of cartopy/pyproj
                basemap_crs = crs.to_string()
            
            ctx.add_basemap(
                ax,
                crs=basemap_crs,
                source=contextily_opts.get("basemap_source", ctx.providers.OpenStreetMap.Mapnik), #ctx.providers.OpenStreetMap.Mapnik
                alpha=contextily_opts.get("basemap_alpha", 0.6),
                zoom=contextily_opts.get("basemap_zoom", "auto")
            )
        except Exception as e:
            logger.warning(f"Failed to add contextily basemap: {e}")
            logger.info("Continuing without background map tiles")

    # production capacity legend
    sizes = [1, 3]
    labels = [f"{s} GW" for s in sizes]
    sizes = [s / bus_size_factor * 1e3 for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0., 1.06), #(-0.1, 1.06),
        labelspacing=0.9,
        frameon=False,
        facecolor="white",
        edgecolor="lightgray",
        framealpha=1.0,  # Ensure frame is fully opaque
        handletextpad=0.5,
        borderpad=1.0,  # Padding inside the legend frame
        columnspacing=1.0,  # Space between columns
        handlelength=2.0,  # Length of legend handles
        title="H2 production capacity",
    )

    add_legend_circles(
        ax,
        sizes,
        labels,
        srid=n.srid,
        patch_kw={"edgecolor":"#6d6d6d","facecolor":"white","alpha":1.0},
        legend_kw=legend_kw,
    )

    # pipeline capacity legend
    sizes = [1000, 2000]
    labels = [f"{s/1000} GW" for s in sizes]
    sizes = [s / branch_width_factor * 1 for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.35, 1.06),
        frameon=False,
        facecolor="white",
        edgecolor="lightgray",
        framealpha=1.0,  # Ensure frame is fully opaque
        labelspacing=0.9,
        handletextpad=1,
        borderpad=1.0,  # Padding inside the legend frame
        columnspacing=1.0,  # Space between columns
        handlelength=2.0,  # Length of legend handles
        title="H2 pipeline capacity",
    )

    add_legend_lines(
        ax,
        sizes,
        labels,
        patch_kw=dict(color=color_h2_pipe),
        legend_kw=legend_kw,
    )

    # Technology legend
    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.70, 1.06), #(-0.1, 0.80),
        frameon=False,
        facecolor="white",
        edgecolor="lightgray",
        framealpha=1.0,  # Ensure frame is fully opaque
        borderpad=1.0,  # Padding inside the legend frame
        columnspacing=1.0,  # Space between columns
        handlelength=2.0,  # Length of legend handles
    )

    if with_legend and 'carriers_list' in locals():
        colors = [tech_colors[c] for c in carriers_list if c in tech_colors] # + [color_h2_pipe, color_retrofit]
        labels = [c for c in carriers_list if c in tech_colors] # + ["H2 pipeline (total)", "H2 pipeline (repurposed)"]

        if colors and labels:  # Only add legend if we have valid colors and labels
            add_legend_patches(
                ax,
                colors,
                labels,
                legend_kw=legend_kw,
            )

    ax.set_facecolor("white")

    plt.show()
    fig.savefig(snakemake.output.hydrogen_map, bbox_inches="tight")
    fig.savefig(snakemake.output.hydrogen_map_svg, bbox_inches="tight")
    plt.close(fig)

    oc.to_csv(snakemake.output.hydrogen_map_csv_optimal_capacity)
    link_widths_total.to_csv(snakemake.output.hydrogen_map_csv_links)
