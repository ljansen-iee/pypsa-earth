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
    rename_electricity, 
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
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        ifind = pd.Series(c.df.index.str.find(" ", start=4), c.df.index)

        for i in ifind.value_counts().index:
            # these have already been assigned defaults
            if i == -1:
                continue

            names = ifind.index[ifind == i]

            c.df.loc[names, "location"] = names.str[:i]


if __name__ == "__main__":
    if "snakemake" not in globals():
        # os.chdir(Path(__file__).parent.resolve().parent)
        snakemake = mock_snakemake(
            "plot_power_network",
            simpl="",
            clusters="10",
            ll="v1.1",
            opts="Co2L0.5",
            planning_horizons="2040",
            sopts="1H",
            discountrate=0.07,
            demand="EL",
            eopts="NH3v1.0+MEOHv1.0",
        )


    n = pypsa.Network(snakemake.input.network)
    n.statistics.set_parameters(nice_names=False, drop_zero=False, round=6)
    regions = gpd.read_file(snakemake.input.regions_onshore).set_index("name")

    map_opts = {}

    map_opts["geomap_colors"] = False
    # {
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

    # bus_scale = {"2030":5e4, "2050":1e5}[yr]
    # line_scale = {"total_transmission":3e3,"expanded_transmission":3e2}
    bus_size_factor=2e5
    transmission = True
    with_legend=True

    transmission_limit = "lv1.1"

    tech_colors = colors["electricity"]

    preferred_order = pd.Index(['Solar PV', 'Wind onshore', 'Gas turbines', 'Coal',  'Oil', 'Hydro'])



    oc = n.statistics.optimal_capacity(comps=["Generator", "Link", "StorageUnit"], bus_carrier="AC", groupby=["bus","carrier"], aggregate_across_components=True,drop_zero=True).dropna().round(2)
    oc = oc[oc>0].reset_index()
    oc = oc[~oc["carrier"].str.contains("shedding")]
    oc["carrier"] = oc["carrier"].map(rename_electricity).map(rename_to_upper_case)
    oc = oc.groupby(["bus","carrier"]).sum().round(2).squeeze().unstack()

    new_columns = preferred_order.intersection(oc.columns).append(
            oc.columns.difference(preferred_order)
        )
    oc = oc[new_columns]
    oc = oc.stack()


    consumption = (
        n.statistics.withdrawal(
            comps=["Load","Link","Generator"],
            bus_carrier="AC",
            groupby=["carrier","bus"],
            drop_zero=True)
        .div(1e6))

    regions["power_consumption"] = consumption.groupby("bus").sum().round(2)


    for item in new_columns:
        if item not in tech_colors:
            logger.warning(f"{item} not in tech_colors dict!")

    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)
    
    n.links.drop(n.links.index[(n.links.carrier != "DC")],inplace=True,)

    to_drop = oc.index.levels[0].symmetric_difference(n.buses.index)
    if len(to_drop) != 0:
        logger.info(f"Dropping non-buses {to_drop.tolist()}")
        oc.drop(to_drop, level=0, inplace=True, axis=0, errors="ignore")

    # make sure they are removed from index
    oc.index = pd.MultiIndex.from_tuples(oc.index.values)



    threshold = 10 # MW  
    carriers = oc.groupby(level=1).sum()
    carriers = carriers.where(carriers > threshold).dropna()
    carriers = preferred_order.intersection(carriers.index).append(
            carriers.index.difference(preferred_order)
        )
    carriers = list(carriers)

    # PDF has minimum width, so set these to zero
    line_lower_threshold = 500.0
    line_upper_threshold = 1e4
    linewidth_factor = 1e3
    ac_color = '#4cc2a6'
    dc_color = "#7c154d"

    title = "Added grid"

    if transmission_limit == "lv1.0":
        # should be zero
        line_widths = n.lines.s_nom_opt - n.lines.s_nom
        link_widths = n.links.p_nom_opt - n.links.p_nom
        if transmission:
            line_widths = n.lines.s_nom_opt
            link_widths = n.links.p_nom_opt
            linewidth_factor = 2e3
            line_lower_threshold = 0.0
            title = "Current grid"
    else:
        line_widths = n.lines.s_nom_opt - n.lines.s_nom_min
        link_widths = n.links.p_nom_opt - n.links.p_nom_min
        if transmission:
            line_widths = n.lines.s_nom_opt
            link_widths = n.links.p_nom_opt
            title = "Total grid"

    line_widths = line_widths.clip(line_lower_threshold, line_upper_threshold)
    link_widths = link_widths.clip(line_lower_threshold, line_upper_threshold)

    line_widths = line_widths.replace(line_lower_threshold, 0)
    link_widths = link_widths.replace(line_lower_threshold, 0)

    # Transform regions to the projection CRS for plotting
    if hasattr(crs, 'proj4_init'):
        regions = regions.to_crs(crs.proj4_init)
    else:
        # For newer versions, use the CRS string representation
        regions = regions.to_crs(crs.to_string())

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": crs})
    # fig.set_size_inches(7, 6)
    # ax.set_extent(regions.total_bounds[[0, 2, 1, 3]] + [-1, 1, -1, 1], crs=crs)

    with plt.rc_context({"patch.linewidth": 0.3}):
        n.plot(
            bus_sizes=oc / bus_size_factor,
            bus_alpha=1.0,
            bus_colors=tech_colors,
            line_colors=ac_color,
            link_colors=dc_color,
            line_widths=line_widths / linewidth_factor,
            link_widths=link_widths / linewidth_factor,
            ax=ax,
            geomap_resolution="10m",
            **map_opts,
        )



        # ax.set_extent(regions.total_bounds[[0, 2, 1, 3]], crs=crs)
        regions.plot(
            ax=ax,
            column="power_consumption",
            cmap="Greys", #"Purples",
            alpha=0.6,
            edgecolor="darkgray", #"whitesmoke", #"darkgray"
            facecolor="white",
            aspect="auto",
            transform=crs,
            linewidth=1,
            vmax=regions["power_consumption"].abs().max()*1 if not (regions["power_consumption"] == 0).all() else 1,
            vmin=0,
            legend=True,
            legend_kwds={
                "label": "Consumption [TWh]",
                "shrink": 0.35,
                "orientation": "vertical", #"horizontal",
                "pad": -0.1,
                "extend": "max",
            }
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


        sizes = [4000, 8000]
        labels = [f"{s / 1000} GW" for s in sizes]
        sizes = [s / bus_size_factor for s in sizes]

        legend_kw = dict(
            loc="upper left",
            bbox_to_anchor=(0., 1.06),
            labelspacing=0.9,
            frameon=False,
            facecolor="white",
            edgecolor="lightgray",
            framealpha=1.0,  # Ensure frame is fully opaque
            handletextpad=0.5,
            borderpad=1.0,  # Padding inside the legend frame
            columnspacing=1.0,  # Space between columns
            handlelength=2.0,  # Length of legend handles
            title="Installed capacity",
        )

        add_legend_circles(
            ax,
            sizes,
            labels,
            srid=n.srid,
            patch_kw={"edgecolor":"#6d6d6d","facecolor":"white","alpha":1.0},
            legend_kw=legend_kw,
        )

        sizes = [2, 4]
        labels = [f"{s} GW" for s in sizes]
        scale = 1e3 / linewidth_factor
        sizes = [s * scale for s in sizes]

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
            title=title,
        )

        add_legend_lines(
            ax, sizes, labels, patch_kw={"color":"#4cc2a6"}, legend_kw=legend_kw
        )

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

        if with_legend:
            colors = [tech_colors[c] for c in carriers] #+ [ac_color, dc_color]
            labels = carriers #+ ["HVAC line", "HVDC link"]

            add_legend_patches(
                ax,
                colors,
                labels,
                legend_kw=legend_kw,
            )
        plt.show()
        fig.savefig(snakemake.output.power_map, bbox_inches="tight")
        fig.savefig(snakemake.output.power_map_svg, bbox_inches="tight")
        plt.close(fig)

        oc.to_csv(snakemake.output.power_map_csv_optimal_capacity)
        line_widths.to_csv(snakemake.output.power_map_csv_lines)
