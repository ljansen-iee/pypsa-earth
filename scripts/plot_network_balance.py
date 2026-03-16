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
from pypsa.plot import add_legend_circles, add_legend_semicircles, add_legend_lines, add_legend_patches
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
            "plot_network_balance",
            simpl="",
            clusters="10",
            ll="v1.1",
            opts="", #Co2L1.09
            planning_horizons="2035",
            sopts="1H",
            discountrate=0.078,
            demand="Exp",
            eopts="H2v1.0",
        )
        
    bus_size_factor={"2030": 2e1, "2035": "auto", "2050":"auto"}[snakemake.wildcards.planning_horizons] #1.5e2

    branch_width_factor = {"2030": 1e2, "2035": "auto", "2050": "auto"}[snakemake.wildcards.planning_horizons]
    branch_flow_factor = 7e8 #3e4


    EB_THRESHOLD = {"2030": 0.1, "2035": 0.1, "2050": 0.1}[snakemake.wildcards.planning_horizons]
    
    # PDF has minimum width, so set these to zero
    branch_lower_threshold = {"2030": 2, "2035": 2, "2050": 2}[snakemake.wildcards.planning_horizons]
    branch_upper_threshold = 1e4

    # legend sizes
    eb_semi_sizes = [8, -8]
    eb_semi_labels = ["+8 TWh", "-8 TWh"]

    flow_sizes = {"2030": [1, 0.5], "2035": [10, 5], "2050": [60, 30]}[snakemake.wildcards.planning_horizons]
    # flow_sizes = {"2030": [1, 0.5], "2035": [1, 0.5], "2050": [5, 10]}[snakemake.wildcards.planning_horizons]
    # flow_sizes = {"2030": [1, 0.5], "2035": [1, 0.5], "2050": [20, 40]}[snakemake.wildcards.planning_horizons]
    
    flow_labels = [f"{s} TWh" for s in flow_sizes]

    tech_colors = colors["hydrogen"]

    # bbox_to_anchors = [(0.05, 1.04), (0.40, 1.04), (0.02,0.85)]
    # bbox_to_anchors = [(0.05, 1.04), (0.40, 1.04), (0.70, 1.06)]  #(0.70, 1.06), #(-0.1, 0.80),
    
    if "ZA" in snakemake.config.get("countries", ""):
        bbox_to_anchors = [(0.97, 0.95), (0.96, 0.75), (0.95, 0.59)]  #(0.70, 1.06), #(-0.1, 0.80),
        # bbox_to_anchors = [(0.05, 1.00), (0.40, 1.00), (0.95, 0.95)]  #(0.70, 1.06), #(-0.1, 0.80),
    elif "EG" in snakemake.config.get("countries", ""):
        bbox_to_anchors = [(0.97, 0.95), (0.96, 0.75), (0.95, 0.59)]  #(0.70, 1.06), #(-0.1, 0.80),
    else:
        bbox_to_anchors = [(0.97, 0.95), (0.96, 0.75), (0.95, 0.59)]  #(0.70, 1.06), #(-0.1, 0.80),
        
        # bbox_to_anchors = [(0.05, 1.10), (0.40, 1.10), (0.95, 1.05)]  #(0.70, 1.06), #(-0.1, 0.80),

    n = pypsa.Network(snakemake.input.network)
    assign_location(n)
    pypsa.options.params.statistics.nice_names = False
    pypsa.options.params.statistics.drop_zero = False
    pypsa.options.params.statistics.round = 6

    map_opts = {}
    map_opts["geomap_colors"] = False
    # map_opts["geomap_colors"] = {
    #                 #"ocean": "lightblue",
    #                 #"land": "whitesmoke",
    #                 "border": "darkgray",
    #                 "coastline": "black",
    #             }
    regions = gpd.read_file(snakemake.input.regions_onshore).set_index("name")

    if "CL" in snakemake.config.get("countries", ""):
        gadm_shapes = gpd.read_file(snakemake.input.gadm_shapes).set_index("GADM_ID")
        gadm_shapes.index = gadm_shapes.index.astype(str) + "_AC"
        
        # Find missing indices and add them from GADM shapes
        missing_indices = gadm_shapes.index.difference(regions.index)
        if len(missing_indices) > 0:
            missing_regions = gadm_shapes.loc[missing_indices]
            regions = pd.concat([regions, missing_regions])

        # Manual boundaries for Chile mainland (excluding outlying islands)
        chile_bounds = [-80, -66, -56, -17]  # [min_lon, max_lon, min_lat, max_lat]
        map_opts["boundaries"] = chile_bounds
    else: 
       map_opts["boundaries"] = regions.total_bounds[[0, 2, 1, 3]] + [-1, 1, -1, 1]

    #%%
    

    # Contextily background map options (separate from map_opts)
    contextily_opts = {
        "add_basemap": False,  # Set to False to disable background maps
        "basemap_source": ctx.providers.Esri.WorldShadedRelief,  # Default basemap
        "basemap_alpha": 0.6,  # Transparency of background map
        "basemap_zoom": "auto"  # Auto-determine zoom level
    }

    crs = ccrs.Mercator(n.buses.query("carrier == 'AC'").x.mean()) #ccrs.Mercator()#ccrs.Mercator(n.buses.query("carrier == 'AC'").x.mean())

    with_legend=True



    preferred_order = pd.Index(['H2 Electrolysis'])


    h2_storage = n.stores.query("carrier == 'H2 Store Tank'")
    regions["H2_storage"] = (
        h2_storage.rename(index=h2_storage.bus.map(n.buses.location)) # removes " H2 Store" from index
        .e_nom_opt.groupby(level=0)
        .sum()
        .div(1e3)
    ) # GWh
    regions["H2_storage"] = regions["H2_storage"].where(regions["H2_storage"] > 0.1)
    regions.fillna(0, inplace=True)

    MIN_CONSUMPTION_THRESHOLD = 1e-3
    TWH_CONVERSION_FACTOR = 1e6
    
    h2_energy_balance = -(
        n.statistics
        .energy_balance(
            groupby=["carrier", "bus"],
            bus_carrier="H2",
            drop_zero=True
        )
        .drop("H2 pipeline", level="carrier")
    )
    
    h2_consumption = (
        h2_energy_balance[h2_energy_balance > MIN_CONSUMPTION_THRESHOLD]
        .groupby(["bus"])
        .sum()
        .div(TWH_CONVERSION_FACTOR)
        .round(2)
    )
    
    h2_consumption.index = h2_consumption.index.str.removesuffix(" H2")
    regions["H2_consumption"] = h2_consumption

    eb = (
        n.statistics
        .energy_balance(
            bus_carrier="H2", 
            groupby=["bus","carrier"],
            aggregate_across_components=True,
            drop_zero=True)
        .drop("H2 pipeline", level="carrier")
        .div(1e6)
    )
    eb = eb[abs(eb)>EB_THRESHOLD].reset_index()
    eb = eb[~eb["carrier"].str.contains("shedding")]
    eb["carrier"] = eb["carrier"].map(rename_h2).map(rename_to_upper_case)
    eb = eb.groupby(["bus","carrier"]).sum().round(2).squeeze().unstack()

    eb.rename(index=lambda x: x.replace(" H2", ""), level=0, inplace=True)
    eb = eb.reindex(n.buses.query("carrier == 'AC'").index, level=0, fill_value=0)
    eb = eb.stack()

    carriers_filtered = eb.groupby(level=1).sum()
    # carriers_filtered = carriers_filtered.where(carriers_filtered > threshold).dropna()
    carriers_filtered = preferred_order.intersection(carriers_filtered.index).append(
            carriers_filtered.index.difference(preferred_order)
        )
    carriers_list = list(carriers_filtered)

    # Check for missing tech_colors
    for item in carriers_list:
        if item not in tech_colors:
            logger.warning(f"{item} not in tech_colors dict!")

    # Data validation - drop non-buses
    to_drop = eb.index.levels[0].symmetric_difference(n.buses.query("carrier == 'AC'").index)
    if len(to_drop) != 0:
        logger.info(f"Dropping non-buses {to_drop.tolist()}")
        eb.drop(to_drop, level=0, inplace=True, axis=0, errors="ignore")

    # make sure they are removed from index
    eb.index = pd.MultiIndex.from_tuples(eb.index.values)

    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    n.remove("Link", n.links[~n.links.carrier.str.contains("H2 pipeline")].index)

    foresight = "overnight"



    carriers = ["H2 Electrolysis", "Methanol steam reforming", "H2 Fuel Cell"] #, "H2 Fuel Cell"


    n.links.bus0 = n.links.bus0.str.replace(" H2", "")
    n.links.bus1 = n.links.bus1.str.replace(" H2", "")

    branch_flow = n.links_t.p0.sum(axis=0)
    branch_flow[abs(branch_flow) < branch_lower_threshold] = 0.0
    branch_width_factor = branch_flow.max() / 4.5 if branch_width_factor == "auto" else branch_width_factor

    # Transform regions to the projection CRS for plotting
    if hasattr(crs, 'proj4_init'):
        regions = regions.to_crs(crs.proj4_init)
    else:
        # For newer versions, use the CRS string representation
        regions = regions.to_crs(crs.to_string())

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": crs})

    # color_h2_pipe = "#7c154d"
    # color_retrofit = "#bb0056"
    color_h2_pipe = "#39c1cd"
    color_retrofit = "#bb0056"

    bus_size_factor = eb.max()/0.8 if bus_size_factor == "auto" else bus_size_factor

    with plt.rc_context({"patch.linewidth": 0.3}):
        n.plot(
            geomap=True,
            bus_alpha=1.0,
            bus_sizes=eb/bus_size_factor,
            bus_colors=tech_colors,
            bus_split_circle=True,
            link_colors=color_h2_pipe,
            link_widths=branch_flow / branch_width_factor,
            link_flow=branch_flow / branch_flow_factor,
            branch_components=["Link"],
            ax=ax,
            geomap_resolution="10m",
            **map_opts,
        )

        # n.plot(
        #     geomap=True,
        #     bus_alpha=1.0,
        #     bus_sizes=0,
        #     link_colors=color_retrofit,
        #     link_widths=link_widths_retro,
        #     branch_components=["Link"],
        #     ax=ax,
        #     geomap_resolution="10m",
        #     **map_opts,
        # )

    regions.plot(
        ax=ax,
        column="H2_storage",
        cmap="Blues",
        edgecolor="darkgray", #"darkgray"whitesmoke
        facecolor="white",
        aspect="auto",
        transform=crs,
        linewidth=1,
        vmax=regions["H2_storage"].abs().max()*1.1 if not (regions["H2_storage"] == 0).all() else 1,
        vmin=0,
        legend=True,
        legend_kwds={
            "label": "Hydrogen storage [GWh]",
            "shrink": 0.35,
            "orientation":"horizontal",
            "pad": -0.03,
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


    legend_kw_common = dict(
        loc="upper left",
        frameon=False,
        facecolor="white",
        edgecolor="lightgray",
        framealpha=1.0,  
        labelspacing=0.9,
        handletextpad=1.0,
        borderpad=1.0,  
        columnspacing=1.0, 
        handlelength=2.0,  
        handleheight=0.8,
        title=""
    )
    legend_kw = dict(
        bbox_to_anchor=bbox_to_anchors[0], #(-0.1, 1.06),
        **legend_kw_common
    )

    eb_semi_sizes = [s / bus_size_factor for s in eb_semi_sizes]
    add_legend_semicircles(
        ax,
        eb_semi_sizes,
        eb_semi_labels,
        srid=n.srid,
        patch_kw={"edgecolor":"#6d6d6d","facecolor":"white","alpha":1.0},
        legend_kw=legend_kw,
    )


    legend_kw = dict(
        bbox_to_anchor=bbox_to_anchors[1],
        **legend_kw_common,
    )

    flow_sizes = [s *1e6 / branch_width_factor for s in flow_sizes]
    add_legend_lines(
        ax,
        flow_sizes,
        flow_labels,
        patch_kw=dict(color=color_h2_pipe),
        legend_kw=legend_kw,
    )

    # Technology legend
    legend_kw = dict(
        bbox_to_anchor=bbox_to_anchors[2],
        # ncol=3,  # Override to use 3 columns for technology legend
        **legend_kw_common
    )

    if with_legend and 'carriers_list' in locals():
        legend_colors = [tech_colors[c] for c in carriers_list if c in tech_colors] # + [color_h2_pipe, color_retrofit]
        labels = [c for c in carriers_list if c in tech_colors] # + ["H2 pipeline (total)", "H2 pipeline (repurposed)"]

        if legend_colors and labels:  # Only add legend if we have valid colors and labels
            add_legend_patches(
                ax,
                legend_colors,
                labels,
                legend_kw=legend_kw,
            )

    ax.set_facecolor("white")

    plt.show()
    fig.savefig(snakemake.output.hydrogen_map, bbox_inches="tight")
    fig.savefig(snakemake.output.hydrogen_map_svg, bbox_inches="tight")
    plt.close(fig)

    eb.to_csv(snakemake.output.hydrogen_map_csv_optimal_capacity)
    branch_flow.to_csv(snakemake.output.hydrogen_map_csv_links)


    # =========================================================================
    # ZUSÄTZLICHE ANALYSE: H2-PRODUKTION, NACHFRAGE UND TRANSPORT
    # =========================================================================
    
    print("\n" + "="*80)
    print("ANALYSE DER WASSERSTOFFPRODUKTION UND -NACHFRAGE")
    print(f"Szenario: {snakemake.wildcards.planning_horizons}")
    print("="*80)
    
    # GADM-Regionsnamen laden für bessere Lesbarkeit
    gadm_mapping = {}
    try:
        # Lade GADM shapes - versuche verschiedene Layer
        gadm_file = "../data/gadm/gadm41_MAR/gadm41_MAR.gpkg"

        gadm_shapes = gpd.read_file(gadm_file, layer="ADM_ADM_1")
        
        logger.info(f"GADM columns: {gadm_shapes.columns.tolist()}")
        
        # Mapping erstellen - verschiedene Formate unterstützen
        for idx, row in gadm_shapes.iterrows():
            # GADM 4.1 Format
            if 'GID_1' in gadm_shapes.columns and 'NAME_1' in gadm_shapes.columns:
                gid = row['GID_1']  # z.B. "MAR.15"
                name = row['NAME_1']
                gadm_mapping[gid.replace("MAR.", "MA.") + "_AC"] = name
        
        logger.info(f"Created GADM mapping with {len(gadm_mapping)} entries")
        if len(gadm_mapping) > 0:
            logger.info(f"Sample mapping: {list(gadm_mapping.items())[:3]}")
        
    except Exception as e:
        logger.warning(f"GADM-Shapes konnten nicht geladen werden: {e}")
        import traceback
        logger.warning(traceback.format_exc())
    
    def get_region_name(bus_id):
        """Konvertiert Bus-ID zu lesbarem Regionsnamen"""
        if bus_id in gadm_mapping:
            return gadm_mapping[bus_id]
        return bus_id
    
    # 1. Zusammenfassung der H2-Produktion (positive Werte)
    print("\n1. WASSERSTOFFPRODUKTION (positive Werte) - TOP 10 REGIONEN")
    print("-"*80)
    production = eb[eb > 0].groupby(level=0).sum().sort_values(ascending=False).head(10)
    for bus, val in production.items():
        region_name = get_region_name(bus)
        print(f"  {region_name:35s} ({bus:15s}) | {val:8.2f} TWh")
    print(f"\n  GESAMT: {eb[eb > 0].sum():.2f} TWh")
    
    # 2. Zusammenfassung der H2-Nachfrage (negative Werte)
    print("\n2. WASSERSTOFFNACHFRAGE (negative Werte) - TOP 10 REGIONEN")
    print("-"*80)
    demand = eb[eb < 0].groupby(level=0).sum().sort_values().head(10)
    for bus, val in demand.items():
        region_name = get_region_name(bus)
        print(f"  {region_name:35s} ({bus:15s}) | {val:8.2f} TWh")
    print(f"\n  GESAMT: {eb[eb < 0].sum():.2f} TWh")
    
    # 3. Dominante Syntheseprozesse für H2-Nachfrage
    print("\n3. SYNTHESEPROZESSE FÜR H2-NACHFRAGE (GESAMT)")
    print("-"*80)
    h2_consumers = eb[eb < 0].groupby(level=1).sum().sort_values()
    for carrier, val in h2_consumers.items():
        print(f"  {carrier:40s} | {val:8.2f} TWh")
    
    # 3b. H2-Nachfrage je Syntheseprozess für Top 2 Regionen
    print("\n3b. H2-NACHFRAGE JE SYNTHESEPROZESS FÜR TOP 2 NACHFRAGE-REGIONEN")
    print("-"*80)
    
    # Finde Top 2 Nachfrage-Regionen
    top_demand_regions = eb[eb < 0].groupby(level=0).sum().sort_values().head(2)
    
    for region_bus in top_demand_regions.index:
        region_name = get_region_name(region_bus)
        region_total = top_demand_regions.loc[region_bus]
        
        print(f"\nRegion: {region_name} ({region_bus})")
        print(f"  Gesamt H2-Nachfrage: {region_total:.2f} TWh")
        print(f"  AufsMARüsselung nach Syntheseprozessen:")
        
        # Hole alle Einträge für diese Region
        region_data = eb.loc[region_bus]
        region_demand = region_data[region_data < 0].sort_values()
        
        for carrier, val in region_demand.items():
            percentage = (val / region_total * 100) if region_total != 0 else 0
            print(f"    - {carrier:35s} | {val:8.2f} TWh ({percentage:5.1f}%)")
    
    # 4. H2-Transportanforderungen
    print("\n4. H2-TRANSPORTANFORDERUNGEN (H2 PIPELINE)")
    print("-"*80)
    
    # Branch flow ist bereits in MWh, umrechnen in TWh
    branch_flow_twh = branch_flow.abs() / 1e6
    
    # Top 10 Verbindungen nach Transportmenge
    print("\nTop 10 H2-Pipeline-Verbindungen nach Transportmenge:")
    top_flows = branch_flow_twh.sort_values(ascending=False).head(10)
    for link, flow in top_flows.items():
        bus0 = n.links.loc[link, 'bus0']
        bus1 = n.links.loc[link, 'bus1']
        region0 = get_region_name(bus0)
        region1 = get_region_name(bus1)
        print(f"  {region0:25s} -> {region1:25s} | {flow:8.2f} TWh")
    
    # Gesamttransport
    total_transport = branch_flow_twh[branch_flow_twh > 0].sum()
    print(f"\n  GESAMT H2-TRANSPORT: {total_transport:.2f} TWh")
    
    # Einordnung
    print("\n5. EINORDNUNG")
    print("-"*80)
    print(f"  Berechneter H2-Transport im Szenario: {total_transport:.2f} TWh/Jahr")
    
    # Zusätzliche Statistiken zu installierten Kapazitäten
    print("\n6. INSTALLIERTE H2-PIPELINE-KAPAZITÄTEN")
    print("-"*80)
    h2_pipes = n.links[n.links.carrier.str.contains("H2 pipeline")]
    total_capacity_gw = h2_pipes.p_nom_opt.sum() / 1e3
    print(f"  Gesamte H2-Pipeline-Kapazität: {total_capacity_gw:.2f} GW")
    print(f"  Anzahl H2-Pipeline-Verbindungen: {len(h2_pipes)}")
    
    if len(h2_pipes) > 0:
        print(f"  Durchschnittliche Kapazität pro Verbindung: {h2_pipes.p_nom_opt.mean()/1e3:.2f} GW")
        print(f"  Maximale Einzelverbindung: {h2_pipes.p_nom_opt.max()/1e3:.2f} GW")
    
    print("\n" + "="*80)


# %%
