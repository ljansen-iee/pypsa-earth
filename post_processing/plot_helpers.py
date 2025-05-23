import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
from pathlib import Path
import os


NA_VALUES = ["NULL", "", "N/A", "NAN", "NaN", "nan", "Nan", "n/a", "null"]

##### Data collection #####

def chdir_to_parent_dir():
    """
    Change the current working directory to the parent folder of the script folder.
    """
    os.chdir(Path(__file__).resolve().parent.parent)


def collect_files_from_directories(all_postnetworks_dir):
    """
    Collects all existing files from the directories specified in all_postnetworks_dir.
    """
    files_in_folder = {}
    for run_name, path in all_postnetworks_dir.items():
        if path.exists() and path.is_dir():
            files_in_folder[f"{run_name}"] = list(path.glob('*'))  # Collect all files in the directory
        else:
            files_in_folder[f"{run_name}"] = []  # If the path doesn't exist or isn't a directory

    # Print the collected files for each country
    for key, files in files_in_folder.items():
        print(f"Files in {key}:")
        for file in files:
            print(f"  {file}")

    return files_in_folder

def init_stats_dict(network_files, keys, name):
    
    stats_dict = {
        key: pd.concat([pd.DataFrame(index=network_files.index)]) #, keys=[key], names=[name]
        for key in keys}

    return stats_dict

def read_csv_nafix(file, **kwargs):
    "Function to open a csv as pandas file and standardize the na value"
    if "keep_default_na" not in kwargs:
        kwargs["keep_default_na"] = False
    if "na_values" not in kwargs:
        kwargs["na_values"] = NA_VALUES

    if os.stat(file).st_size > 0:
        return pd.read_csv(file, **kwargs)
    else:
        return pd.DataFrame()


def to_csv_nafix(df, path, **kwargs):
    if "na_rep" in kwargs:
        del kwargs["na_rep"]
    # if len(df) > 0:
    if not df.empty or not df.columns.empty:
        return df.to_csv(path, **kwargs, na_rep=NA_VALUES[0])
    else:
        with open(path, "w") as fp:
            pass

def save_stats_dict(stats_dict, stats_name, summary_dir):
    for key, df in stats_dict.items():
        to_csv_nafix(df, summary_dir / f"{stats_name}_{key}.csv")
        print(f"Saved {key} to {summary_dir / f'{stats_name}_{key}.csv'}")

def read_stats_dict(stats_name, summary_dir, keys=[]):
    stats_dict = {}
    index_cols = ["run_name_prefix", "country", "year", "simpl", "clusters", "ll", "opts", "sopts", "discountrate", "demand", "h2export"]
    for key in keys:
        stats_dict[key] = read_csv_nafix(summary_dir / f"{stats_name}_{key}.csv", index_col=index_cols)
        stats_dict[key].index.set_names(index_cols, inplace=True)
        print(f"Imported {key} from {summary_dir / f'{stats_name}_{key}.csv'}")
    return stats_dict


def consistency_check(df):

    df = df.copy()
    # Check if variable values are unique for each run_name_prefix, scen, and year
    duplicates = df.duplicated(subset=["run_name_prefix", "scen", "year", "country", "variable"], keep=False)

    if duplicates.any():
        raise ValueError("Duplicate variable values found for the same run_name_prefix, scen, year, country, variable.")


def drop_index_levels(df, to_drop=[]):
    """
    Drop index levels from the dataframe.
    """
    df = df.copy()

    for level in to_drop:
        df = df.droplevel(level)
        # if level in df.index.names and df.index.get_level_values(level).nunique() == 1:
        #     print(f"Dropping index level {level} with only one unique value: ",
        #           f"{df.index.get_level_values(level).unique()[0]}")
        if level in df.index.names and df.index.get_level_values(level).nunique() > 1:
            print(f"Index level {level} has multiple unique values and should maybe be maintained: ",
                  f"{df.index.get_level_values(level).unique()}")

    return df

def set_scen_col_for_h2g_a(df):
    """
    Set combined scenario str (for plots).
    NB: This is specific to the experiment and should be adapted for other experiments.
    """
    df = df.copy()
    
    df["scen"] = df["h2export"].div(33.3333).round(1).astype(str) + "MtH2export"
    df = df.drop(columns=["h2export"])
    #df["scen"] = df.apply(lambda row: "_".join([f"{col}_{row[col]}" for col in df.columns if col not in cols_to_ignore]), axis=1)
    #df["scen"] = df.apply(lambda row: "_".join([f"{row[col]}" for col in df.columns if col not in cols_to_ignore]), axis=1)

    return df

def prepare_dataframe(stats_df, idx_group, round=1, drop_zero=True):
    df = stats_df.copy().loc[idx_group].reset_index()
    df = set_scen_col_for_h2g_a(df)
    df = df.melt(id_vars=["run_name_prefix", "scen", "year", "country"]).groupby(
        ["run_name_prefix", "scen", "year", "country", "variable"], as_index=False
    ).sum().round(round)

    if drop_zero:
        df = df[df["value"] != 0]

    return df

def update_layout(fig):
    fig.update_traces(textposition='inside', textangle=0)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return



def get_supply_demand_from_balance(stats_df, threshold=0.01, round=1):
    """
    Get supply and demand from balance.
    """
    supply_df = stats_df[stats_df["value"]>=threshold].copy()
    supply_df = supply_df.groupby(["run_name_prefix","scen","year","country", "variable"], as_index=False).sum().round(round)
    supply_sum_df = supply_df.groupby(["scen","year","country"]).sum(numeric_only=True).round(round)

    demand_df = stats_df[stats_df["value"]<=-threshold].copy()
    demand_df["value"] *= -1 
    demand_df = demand_df.groupby(["run_name_prefix","scen","year","country", "variable"], as_index=False).sum().round(round)
    demand_sum_df = demand_df.groupby(["scen","year","country"]).sum(numeric_only=True).round(round)

    return supply_df, supply_sum_df, demand_df, demand_sum_df


##### Renaming and consolidation #####


def rename_techs(label):
    prefix_to_remove = [
        # "residential ",
        # "services ",
        "urban ",
        "rural ",
        "central ",
        "decentral ",
    ]

    rename_if_contains = [
        "CHP",
        "gas boiler",
        "biogas",
        "solar thermal",
        "air heat pump",
        "ground heat pump",
        "resistive heater",
        "Fischer-Tropsch",
    ]

    rename_if_contains_dict = {
        "water tanks": "hot water storage",
        "retrofitting": "building retrofitting",
        # "H2 Electrolysis": "hydrogen storage",
        # "H2 Fuel Cell": "hydrogen storage",
        # "H2 pipeline": "hydrogen storage",
        "battery": "battery",
        "H2 for industry": "H2 for industry",
        "land transport fuel cell": "land transport fuel cell",
        "land transport oil": "land transport oil",
        "oil shipping": "shipping oil",
        "Hydro": "Hydro",
        # "CC": "CC"
    }

    rename = {
        "solar": "solar PV",
        "solar rooftop": "solar PV",
        # "Sabatier": "methanation",
        "offwind": "offshore wind",
        "offwind-ac": "offshore wind (AC)",
        "offwind-dc": "offshore wind (DC)",
        "onwind": "wind onshore",
        "ror": "hydroelectricity",
        "hydro": "hydroelectricity",
        "PHS": "Pumped hydro storage",
        "NH3": "ammonia",
        "co2 Store": "DAC",
        "co2 stored": "CO2 sequestration",
        "AC": "transmission lines",
        "DC": "transmission lines",
        "B2B": "transmission lines",
    }

    for ptr in prefix_to_remove:
        if label[: len(ptr)] == ptr:
            label = label[len(ptr) :]

    for rif in rename_if_contains:
        if rif in label:
            label = rif

    for old, new in rename_if_contains_dict.items():
        if old in label:
            label = new

    for old, new in rename.items():
        if old == label:
            label = new
    return label


def rename_techs_study(tech):
    tech = rename_techs(tech)
    if "heat pump" in tech or "resistive heater" in tech:
        return "power-to-heat"
    # elif tech in ["H2 Electrolysis"]:  # , "H2 liquefaction"]:
    #     return "power-to-hydrogen"
    elif "H2 pipeline" in tech:
        return "H2 pipeline"
    # elif tech == "H2":
    #     return "H2 storage"
    elif tech in ["OCGT", "CCGT","OCGT (Diesel)","sasol_gas","ocgt_diesel"]:
        return "Gas turbines"
    elif tech == 'Combined-Cycle Gas':
        return "CHP"
    elif "V2G" in tech:
        return "Vehicle-to-Grid"
    elif tech == "battery":
        return "Battery"
    # elif tech == "battery charger":
    #     return "Battery charger"
    elif "Hydro" in tech or "hydro" in tech:
        return "Hydro"
    elif "Haber-Bosch" in tech:
       return "ammonia synthesis"
    elif tech in ["Fischer-Tropsch", "methanolisation"]:
        return "liquid fuel synthesis"
    elif "offshore wind" in tech:
        return "offshore wind"
    elif "SMR" in tech:
        return tech.replace("SMR", "steam methane reforming")
    elif "DAC" in tech:
        return "direct air capture"
    elif "CC" in tech or "sequestration" in tech:
        return "carbon capture"
    # elif tech == "oil" or tech == "gas":
    #     return "fossil oil and gas"
    elif tech in ["solar PV", "solar_pv"]:
        return "Solar PV"
    elif tech in ["solar CSP", "solar_csp"]:
        return "Solar CSP"
    elif 'biomass' in tech:
        return "Biomass"
    elif "nuclear" in tech:
        return "Nuclear"
    # elif tech in ["wind"]:
    #     return "Wind"
    elif tech in ["coal","Sasol_coal"]:
        return "Coal"
    elif tech == "Load_Shedding":
        return "Backup/load-shedding"
    elif tech == "helmeth":
        return "Power-to-gas"
    elif tech == "H2 Fuel Cell":
        return "H2-to-power"
    else:
        return tech
    
def rename_costs(tech):
    """
    """
    tech = rename_techs_study(tech)
    if tech in ["Solar PV", "Solar CSP", "solar rooftop"]:
        return "Solar"
    elif tech in ["Onshore Wind", "Offshore Wind (DC)", "Wind onshore", "wind onshore"]:
        return "Wind"
    elif tech in ["H2 Electrolysis"]:
        return "H2 Electrolysis"
    elif tech in ["Hydro Power"]:
        return "Hydro"    
    elif tech in ["Battery"]:
        return "Battery"
    elif tech in ['transmission lines']:
        return "Transmission line"
    # elif tech in ["H2 pipeline"]:
    #     return "H2 pipeline"
    # TODO maybe group "opex" expenditures?
    elif tech in ["Coal"]:
        return "Coal"
    elif tech in ["oil"]:
        return "Oil"
    elif tech in ["gas"]:
        return "Gas"
    elif tech in ["Nuclear"]:
        return "Nuclear"
    # elif tech in ["Gas turbines"]:
    #     return "Gas turbines"    
    elif tech in ["Fischer-Tropsch export"]:
        return "PtL export"    
    elif tech in ["Haber-Bosch"]:
        return "Ammonia synthesis"
    elif tech in ["NH3 export"]:
        return "Ammonia export"
    elif tech in ["Power-to-gas"]:
        return "Power-to-gas"
    elif tech in ["liquid fuel synthesis"]:
        return "Power-to-liquid"
    elif tech in ["power-to-heat"]:
        return "Power-to-heat"
    elif tech in ["H2 export", "H2"]:
        return "H2 export"
    elif tech in ["H2 Store Tank"]:
        return "H2 storage"
    elif tech in ["process emissions CC", "carbon capture", "direct air capture"]:
        return "Carbon capture"
    # elif tech in ["direct air capture"]:
    #     return "DAC"
    elif tech in ["electricity distribution grid"]:
        return "Distribution grid"
    else:
        return tech
    
def rename_oil(tech):
    if "rail transport" in tech:
        return "Rail transport"
    elif "shipping" in tech:
        return "Maritime"
    elif "industry" in tech:
        return "Industry"
    elif "land transport" in tech:
        return "Land transport"
    elif "aviation" in tech:
        return "Aviation"
    elif "services" in tech:
        return "Commerce"
    elif "agriculture" in tech:
        return "Agriculture"
    elif "residential" in tech:
        return "Residential"
    elif tech in ["oil"]:
        return "Fossil fuel import" # or CtL
    elif tech in ["Fischer-Tropsch","Fischer-Tropsch -> oil"]:
        return "Power-to-liquid"
    elif tech in ["Fischer-Tropsch export"]:
        return "Power-to-liquid export"
    else:
        return tech
    
def rename_gas(tech):
    tech = rename_techs_study(tech)
    if tech in ["gas"]:
        return "Gas import"
    elif "methanation" in tech:
        return "Power-to-gas"
    elif "residential" in tech:
         return "Residential"
    elif "services" in tech:
        return "Commerce"
    elif tech in ["Sabatier"]:
        return "Power-to-gas"
    elif tech in ["OCGT", "CCGT"]:
        return "Gas turbines"
    elif tech == "gas for industry CC":
        return "Industry with CC"
    elif tech == "gas for industry":
        return "Industry"    
    elif tech == "biogas":
        return "Biogas"
    elif tech == "gas boiler":
        return "Gas boiler"
    else:
        return tech
    
def rename_h2(tech):
        
    if tech == 'Fischer-Tropsch':
        return "Power-to-liquid"
    elif "Haber-Bosch" in tech:
       return "Ammonia synthesis"
    # elif tech == 'H2 export':
    #     return "ammonia synthesis"
    elif tech == 'Sabatier':
        return "Power-to-gas"
    elif "shipping" in tech:
        return "Shipping"
    elif "industry" in tech:
        return "Industry"
    elif "land transport" in tech:
        return "Land transport"    
    elif tech == "H2":
        return "H2 export"
    else:
        return tech

def rename_electricity(tech):
    tech = rename_techs_study(tech)
    
    suffix_to_remove = [
        " electricity",
    ]
    for sfx in suffix_to_remove:
        tech = tech.removesuffix(sfx)
    
    if tech == 'BEV charger':
        return 'Electric vehicles'
    elif tech == 'seawater desalination':
        return "Desalination"
    elif tech == 'services':
        return "Commerce"
    elif tech == 'transmission lines':
        return "Other"
    else:
        return tech

def rename_co2(tech):

    if tech == 'solid biomass for industry CC':
        return 'Biomass for industry with CC'        
    if tech == 'urban central solid biomass CHP CC':
        return 'Biomass for CHP with CC'   
    elif tech == 'gas for industry CC':
        return "Gas for industry with CC"
    elif tech == 'process emissions CC':
        return "Process emissions with CC"
    elif tech == 'Fischer-Tropsch':
        return "Power-to-liquid"
    elif tech == 'DAC':
        return "Direct air capture"
    elif tech == 'Sabatier':
        return "Power-to-gas"    
    elif tech == 'helmeth':
        return "Power-to-gas"
    else:
        return tech

def rename_to_upper_case(tech):
    tech = tech[0].upper() + tech[1:]
    return tech


##### Plotting #####

def get_missing_colors(df, colors_dict):
    """
    Find missing color specifications for variables
    """
    missing_colors = set(df["variable"]) - set(colors_dict.keys())
    if missing_colors:
        print(f"Missing color specifications for variables: {missing_colors}")
    else:
        print("All variables have color specifications.")
    return missing_colors

my_template = go.layout.Template(
    layout=dict(
        font_color='#000000',
        #font_family="Open Sans",
        plot_bgcolor = "#ffffff",  #rgba(212,218,220,255)
        paper_bgcolor = "#ffffff", 
        legend=dict(bgcolor='rgba(0,0,0,0)'), #"#ffffff"
        title={'y': 0.95, 'x': .06},
        font_size=14,
        uniformtext_minsize=11, 
        uniformtext_mode='hide',
        margin=dict(l=15, r=0, t=80, b=0),
))

pio.renderers.default = 'plotly_mimetype+notebook_connected'
pio.templates["my"] = my_template
pio.templates.default = "simple_white+xgridoff+ygridoff+my"
#pio.kaleido.scope.mathjax= None

def nice_title(title, subtitle):
    return f'{title}' + '<br>' +  f'<span style="font-size: 12px;">{subtitle}</span>'

def save_plotly_fig(df, fig, output_dir, fig_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir/f"{fig_name}.csv", index=False)
    fig.write_image(output_dir/f"{fig_name}.svg", engine="kaleido")


colors = {
    "electricity": {
        "Industry": '#f58220',
        "Commerce": "#b2d235",
        "Residential": "#d3c7ae",
        #"Solar": '#fdb913',
        "Solar PV": '#fdb913',
        "Solar CSP": '#face61',
        #"Wind": '#005b7f',
        "Wind onshore": '#005b7f',
        "Coal": '#454545',
        "Hydro": '#a6bbc8',
        "Nuclear": "#bb0056",
        "Gas turbines": '#d3c7ae',
        "CHP": "#d6a67c",
        'Ammonia synthesis': '#fce356',
        "Agriculture": '#f08591',
        "Rail transport": '#008598',
        "Desalination": "#39c1cd",
        "Direct air capture": "#1c3f52",
        "Electric vehicles": '#a8508c',
        "Battery": '#836bad',
        "Battery charger": '#836bad',
        "Battery discharger": '#836bad',
        "Vehicle-to-Grid": '#a8508c',
        "Biomass": "#b2d235",
        "Other": "#a6bbc8",
        "Power-to-heat": "#FCD80E",
        "Power-to-gas": "#d3c7ae",
        'H2 Electrolysis': "#179c7d",
        "H2-to-power": "#179c7d",
        "Liquid fuel synthesis": "#bb0056",
        "Power-to-liquid": "#bb0056",
        "Oil": "#1c3f52",
    },
    "hydrogen": {
        'H2 Electrolysis': "#179c7d",
        'Industry': '#a6bbc8',
        'Land transport': '#669db2',
        'Ammonia synthesis': '#fce356',
        'Power-to-gas': '#d3c7ae',
        'Power-to-liquid': '#bb0056',
        "H2 export": "#4CC2A6",
    },
    "oil": {
        "Fuel import or CtL": "#1c3f52",
        "Fossil fuel import": "#1c3f52",
        'Power-to-liquid': '#bb0056',
        'Land transport': '#669db2',
        "Maritime": "#39c1cd",
        "Aviation": "#fce356",
        "Industry": '#f58220',
        "Commerce": "#b2d235",
        "Residential": "#d3c7ae",
        "Agriculture": '#f08591',
        "Rail transport": '#008598',
        'Power-to-gas': '#d3c7ae',
        "Power-to-liquid export": "#7c154d",
    },
    "gas": {
        'Power-to-gas': '#d3c7ae',
        "Commerce": "#b2d235",
        'Gas import': "#fce356",
        'Gas turbines': '#d3c7ae',
        'Residential': "#f58220",
        'Industry': '#a6bbc8',
        'Industry with CC': "#005b7f",
        "Biogas": "#b2d235",
        "Gas boiler": "#d6a67c",
    },
    "co2": {
        'Process emissions with CC': '#1c3f52',
        'Biomass for industry with CC': '#b2d235',
        "Biomass for CHP with CC": "#C2D05C",
        'Gas for industry with CC': '#005b7f',
        'Direct air capture': '#7c154d',
        'Power-to-liquid': '#bb0056',
        "Power-to-gas": '#d3c7ae',
    },
    "costs": {
        "Ammonia export": '#bb0056',
        "PtL export": '#7c154d',
        "Solar": '#fdb913',
        "Wind": '#005b7f',
        "H2 Electrolysis": "#179c7d",
        "H2 export": "#7c154d",
        "Ammonia synthesis": '#fce356',
        "Transmission line": '#4cc2a6',
        "Battery": '#836bad',
        "Oil": '#C0C0C0',
        "Coal": '#454545',
        "Gas": "#d3c7ae",
        "Hydro": '#a6bbc8',
        "Nuclear": "#bb0056",
        'Carbon capture': '#7c154d',
        "Other": "#a6bbc8",
    }
}