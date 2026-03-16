from pathlib import Path
import yaml
from pypsa.definitions.structures import Dict

def mock_snakemake(rulename, **wildcards):
    """
    This function is expected to be executed from the "scripts"-directory of "
    the snakemake project. It returns a snakemake.script.Snakemake object,
    based on the Snakefile.

    If a rule has wildcards, you have to specify them in **wildcards.

    Parameters
    ----------
    rulename: str
        name of the rule for which the snakemake object should be generated
    **wildcards:
        keyword arguments fixing the wildcards. Only necessary if wildcards are
        needed.
    """
    import os

    import snakemake as sm
    #from pypsa.descriptors import Dict
    from pypsa.definitions.structures import Dict
    from snakemake.script import Snakemake

    script_dir = Path(__file__).parent.resolve()
    assert (
        Path.cwd().resolve() == script_dir
    ), f"mock_snakemake has to be run from the repository scripts directory {script_dir}"
    os.chdir(script_dir.parent)
    for p in sm.SNAKEFILE_CHOICES:
        if os.path.exists(p):
            snakefile = p
            break
    workflow = sm.Workflow(snakefile, overwrite_configfiles=[], rerun_triggers=[])
    workflow.include(snakefile)
    workflow.global_resources = {}
    try:
        rule = workflow.get_rule(rulename)
    except Exception as exception:
        print(
            exception,
            f"The {rulename} might be a conditional rule in the Snakefile.\n"
            f"Did you enable {rulename} in the config?",
        )
        raise
    dag = sm.dag.DAG(workflow, rules=[rule])
    wc = Dict(wildcards)
    job = sm.jobs.Job(rule, dag, wc)

    def make_accessable(*ios):
        for io in ios:
            for i in range(len(io)):
                io[i] = os.path.abspath(io[i])

    make_accessable(job.input, job.output, job.log)
    snakemake = Snakemake(
        job.input,
        job.output,
        job.params,
        job.wildcards,
        job.threads,
        job.resources,
        job.log,
        job.dag.workflow.config,
        job.rule.name,
        None,
    )
    # create log and output dir if not existent
    for path in list(snakemake.log) + list(snakemake.output):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    os.chdir(script_dir)
    return snakemake

#%%
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
        "AC": "Residential",
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
    elif tech in ["OCGT", "CCGT","OCGT (Diesel)","sasol_gas","ocgt_diesel", "CHP"]:
        return "Gas turbines"
    elif tech == 'Combined-Cycle Gas':
        return "Gas turbines" #"CHP"
    elif "V2G" in tech:
        return "Vehicle-to-Grid"
    elif tech == "battery":
        return "Battery"
    # elif tech == "battery charger":
    #     return "Battery charger"
    elif "Hydro" in tech or "hydro" in tech:
        return "Hydro"
    elif "Haber-Bosch" in tech:
       return "NH3 Synthesis"
    elif tech in ["Fischer-Tropsch", "CH3OH Synthesis"]:
        return "FT Synthesis"
    elif "offshore wind" in tech:
        return "offshore wind"
    elif "SMR" in tech:
        return tech.replace("SMR", "steam reforming")
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
        return "CH4 Synthesis"
    elif tech == "H2 Fuel Cell":
        return "H2 Fuel Cell"
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
        return "NH3 Synthesis"
    elif tech in ["NH3 export"]:
        return "Ammonia export"
    elif tech in ["CH4 Synthesis"]:
        return "CH4 Synthesis"
    elif tech in ["FT Synthesis"]:
        return "FT Synthesis"
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
        return "Navigation"
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
        return "FT Synthesis"
    elif tech in ["Fischer-Tropsch export"]:
        return "FT Synthesis export"
    else:
        return tech
    
def rename_gas(tech):
    tech = rename_techs_study(tech)
    if tech in ["gas"]:
        return "Gas import"
    elif "methanation" in tech:
        return "CH4 Synthesis"
    elif "residential" in tech:
         return "Residential"
    elif "services" in tech:
        return "Commerce"
    elif tech in ["Sabatier"]:
        return "CH4 Synthesis"
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
        return "FT Synthesis"
    elif "Haber-Bosch" in tech:
       return "NH3 Synthesis"
    elif "methanolisation" in tech:
       return "CH3OH Synthesis"
    elif tech == 'SMR':
        return "Steam reforming"
    elif tech == 'Sabatier':
        return "CH4 Synthesis"
    elif "shipping" in tech:
        return "Navigation"
    elif "industry" in tech:
        return "Other industry"
    elif "land transport" in tech:
        # return "Land transport"   
        return "H2 Fuel Cell"  
    elif "H2 Fuel Cell" in tech:
        return "H2 Fuel Cell"
    elif tech == "H2":
        return "H2 export"
    elif tech == "DRI":
        return "H2 DRI"
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
        return "Residential"
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
        return "FT Synthesis"
    elif tech == 'DAC':
        return "Direct air capture"
    elif tech == 'Sabatier':
        return "CH4 Synthesis"    
    elif tech == 'helmeth':
        return "CH4 Synthesis"
    else:
        return tech

def rename_to_upper_case(tech):
    tech = tech[0].upper() + tech[1:]
    return tech


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
        'NH3 Synthesis': '#fce356',
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
        "CH4 Synthesis": "#d3c7ae",
        'H2 Electrolysis': "#179c7d",
        "H2 Fuel Cell": "#179c7d",
        "H2 turbine": "#179c7d",
        "CH3OH Synthesis": "#bb0056",
        "FT Synthesis": "#7c154d",
        "Oil": "#1c3f52",
        "Geothermal": '#f08591',
    },
    "hydrogen": {
        'H2 Electrolysis': "#179c7d",
        'Other industry': '#a6bbc8',
        'Land transport': '#669db2',
        "H2 Fuel Cell": "#179c7d",
        'NH3 Synthesis': '#fce356',
        'CH4 Synthesis': '#d3c7ae',
        'FT Synthesis': '#a8508c',
        "CH3OH Synthesis": "#836bad",
        "H2 DRI": '#1c3f52',
        "H2 export": "#4CC2A6",
        "Navigation": "#39c1cd",
        "Methanol steam reforming": "#39c1cd",
        'Steam reforming': '#d3c7ae',
    },
    "oil": {
        "Fuel import or CtL": "#1c3f52",
        "Fossil fuel import": "#1c3f52",
        'FT Synthesis': '#a8508c',
        'Land transport': '#669db2',
        "Navigation": "#39c1cd",
        "Aviation": "#fce356",
        "Industry": '#f58220',
        "Commerce": "#b2d235",
        "Residential": "#d3c7ae",
        "Agriculture": '#f08591',
        "Rail transport": '#008598',
        'CH4 Synthesis': '#d3c7ae',
        "FT Synthesis export": "#7c154d",
    },
    "gas": {
        'CH4 Synthesis': '#d3c7ae',
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
        'FT Synthesis': '#bb0056',
        "CH4 Synthesis": '#d3c7ae',
    },
    "costs": {
        "Ammonia export": '#bb0056',
        "PtL export": '#7c154d',
        "Solar": '#fdb913',
        "Wind": '#005b7f',
        "H2 Electrolysis": "#179c7d",
        "H2 export": "#7c154d",
        "NH3 Synthesis": '#fce356',
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