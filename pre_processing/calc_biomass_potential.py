"""
Calculate biomass potential per year and per demand scenario [(2030, RF),(2030, EL), (2050, RF), (2050, EL)].
Do this, based on the total demand (load) of biomass per scenario, which is calculated and saved in calc_potential_biomass_use.py.

Then define an offset (additional biomass potential) for replacing fossil oil in maritime / navigation and aviation transport.
This offset can be based on the estimated demand for biofuels in these sectors. Efficiency factor for converting Biomass to Liquid fuel is 0.38333.
We define that 4% of oil demands of navigation and aviation transport oil demand could be replaced with biofuels in 2030; and 20% in (RF, 2050); and 40% in (EL, 2050).

Also calculate how much CO2 emissions could be avoided by this substitution.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# Change to the parent directory to access data files
os.chdir(Path(__file__).resolve().parent.parent)

# CO2 emission factor for oil (tCO2/MWh)
OIL_EMISSION_FACTOR = 0.26647

# BtL conversion efficiency (from costs data)
BTL_EFFICIENCY = 0.38333

# Biofuel replacement percentages by scenario
BIOFUEL_REPLACEMENT = {
    "RF_2030": 0.02,  # 2%
    "EL_2030": 0.04,  # 4%
    "RF_2050": 0.20,  # 20%
    "EL_2050": 0.30,  # 30%
}

def load_biomass_demand():
    """Load existing biomass demand from calc_potential_biomass_use.py output."""
    try:
        # Load the solid biomass loads analysis data from Excel file
        excel_path = Path(__file__).resolve().parent / "solid_biomass_loads_analysis.xlsx"
        biomass_data = pd.read_excel(excel_path, sheet_name='Biomass_Loads_Data', index_col=[0,1])
        
        # Extract total biomass demand (total_all_biomass column) by scenario and country
        biomass_demand = biomass_data['total_all_biomass'].unstack(level=0)
        
        print("Loaded existing biomass demand data (TWh):")
        print(biomass_demand)
        return biomass_demand
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Required file '{excel_path}' not found. "
            "Please run 'calc_potential_biomass_use.py' first to generate biomass demand data."
        )

def load_transport_oil_demand():
    """Load oil demand for aviation and navigation transport from energy totals."""
    totals_files = [
        ("RF_2030", "data/custom/energy_totals_RF_2030.csv"),
        ("EL_2030", "data/custom/energy_totals_EL_2030.csv"),
        ("RF_2050", "data/custom/energy_totals_RF_2050.csv"),
        ("EL_2050", "data/custom/energy_totals_EL_2050.csv"),
    ]
    
    # Columns for aviation and navigation oil demand
    transport_oil_cols = [
        "total domestic aviation",
        "total international aviation", 
        "total domestic navigation",
        "total international navigation",
    ]
    
    oil_demand = {}
    
    for scenario, file_path in totals_files:
        try:
            df = pd.read_csv(file_path, index_col=0)
            # Sum aviation and navigation oil demand
            available_cols = [col for col in transport_oil_cols if col in df.columns]
            if available_cols:
                oil_demand[scenario] = df[available_cols].sum(axis=1)
            else:
                print(f"Warning: No transport oil columns found in {file_path}")
                oil_demand[scenario] = pd.Series(index=df.index, data=0.0)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Required file '{file_path}' not found. "
                "Please ensure all energy totals files are available in data/custom/."
            )
    
    return pd.DataFrame(oil_demand)

def calculate_biofuel_potential(oil_demand):
    """Calculate biofuel potential and required biomass for each scenario."""
    biofuel_potential = pd.DataFrame(index=oil_demand.index, columns=oil_demand.columns)
    biomass_for_biofuel = pd.DataFrame(index=oil_demand.index, columns=oil_demand.columns)
    
    for scenario in oil_demand.columns:
        replacement_pct = BIOFUEL_REPLACEMENT[scenario]
        
        # Calculate biofuel potential (TWh of liquid fuel)
        biofuel_potential[scenario] = oil_demand[scenario] * replacement_pct
        
        # Calculate required biomass input (TWh of biomass)
        # biomass_input = biofuel_output / efficiency
        biomass_for_biofuel[scenario] = biofuel_potential[scenario] / BTL_EFFICIENCY
    
    return biofuel_potential, biomass_for_biofuel

def calculate_co2_savings(biofuel_potential):
    """Calculate CO2 emissions avoided by replacing oil with biofuels."""
    # Assuming biofuels are carbon neutral (biomass CO2 is recaptured when growing)
    # CO2 savings = oil replaced × oil emission factor
    co2_savings = biofuel_potential * OIL_EMISSION_FACTOR
    return co2_savings

def calculate_total_biomass_potential(biomass_demand, biomass_for_biofuel):
    """Calculate total biomass potential including base demand and biofuel offset."""
    total_biomass_potential = biomass_demand + biomass_for_biofuel
    return total_biomass_potential

def main():
    print("=== Calculating Biomass Potential ===")
    
    # 1. Load existing biomass demand
    print("\n1. Loading biomass demand data...")
    biomass_demand = load_biomass_demand()
    
    # 2. Load transport oil demand
    print("\n2. Loading transport oil demand data...")
    oil_demand = load_transport_oil_demand()
    print("Transport oil demand (TWh):")
    print(oil_demand)
    
    # 3. Calculate biofuel potential and biomass requirements
    print("\n3. Calculating biofuel potential...")
    biofuel_potential, biomass_for_biofuel = calculate_biofuel_potential(oil_demand)
    
    print("Biofuel potential (TWh liquid fuel):")
    print(biofuel_potential)
    print("\nBiomass required for biofuel (TWh biomass):")
    print(biomass_for_biofuel)
    
    # 4. Calculate CO2 savings
    print("\n4. Calculating CO2 emissions savings...")
    co2_savings = calculate_co2_savings(biofuel_potential)
    print("CO2 emissions avoided (tCO2):")
    print(co2_savings)
    
    # 5. Calculate total biomass potential
    print("\n5. Calculating total biomass potential...")
    total_biomass_potential = calculate_total_biomass_potential(biomass_demand, biomass_for_biofuel)
    print("Total biomass potential (TWh):")
    print(total_biomass_potential)
    
    # 6. Save results
    print("\n6. Saving results...")
    
    # Create output directory in pre_processing folder
    output_dir = Path(__file__).resolve().parent  # Save in pre_processing folder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create detailed summary report by country and scenario
    summary_data = []
    for scenario in oil_demand.columns:
        for country in oil_demand.index:
            summary_data.append({
                'Country': country,
                'Scenario': scenario,
                'Replacement_Percentage': f"{BIOFUEL_REPLACEMENT[scenario]*100:.0f}%",
                'Oil_Demand_TWh': oil_demand.loc[country, scenario],
                'Biofuel_Potential_TWh': biofuel_potential.loc[country, scenario],
                'Biomass_for_Biofuel_TWh': biomass_for_biofuel.loc[country, scenario],
                'CO2_Savings_tCO2': co2_savings.loc[country, scenario],
                'Base_Biomass_Demand_TWh': biomass_demand.loc[country, scenario],
                'Total_Biomass_Potential_TWh': total_biomass_potential.loc[country, scenario],
            })
    
    summary = pd.DataFrame(summary_data)
    
    # Save all results to a single Excel file with multiple sheets
    excel_path = output_dir / "biomass_potential_analysis.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Info sheet (first sheet)
        info_data = [
            ["Biomass Potential Analysis", ""],
            ["", ""],
            ["Description", "This analysis calculates biomass potential and biofuel production scenarios"],
            ["", "for replacing fossil oil in aviation and maritime transport sectors."],
            ["", ""],
            ["Analysis Date", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["", ""],
            ["Scenarios Analyzed", "RF_2030, EL_2030, RF_2050, EL_2050"],
            ["Countries", "South Africa (ZA), Egypt (EG), Morocco (MA), Chile (CL)"],
            ["", ""],
            ["Biofuel Replacement Rates", ""],
            ["RF_2030", f"{BIOFUEL_REPLACEMENT['RF_2030']*100:.0f}% of transport oil"],
            ["EL_2030", f"{BIOFUEL_REPLACEMENT['EL_2030']*100:.0f}% of transport oil"],
            ["RF_2050", f"{BIOFUEL_REPLACEMENT['RF_2050']*100:.0f}% of transport oil"],
            ["EL_2050", f"{BIOFUEL_REPLACEMENT['EL_2050']*100:.0f}% of transport oil"],
            ["", ""],
            ["Technical Parameters", ""],
            ["BtL Efficiency", f"{BTL_EFFICIENCY:.5f} (biomass to liquid fuel conversion)"],
            ["Oil CO2 Factor", f"{OIL_EMISSION_FACTOR:.5f} tCO2/MWh"],
            ["", ""],
            ["Transport Sectors", "Domestic & International Aviation, Domestic & International Navigation"],
            ["Biomass Assumption", "Carbon neutral (CO2 recaptured during biomass growth)"],
            ["", ""],
            ["Sheet Descriptions", ""],
            ["Summary", "Detailed results by country and scenario"],
            ["Transport_Oil_Demand", "Oil demand for aviation and navigation (TWh)"],
            ["Biofuel_Potential", "Biofuel production potential (TWh liquid fuel)"],
            ["Biomass_for_Biofuel", "Biomass required for biofuel production (TWh biomass)"],
            ["CO2_Savings", "CO2 emissions avoided by biofuel substitution (tCO2)"],
            ["Total_Biomass_Potential", "Total biomass potential including biofuel offset (TWh)"],
            ["Base_Biomass_Demand", "Original biomass demand from energy and industry sectors (TWh)"],
            ["", ""],
            ["Notes", ""],
            ["", "• Total biomass potential = Base biomass demand + Biomass for biofuel"],
            ["", "• CO2 savings assume biofuels are carbon neutral"],
            ["", "• Analysis based on energy totals and industry totals data"],
            ["", "• Results are for PyPSA-Earth energy system modeling"]
        ]
        
        info_df = pd.DataFrame(info_data, columns=['Parameter', 'Value'])
        info_df.to_excel(writer, sheet_name='Info', index=False)
        
        # Summary sheet
        summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # Detailed data sheets
        oil_demand.to_excel(writer, sheet_name='Transport_Oil_Demand')
        biofuel_potential.to_excel(writer, sheet_name='Biofuel_Potential')
        biomass_for_biofuel.to_excel(writer, sheet_name='Biomass_for_Biofuel')
        co2_savings.to_excel(writer, sheet_name='CO2_Savings')
        total_biomass_potential.to_excel(writer, sheet_name='Total_Biomass_Potential')
        biomass_demand.to_excel(writer, sheet_name='Base_Biomass_Demand')
    
    print(f"\nAll results saved to: {excel_path}")
    print("Excel file contains the following sheets:")
    print("- Info: Analysis documentation and parameters")
    print("- Summary: Aggregated results by scenario")
    print("- Transport_Oil_Demand: Oil demand for aviation and navigation (TWh)")
    print("- Biofuel_Potential: Biofuel production potential (TWh)")
    print("- Biomass_for_Biofuel: Biomass required for biofuel production (TWh)")
    print("- CO2_Savings: CO2 emissions avoided by biofuel substitution (tCO2)")
    print("- Total_Biomass_Potential: Total biomass potential including biofuel offset (TWh)")
    print("- Base_Biomass_Demand: Original biomass demand from energy and industry sectors (TWh)")
    
    print("\nSummary:")
    print(summary)
    
    print(f"\n=== Results saved to {excel_path} ===")

if __name__ == "__main__":
    main()

