"""
Calculate exogenous solid biomass loads in energy totals and industry totals.

This script processes energy totals and industry totals data to calculate
solid biomass loads in both energy and industry sectors. The results are 
aggregated by country and scenario. The calculation considers the actually 
used columns and biomass loads in prepare_sector_network.py.

Energy sector biomass includes:
- Agriculture biomass
- Services biomass  
- Residential biomass (including residential heat biomass)
- Road biomass (transport)
- Other biomass
- Non-energy biomass (industrial feedstock)

Industry sector biomass includes:
- Solid biomass consumption across all industrial sectors
- Aggregated from detailed industry totals by sector

The script outputs biomass loads in TWh for use as exogenous demands 
in the PyPSA-Earth energy system model.
"""

import os
from pathlib import Path
import pandas as pd

os.chdir(Path(__file__).resolve().parent.parent)

totals_files = [
    ("RF_2030","data/custom/energy_totals_RF_2030.csv"),
    ("EL_2030","data/custom/energy_totals_EL_2030.csv"),
    ("RF_2050","data/custom/energy_totals_RF_2050.csv"),
    ("EL_2050","data/custom/energy_totals_EL_2050.csv"),
]

industry_files = [
    ("RF_2030","data/custom/industry_totals_2030_RF.csv"),
    ("EL_2030","data/custom/industry_totals_2030_EL.csv"),
    ("RF_2050","data/custom/industry_totals_2050_RF.csv"),
    ("EL_2050","data/custom/industry_totals_2050_EL.csv"),
]

totals_df = pd.concat(
    [pd.read_csv(f, index_col=0) for _, f in totals_files],
    keys=[k for k, _ in totals_files],
    names=['source']
)

industry_df = pd.concat(
    [pd.read_csv(f, index_col=[0,1]) for _, f in industry_files],
    keys=[k for k, _ in industry_files],
    names=['source']
)

# Define energy sector biomass categories based on prepare_sector_network.py usage
energy_biomass_cols = [
    "agriculture biomass",
    "services biomass", 
    "residential biomass",
    "residential heat biomass",
    "road biomass",
    "other biomass"
]

# Non-energy biomass (industrial feedstock, not converted to energy)
non_energy_biomass_cols = [
    "non energy biomass"
]

# All energy-related biomass (excludes non-energy biomass)
all_energy_biomass_cols = energy_biomass_cols.copy()

# Check which columns actually exist in the data
available_cols = totals_df.columns.tolist()
energy_biomass_cols = [col for col in energy_biomass_cols if col in available_cols]
non_energy_biomass_cols = [col for col in non_energy_biomass_cols if col in available_cols]

print(f"Found energy biomass columns: {energy_biomass_cols}")
print(f"Found non-energy biomass columns: {non_energy_biomass_cols}")

# Calculate energy sector biomass totals
biomass_df = pd.DataFrame({
    'energy_biomass_total': totals_df[energy_biomass_cols].sum(axis=1),
    'non_energy_biomass_total': totals_df[non_energy_biomass_cols].sum(axis=1) if non_energy_biomass_cols else 0,
}, index=totals_df.index)

# Process industry biomass data
industry_biomass_data = industry_df.xs('biomass', level='carrier')
biomass_df['industry_biomass_total'] = industry_biomass_data.sum(axis=1).div(1e6)  # Convert MWh to TWh

# Calculate totals
biomass_df['total_energy_biomass'] = biomass_df['energy_biomass_total']
biomass_df['total_all_biomass'] = biomass_df['total_energy_biomass'] + biomass_df['industry_biomass_total'] + biomass_df['non_energy_biomass_total']

# Add individual energy biomass components for detailed analysis
for col in energy_biomass_cols:
    biomass_df[col] = totals_df[col]

# Add individual industry biomass components for detailed analysis  
industry_sectors = industry_biomass_data.columns.tolist()
for sector in industry_sectors:
    biomass_df[f'industry_{sector}_biomass'] = (industry_biomass_data[sector] / 1e6)  # Convert to TWh

# Save results to Excel file instead of CSV
output_dir = Path(__file__).resolve().parent  # Save in pre_processing folder
output_dir.mkdir(parents=True, exist_ok=True)

# Save to Excel file
excel_path = output_dir / "solid_biomass_loads_analysis.xlsx"
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    # Main data sheet
    biomass_df.to_excel(writer, sheet_name='Biomass_Loads_Data')
    
    # Create summary sheets for each scenario
    for source in biomass_df.index.get_level_values('source').unique():
        scenario_data = biomass_df.xs(source, level='source')
        scenario_data.to_excel(writer, sheet_name=f'Summary_{source}')

print(f"Solid biomass loads analysis saved to {excel_path}")
print("Excel file contains the following sheets:")
print("- Biomass_Loads_Data: Complete dataset with all countries and scenarios")
print("- Summary_[Scenario]: Individual sheets for each scenario (RF_2030, EL_2030, RF_2050, EL_2050)")
print("\nThis analysis provides exogenous biomass loads for PyPSA-Earth modeling:")
print("- Energy sector biomass: Residential heating, services, agriculture, transport")
print("- Industry sector biomass: Process heat and feedstock across industrial sectors")
print("- Data can be used to set fixed biomass demands in prepare_sector_network.py")
print("- Units are in TWh/year for direct use in energy system modeling")

print("\nSolid Biomass Loads Summary (TWh):")
print("=" * 80)

countries = ['ZA', 'EG', 'MA', 'CL']

for source in biomass_df.index.get_level_values('source').unique():
    print(f"\n{source} Scenario:")
    print("-" * 80)
    
    summary_data = []
    
    for country in countries:
        try:
            country_data = biomass_df.xs((source, country))
            
            energy_biomass = country_data['energy_biomass_total']
            industry_biomass = country_data['industry_biomass_total']
            non_energy_biomass = country_data['non_energy_biomass_total']
            total_energy = country_data['total_energy_biomass']
            total_all = country_data['total_all_biomass']
            
            summary_data.append({
                'Country': country,
                'Energy Biomass': f"{energy_biomass:.1f}",
                'Industry Biomass': f"{industry_biomass:.1f}",
                'Non-Energy Biomass': f"{non_energy_biomass:.1f}",
                'Total Energy': f"{total_energy:.1f}",
                'Total All': f"{total_all:.1f}",
                'Industry %': f"{(industry_biomass/total_all*100):.1f}%" if total_all > 0 else "0.0%",
                'Energy %': f"{(energy_biomass/total_all*100):.1f}%" if total_all > 0 else "0.0%"
            })
        except KeyError:
            summary_data.append({
                'Country': country,
                'Energy Biomass': 'N/A',
                'Industry Biomass': 'N/A', 
                'Non-Energy Biomass': 'N/A',
                'Total Energy': 'N/A',
                'Total All': 'N/A',
                'Industry %': 'N/A',
                'Energy %': 'N/A'
            })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

print("\n" + "="*80)
print("Energy Sector Biomass Breakdown by Category (TWh):")
print("="*80)

for source in biomass_df.index.get_level_values('source').unique():
    print(f"\n{source} Scenario:")
    print("-" * 60)
    
    breakdown_data = []
    
    for country in countries:
        try:
            country_data = biomass_df.xs((source, country))
            
            breakdown_data.append({
                'Country': country,
                'Agriculture': f"{country_data.get('agriculture biomass', 0):.2f}",
                'Services': f"{country_data.get('services biomass', 0):.2f}",
                'Residential': f"{country_data.get('residential biomass', 0):.2f}",
                'Res. Heat': f"{country_data.get('residential heat biomass', 0):.2f}",
                'Transport': f"{country_data.get('road biomass', 0):.2f}",
                'Other': f"{country_data.get('other biomass', 0):.2f}",
                'Total Energy': f"{country_data['energy_biomass_total']:.2f}"
            })
        except KeyError:
            breakdown_data.append({
                'Country': country,
                'Agriculture': 'N/A',
                'Services': 'N/A',
                'Residential': 'N/A',
                'Res. Heat': 'N/A',
                'Transport': 'N/A',
                'Other': 'N/A',
                'Total Energy': 'N/A'
            })
    
    breakdown_df = pd.DataFrame(breakdown_data)
    print(breakdown_df.to_string(index=False))

print("\n" + "="*80)
print("Industry Sector Biomass by Sector (TWh):")
print("="*80)

# Get industry sector names
industry_sectors = [col.replace('industry_', '').replace('_biomass', '') 
                   for col in biomass_df.columns if col.startswith('industry_') and col.endswith('_biomass')]

for source in biomass_df.index.get_level_values('source').unique():
    print(f"\n{source} Scenario:")
    print("-" * 60)
    
    industry_data = []
    
    for country in countries:
        try:
            country_data = biomass_df.xs((source, country))
            
            row = {'Country': country}
            for sector in industry_sectors:
                col_name = f'industry_{sector}_biomass'
                if col_name in country_data:
                    row[sector.title()] = f"{country_data[col_name]:.2f}"
                else:
                    row[sector.title()] = "0.00"
            
            row['Total Industry'] = f"{country_data['industry_biomass_total']:.2f}"
            industry_data.append(row)
            
        except KeyError:
            row = {'Country': country}
            for sector in industry_sectors:
                row[sector.title()] = 'N/A'
            row['Total Industry'] = 'N/A'
            industry_data.append(row)
    
    industry_df_display = pd.DataFrame(industry_data)
    print(industry_df_display.to_string(index=False))

print("\n" + "="*80)
print("Total Biomass Summary for all countries:")
print("="*80)

for source in biomass_df.index.get_level_values('source').unique():
    print(f"\n{source} Scenario Totals:")
    print("-" * 40)
    
    scenario_data = biomass_df.xs(source, level='source')
    
    total_energy_biomass = scenario_data['energy_biomass_total'].sum()
    total_industry_biomass = scenario_data['industry_biomass_total'].sum()
    total_non_energy_biomass = scenario_data['non_energy_biomass_total'].sum()
    total_all_energy = scenario_data['total_energy_biomass'].sum()
    grand_total = scenario_data['total_all_biomass'].sum()
    
    print(f"  Total Energy Sector Biomass: {total_energy_biomass:.1f} TWh")
    print(f"  Total Industry Sector Biomass: {total_industry_biomass:.1f} TWh")
    print(f"  Total Non-Energy Biomass: {total_non_energy_biomass:.1f} TWh")
    print(f"  Total Energy-Related Biomass: {total_all_energy:.1f} TWh")
    print(f"  Grand Total All Biomass: {grand_total:.1f} TWh")
    
    if total_all_energy > 0:
        print(f"  Shares - Industry: {(total_industry_biomass/total_all_energy*100):.1f}%, Energy: {(total_energy_biomass/total_all_energy*100):.1f}%")

print("\n" + "="*80)
print("Notes for PyPSA-Earth Integration:")
print("="*80)
print("1. These exogenous biomass loads represent FIXED demands in the energy system")
print("2. They should be compared against biomass POTENTIAL (supply) configured in sector_options")
print("3. In prepare_sector_network.py, biomass potential is spatially distributed across nodes")
print("4. Energy sector loads are added as Load components on biomass buses")
print("5. Industry sector loads are added as Load components on 'solid biomass for industry' buses")
print("6. The biomass EOP (electricity-only power) link converts biomass to electricity")
print("7. Consider biomass transport costs if spatial_biomass=True in configuration")
print("\nKey model considerations:")
print("- Ensure biomass potential >= biomass demand to avoid infeasibility")
print("- Biomass competes with other renewable sources for land use") 
print("- CO2 neutrality of biomass can be modeled via CO2 emissions coefficient")
print("- Non-energy biomass (feedstock) may not require energy conversion")


