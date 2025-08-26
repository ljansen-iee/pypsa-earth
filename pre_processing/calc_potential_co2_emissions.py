"""
Calculate potential CO2 emissions of gas, oil, coal in energy totals and industry totals.

This script processes energy totals and industry totals data to calculate
CO2 emissions from oil, gas, and coal consumption in both energy and 
industry sectors. The results are aggregated by country and scenario.

Energy sector includes:
- Oil: agriculture, services, residential, transport (road ICE, rail, aviation, navigation)
- Gas: services, residential
- Coal: agriculture, residential (not used in model)

International transport includes:
- International aviation
- International navigation/shipping
- Accounts for biofuel substitution as defined in calc_biomass_potential.py

Industry sector includes:
- Oil, gas, and coal consumption across all industrial sectors
- Aggregated from detailed industry totals by sector

CO2 emission factors (tCO2/MWh):
- Oil: 0.26647
- Gas: 0.20098  
- Coal: 0.33613

Biofuel replacement rates (from calc_biomass_potential.py):
- RF_2030: 2% of international transport oil
- EL_2030: 4% of international transport oil
- RF_2050: 20% of international transport oil
- EL_2050: 30% of international transport oil
"""

import os
from pathlib import Path
import pandas as pd

os.chdir(Path(__file__).resolve().parent.parent)

# Biofuel replacement percentages by scenario (from calc_biomass_potential.py)
BIOFUEL_REPLACEMENT = {
    "RF_2030": 0.02,  # 2%
    "EL_2030": 0.04,  # 4%
    "RF_2050": 0.20,  # 20%
    "EL_2050": 0.30,  # 30%
}

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

# Define energy sector fuel categories
oil_cols = [
    "agriculture oil", 
    "services oil", 
    "residential oil", 
    "total road ice",
    "total rail",
    "total domestic aviation", 
    "total international aviation", 
    "total domestic navigation", 
    "total international navigation",
]

international_transport_cols = [
    "total international aviation",
    "total international navigation",
]

gas_cols = [
    "services gas",
    "residential gas", 
]

# Coal columns not used in the energy model but included for emissions accounting
coal_cols_not_used_in_model = [
    "agriculture coal",
    "residential coal",
]



agg_df = pd.DataFrame({
    'oil_total': totals_df[oil_cols].sum(axis=1),
    'gas_total': totals_df[gas_cols].sum(axis=1),
    'coal_total_not_used': totals_df[coal_cols_not_used_in_model].sum(axis=1),
    'international_transport_total': totals_df[international_transport_cols].sum(axis=1)
}, index=totals_df.index)

# Calculate biofuel potential for international transport
biofuel_potential = pd.Series(index=agg_df.index, dtype=float)
for idx in agg_df.index:
    scenario = idx[0]  # Extract scenario from multi-index
    replacement_pct = BIOFUEL_REPLACEMENT.get(scenario, 0.0)
    biofuel_potential[idx] = agg_df.loc[idx, 'international_transport_total'] * replacement_pct

# Calculate remaining fossil oil after biofuel substitution
agg_df['international_transport_biofuel'] = biofuel_potential
agg_df['international_transport_fossil_oil'] = agg_df['international_transport_total'] - agg_df['international_transport_biofuel']

# Process industry data - group by carrier and sum across all industry sectors
industry_carriers = ['oil', 'gas', 'coal']
industry_aggregated = {}

for carrier in industry_carriers:
    carrier_data = industry_df.xs(carrier, level='carrier')
    industry_aggregated[f'industry_{carrier}_total'] = carrier_data.sum(axis=1)

industry_agg_df = pd.DataFrame(industry_aggregated).div(1e6)

# CO2 emission factors (tCO2/MWh)
co2_factors = {
    'oil': 0.26647,
    'gas': 0.20098,
    'coal': 0.33613
}

agg_df['oil_total_co2'] = agg_df['oil_total'] * co2_factors['oil']
agg_df['gas_total_co2'] = agg_df['gas_total'] * co2_factors['gas']
agg_df['coal_not_used_total_co2'] = agg_df['coal_total_not_used'] * co2_factors['coal']
# Calculate CO2 only from remaining fossil oil (after biofuel substitution)
agg_df['international_transport_co2'] = agg_df['international_transport_fossil_oil'] * co2_factors['oil']
# Calculate CO2 savings from biofuel substitution (biofuels are carbon neutral)
agg_df['international_transport_co2_savings'] = agg_df['international_transport_biofuel'] * co2_factors['oil']

agg_df['industry_oil_total_co2'] = industry_agg_df['industry_oil_total'] * co2_factors['oil']
agg_df['industry_gas_total_co2'] = industry_agg_df['industry_gas_total'] * co2_factors['gas']
agg_df['industry_coal_total_co2'] = industry_agg_df['industry_coal_total'] * co2_factors['coal']

agg_df['total_co2'] = (agg_df['oil_total_co2'] + agg_df['gas_total_co2'] + 
                             agg_df['coal_not_used_total_co2'])

agg_df['total_co2_with_international'] = agg_df['total_co2'] + agg_df['international_transport_co2']

agg_df['industry_total_co2'] = (agg_df['industry_oil_total_co2'] + 
                                      agg_df['industry_gas_total_co2'] + 
                                      agg_df['industry_coal_total_co2'])

agg_df['total_co2_with_industry'] = agg_df['total_co2'] + agg_df['industry_total_co2']

agg_df['total_co2_all_sectors'] = agg_df['total_co2'] + agg_df['industry_total_co2'] + agg_df['international_transport_co2']

for col in industry_agg_df.columns:
    agg_df[col] = industry_agg_df[col]

# Save results to Excel file instead of CSV
output_dir = Path(__file__).resolve().parent  # Save in pre_processing folder
output_dir.mkdir(parents=True, exist_ok=True)

# Save to Excel file
excel_path = output_dir / "co2_emissions_analysis.xlsx"
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    # Main data sheet
    agg_df.to_excel(writer, sheet_name='CO2_Emissions_Data')
    
    # Create summary sheets for each scenario
    for source in agg_df.index.get_level_values('source').unique():
        scenario_data = agg_df.xs(source, level='source')
        scenario_data.to_excel(writer, sheet_name=f'Summary_{source}')

print(f"CO2 emissions analysis saved to {excel_path}")
print("Excel file contains the following sheets:")
print("- CO2_Emissions_Data: Complete dataset with all countries and scenarios")
print("- Summary_[Scenario]: Individual sheets for each scenario (RF_2030, EL_2030, RF_2050, EL_2050)")

print("\nPotential CO2 Emissions Summary (MtCO2):")
print("=" * 80)

countries = ['ZA', 'EG', 'MA', 'CL']

for source in agg_df.index.get_level_values('source').unique():
    print(f"\n{source} Scenario:")
    print("-" * 80)
    
    summary_data = []
    
    for country in countries:
        try:
            country_data = agg_df.xs((source, country))
            
            domestic_energy = country_data['total_co2']
            international_transport = country_data['international_transport_co2']
            industry = country_data['industry_total_co2']
            total = country_data['total_co2_all_sectors']
            
            summary_data.append({
                'Country': country,
                'Domestic Energy': f"{domestic_energy:.1f}",
                'Int. Transport': f"{international_transport:.1f}",
                'Industry': f"{industry:.1f}",
                'Total': f"{total:.1f}",
                'Industry %': f"{(industry/total*100):.1f}%",
                'Int. Transport %': f"{(international_transport/total*100):.1f}%",
                'Energy %': f"{(domestic_energy/total*100):.1f}%"
            })
        except KeyError:
            summary_data.append({
                'Country': country,
                'Domestic Energy': 'N/A',
                'Int. Transport': 'N/A',
                'Industry': 'N/A',
                'Total': 'N/A',
                'Industry %': 'N/A',
                'Int. Transport %': 'N/A',
                'Energy %': 'N/A'
            })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

print("\n" + "="*80)
print("Biofuel Impact on International Transport CO2 Emissions:")
print("="*80)

for source in agg_df.index.get_level_values('source').unique():
    print(f"\n{source} Scenario (Biofuel replacement: {BIOFUEL_REPLACEMENT[source]*100:.0f}%):")
    print("-" * 80)
    
    biofuel_data = []
    
    for country in countries:
        try:
            country_data = agg_df.xs((source, country))
            
            total_transport_energy = country_data['international_transport_total']
            biofuel_energy = country_data['international_transport_biofuel']
            fossil_oil_energy = country_data['international_transport_fossil_oil']
            fossil_co2 = country_data['international_transport_co2']
            co2_savings = country_data['international_transport_co2_savings']
            
            biofuel_data.append({
                'Country': country,
                'Total Energy (TWh)': f"{total_transport_energy:.2f}",
                'Biofuel (TWh)': f"{biofuel_energy:.2f}",
                'Fossil Oil (TWh)': f"{fossil_oil_energy:.2f}",
                'Fossil CO2 (MtCO2)': f"{fossil_co2:.1f}",
                'CO2 Savings (MtCO2)': f"{co2_savings:.1f}",
                'Biofuel %': f"{(biofuel_energy/total_transport_energy*100):.1f}%" if total_transport_energy > 0 else "0.0%"
            })
        except KeyError:
            biofuel_data.append({
                'Country': country,
                'Total Energy (TWh)': 'N/A',
                'Biofuel (TWh)': 'N/A',
                'Fossil Oil (TWh)': 'N/A',
                'Fossil CO2 (MtCO2)': 'N/A',
                'CO2 Savings (MtCO2)': 'N/A',
                'Biofuel %': 'N/A'
            })
    
    biofuel_df = pd.DataFrame(biofuel_data)
    print(biofuel_df.to_string(index=False))
    
    # # Print totals across all countries
    # print("\nTotals:")
    # scenario_data = agg_df.xs(source, level='source')
    # total_domestic = scenario_data['total_co2'].sum()
    # total_international = scenario_data['international_transport_co2'].sum()
    # total_industry = scenario_data['industry_total_co2'].sum()
    # grand_total = scenario_data['total_co2_all_sectors'].sum()
    
    # print(f"  Total Domestic Energy CO2: {total_domestic:.1f} MtCO2")
    # print(f"  Total International Transport CO2: {total_international:.1f} MtCO2") 
    # print(f"  Total Industry CO2: {total_industry:.1f} MtCO2")
    # print(f"  Grand Total CO2: {grand_total:.1f} MtCO2")
    # print(f"  Shares - Industry: {(total_industry/grand_total*100):.1f}%, Int. Transport: {(total_international/grand_total*100):.1f}%, Energy: {(total_domestic/grand_total*100):.1f}%")

print("\n" + "="*80)
print("International Transport Energy Consumption Summary (TWh):")
print("="*80)

# Create a comprehensive table for international transport energy consumption
transport_summary = []

for source in agg_df.index.get_level_values('source').unique():
    for country in countries:
        try:
            country_data = agg_df.xs((source, country))
            
            # Get breakdown by aviation and navigation (already in TWh)
            country_totals = totals_df.xs((source, country))
            aviation_twh = country_totals['total international aviation']
            navigation_twh = country_totals['total international navigation']
            transport_energy_twh = aviation_twh + navigation_twh
            
            transport_summary.append({
                'Scenario': source,
                'Country': country,
                'Aviation (TWh)': f"{aviation_twh:.2f}",
                'Navigation (TWh)': f"{navigation_twh:.2f}",
                'Total Int. Transport (TWh)': f"{transport_energy_twh:.2f}",
                'Biofuel (TWh)': f"{country_data['international_transport_biofuel']:.2f}",
                'Fossil Oil (TWh)': f"{country_data['international_transport_fossil_oil']:.2f}",
                'CO2 (MtCO2)': f"{country_data['international_transport_co2']:.1f}",
                'CO2 Savings (MtCO2)': f"{country_data['international_transport_co2_savings']:.1f}"
            })
        except KeyError:
            transport_summary.append({
                'Scenario': source,
                'Country': country,
                'Aviation (TWh)': 'N/A',
                'Navigation (TWh)': 'N/A',
                'Total Int. Transport (TWh)': 'N/A',
                'Biofuel (TWh)': 'N/A',
                'Fossil Oil (TWh)': 'N/A',
                'CO2 (MtCO2)': 'N/A',
                'CO2 Savings (MtCO2)': 'N/A'
            })

transport_df = pd.DataFrame(transport_summary)
print(transport_df.to_string(index=False))

# print("\nInternational Transport Totals by Scenario:")
# print("-" * 50)
# for source in agg_df.index.get_level_values('source').unique():
#     scenario_data = agg_df.xs(source, level='source')
#     total_transport_co2 = scenario_data['international_transport_co2'].sum()
    
#     # Get breakdown totals (already in TWh)
#     scenario_totals = totals_df.xs(source, level='source')
#     total_aviation = scenario_totals['total international aviation'].sum()
#     total_navigation = scenario_totals['total international navigation'].sum()
#     total_transport_energy = total_aviation + total_navigation
    
#     print(f"{source}: {total_transport_energy:.2f} TWh total ({total_aviation:.2f} TWh aviation, {total_navigation:.2f} TWh navigation) = {total_transport_co2:.1f} MtCO2")


