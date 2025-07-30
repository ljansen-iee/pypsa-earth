import pandas as pd



demands = ["RF", "EL"]

# runs = ["DKS_CL_2030","DKS_EG_2030","DKS_MA_2030","DKS_ZA_2030"]
# yr = 2030

runs = ["DKS_CL_2050","DKS_EG_2050","DKS_MA_2050","DKS_ZA_2050"]
yr = 2050

merged_all_energy_totals = pd.DataFrame()

for d in demands:
    
    merged_energy_totals = pd.DataFrame()
    
    for run in runs:
        
        df = pd.read_csv(f"../resources/{run}/energy_totals_{d}_{yr}.csv", index_col=0)
        merged_energy_totals = pd.concat([merged_energy_totals, df])

    merged_energy_totals = merged_energy_totals.fillna(0)
    merged_energy_totals.to_csv(f"../data/custom/energy_totals_{d}_{yr}.csv")

    merged_all_energy_totals = pd.concat(
        [merged_all_energy_totals, merged_energy_totals])
    

merged_all_industry_totals = pd.DataFrame()

for d in demands:

    merged_industry_totals = pd.DataFrame()

    for run in runs:

        df = pd.read_csv(f"../resources/{run}/demand/industrial_totals_{yr}_{d}.csv", index_col=0)
        merged_industry_totals = pd.concat([merged_industry_totals, df])

    merged_industry_totals = merged_industry_totals.fillna(0)
    merged_industry_totals.to_csv(f"../data/custom/industry_totals_{yr}_{d}.csv")

    merged_all_industry_totals = pd.concat(
        [merged_all_industry_totals, merged_industry_totals])