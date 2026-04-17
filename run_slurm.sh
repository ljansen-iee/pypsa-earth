# Extract country code from config file name (e.g. H2G_A_KE_2035 -> KE)
# CONFIGFILE="configs/GSA/config.GSA_ZA_morris.yaml"
CONFIGFILE="configs/DKS/config.DKS_NA_2035_CNTRL.yaml"

# WSA_MA_2050_low50
# WSA_MA_2050_vestas3

# configs/DKS/config.DKS_NA_2035_h2lim.yaml
# config.DKS_NA_2035_AB.yaml
#  configs/DKS/config.DKS_CL_2030.yaml
#  configs/DKS/config.DKS_EG_2030.yaml
#  configs/DKS/config.DKS_MA_2030.yaml
#  configs/DKS/config.DKS_ZA_2030.yaml
#  configs/DKS/config.DKS_CL_2035.yaml
#  configs/DKS/config.DKS_EG_2035.yaml
#  configs/DKS/config.DKS_MA_2035.yaml
#  configs/DKS/config.DKS_ZA_2035.yaml
#  configs/DKS/config.DKS_CL_2050.yaml
#  configs/DKS/config.DKS_EG_2050.yaml
#  configs/DKS/config.DKS_MA_2050.yaml
#  configs/DKS/config.DKS_ZA_2050.yaml
#  configs/DKS/config.DKS_CL_2030_AB.yaml
#  configs/DKS/config.DKS_EG_2030_AB.yaml
#  configs/DKS/config.DKS_MA_2030_AB.yaml
#  configs/DKS/config.DKS_ZA_2030_AB.yaml
#  configs/DKS/config.DKS_CL_2035_AB.yaml
#  configs/DKS/config.DKS_EG_2035_AB.yaml
#  configs/DKS/config.DKS_MA_2035_AB.yaml
#  configs/DKS/config.DKS_ZA_2035_AB.yaml
#  configs/DKS/config.DKS_CL_2050_AB.yaml
#  configs/DKS/config.DKS_EG_2050_AB.yaml
#  configs/DKS/config.DKS_MA_2050_AB.yaml
#  configs/DKS/config.DKS_ZA_2050_AB.yaml

# MA Paper
# configs/MAPaper/config.MAPaper_2035_Exp.yaml
# configs/MAPaper/config.MAPaper_2035_best.yaml
# configs/MAPaper/config.MAPaper_2035_worst.yaml
# configs/MAPaper/config.MAPaper_2035_IR.yaml



COUNTRY=$(basename "$CONFIGFILE" | grep -oP '(?<=DKS_)[A-Z]+')
# COUNTRY=$(basename "$CONFIGFILE" | grep -oP '(?<=WSA_)[A-Z]+')
# COUNTRY=$(basename "$CONFIGFILE" | grep -oP '(?<=GSA_)[A-Z]+')


# Memory map in MB (rounded up from worst-case observed RAM usage)
declare -A MEM_MAP=(
    ["DZ"]=130000
    ["NG"]=92000
    ["KE"]=82000
    ["EG"]=72000
    ["TN"]=66000
    ["TZ"]=60000
    ["GH"]=42000
    ["MA"]=130000
    ["NA"]=40000
    ["ET"]=28000
    ["CD"]=28000
    ["ZA"]=36000
    ["MR"]=18000
)

MEM_MB=${MEM_MAP[$COUNTRY]:-96000}
echo "Country: $COUNTRY  ->  mem_mb: $MEM_MB"

# SLURM specifications made in default.cluster.yaml & the individual rules
#snakemake --cluster-config config/config.cluster.yaml --cluster "sbatch -p {cluster.partition} -t {cluster.walltime} -o {cluster.output} -e {cluster.error} -c {threads} --mem {resources.mem_mb}" --jobs 199 --latency-wait 60 --keep-going --configfile config/config.yaml
# snakemake solve_sector_networks --cluster-config configs/cluster_config.yaml \
# Create NaN placeholder CSVs for any Morris metric files that are missing
# (e.g. because the solve job failed/timed out on SLURM).
# morris_analyze.py already drops incomplete trajectories automatically.
# conda run -n pypsa-earth-morris python scripts/morris/morris_fill_missing.py --configfile configs/GSA/config.GSA_ZA_morris.yaml

snakemake solve_sector_networks --cluster-config configs/cluster_config.yaml \
--cluster "sbatch -p {cluster.partition} -t {cluster.walltime} -c {cluster.cpus_per_task} --mem $MEM_MB -x {cluster.exclude}" \
--jobs 199 --latency-wait 60 --keep-going \
--configfile "$CONFIGFILE" \
# --rerun-trigger mtime \
# --forcerun build_renewable_profiles \
# --forceall \
# --allowed-rules solve_sector_networks add_electricity add_export add_extra_components cluster_network copy_custom_costs overwrite_renewables prepare_network prepare_sector_network simplify_network solve_sector_network solve_sector_networks
# -n \
# --rerun-incomplete \

# squeue --sort=t,p,-S --format="%.7i %.9P %.8u %.8j %.4Q %.7T %.19V %.19S %.11M %.11l %.19e %.2D %.3C %R" -p progress
# ps -u ljansen -o pid,%cpu,%mem,rss,vsz,comm --sort=-rss | head -20
# sinfo -o "%n %e %m %a %c %C" | sort -k5,5nr
# cp configs/DKS/config.DKS_MA_2050_AB.yaml results/DKS_MA_2030/configs/

# echo "=== Step 2: Filling NaN placeholders for missing solved networks ==="
# python scripts/morris/morris_fill_missing.py --configfile "$CONFIGFILE"

# rm -rf /mnt/data/gsa/pypsa-earth-lukas/results/*/configs /mnt/data/gsa/pypsa-earth-lukas/results/*/prenetworks
# rm -Recurse -Force results\*\configs, results\*\prenetworks

# conda run -n pypsa-earth-morris python scripts/morris/morris_fill_missing.py --configfile configs/GSA/config.GSA_ZA_morris.yaml


# conda activate pypsa-earth

# bash run_slurm.sh

# snakemake -j8 plot_hydrogen_networks --configfile configs/DKS/config.DKS_ZA_2050.yaml --allowed-rules plot_hydrogen_networks plot_hydrogen_network --force
# snakemake -j8 plot_power_networks --configfile configs/DKS/config.DKS_ZA_2050.yaml --allowed-rules plot_power_networks plot_power_network --force

# snakemake -j8 plot_network_balances --configfile configs/DKS/config.DKS_EG_2050.yaml --allowed-rules plot_network_balances plot_network_balance --force

# snakemake -j8 plot_network_balances --configfile configs/MAPaper/config.MAPaper_2035_IR.yaml --allowed-rules plot_network_balances plot_network_balance --force
# snakemake -j8 plot_network_balances --configfile configs/MAPaper/config.MAPaper_2035_Exp.yaml --allowed-rules plot_network_balances plot_network_balance --force