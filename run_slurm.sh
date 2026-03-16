# SLURM specifications made in default.cluster.yaml & the individual rules
#snakemake --cluster-config config/config.cluster.yaml --cluster "sbatch -p {cluster.partition} -t {cluster.walltime} -o {cluster.output} -e {cluster.error} -c {threads} --mem {resources.mem_mb}" --jobs 199 --latency-wait 60 --keep-going --configfile config/config.yaml
snakemake solve_sector_networks --cluster-config configs/cluster_config.yaml \
--cluster "sbatch -p {cluster.partition} -t {cluster.walltime} -c {cluster.cpus_per_task} --mem {cluster.mem_mb} -x {cluster.exclude}" \
--jobs 199 --latency-wait 60 --keep-going \
--configfile configs/MAPaper/config.MAPaper_2035_IR.yaml \
--rerun-trigger mtime \
# --allowed-rules solve_sector_networks add_electricity add_export add_extra_components cluster_network copy_custom_costs overwrite_renewables prepare_network prepare_sector_network simplify_network solve_sector_network solve_sector_networks
# -n \
# --forceall \
# --rerun-incomplete \

# --rerun-incomplete \
#--until prepare_network \
# cp configs/DKS/config.DKS_MA_2050_AB.yaml results/DKS_MA_2030/configs/

# conda activate pypsa-earth

# WACC Summary:
# country_code  wacc  wacc_real
#          CHL 0.100      0.079
#          EGY 0.139      0.117
#          DEU 0.094      0.072
#          KEN 0.134      0.111
#          MAR 0.106      0.084
#          ZAF 0.112      0.090

#  configs/DKS/config.DKS_CL_2030.yaml
#  configs/DKS/config.DKS_EG_2030.yaml
#  configs/DKS/config.DKS_MA_2030.yaml
#  configs/DKS/config.DKS_ZA_2030.yaml
#  configs/DKS/config.DKS_CL_2050.yaml
#  configs/DKS/config.DKS_EG_2050.yaml
#  configs/DKS/config.DKS_MA_2050.yaml
#  configs/DKS/config.DKS_ZA_2050.yaml
#  configs/DKS/config.DKS_CL_2030_AB.yaml
#  configs/DKS/config.DKS_EG_2030_AB.yaml
#  configs/DKS/config.DKS_MA_2030_AB.yaml
#  configs/DKS/config.DKS_ZA_2030_AB.yaml
#  configs/DKS/config.DKS_CL_2050_AB.yaml
#  configs/DKS/config.DKS_EG_2050_AB.yaml
#  configs/DKS/config.DKS_MA_2050_AB.yaml
#  configs/DKS/config.DKS_ZA_2050_AB.yaml

# MA Paper
# configs/MAPaper/config.MAPaper_2035_Exp.yaml
# configs/MAPaper/config.MAPaper_2035_best.yaml
# configs/MAPaper/config.MAPaper_2035_worst.yaml
# configs/MAPaper/config.MAPaper_2035_IR.yaml

# bash run_slurm.sh

# snakemake -j8 plot_hydrogen_networks --configfile configs/DKS/config.DKS_ZA_2050.yaml --allowed-rules plot_hydrogen_networks plot_hydrogen_network --force
# snakemake -j8 plot_power_networks --configfile configs/DKS/config.DKS_ZA_2050.yaml --allowed-rules plot_power_networks plot_power_network --force

# snakemake -j8 plot_network_balances --configfile configs/DKS/config.DKS_EG_2050.yaml --allowed-rules plot_network_balances plot_network_balance --force

# snakemake -j8 plot_network_balances --configfile configs/MAPaper/config.MAPaper_2035_IR.yaml --allowed-rules plot_network_balances plot_network_balance --force
# snakemake -j8 plot_network_balances --configfile configs/MAPaper/config.MAPaper_2035_Exp.yaml --allowed-rules plot_network_balances plot_network_balance --force