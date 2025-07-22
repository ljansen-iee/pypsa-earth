# SLURM specifications made in default.cluster.yaml & the individual rules
#snakemake --cluster-config config/config.cluster.yaml --cluster "sbatch -p {cluster.partition} -t {cluster.walltime} -o {cluster.output} -e {cluster.error} -c {threads} --mem {resources.mem_mb}" --jobs 199 --latency-wait 60 --keep-going --configfile config/config.yaml
snakemake solve_sector_networks --cluster-config configs/cluster_config.yaml \
--cluster "sbatch -p {cluster.partition} -t {cluster.walltime} -c {cluster.cpus_per_task} --mem {cluster.mem_mb} -x {cluster.exclude}" \
--jobs 199 --latency-wait 60 --keep-going \
--configfile configs/DKS/config.DKS_MA_2030.yaml \
--rerun-trigger mtime 
#--until prepare_network \
# cp configs/DKS/config.DKS_MA_2030.yaml results/DKS_MA_2030/configs/

#  configs/DKS/config.DKS_CL_2030.yaml
#  configs/DKS/config.DKS_CL_2050.yaml
#  configs/DKS/config.DKS_EG_2030.yaml
#  configs/DKS/config.DKS_MA_2030.yaml
#  configs/DKS/config.DKS_MA_2050.yaml
#  configs/DKS/config.DKS_ZA_2030.yaml
#  configs/DKS/config.DKS_ZA_2050.yaml