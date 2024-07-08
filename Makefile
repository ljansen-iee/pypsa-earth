# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

.PHONY: test setup clean

test:
	set -e
	snakemake solve_all_networks -call --configfile config.tutorial.yaml
	snakemake solve_all_networks -call --configfile config.tutorial.yaml test/config.custom.yaml
	snakemake solve_all_networks -call --configfile config.tutorial.yaml configs/scenarios/config.NG.yaml
	snakemake solve_all_networks_monte -call --configfile config.tutorial.yaml test/config.monte_carlo.yaml
	snakemake solve_all_networks -call --configfile config.tutorial.yaml test/config.landlock.yaml
	echo "All tests completed successfully."

setup:
	# Add setup commands here
	echo "Setup complete."

clean:
	# Add clean-up commands here
	snakemake -j1 solve_all_networks --delete-all-output --configfile config.tutorial.yaml test/config.custom.yaml
	snakemake -j1 solve_all_networks --delete-all-output --configfile config.tutorial.yaml test/config.tutorial_noprogress.yaml
	snakemake -j1 solve_all_networks_monte --delete-all-output --configfile test/config.monte_carlo.yaml
	snakemake -j1 run_all_scenarios --delete-all-output --configfile test/config.landlock.yaml
	echo "Clean-up complete."