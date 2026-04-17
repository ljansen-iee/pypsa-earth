# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Generates the Morris Method sample matrix using SALib.

Reads the ``morris`` config section, builds a SALib problem dict from the
parameter names and bounds, and produces a Morris sample matrix of shape
(N * (D + 1), D) where N is the number of trajectories and D is the number
of parameters.

Outputs
-------
- ``resources/{RDIR}/morris/sample_matrix.npy``
- ``resources/{RDIR}/morris/problem.json``
"""

import json
import logging

import numpy as np
from SALib.sample.morris import sample as morris_sample

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("morris_sample")

    morris_config = snakemake.params.morris
    parameters = morris_config["parameters"]
    options = morris_config["options"]

    N = options["N"]
    num_levels = options["num_levels"]
    seed = options.get("seed", 42)

    # Build SALib problem definition
    problem = {
        "num_vars": len(parameters),
        "names": [p["name"] for p in parameters],
        "bounds": [p["bounds"] for p in parameters],
    }

    logger.info(
        f"Morris sampling: N={N}, num_levels={num_levels}, D={problem['num_vars']}, "
        f"total runs={N * (problem['num_vars'] + 1)}"
    )
    logger.info(f"Parameter names: {problem['names']}")
    logger.info(f"Parameter bounds: {problem['bounds']}")

    # Generate Morris sample matrix
    # Shape: (N * (D + 1), D)  — each row is a set of multiplicative scaling factors
    sample_matrix = morris_sample(
        problem,
        N=N,
        num_levels=num_levels,
        seed=seed,
    )

    logger.info(f"Sample matrix shape: {sample_matrix.shape}")

    # Save outputs
    np.save(snakemake.output.sample_matrix, sample_matrix)
    logger.info(f"Saved sample matrix to {snakemake.output.sample_matrix}")

    with open(snakemake.output.problem, "w") as f:
        json.dump(problem, f, indent=2)
    logger.info(f"Saved problem definition to {snakemake.output.problem}")
