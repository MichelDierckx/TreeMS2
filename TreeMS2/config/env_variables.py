"""
Module related to relevant environment variables
"""

import os

from TreeMS2.config.logger_config import get_logger, log_parameter, log_section_title

logger = get_logger(__name__)

# Define constants for environment variable names

# TreeMS2
TREEMS2_NUM_CPUS = "TREEMS2_NUM_CPUS"
TREEMS2_MEM_PER_CPU = "TREEMS2_MEM_PER_CPU"

# Lance
LANCE_IO_THREADS = "LANCE_IO_THREADS"
LANCE_CPU_THREADS = "LANCE_CPU_THREADS"

# Numba
NUMBA_NUM_THREADS = "NUMBA_NUM_THREADS"

# SciPy, Numpy, Faiss, ...
OMP_NUM_THREADS = "OMP_NUM_THREADS"
MKL_NUM_THREADS = "MKL_NUM_THREADS"
NUMEXPR_NUM_THREADS = "NUMEXPR_NUM_THREADS"
BLIS_NUM_THREADS = "BLIS_NUM_THREADS"
OPENBLAS_NUM_THREADS = "OPENBLAS_NUM_THREADS"
OMP_WAIT_POLICY = "OMP_WAIT_POLICY"  # OMP_WAIT_POLICY=PASSIVE

# List of environment variables
env_vars = [
    TREEMS2_NUM_CPUS,
    TREEMS2_MEM_PER_CPU,
    LANCE_IO_THREADS,
    LANCE_CPU_THREADS,
    NUMBA_NUM_THREADS,
    OMP_NUM_THREADS,
    MKL_NUM_THREADS,
    NUMEXPR_NUM_THREADS,
    BLIS_NUM_THREADS,
    OPENBLAS_NUM_THREADS,
    OMP_WAIT_POLICY,
]


def log_environment_variables():
    """
    Helper function to log environment variables.
    :return:
    """
    log_section_title(logger=logger, title="[ ENVIRONMENT VARIABLES ]")
    for var in env_vars:
        log_parameter(
            logger=logger,
            parameter_name=var,
            parameter_value=os.environ.get(var, "Not Set"),
        )
