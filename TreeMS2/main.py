import multiprocessing
import time
from typing import Union, List

import faiss
import pandas as pd
from dotenv import load_dotenv

from TreeMS2.config.env_variables import log_environment_variables
from TreeMS2.config.logger_config import setup_logging, get_logger
from TreeMS2.config.treems2_config import Config
from TreeMS2.tree_ms2 import TreeMS2


def main(args: Union[str, List[str]] = None) -> int:
    load_dotenv()  # Load environment variables from a .env file
    # setup up config
    config = Config()
    config.parse(args)  # Parse arguments from config file or command-line

    # setup logging
    console_level = config.log_level
    setup_logging(work_dir=config.work_dir, console_level=console_level)

    # log config parameters
    config.log_parameters()

    logger = get_logger(__name__)
    # log environment variables
    log_environment_variables()

    multiprocessing.set_start_method("spawn")  # lance does not work with FORK method
    # get FAISS number of threads
    logger.debug(f"FAISS threads: {faiss.omp_get_max_threads()}")

    n_runs = 10
    # Read original mapping CSV (tab-separated)
    original_csv_path = "full_dataset.csv"
    df = pd.read_csv(original_csv_path, sep="\t")
    # For saving benchmark timings in current directory
    timing_records = []

    for run in range(1, n_runs + 1):
        print(f"Run {run}/{n_runs}")
        config.work_dir = config.work_dir + f"_run{run}"

        subset_csv_path = config.sample_to_group_file
        # Get subset of the data
        subset_size = int(len(df) * run / n_runs)
        subset_df = df.iloc[:subset_size]

        # Write subset CSV (tab-separated)
        subset_df.to_csv(subset_csv_path, sep="\t", index=False)

        # Run the app and time it
        start_time = time.time()
        app = TreeMS2(config)
        num_spectra = app.run()
        elapsed = time.time() - start_time

        # Record results
        timing_records.append({
            "run": run,
            "num_rows": subset_size,
            "num_spectra": num_spectra,
            "time_seconds": round(elapsed, 2),
        })
        # Save timings to CSV in current directory
        timing_df = pd.DataFrame(timing_records)
        timing_df.to_csv("timings.csv", index=False, sep="\t")

    return 0
