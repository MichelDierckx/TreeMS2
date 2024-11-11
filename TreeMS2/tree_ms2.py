import functools
from typing import Union, List

from .cluster import spectrum
from .config import config


def main(args: Union[str, List[str]] = None) -> int:
    config.parse(args)  # Parse arguments from config file or command-line

    _, min_mz, max_mz = spectrum.get_dim(
        config.min_mz, config.max_mz, config.fragment_tol
    )

    process_spectrum = functools.partial(
        spectrum.process_spectrum,
        min_peaks=config.min_peaks,
        min_mz_range=config.min_mz_range,
        mz_min=min_mz,
        mz_max=max_mz,
        remove_precursor_tolerance=config.remove_precursor_tol,
        min_intensity=config.min_intensity,
        max_peaks_used=config.max_peaks_used,
        scaling=None if config.scaling == "off" else config.scaling,
    )

    return 0
