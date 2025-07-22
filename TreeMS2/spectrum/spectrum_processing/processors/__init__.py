from typing import Optional

from TreeMS2.ingestion.spectra_dataset.treems2_spectrum import TreeMS2Spectrum


class MZRangeFilterProcessor:
    def __init__(self, mz_min: Optional[float], mz_max: Optional[float]):
        self.mz_min = mz_min
        self.mz_max = mz_max

    def process(self, spectrum: TreeMS2Spectrum) -> TreeMS2Spectrum:
        return spectrum.set_mz_range(self.mz_min, self.mz_max)
