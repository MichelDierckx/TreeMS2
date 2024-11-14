from typing import Optional

from ...group_spectrum import GroupSpectrum


class MZRangeFilterProcessor:
    def __init__(self, mz_min: Optional[float], mz_max: Optional[float]):
        self.mz_min = mz_min
        self.mz_max = mz_max

    def process(self, spectrum: GroupSpectrum) -> GroupSpectrum:
        return spectrum.set_mz_range(self.mz_min, self.mz_max)
