from typing import Optional

import spectrum_utils.spectrum as sus

from ..spectrum_processor import SpectrumProcessor


class IntensityScalingProcessor(SpectrumProcessor):
    def __init__(self, scaling: Optional[str], max_rank: Optional[int]):
        super().__init__()
        self.scaling = scaling
        self.max_rank = max_rank

    def change(self, spectrum: sus.MsmsSpectrum) -> sus.MsmsSpectrum:
        return spectrum.scale_intensity(self.scaling, max_rank=self.max_rank)
