from typing import Optional, Literal

import spectrum_utils.spectrum as sus

from ..spectrum_processor import SpectrumProcessor
from ....config.spectrum_processing_config import ScalingMethod


class IntensityScalingProcessor(SpectrumProcessor):
    def __init__(self, scaling: Literal[ScalingMethod.ROOT, ScalingMethod.LOG, ScalingMethod.RANK],
                 max_rank: Optional[int]):
        super().__init__()
        self.scaling = scaling
        self.max_rank = max_rank

    def change(self, spectrum: sus.MsmsSpectrum) -> sus.MsmsSpectrum:
        return spectrum.scale_intensity(self.scaling.value, max_rank=self.max_rank)