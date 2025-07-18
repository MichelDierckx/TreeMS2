from enum import Enum
from typing import Optional, Literal

import spectrum_utils.spectrum as sus

from TreeMS2.spectrum.spectrum_processing.spectrum_processor import SpectrumProcessor


class ScalingMethod(Enum):
    OFF = "off"
    ROOT = "root"
    LOG = "log"
    RANK = "rank"


class IntensityScalingProcessor(SpectrumProcessor):
    def __init__(self, scaling: Literal[ScalingMethod.ROOT, ScalingMethod.LOG, ScalingMethod.RANK],
                 max_rank: Optional[int]):
        super().__init__()
        self.scaling = scaling
        self.max_rank = max_rank

    def change(self, spectrum: sus.MsmsSpectrum) -> sus.MsmsSpectrum:
        return spectrum.scale_intensity(self.scaling.value, max_rank=self.max_rank)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scaling={self.scaling}, max_rank={self.max_rank})"
