from typing import Optional

from ...spectrum import Spectrum


class IntensityScalingProcessor:
    def __init__(self, scaling: Optional[str], max_rank: Optional[int]):
        self.scaling = scaling
        self.max_rank = max_rank

    def process(self, spectrum: Spectrum) -> Spectrum:
        return spectrum.scale_intensity(self.scaling, max_rank=self.max_rank)
