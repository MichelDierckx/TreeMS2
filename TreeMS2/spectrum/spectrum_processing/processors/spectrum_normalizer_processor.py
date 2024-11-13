from ..spectrum_processor import SpectrumProcessor
from ...spectrum import Spectrum


class SpectrumNormalizerProcessor(SpectrumProcessor):
    def process(self, spectrum: Spectrum) -> Spectrum:
        spectrum.intensity = _norm_intensity(spectrum.intensity)
        return spectrum
