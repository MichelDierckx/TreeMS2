from ...spectrum import Spectrum


class SpectrumNormalizerProcessor:
    def process(self, spectrum: Spectrum) -> Spectrum:
        spectrum.intensity = _norm_intensity(spectrum.intensity)
        return spectrum
