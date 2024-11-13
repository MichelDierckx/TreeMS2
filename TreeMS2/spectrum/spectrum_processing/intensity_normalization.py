# Example processor: Normalizes intensities
class IntensityNormalizationProcessor(SpectrumProcessor):
    def process(self, spectrum: 'Spectrum') -> 'Spectrum':
        spectrum.spectrum.intensity_array /= max(spectrum.spectrum.intensity_array)
        return spectrum
