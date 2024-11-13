# Base class for a processing step in the pipeline
class SpectrumProcessor(ABC):
    @abstractmethod
    def process(self, spectrum: 'Spectrum') -> 'Spectrum':
        pass
