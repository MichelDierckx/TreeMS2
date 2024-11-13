# Processing pipeline that applies each processor in sequence
class SpectrumProcessingPipeline:
    def __init__(self, processors: List[SpectrumProcessor]):
        self.processors = processors

    def process(self, spectrum: 'Spectrum') -> 'Spectrum':
        for processor in self.processors:
            spectrum = processor.process(spectrum)
        return spectrum


# Pipeline Factory that creates the processing pipeline based on configuration
class ProcessingPipelineFactory:
    @staticmethod
    def create_pipeline(config: Optional[dict]) -> SpectrumProcessingPipeline:
        processors = []

        # Dynamically add processors based on config settings
        if config is not None and "processing" in config:
            processing_steps = config["processing"]

            if processing_steps.get("normalize_intensity", False):
                processors.append(IntensityNormalizationProcessor())

            if "filter_low_intensity" in processing_steps:
                threshold = processing_steps["filter_low_intensity"].get("threshold", 0.1)
                processors.append(LowIntensityFilterProcessor(threshold))

            # Add more processors here as needed based on the config options
            # For example:
            # if processing_steps.get("other_processing", False):
            #     processors.append(OtherProcessor(...))

        # Return the constructed pipeline with all relevant processors
        return SpectrumProcessingPipeline(processors)
