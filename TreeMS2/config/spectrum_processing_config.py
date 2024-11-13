from enum import Enum


class ScalingMethod(Enum):
    OFF = "off"
    ROOT = "root"
    LOG = "log"
    RANK = "rank"


class SpectrumProcessingConfig:
    def __init__(
            self,
            fragment_tol: float = 0.05,
            min_peaks: int = 5,
            min_mz_range: float = 250.0,
            min_mz: float = 101.0,
            max_mz: float = 1500.0,
            remove_precursor_tol: float = 1.5,
            min_intensity: float = 0.01,
            max_peaks_used: int = 50,
            scaling: ScalingMethod = ScalingMethod.OFF,
            low_dim: int = 400
    ):
        """
        Configuration for spectrum processing parameters.

        Parameters:
        ----------
        fragment_tol : float
            Fragment mass tolerance in m/z (default: 0.05 m/z).

        min_peaks : int
            Minimum number of peaks required in each spectrum. Spectra with fewer peaks are discarded (default: 5).

        min_mz_range : float
            Minimum m/z range required in each spectrum. Spectra with smaller ranges are discarded (default: 250.0 m/z).

        min_mz : float
            Minimum allowed m/z value for peaks in spectra (default: 101.0 m/z).

        max_mz : float
            Maximum allowed m/z value for peaks in spectra (default: 1500.0 m/z).

        remove_precursor_tol : float
            Tolerance range around the precursor m/z to remove peaks, to avoid interference (default: 1.5 m/z).

        min_intensity : float
            Minimum relative intensity threshold for peaks. Peaks below this threshold (as a fraction of the base intensity) are removed (default: 0.01).

        max_peaks_used : int
            Maximum number of peaks to retain in spectra after filtering for intensity. Only the most intense peaks are kept (default: 50).

        scaling : str
            Method for scaling peak intensities to reduce the influence of very intense peaks. Options are ScalingMethod.OFF, ScalingMethod.ROOT, ScalingMethod.LOG, and ScalingMethod.RANK (default: ScalingMethod.OFF).

        low_dim : int
            Target dimensionality for vectorization, representing the length of low-dimensional spectra vectors (default: 400).
        """
        self.fragment_tol = fragment_tol
        self.min_peaks = min_peaks
        self.min_mz_range = min_mz_range
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.remove_precursor_tol = remove_precursor_tol
        self.min_intensity = min_intensity
        self.max_peaks_used = max_peaks_used
        self.scaling = scaling
        self.low_dim = low_dim

    @classmethod
    def from_parser(cls, parser) -> "SpectrumProcessingConfig":
        """
        Creates an instance of SpectrumProcessingConfig by fetching values from the parser.

        Parameters:
        ----------
        parser : configargparse.ArgParser or similar
            The parser instance containing the parsed configuration.

        Returns:
        -------
        SpectrumProcessingConfig
            A new instance of SpectrumProcessingConfig initialized with parser values.
        """
        return cls(
            fragment_tol=parser.get("fragment_tol"),
            min_peaks=parser.get("min_peaks"),
            min_mz_range=parser.get("min_mz_range"),
            min_mz=parser.get("min_mz"),
            max_mz=parser.get("max_mz"),
            remove_precursor_tol=parser.get("remove_precursor_tol"),
            min_intensity=parser.get("min_intensity"),
            max_peaks_used=parser.get("max_peaks_used"),
            scaling=ScalingMethod(parser.get("scaling")),
            low_dim=parser.get("low_dim")
        )
