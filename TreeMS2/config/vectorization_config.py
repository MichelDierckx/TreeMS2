class VectorizationConfig:
    def __init__(self, min_mz: float, max_mz: float, fragment_tol: float, low_dim: int):
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.fragment_tol = fragment_tol
        self.low_dim = low_dim

    @classmethod
    def from_parser(cls, parser) -> "VectorizationConfig":
        return cls(
            min_mz=parser.get("min_mz"),
            max_mz=parser.get("max_mz"),
            fragment_tol=parser.get("fragment_tol"),
            low_dim=parser.get("low_dim")
        )
