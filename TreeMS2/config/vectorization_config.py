class VectorizationConfig:
    def __init__(self, min_mz: float, max_mz: float, fragment_tol: float, low_dim: int):
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.fragment_tol = fragment_tol
        self.low_dim = low_dim
