import numpy as np
from scipy.sparse import csr_matrix


class SpectraMatrix:
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], int):
            # Case 1: Initialize an empty matrix with given size
            nr_spectra = args[0]
            self.matrix = csr_matrix((nr_spectra, nr_spectra), dtype=np.bool_)

        elif len(args) == 1 and isinstance(args[0], csr_matrix):
            # Case 2: Initialize from an existing csr_matrix
            self.matrix = args[0]

        elif len(args) == 4:
            # Case 3: Initialize from data, rows, columns, and shape
            data, rows, cols, shape = args
            self.matrix = csr_matrix((data, (rows, cols)), shape=shape, dtype=np.bool_)

        else:
            raise ValueError("Invalid arguments for initializing SpectraMatrix.")

    def subtract(self, spectra_matrix: "SpectraMatrix"):
        self.matrix -= spectra_matrix.matrix
