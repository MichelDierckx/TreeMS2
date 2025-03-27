from enum import Enum


class StateType(Enum):
    PROCESS_SPECTRA = "process_spectra"
    CREATE_INDEX = "create_index"
    QUERY_INDEX = "query_index"
    COMPUTE_DISTANCES = "compute_distances"
