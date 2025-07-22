from enum import Enum


class StateType(Enum):
    INGESTION_STATE = "ingestion_state"
    CREATE_INDEX = "create_index"
    QUERY_INDEX = "query_index"
    COMPUTE_DISTANCES = "compute_distances"
