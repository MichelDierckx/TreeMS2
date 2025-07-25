from enum import Enum


class StateType(Enum):
    INGESTION_STATE = "ingestion_state"
    INDEXING_STATE = "indexing_state"
    SEARCH_STATE = "search_state"
    SEARCH_RESULT_AGGREGATION_STATE = "search_result_aggregation_state"
    DISTANCE_MATRIX_COMPUTATION_STATE = "distance_matrix_computation_state"
