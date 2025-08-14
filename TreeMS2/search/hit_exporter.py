import os
import numpy as np
import numpy.typing as npt
import pandas as pd


class HitExporter:
    def __init__(
        self,
        output_file_path: str,
        file_ids: npt.NDArray[np.uint16],
        group_ids: npt.NDArray[np.uint16],
        scan_numbers: npt.NDArray[np.uint32],
    ):
        self.output_file_path = output_file_path

        self.file_ids = file_ids
        self.group_ids = group_ids
        self.scan_numbers = scan_numbers

    def export_hits(self, query_ids, target_ids, distances):
        # Create dataframe directly with dict for speed
        df = pd.DataFrame(
            {
                "query_group_id": self.group_ids[query_ids],
                "query_file_id": self.file_ids[query_ids],
                "query_scan_number": self.scan_numbers[query_ids],
                "hit_group_id": self.group_ids[target_ids],
                "hit_file_id": self.file_ids[target_ids],
                "hit_scan_number": self.scan_numbers[target_ids],
                "similarity": distances,
            }
        )

        write_header = not os.path.exists(self.output_file_path)
        df.to_csv(self.output_file_path, mode="a", header=write_header, index=False)
