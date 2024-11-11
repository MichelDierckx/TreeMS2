from typing import Dict, Optional


class PeakFileGroupMapping:
    """Class to manage mapping from samples (ms/ms files) to group IDs, and from group IDs to group names."""

    def __init__(self):
        # Mapping of group name to group ID
        self.name_to_group_id: Dict[str, int] = {}
        # Mapping of group ID to group name
        self.group_id_to_name: Dict[int, str] = {}
        # Mapping of sample (file) to group ID
        self.sample_to_group_id: Dict[str, int] = {}
        # The number of groups
        self.nr_groups = 0
        # The number of peak files
        self.nr_files = 0

    def add(self, peak_file: str, group_name: str) -> None:
        """
        Add a sample file to the associated group.
        :param peak_file: the path to the sample file
        :param group_name: the name of group associated with the sample file
        :return: None
        """
        if group_name not in self.name_to_group_id:
            # make new group
            self.name_to_group_id[group_name] = self.nr_groups + 1
            self.group_id_to_name[self.nr_groups + 1] = group_name
            self.sample_to_group_id[peak_file] = self.nr_groups + 1
            self.nr_groups += 1
        else:
            # add to existing group
            group_id = self.name_to_group_id[group_name]
            self.sample_to_group_id[peak_file] = group_id
        self.nr_files += 1

    def get_group_id_for_peak_file(self, peak_file: str) -> Optional[int]:
        """Retrieve the group ID associated with a sample file."""
        return self.sample_to_group_id.get(peak_file)

    def get_group_name_for_peak_file(self, peak_file: str) -> Optional[str]:
        """Retrieve the group name associated with a sample file."""
        group_id = self.get_group_id_for_peak_file(peak_file)
        if group_id is not None:
            return self.group_id_to_name.get(group_id)
        return None

    def get_number_of_files_for_group(self, group_name: str) -> int:
        """
        Return the number of peak files associated with a specific group.
        :param group_name: The name of the group.
        :return: The number of files in the group.
        """
        group_id = self.name_to_group_id.get(group_name)
        if group_id is not None:
            return sum(1 for group_id_in_sample in self.sample_to_group_id.values() if group_id_in_sample == group_id)
        return 0
