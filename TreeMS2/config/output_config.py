class OutputConfig:
    def __init__(self, work_dir: str):
        self.work_dir = work_dir

    @classmethod
    def from_parser(cls, parser) -> "OutputConfig":
        return cls(
            work_dir=parser.get("work_dir")
        )