from pathlib import Path

from .data_point import DataPoint


class Dataset:
    def __init__(self, dir: Path, data: list[DataPoint]):
        self.dir = dir
        self.data = data

    @staticmethod
    def parse_datasets(dataset: str) -> list["Dataset"]:
        data_path = Path(dataset) / "data"
        if not data_path.exists():
            return []

        data = []
        for sub_dir in data_path.iterdir():
            txt_file = list(sub_dir.glob("*.txt"))
            if not txt_file:
                continue
            data.append(
                Dataset(sub_dir, DataPoint.parse_datapoints(txt_file[0]))
            )

        return data
