from pathlib import Path


class DataPoint:
    def __init__(self, image: str, num: int, text: str):
        self.image = image
        self.num = num
        self.text = text

    @staticmethod
    def parse_datapoints(path: Path) -> list["DataPoint"]:
        data = []
        file = open(path, "r")
        for line in file:
            line = line.strip()
            if not line:
                continue

            parts = line.split(" ", 2)
            if len(parts) < 3:
                continue

            data.append(DataPoint(parts[0], int(parts[1]), parts[2]))

        file.close()
        return data
