from pathlib import Path


def get_columns_in_csv(path_to_scv: Path) -> list[str]:
    with open(path_to_scv, "r") as f:
        columns = f.readline().strip().split(",")
    return columns
