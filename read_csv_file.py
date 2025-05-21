import csv

import numpy as np


def string_to_float(v: str):
    v = v.strip(" ()<>聽\ufeff")
    return float(v) if v != "" else 0


def read_csv_file(file_path, skip_header=True) -> list[list[float]]:
    print(f"read file: {file_path}")
    # Read CSV file and extract data
    ans = []

    with open(file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        if skip_header:
            next(reader)  # Skip the header row
        for row in reader:
            row1 = map(string_to_float, row)
            ans.append(list(row1))

    return ans


def read_csv_file_np(file_path, skip_header=True) -> np.ndarray[float]:
    return np.array(read_csv_file(file_path, skip_header))
