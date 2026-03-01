from __future__ import annotations

import csv
from pathlib import Path

from Code.Utils.CSVGenerator import GenerateCSV, checkCSV


def _touch(path: Path) -> None:
    path.write_text("", encoding="utf-8")


def test_generate_csv_all_files(tmp_path: Path) -> None:
    _touch(tmp_path / "SUB1_1-0.10.nii.gz")
    _touch(tmp_path / "SUB1_2-0.90.nii.gz")
    csv_name = "train.csv"

    GenerateCSV(dataset_Path=f"{tmp_path}/", csv_FileName=csv_name, subjects=None)
    rows = list(csv.reader((tmp_path / csv_name).open("r", encoding="utf-8")))
    assert rows == [["SUB1_1-0.10.nii.gz", "0.1"], ["SUB1_2-0.90.nii.gz", "0.9"]]


def test_generate_csv_selected_subjects(tmp_path: Path) -> None:
    _touch(tmp_path / "SUB1_1-0.10.nii.gz")
    _touch(tmp_path / "SUB2_1-0.30.nii.gz")
    csv_name = "filtered.csv"

    GenerateCSV(dataset_Path=f"{tmp_path}/", csv_FileName=csv_name, subjects=["SUB2"])
    rows = list(csv.reader((tmp_path / csv_name).open("r", encoding="utf-8")))
    assert rows == [["SUB2_1-0.30.nii.gz", "0.3"]]


def test_check_csv_creates_missing_file(tmp_path: Path) -> None:
    _touch(tmp_path / "SUB1_1-0.10.nii.gz")
    checkCSV(dataset_Path=f"{tmp_path}/", csv_FileName="check.csv", subjects=None, overwrite=False)
    assert (tmp_path / "check.csv").exists()
