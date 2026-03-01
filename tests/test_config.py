from __future__ import annotations

import json
from pathlib import Path

import pytest

from blur_detection.config import load_runtime_config


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    payload = {
        "StudentPC": {
            "Dataset_Path": "/tmp/train/",
            "Test_Dataset_Path": "/tmp/test/",
            "model_Path": {"1": "a.pth", "2": "b.pth", "3": "c.pth"},
            "model_bestweight_Path": {"1": "a_bw.pth", "2": "b_bw.pth", "3": "c_bw.pth"},
            "log_dir": {"1": "runs/1/{}", "2": "runs/2/{}", "3": "runs/3/{}"},
            "tempdir": "/tmp/",
            "tempdirTestDataset": "/tmp/test_temp/",
        }
    }
    file_path = tmp_path / "Config.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")
    return file_path


def test_load_runtime_config_defaults(config_file: Path) -> None:
    cfg = load_runtime_config(config_path=config_file)
    assert cfg.system_to_run == "StudentPC"
    assert cfg.model_selection == 1
    assert cfg.batch_size == 64
    assert cfg.validation_split == 0.3
    assert cfg.output_path == "Outputs/"


def test_load_runtime_config_overrides(config_file: Path) -> None:
    cfg = load_runtime_config(
        config_path=config_file,
        system_to_run="StudentPC",
        model_selection=3,
        enable_multi_gpu=True,
        device_ids=[0, 1],
        default_gpu_id="cuda:3",
        tensorboard=True,
        epochs=12,
        batch_size=8,
        validation_split=0.2,
        num_class_confusion_matrix=4,
        path_single_file="/tmp/in.nii.gz",
        output_path="/tmp/out/",
        custom_model_path="/tmp/weights.pth",
        transform_images=False,
        train_enabled=True,
        test_enabled=True,
    )
    assert cfg.model_selection == 3
    assert cfg.enable_multi_gpu is True
    assert cfg.device_ids == [0, 1]
    assert cfg.default_gpu_id == "cuda:3"
    assert cfg.tensorboard is True
    assert cfg.epochs == 12
    assert cfg.batch_size == 8
    assert cfg.validation_split == 0.2
    assert cfg.num_class_confusion_matrix == 4
    assert cfg.path_single_file == "/tmp/in.nii.gz"
    assert cfg.output_path == "/tmp/out/"
    assert cfg.custom_model_path == "/tmp/weights.pth"
    assert cfg.transform_images is False
    assert cfg.train_enabled is True
    assert cfg.test_enabled is True


def test_load_runtime_config_invalid_system(config_file: Path) -> None:
    with pytest.raises(ValueError):
        load_runtime_config(config_path=config_file, system_to_run="Unknown")
