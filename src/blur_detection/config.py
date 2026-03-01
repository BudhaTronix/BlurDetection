from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_device_ids(value: str, default: List[int]) -> List[int]:
    if not value:
        return default
    try:
        return [int(token.strip()) for token in value.split(",") if token.strip()]
    except ValueError:
        return default


def _config_default_path() -> Path:
    env_override = os.getenv("BLUR_CONFIG_PATH")
    if env_override:
        return Path(env_override).expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / "Code" / "Config.json").resolve()


@dataclass(frozen=True)
class RuntimeConfig:
    config_path: Path
    config_data: Dict[str, Any]
    system_to_run: str
    model_selection: int
    enable_multi_gpu: bool
    device_ids: List[int]
    default_gpu_id: str
    tensorboard: bool
    epochs: int
    batch_size: int
    validation_split: float
    num_class_confusion_matrix: int
    path_single_file: str
    output_path: str
    custom_model_path: str
    transform_images: bool
    train_enabled: bool
    test_enabled: bool


def load_runtime_config(
    *,
    config_path: Optional[Path] = None,
    system_to_run: Optional[str] = None,
    model_selection: Optional[int] = None,
    enable_multi_gpu: Optional[bool] = None,
    device_ids: Optional[List[int]] = None,
    default_gpu_id: Optional[str] = None,
    tensorboard: Optional[bool] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    validation_split: Optional[float] = None,
    num_class_confusion_matrix: Optional[int] = None,
    path_single_file: Optional[str] = None,
    output_path: Optional[str] = None,
    custom_model_path: Optional[str] = None,
    transform_images: Optional[bool] = None,
    train_enabled: Optional[bool] = None,
    test_enabled: Optional[bool] = None,
) -> RuntimeConfig:
    """Load runtime configuration with env-variable overrides.

    Existing model logic remains unchanged. This only centralizes configuration.
    """

    final_config_path = (config_path or _config_default_path()).expanduser().resolve()
    if not final_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {final_config_path}")

    with final_config_path.open("r", encoding="utf-8") as file:
        config_data = json.load(file)

    sys_name = (
        system_to_run
        or os.getenv("BLUR_SYSTEM")
        or "StudentPC"
    )
    model_id = (
        model_selection
        if model_selection is not None
        else int(os.getenv("BLUR_MODEL_SELECTION", "1"))
    )
    multi_gpu = (
        enable_multi_gpu
        if enable_multi_gpu is not None
        else _parse_bool(os.getenv("BLUR_ENABLE_MULTI_GPU"), False)
    )
    ids = (
        device_ids
        if device_ids is not None
        else _parse_device_ids(os.getenv("BLUR_DEVICE_IDS", ""), [3, 4])
    )
    default_gpu = default_gpu_id or os.getenv("BLUR_DEFAULT_GPU", "cuda:0")
    tb_enabled = (
        tensorboard
        if tensorboard is not None
        else _parse_bool(os.getenv("BLUR_TENSORBOARD"), False)
    )
    epoch_count = epochs if epochs is not None else int(os.getenv("BLUR_EPOCHS", "1000"))
    batch = batch_size if batch_size is not None else int(os.getenv("BLUR_BATCH_SIZE", "64"))
    val_split = (
        validation_split
        if validation_split is not None
        else float(os.getenv("BLUR_VALIDATION_SPLIT", "0.3"))
    )
    confusion_classes = (
        num_class_confusion_matrix
        if num_class_confusion_matrix is not None
        else int(os.getenv("BLUR_NUM_CLASS_CONFUSION_MATRIX", "9"))
    )
    single_file = path_single_file or os.getenv("BLUR_PATH_SINGLE_FILE", "")
    out_path = output_path or os.getenv("BLUR_OUTPUT_PATH", "Outputs/")
    custom_model = custom_model_path or os.getenv("BLUR_CUSTOM_MODEL_PATH", "")
    transform = (
        transform_images
        if transform_images is not None
        else _parse_bool(os.getenv("BLUR_TRANSFORM_IMAGES"), True)
    )
    train_flag = (
        train_enabled
        if train_enabled is not None
        else _parse_bool(os.getenv("BLUR_TRAIN"), False)
    )
    test_flag = (
        test_enabled
        if test_enabled is not None
        else _parse_bool(os.getenv("BLUR_TEST"), False)
    )

    if sys_name not in config_data:
        valid = ", ".join(sorted(config_data.keys()))
        raise ValueError(f"Unknown system '{sys_name}'. Valid values: {valid}")
    if model_id not in {1, 2, 3}:
        raise ValueError("model_selection must be one of: 1, 2, 3")
    if not 0 <= val_split < 1:
        raise ValueError("validation_split must be between 0 (inclusive) and 1 (exclusive)")

    return RuntimeConfig(
        config_path=final_config_path,
        config_data=config_data,
        system_to_run=sys_name,
        model_selection=model_id,
        enable_multi_gpu=multi_gpu,
        device_ids=ids,
        default_gpu_id=default_gpu,
        tensorboard=tb_enabled,
        epochs=epoch_count,
        batch_size=batch,
        validation_split=val_split,
        num_class_confusion_matrix=confusion_classes,
        path_single_file=single_file,
        output_path=out_path,
        custom_model_path=custom_model,
        transform_images=transform,
        train_enabled=train_flag,
        test_enabled=test_flag,
    )
