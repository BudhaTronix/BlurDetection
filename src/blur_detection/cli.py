from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from .config import load_runtime_config
from .logging_utils import setup_logging


def _parse_device_ids(ids: Optional[str]) -> Optional[List[int]]:
    if ids is None:
        return None
    parsed: List[int] = []
    for item in ids.split(","):
        value = item.strip()
        if not value:
            continue
        parsed.append(int(value))
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Production CLI for BlurDetection")
    parser.add_argument(
        "--mode",
        default="test",
        choices=["train", "test", "single-file", "legacy"],
        help="Execution mode",
    )
    parser.add_argument("--config-path", type=Path, default=None, help="Path to Config.json")
    parser.add_argument("--system", type=str, default=None, help="System key in config (e.g., StudentPC)")
    parser.add_argument(
        "--model-selection",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="1=ResNet18, 2=ResNet50, 3=ResNet101",
    )
    parser.add_argument("--enable-multi-gpu", action="store_true", help="Enable multi-GPU execution")
    parser.add_argument(
        "--device-ids",
        type=str,
        default=None,
        help='Comma-separated CUDA device IDs (e.g. "0,1")',
    )
    parser.add_argument("--default-gpu-id", type=str, default=None, help='Default GPU device (e.g. "cuda:0")')
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--validation-split", type=float, default=None, help="Validation split")
    parser.add_argument("--num-class-confusion-matrix", type=int, default=None, help="Class count for confusion matrix")
    parser.add_argument("--path-single-file", type=str, default=None, help="Single NIfTI file path for inference")
    parser.add_argument("--output-path", type=str, default=None, help="Output directory")
    parser.add_argument("--custom-model-path", type=str, default=None, help="Override model path for single-file mode")
    parser.add_argument(
        "--no-transform-images",
        action="store_true",
        help="Disable image transform in single-file mode",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging()

    config = load_runtime_config(
        config_path=args.config_path,
        system_to_run=args.system,
        model_selection=args.model_selection,
        enable_multi_gpu=args.enable_multi_gpu if args.enable_multi_gpu else None,
        device_ids=_parse_device_ids(args.device_ids),
        default_gpu_id=args.default_gpu_id,
        tensorboard=args.tensorboard if args.tensorboard else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        num_class_confusion_matrix=args.num_class_confusion_matrix,
        path_single_file=args.path_single_file,
        output_path=args.output_path,
        custom_model_path=args.custom_model_path,
        transform_images=False if args.no_transform_images else None,
    )
    from .service import BlurDetectionService

    service = BlurDetectionService(config)

    if args.mode == "train":
        service.run_train()
        return
    if args.mode == "test":
        service.run_test()
        return
    if args.mode == "single-file":
        service.run_single_file_test()
        return
    service.run_legacy_default()


if __name__ == "__main__":
    main()
