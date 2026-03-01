from __future__ import annotations

import logging

from Code.src.Pipeline import BlurDetection

from .config import RuntimeConfig

LOGGER = logging.getLogger(__name__)


class BlurDetectionService:
    """Thin production wrapper over legacy pipeline logic."""

    def __init__(self, config: RuntimeConfig):
        self.config = config

    def build_detector(self) -> BlurDetection:
        detector = BlurDetection(
            self.config.config_data,
            self.config.system_to_run,
            self.config.model_selection,
            self.config.device_ids,
            self.config.enable_multi_gpu,
            self.config.default_gpu_id,
            self.config.epochs,
            self.config.tensorboard,
            self.config.batch_size,
            self.config.validation_split,
            self.config.num_class_confusion_matrix,
            self.config.path_single_file,
            self.config.output_path,
        )
        return detector

    def run_train(self) -> None:
        LOGGER.info("Running training workflow")
        detector = self.build_detector()
        detector.train()

    def run_test(self) -> None:
        LOGGER.info("Running dataset test workflow")
        detector = self.build_detector()
        detector.test()

    def run_single_file_test(self) -> None:
        detector = self.build_detector()
        if not self.config.path_single_file:
            raise ValueError("path_single_file is required for single-file mode")

        LOGGER.info("Running single-file test workflow for %s", self.config.path_single_file)
        if self.config.custom_model_path:
            detector.test_singleFile(
                transform_Images=self.config.transform_images,
                custom_model_path=self.config.custom_model_path,
            )
            return
        detector.test_singleFile(transform_Images=self.config.transform_images)

    def run_legacy_default(self) -> None:
        """Mimic legacy Pipeline_executer behavior exactly."""
        detector = self.build_detector()
        print("Model selection : ", self.config.model_selection)
        detector.test()
