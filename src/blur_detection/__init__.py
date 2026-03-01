"""Production wrapper package for the BlurDetection project."""

from .config import RuntimeConfig, load_runtime_config

__all__ = ["RuntimeConfig", "load_runtime_config", "BlurDetectionService"]


def __getattr__(name: str):
    if name == "BlurDetectionService":
        from .service import BlurDetectionService

        return BlurDetectionService
    raise AttributeError(name)
