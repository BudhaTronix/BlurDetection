from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blur_detection.config import load_runtime_config  # noqa: E402
from blur_detection.service import BlurDetectionService  # noqa: E402


st.set_page_config(page_title="Blur Detection", page_icon="MRI", layout="centered")
st.title("Blur Detection UI")
st.caption("Minimal Streamlit UI for running existing BlurDetection workflows.")

default_config = PROJECT_ROOT / "Code" / "Config.json"
mode = st.selectbox("Mode", ["single-file", "test", "train"], index=0)
config_path = st.text_input("Config Path", str(default_config))
system = st.text_input("System Key", "StudentPC")
model_selection = st.selectbox("Model", [1, 2, 3], index=0)
single_file = st.text_input("Single NIfTI File Path", "")
output_path = st.text_input("Output Directory", "Outputs/")
custom_model_path = st.text_input("Custom Model Path (optional)", "")
transform_images = st.checkbox("Apply Transform", value=True)
enable_multi_gpu = st.checkbox("Enable Multi-GPU", value=False)
device_ids = st.text_input("Device IDs (comma-separated)", "3,4")
default_gpu_id = st.text_input("Default GPU", "cuda:0")
tensorboard = st.checkbox("Enable TensorBoard", value=False)
epochs = st.number_input("Epochs", min_value=1, value=1000, step=1)
batch_size = st.number_input("Batch Size", min_value=1, value=64, step=1)
validation_split = st.number_input("Validation Split", min_value=0.0, max_value=0.99, value=0.3, step=0.01)
num_class_confusion_matrix = st.number_input(
    "Confusion Matrix Class Count",
    min_value=1,
    value=9,
    step=1,
)

if st.button("Run Workflow"):
    try:
        ids = [int(token.strip()) for token in device_ids.split(",") if token.strip()]
        runtime_config = load_runtime_config(
            config_path=Path(config_path),
            system_to_run=system,
            model_selection=int(model_selection),
            enable_multi_gpu=enable_multi_gpu,
            device_ids=ids,
            default_gpu_id=default_gpu_id,
            tensorboard=tensorboard,
            epochs=int(epochs),
            batch_size=int(batch_size),
            validation_split=float(validation_split),
            num_class_confusion_matrix=int(num_class_confusion_matrix),
            path_single_file=single_file,
            output_path=output_path,
            custom_model_path=custom_model_path,
            transform_images=transform_images,
        )
        service = BlurDetectionService(runtime_config)

        if mode == "single-file":
            service.run_single_file_test()
        elif mode == "test":
            service.run_test()
        else:
            service.run_train()
        st.success("Workflow completed.")
    except Exception as exc:  # pragma: no cover - UI surface
        st.error(f"Workflow failed: {exc}")
