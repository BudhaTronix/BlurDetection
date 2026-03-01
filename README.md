# BlurDetection

Productionized MRI blur-detection project based on the existing repository logic.

The core training/inference logic in `Code/` is preserved. New files add configuration management, CLI, tests, containerization, and a minimal UI for safer production usage.

## Features

- Legacy-compatible blur detection pipeline (training, dataset testing, single-file inference)
- Typed runtime configuration with environment variable overrides
- Production CLI entrypoint
- Minimal Streamlit UI
- Unit tests for config and utility behavior
- Dockerized runtime

## Installation

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Optional editable install (adds `blur-detection` CLI command):

```bash
pip install -e .
```

## Running Locally

Run CLI help:

```bash
python main.py --help
```

Dataset test mode:

```bash
python main.py \
  --mode test \
  --config-path Code/Config.json \
  --system StudentPC \
  --model-selection 1
```

Single-file mode:

```bash
python main.py \
  --mode single-file \
  --config-path Code/Config.json \
  --system StudentPC \
  --model-selection 1 \
  --path-single-file /path/to/input.nii.gz \
  --output-path /path/to/output/
```

Train mode:

```bash
python main.py \
  --mode train \
  --config-path Code/Config.json \
  --system StudentPC \
  --model-selection 1
```

Legacy executor behavior:

```bash
python Code/Pipeline_executer.py
```

## Environment Variables

Supported overrides include:

- `BLUR_CONFIG_PATH`
- `BLUR_SYSTEM`
- `BLUR_MODEL_SELECTION`
- `BLUR_ENABLE_MULTI_GPU`
- `BLUR_DEVICE_IDS`
- `BLUR_DEFAULT_GPU`
- `BLUR_TENSORBOARD`
- `BLUR_EPOCHS`
- `BLUR_BATCH_SIZE`
- `BLUR_VALIDATION_SPLIT`
- `BLUR_NUM_CLASS_CONFUSION_MATRIX`
- `BLUR_PATH_SINGLE_FILE`
- `BLUR_OUTPUT_PATH`
- `BLUR_CUSTOM_MODEL_PATH`
- `BLUR_TRANSFORM_IMAGES`
- `BLUR_LOG_LEVEL`

## Running the UI

```bash
streamlit run ui/app.py
```

The UI exposes:

- Mode selection (`single-file`, `test`, `train`)
- Config/system/model fields
- Single-file input and output paths
- GPU and training runtime options

## Testing

Run:

```bash
pytest
```

Current automated tests cover:

- Runtime configuration loading and validation
- CSV generation logic
- Class binning utility behavior

## Docker

Build:

```bash
docker build -t blur-detection:latest .
```

Run CLI help:

```bash
docker run --rm blur-detection:latest
```

Run single-file inference (example):

```bash
docker run --rm \
  -v /host/path/to/data:/data \
  blur-detection:latest \
  python main.py \
  --mode single-file \
  --config-path /app/Code/Config.json \
  --path-single-file /data/input.nii.gz \
  --output-path /data/output/
```

## Project Structure

```text
BlurDetection/
├── Code/                      # Legacy core implementation (preserved logic)
├── src/
│   └── blur_detection/        # Production wrapper package
├── tests/                     # Unit tests
├── ui/                        # Streamlit UI
├── docker/                    # Docker notes
├── requirements.txt
├── pyproject.toml
├── Dockerfile
└── README.md
```
