# Docker Notes

The production image is defined in the repository root `Dockerfile`.

Build:

```bash
docker build -t blur-detection:latest .
```

Run CLI help:

```bash
docker run --rm blur-detection:latest
```

Run single-file mode (example):

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
