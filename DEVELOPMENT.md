# Development

## Install

```
pip install '.[dev]'
```

## Test

```
pytest -vv
```

## Format

```
ruff format . && ruff check --fix .
```

## Build and Push to PyPi

```
python -m docketanalyzer_core dev build
python -m docketanalyzer_core dev build --push
```

## Docker Container

Build and run:

```
DOCKER_BUILDKIT=1 docker build -t docketanalyzer-ocr .
docker run --gpus all -p 8000:8000 docketanalyzer-ocr
```

Push:

```
docker tag docketanalyzer-ocr nadahlberg/docketanalyzer-ocr:latest
docker push nadahlberg/docketanalyzer-ocr:latest
```
