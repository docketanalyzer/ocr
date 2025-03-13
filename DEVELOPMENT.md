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
python -m docketanalyzer_core build
python -m docketanalyzer_core build --push
```
