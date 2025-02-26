# Docket Analyzer OCR

## Installation

```bash
# Install standard version
pip install .

# Install with GPU support
pip install '.[gpu]'
```

## Testing

```bash
pytest --cov=docketanalyzer_ocr tests/ --cov-report=xml --cov-branch --junitxml=junit.xml -o junit_family=legacy
```

## Code Quality

```bash
ruff format . && ruff check --fix .
```

## CI

This project uses GitHub Actions for continuous integration:

- **Testing**: Runs tests and generates coverage reports first
- **Linting with Auto-fix**: Only if tests pass, checks and automatically fixes code formatting/linting issues

The workflow runs on pushes and PRs to main/master branches. Auto-fixes are only applied after tests have passed, ensuring test failures remain visible.

See `.github/workflows/ci.yml` for configuration details.

### Code Coverage


[![codecov](https://codecov.io/gh/docketanalyzer/ocr/graph/badge.svg?token=XRATNOME24)](https://codecov.io/gh/docketanalyzer/ocr)


### GitHub Secrets

Required secrets for CI:
- `RUNPOD_API_KEY`
- `RUNPOD_ENDPOINT_ID`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `S3_BUCKET_NAME`
- `S3_ENDPOINT_URL`
- `CODECOV_TOKEN` (only for private repositories)

Add these in your repository's Settings > Secrets and variables > Actions.

