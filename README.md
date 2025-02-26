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
# Run all tests
pytest
```

## Code Quality

```bash
# Format code and fix linting issues
ruff format . && ruff check --fix .
```

## CI

This project uses GitHub Actions for continuous integration:

- **Linting with Auto-fix**: Checks and automatically fixes code formatting/linting issues
- **Testing**: Runs tests and generates coverage reports

The workflow runs on pushes and PRs to main/master branches. If linting issues are found, they're automatically fixed, committed, and the tests run on the fixed code.

See `.github/workflows/ci.yml` for configuration details.

### GitHub Secrets

Required secrets for CI:
- `RUNPOD_API_KEY`
- `RUNPOD_ENDPOINT_ID`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `S3_BUCKET_NAME`
- `S3_ENDPOINT_URL`

Add these in your repository's Settings > Secrets and variables > Actions.

