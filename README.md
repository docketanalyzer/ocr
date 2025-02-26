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

- **Linting**: Automatically checks code formatting and linting with Ruff
- **Testing**: Runs all tests with pytest and generates coverage reports

The CI workflow runs on every push to main/master branches and on pull requests.

You can view the workflow configuration in the `.github/workflows/ci.yml` file.

### Setting Up GitHub Secrets

The CI workflow requires the following environment variables to be set as GitHub Secrets:

1. Go to your GitHub repository
2. Navigate to Settings > Secrets and variables > Actions
3. Click on "New repository secret"
4. Add each of the following secrets (see `env.example` for the required variables):
   - `RUNPOD_API_KEY`
   - `RUNPOD_ENDPOINT_ID`
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `S3_BUCKET_NAME`
   - `S3_ENDPOINT_URL`

These secrets will be securely used by the GitHub Actions workflow when running tests.

