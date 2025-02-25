# OCR Project

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

## Continuous Integration

This project uses GitHub Actions for continuous integration:

- **Linting**: Automatically checks code formatting and linting with Ruff
- **Testing**: Runs all tests with pytest and generates coverage reports

The CI workflow runs on every push to main/master branches and on pull requests.

You can view the workflow configuration in the `.github/workflows/ci.yml` file.

