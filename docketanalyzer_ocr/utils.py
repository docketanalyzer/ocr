import tempfile
from pathlib import Path

from docketanalyzer_core import load_s3


def load_pdf(
    file: bytes | None = None,
    s3_key: str | None = None,
    filename: str | None = None,
) -> tuple[bytes, str]:
    """Loads a PDF file either from binary content or S3.

    This function handles loading a PDF file from either binary or from an S3 bucket.
    It returns the binary content of the PDF file and the filename.

    Args:
        file: PDF file content as bytes. Defaults to None.
        s3_key: S3 key if the PDF should be fetched from S3. Defaults to None.
        filename: Optional filename to use. If not provided, will be derived
            from s3_key or set to a default.

    Returns:
        tuple[bytes, str]: A tuple containing:
            - The binary content of the PDF file
            - The filename of the PDF

    Raises:
        ValueError: If neither file nor s3_key is provided.
    """
    if file is None and s3_key is None:
        raise ValueError("Either file or s3_key must be provided")

    if filename is None:
        filename = Path(s3_key).name if s3_key else "document.pdf"

    # If we already have the file content, just return it
    if file is not None:
        return file, filename

    # Otherwise, we need to download from S3
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_path = Path(temp_file.name)
        load_s3().download(s3_key, str(temp_path))
        return temp_path.read_bytes(), filename
