import os
import tempfile
from pathlib import Path

import pytest

from docketanalyzer_ocr.utils import load_pdf


class TestUtils:
    """Tests for utility functions."""

    def test_load_pdf_from_bytes(self, sample_pdf_bytes):
        """Test loading a PDF from bytes."""
        pdf_bytes, filename = load_pdf(file=sample_pdf_bytes, filename="test.pdf")

        assert pdf_bytes == sample_pdf_bytes
        assert filename == "test.pdf"

    def test_load_pdf_missing_params(self):
        """Test loading a PDF with missing parameters."""
        with pytest.raises(ValueError):
            load_pdf()  # No file or s3_key provided

    def test_s3_operations(self):
        """Test S3 operations with real AWS resources."""
        # Skip if credentials are missing
        if (
            not os.getenv("AWS_ACCESS_KEY_ID")
            or not os.getenv("AWS_SECRET_ACCESS_KEY")
            or not os.getenv("S3_BUCKET_NAME")
        ):
            pytest.skip("AWS credentials or S3 bucket name not available")

        # Create a test file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"Test S3 file content")
            tmp_path = Path(tmp.name)

        try:
            # Import here to avoid errors if AWS credentials are not set
            from docketanalyzer_ocr.utils import download_from_s3, upload_to_s3

            # Upload to S3
            s3_key = f"test/test_file_{os.urandom(4).hex()}.txt"
            upload_success = upload_to_s3(tmp_path, s3_key, overwrite=True)

            assert upload_success is True

            # Download from S3
            with tempfile.TemporaryDirectory() as tmp_dir:
                download_path = Path(tmp_dir) / "downloaded.txt"
                result_path = download_from_s3(s3_key, download_path)

                assert result_path == download_path
                assert download_path.exists()
                assert download_path.read_bytes() == b"Test S3 file content"

        finally:
            # Clean up
            if tmp_path.exists():
                tmp_path.unlink()
