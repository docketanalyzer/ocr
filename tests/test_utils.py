import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from docketanalyzer_ocr.utils import download_from_s3, load_pdf, upload_to_s3


class TestS3Functions:
    """Tests for S3-related utility functions."""

    @patch("docketanalyzer_ocr.utils.s3_client")
    def test_upload_to_s3_success(self, mock_s3_client, mock_s3_file):
        """Test successful file upload to S3."""
        # Setup
        mock_s3_client.head_object.side_effect = Exception("Not found")

        # Test
        result = upload_to_s3(mock_s3_file, "test/file.txt")

        # Assertions
        assert result is True
        mock_s3_client.upload_file.assert_called_once_with(
            str(mock_s3_file), os.getenv("S3_BUCKET_NAME", "default-bucket"), "test/file.txt"
        )

    @patch("docketanalyzer_ocr.utils.s3_client")
    def test_upload_to_s3_file_exists(self, mock_s3_client, mock_s3_file):
        """Test upload when file already exists in S3."""
        # Setup - file exists
        mock_s3_client.head_object.return_value = {"ContentLength": 100}

        # Test
        result = upload_to_s3(mock_s3_file, "test/file.txt", overwrite=False)

        # Assertions
        assert result is False
        mock_s3_client.upload_file.assert_not_called()

    @patch("docketanalyzer_ocr.utils.s3_client")
    def test_upload_to_s3_overwrite(self, mock_s3_client, mock_s3_file):
        """Test overwriting an existing file in S3."""
        # Test
        result = upload_to_s3(mock_s3_file, "test/file.txt", overwrite=True)

        # Assertions
        assert result is True
        mock_s3_client.upload_file.assert_called_once()
        mock_s3_client.head_object.assert_not_called()

    @patch("docketanalyzer_ocr.utils.s3_client")
    def test_download_from_s3_success(self, mock_s3_client, tmp_path):
        """Test successful file download from S3."""
        # Setup
        s3_key = "test/file.txt"
        local_path = tmp_path / "downloaded.txt"

        # Test
        result = download_from_s3(s3_key, local_path)

        # Assertions
        assert result == local_path
        mock_s3_client.download_file.assert_called_once_with(
            os.getenv("S3_BUCKET_NAME", "default-bucket"), s3_key, str(local_path)
        )

    @patch("docketanalyzer_ocr.utils.s3_client")
    def test_download_from_s3_default_path(self, mock_s3_client):
        """Test download with default local path."""
        # Setup
        s3_key = "test/file.txt"

        # Test
        result = download_from_s3(s3_key)

        # Assertions
        assert result.name == "file.txt"
        mock_s3_client.download_file.assert_called_once_with(
            os.getenv("S3_BUCKET_NAME", "default-bucket"), s3_key, "file.txt"
        )

    @patch("docketanalyzer_ocr.utils.s3_client")
    def test_download_from_s3_error(self, mock_s3_client):
        """Test download with an error."""
        # Setup
        mock_s3_client.download_file.side_effect = Exception("Download error")

        # Test
        result = download_from_s3("test/file.txt")

        # Assertions
        assert result is None


class TestPDFLoading:
    """Tests for PDF loading functions."""

    def test_load_pdf_from_bytes(self, sample_pdf_bytes):
        """Test loading a PDF from bytes."""
        # Test
        pdf_bytes, filename = load_pdf(file=sample_pdf_bytes, filename="test.pdf")

        # Assertions
        assert pdf_bytes == sample_pdf_bytes
        assert filename == "test.pdf"

    @patch("docketanalyzer_ocr.utils.download_from_s3")
    @patch("docketanalyzer_ocr.utils.tempfile.NamedTemporaryFile")
    def test_load_pdf_from_s3(self, mock_temp_file, mock_download, tmp_path, sample_pdf_bytes):
        """Test loading a PDF from S3."""
        # Setup
        s3_key = "test/document.pdf"
        local_path = tmp_path / "document.pdf"

        # Write sample PDF bytes to the temp file
        local_path.write_bytes(sample_pdf_bytes)

        # Mock the temporary file
        mock_temp = MagicMock()
        mock_temp.name = str(local_path)
        mock_temp.__enter__.return_value = mock_temp
        mock_temp_file.return_value = mock_temp

        # Mock the download to return our local path
        mock_download.return_value = local_path

        # Test
        with patch("docketanalyzer_ocr.utils.Path.read_bytes", return_value=sample_pdf_bytes):
            pdf_bytes, filename = load_pdf(s3_key=s3_key)

        # Assertions
        assert pdf_bytes == sample_pdf_bytes  # Should match our sample bytes
        assert filename == "document.pdf"
        mock_download.assert_called_once_with(s3_key, Path(mock_temp.name))

    def test_load_pdf_missing_params(self):
        """Test loading a PDF with missing parameters."""
        with pytest.raises(ValueError):
            load_pdf()  # No file or s3_key provided
