from datetime import datetime

from docketanalyzer_ocr.document import PDFDocument
from docketanalyzer_ocr.remote import RemoteClient
from docketanalyzer_ocr.utils import (
    RUNPOD_API_KEY,
    RUNPOD_OCR_ENDPOINT_ID,
)


class TestRemote:
    """Tests for remote processing functionality."""

    def test_client_initialization(self):
        """Test remote client initialization."""
        client = RemoteClient()

        assert client.api_key == RUNPOD_API_KEY
        assert client.base_url == f"https://api.runpod.ai/v2/{RUNPOD_OCR_ENDPOINT_ID}"
        assert client.headers == {
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json",
        }

    def test_client_initialization_with_custom_url(self):
        """Test remote client initialization with custom endpoint URL."""
        custom_url = "http://example.com/api"
        client = RemoteClient(endpoint_url=custom_url)

        assert client.base_url == custom_url
        assert client.headers["Content-Type"] == "application/json"

    def test_client_health_check(self):
        """Test client health check."""
        client = RemoteClient()
        health = client.get_health()

        assert isinstance(health, dict)
        assert "workers" in health
        assert "jobs" in health

    def test_remote_processing(self, test_pdf_path):
        """Test remote document processing."""
        # Load the test PDF
        test_pdf_bytes = test_pdf_path.read_bytes()

        # Process with remote=True
        start = datetime.now()
        doc = PDFDocument(test_pdf_bytes, remote=True)

        # Test streaming API
        processed_pages = []
        for page in doc.stream():
            processed_pages.append(page)
            assert len(page.text) > 0
            assert len(page.blocks) > 0

        # Verify all pages were processed
        assert len(processed_pages) == len(doc.pages)

        # The test should take a reasonable amount of time for real streaming
        elapsed_time = (datetime.now() - start).total_seconds()
        assert elapsed_time > 1.0, "Test completed too quickly for real streaming"
