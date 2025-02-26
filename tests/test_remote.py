import os
import time
from datetime import datetime
from pathlib import Path

import pytest

from docketanalyzer_ocr.document import PDFDocument
from docketanalyzer_ocr.remote import RunPodClient
from docketanalyzer_ocr.utils import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    RUNPOD_API_KEY,
    RUNPOD_ENDPOINT_ID,
    S3_BUCKET_NAME,
)


class TestRemote:
    """Tests for remote processing functionality."""
    
    def setup_method(self):
        """Skip tests if credentials are missing."""
        if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
            pytest.skip("RunPod API key or endpoint ID not available")
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY or not S3_BUCKET_NAME:
            pytest.skip("AWS credentials or S3 bucket name not available")
    
    def test_client_initialization(self):
        """Test RunPod client initialization."""
        client = RunPodClient()
        
        assert client.api_key == RUNPOD_API_KEY
        assert client.endpoint_id == RUNPOD_ENDPOINT_ID
        assert client.base_url == f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"
        assert client.headers == {
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json",
        }
    
    def test_client_health_check(self):
        """Test client health check."""
        client = RunPodClient()
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
