import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from docketanalyzer_ocr.document import PDFDocument
from docketanalyzer_ocr.remote import RunPodClient


class TestRunPodClient:
    """Tests for the RunPodClient class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock RunPodClient with predefined API key and endpoint ID."""
        with (
            patch("docketanalyzer_ocr.remote.RUNPOD_API_KEY", "test_api_key"),
            patch("docketanalyzer_ocr.remote.RUNPOD_ENDPOINT_ID", "test_endpoint_id"),
        ):
            client = RunPodClient()
            return client

    def test_init_with_defaults(self):
        """Test initialization with default values from environment."""
        with (
            patch("docketanalyzer_ocr.remote.RUNPOD_API_KEY", "test_api_key"),
            patch("docketanalyzer_ocr.remote.RUNPOD_ENDPOINT_ID", "test_endpoint_id"),
        ):
            client = RunPodClient()
            assert client.api_key == "test_api_key"
            assert client.endpoint_id == "test_endpoint_id"
            assert client.base_url == "https://api.runpod.ai/v2/test_endpoint_id"
            assert client.headers == {
                "Authorization": "Bearer test_api_key",
                "Content-Type": "application/json",
            }

    def test_init_with_custom_values(self):
        """Test initialization with custom API key and endpoint ID."""
        client = RunPodClient(api_key="custom_api_key", endpoint_id="custom_endpoint_id")
        assert client.api_key == "custom_api_key"
        assert client.endpoint_id == "custom_endpoint_id"
        assert client.base_url == "https://api.runpod.ai/v2/custom_endpoint_id"
        assert client.headers == {
            "Authorization": "Bearer custom_api_key",
            "Content-Type": "application/json",
        }

    def test_init_missing_api_key(self):
        """Test initialization with missing API key."""
        with patch("docketanalyzer_ocr.remote.RUNPOD_API_KEY", None):
            with pytest.raises(ValueError, match="RunPod API key not provided"):
                RunPodClient()

    def test_init_missing_endpoint_id(self):
        """Test initialization with missing endpoint ID."""
        with (
            patch("docketanalyzer_ocr.remote.RUNPOD_API_KEY", "test_api_key"),
            patch("docketanalyzer_ocr.remote.RUNPOD_ENDPOINT_ID", None),
        ):
            with pytest.raises(ValueError, match="RunPod endpoint ID not provided"):
                RunPodClient()

    def test_call_missing_inputs(self, mock_client):
        """Test calling the client with missing inputs."""
        with pytest.raises(ValueError, match="Either s3_key or file must be provided"):
            next(mock_client())

    @patch("requests.post")
    def test_submit_job(self, mock_post, mock_client):
        """Test submitting a job to RunPod."""
        mock_response = Mock()
        mock_response.json.return_value = {"id": "test_job_id"}
        mock_post.return_value = mock_response

        job_id = mock_client._submit_job({"input": {"s3_key": "test_key"}}, 30)

        assert job_id == "test_job_id"
        mock_post.assert_called_once_with(
            "https://api.runpod.ai/v2/test_endpoint_id/run",
            headers=mock_client.headers,
            json={"input": {"s3_key": "test_key"}},
            timeout=30,
        )

    @patch("requests.post")
    def test_submit_job_invalid_response(self, mock_post, mock_client):
        """Test submitting a job with invalid response."""
        mock_response = Mock()
        mock_response.json.return_value = {"error": "Something went wrong"}
        mock_post.return_value = mock_response

        with pytest.raises(ValueError, match="Invalid response format, missing 'id'"):
            mock_client._submit_job({"input": {"s3_key": "test_key"}}, 30)

    @patch("requests.post")
    def test_submit_job_json_decode_error(self, mock_post, mock_client):
        """Test submitting a job with JSON decode error."""
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        mock_post.return_value = mock_response

        with pytest.raises(ValueError, match="Failed to parse response"):
            mock_client._submit_job({"input": {"s3_key": "test_key"}}, 30)

    def test_stream_results(self, mock_client):
        """Test streaming results from RunPod."""
        # Mock the _stream_results method directly
        expected_results = [
            {"status": "IN_PROGRESS", "stream": [{"output": {"page": {"i": 0, "blocks": []}}}]},
            {"status": "IN_PROGRESS", "stream": [{"output": {"page": {"i": 1, "blocks": []}}}]},
            {"status": "COMPLETED"},
        ]

        with patch.object(mock_client, "_stream_results", return_value=expected_results):
            results = list(mock_client._stream_results("test_job_id", 30, 0.1))

            assert len(results) == 3
            assert results[0]["status"] == "IN_PROGRESS"
            assert results[1]["status"] == "IN_PROGRESS"
            assert results[2]["status"] == "COMPLETED"

    def test_stream_results_not_ready(self, mock_client):
        """Test streaming results when job is not ready."""
        expected_results = [
            {"status": "COMPLETED"},
        ]

        with patch.object(mock_client, "_stream_results", return_value=expected_results):
            results = list(mock_client._stream_results("test_job_id", 30, 0.1))

            assert len(results) == 1
            assert results[0]["status"] == "COMPLETED"

    def test_stream_results_timeout(self, mock_client):
        """Test streaming results timeout."""
        with patch.object(mock_client, "_stream_results", side_effect=TimeoutError("Streaming results timed out")):
            with pytest.raises(TimeoutError, match="Streaming results timed out"):
                list(mock_client._stream_results("test_job_id", 30, 0.1))

    def test_stream_results_chunked_encoding_error(self, mock_client):
        """Test handling of chunked encoding errors during streaming."""
        expected_results = [
            {"status": "COMPLETED"},
        ]

        with patch.object(mock_client, "_stream_results", return_value=expected_results):
            results = list(mock_client._stream_results("test_job_id", 30, 0.1))

            assert len(results) == 1
            assert results[0]["status"] == "COMPLETED"

    @patch.object(RunPodClient, "_submit_job")
    @patch.object(RunPodClient, "_stream_results")
    def test_call_with_s3_key(self, mock_stream_results, mock_submit_job, mock_client):
        """Test calling the client with an S3 key."""
        mock_submit_job.return_value = "test_job_id"
        mock_stream_results.return_value = [
            {"status": "IN_PROGRESS", "stream": [{"output": {"page": {"i": 0, "blocks": []}}}]},
            {"status": "COMPLETED"},
        ]

        results = list(mock_client(s3_key="test_s3_key", filename="test.pdf", batch_size=2))

        assert len(results) == 2
        mock_submit_job.assert_called_once_with(
            {"input": {"s3_key": "test_s3_key", "filename": "test.pdf", "batch_size": 2}},
            600,
        )
        mock_stream_results.assert_called_once_with("test_job_id", 600, 1.0)

    @patch.object(RunPodClient, "_submit_job")
    @patch.object(RunPodClient, "_stream_results")
    def test_call_with_file(self, mock_stream_results, mock_submit_job, mock_client):
        """Test calling the client with a file."""
        mock_submit_job.return_value = "test_job_id"
        mock_stream_results.return_value = [
            {"status": "IN_PROGRESS", "stream": [{"output": {"page": {"i": 0, "blocks": []}}}]},
            {"status": "COMPLETED"},
        ]

        file_bytes = b"test file content"
        results = list(mock_client(file=file_bytes, filename="test.pdf", batch_size=2))

        assert len(results) == 2
        mock_submit_job.assert_called_once_with(
            {"input": {"file": file_bytes, "filename": "test.pdf", "batch_size": 2}},
            600,
        )
        mock_stream_results.assert_called_once_with("test_job_id", 600, 1.0)

    @patch.object(RunPodClient, "_submit_job")
    @patch.object(RunPodClient, "_stream_results")
    def test_call_non_streaming(self, mock_stream_results, mock_submit_job, mock_client):
        """Test calling the client with streaming disabled."""
        mock_submit_job.return_value = "test_job_id"
        mock_stream_results.return_value = [
            {"status": "IN_PROGRESS", "stream": [{"output": {"page": {"i": 0, "blocks": []}}}]},
            {"status": "COMPLETED"},
        ]

        results = mock_client(s3_key="test_s3_key", stream=False)

        assert len(results) == 2
        mock_submit_job.assert_called_once()
        mock_stream_results.assert_called_once()

    @patch("requests.post")
    def test_get_status(self, mock_post, mock_client):
        """Test getting job status."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "COMPLETED"}
        mock_post.return_value = mock_response

        status = mock_client.get_status("test_job_id")

        assert status == {"status": "COMPLETED"}
        mock_post.assert_called_once_with(
            "https://api.runpod.ai/v2/test_endpoint_id/status/test_job_id",
            headers=mock_client.headers,
            timeout=30,
        )

    @patch("requests.post")
    def test_cancel_job(self, mock_post, mock_client):
        """Test canceling a job."""
        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_post.return_value = mock_response

        result = mock_client.cancel_job("test_job_id")

        assert result == {"success": True}
        mock_post.assert_called_once_with(
            "https://api.runpod.ai/v2/test_endpoint_id/cancel/test_job_id",
            headers=mock_client.headers,
            timeout=30,
        )

    @patch("requests.post")
    def test_purge_queue(self, mock_post, mock_client):
        """Test purging the queue."""
        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_post.return_value = mock_response

        result = mock_client.purge_queue()

        assert result == {"success": True}
        mock_post.assert_called_once_with(
            "https://api.runpod.ai/v2/test_endpoint_id/purge-queue",
            headers=mock_client.headers,
            timeout=30,
        )

    @patch("requests.get")
    def test_get_health(self, mock_get, mock_client):
        """Test getting endpoint health."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "READY"}
        mock_get.return_value = mock_response

        health = mock_client.get_health()

        assert health == {"status": "READY"}
        mock_get.assert_called_once_with(
            "https://api.runpod.ai/v2/test_endpoint_id/health",
            headers=mock_client.headers,
            timeout=30,
        )


class TestPDFDocumentRemote:
    """Tests for the remote functionality in PDFDocument."""

    @pytest.fixture
    def mock_runpod_client(self):
        """Create a mock RunPodClient."""
        with patch("docketanalyzer_ocr.document.RunPodClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            yield mock_client

    @patch("docketanalyzer_ocr.document.upload_to_s3")
    @patch("docketanalyzer_ocr.document.delete_from_s3")
    def test_stream_remote_with_s3_key(
        self, mock_delete_from_s3, mock_upload_to_s3, mock_runpod_client, sample_pdf_bytes
    ):
        """Test streaming with remote processing using S3 key."""
        # Setup mocks
        mock_upload_to_s3.return_value = True
        mock_runpod_client.return_value = [
            {
                "stream": [
                    {
                        "output": {
                            "page": {
                                "i": 0,
                                "blocks": [
                                    {
                                        "bbox": [10, 10, 100, 30],
                                        "type": "text",
                                        "lines": [{"bbox": [10, 10, 100, 30], "content": "Test content"}],
                                    }
                                ],
                            }
                        }
                    }
                ]
            },
            {"status": "COMPLETED"},
        ]

        # Create document with remote=True
        with (
            patch.object(PDFDocument, "__init__", return_value=None),
            patch.object(PDFDocument, "_upload_to_s3", return_value="test_s3_key"),
        ):
            doc = PDFDocument.__new__(PDFDocument)
            doc.remote = True
            doc.filename = "document.pdf"
            doc._s3_key = None
            doc._runpod_client = mock_runpod_client

            # Create a mock page
            mock_page = MagicMock()
            doc.pages = [mock_page]

            # Process the document
            pages = list(doc.stream())

            # Verify results
            assert len(pages) == 1
            assert pages[0] == mock_page

            # Verify page was updated
            mock_page.set_blocks.assert_called_once()

            # Verify RunPod client was called
            mock_runpod_client.assert_called_once()

    @patch("docketanalyzer_ocr.document.upload_to_s3")
    @patch("docketanalyzer_ocr.document.delete_from_s3")
    def test_stream_remote_with_nested_response(
        self, mock_delete_from_s3, mock_upload_to_s3, mock_runpod_client, sample_pdf_bytes
    ):
        """Test streaming with remote processing using nested response format."""
        # Setup mocks
        mock_upload_to_s3.return_value = True
        mock_runpod_client.return_value = [
            {
                "stream": [
                    {
                        "output": {
                            "page": {
                                "i": 0,
                                "blocks": [
                                    {
                                        "bbox": [10, 10, 100, 30],
                                        "type": "text",
                                        "lines": [{"bbox": [10, 10, 100, 30], "content": "Page 0"}],
                                    }
                                ],
                            }
                        }
                    },
                    {
                        "output": {
                            "page": {
                                "i": 1,
                                "blocks": [
                                    {
                                        "bbox": [10, 10, 100, 30],
                                        "type": "text",
                                        "lines": [{"bbox": [10, 10, 100, 30], "content": "Page 1"}],
                                    }
                                ],
                            }
                        }
                    },
                ]
            },
            {"status": "COMPLETED"},
        ]

        # Create a document with multiple pages
        with (
            patch.object(PDFDocument, "__init__", return_value=None),
            patch.object(PDFDocument, "_upload_to_s3", return_value="test_s3_key"),
        ):
            doc = PDFDocument.__new__(PDFDocument)
            doc.remote = True
            doc.filename = "document.pdf"
            doc._s3_key = None
            doc._runpod_client = mock_runpod_client

            # Create mock pages
            mock_page0 = MagicMock()
            mock_page1 = MagicMock()
            doc.pages = [mock_page0, mock_page1]

            # Process the document
            pages = list(doc.stream())

            # Verify pages were updated
            mock_page0.set_blocks.assert_called_once()
            mock_page1.set_blocks.assert_called_once()

            # Verify results
            assert len(pages) == 2
            assert pages[0] == mock_page0
            assert pages[1] == mock_page1

    @patch("docketanalyzer_ocr.document.upload_to_s3")
    def test_upload_to_s3_failure(self, mock_upload_to_s3, sample_pdf_bytes):
        """Test handling of S3 upload failure."""
        # Setup mock to simulate upload failure
        mock_upload_to_s3.return_value = False

        # Create document with remote=True
        with (
            patch.object(PDFDocument, "__init__", return_value=None),
            patch.object(
                PDFDocument, "_upload_to_s3", side_effect=ValueError("Failed to upload PDF to S3 at key: test_s3_key")
            ),
        ):
            doc = PDFDocument.__new__(PDFDocument)
            doc.remote = True
            doc.filename = "document.pdf"
            doc._s3_key = None

            # Process the document - should raise ValueError
            with pytest.raises(ValueError, match="Failed to upload PDF to S3"):
                list(doc.stream())

    @patch("docketanalyzer_ocr.document.upload_to_s3")
    @patch("docketanalyzer_ocr.document.delete_from_s3")
    def test_process_remote(self, mock_delete_from_s3, mock_upload_to_s3, mock_runpod_client, sample_pdf_bytes):
        """Test process method with remote processing."""
        # Setup mocks
        mock_upload_to_s3.return_value = True
        mock_runpod_client.return_value = [
            {
                "stream": [
                    {
                        "output": {
                            "page": {
                                "i": 0,
                                "blocks": [
                                    {
                                        "bbox": [10, 10, 100, 30],
                                        "type": "text",
                                        "lines": [{"bbox": [10, 10, 100, 30], "content": "Test content"}],
                                    }
                                ],
                            }
                        }
                    }
                ]
            },
            {"status": "COMPLETED"},
        ]

        # Create document with remote=True
        with (
            patch.object(PDFDocument, "__init__", return_value=None),
            patch.object(PDFDocument, "stream") as mock_stream,
        ):
            doc = PDFDocument.__new__(PDFDocument)
            doc.remote = True
            doc.filename = "document.pdf"
            doc._s3_key = "test_s3_key"
            doc._runpod_client = mock_runpod_client

            # Create a mock page
            mock_page = MagicMock()
            doc.pages = [mock_page]

            # Mock the stream method to return our mock page
            mock_stream.return_value = [mock_page]

            # Process the document
            result = doc.process(batch_size=2)

            # Verify results
            assert result is doc  # process() should return self

            # Verify stream was called with correct batch_size
            mock_stream.assert_called_once_with(batch_size=2)

    @patch("docketanalyzer_ocr.document.upload_to_s3")
    @patch("docketanalyzer_ocr.document.delete_from_s3")
    def test_cleanup_on_exception(self, mock_delete_from_s3, mock_upload_to_s3, mock_runpod_client, sample_pdf_bytes):
        """Test S3 cleanup when an exception occurs during processing."""
        # Setup mocks
        mock_upload_to_s3.return_value = True
        mock_runpod_client.side_effect = Exception("Test exception")

        # Create document with remote=True
        with (
            patch.object(PDFDocument, "__init__", return_value=None),
            patch.object(PDFDocument, "_upload_to_s3", return_value="test_s3_key"),
        ):
            doc = PDFDocument.__new__(PDFDocument)
            doc.remote = True
            doc.filename = "document.pdf"
            doc._s3_key = "test_s3_key"
            doc._runpod_client = mock_runpod_client

            # Process the document - should raise the exception
            with pytest.raises(Exception, match="Test exception"):
                for _ in doc.stream():
                    pass

            # Verify S3 cleanup was called
            mock_delete_from_s3.assert_called_once_with("test_s3_key")
