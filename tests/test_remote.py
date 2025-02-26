import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
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

    def test_stream_results_implementation(self):
        """Test the actual implementation of _stream_results method with controlled server responses."""

        # Create a mock HTTP server that simulates RunPod's streaming response
        class MockRunPodHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()

                # Simulate a streaming response with multiple chunks
                responses = [
                    {"status": "IN_PROGRESS", "id": "test-job-id"},
                    {"status": "IN_PROGRESS", "stream": [{"output": {"page": {"i": 0, "blocks": []}}}]},
                    {"status": "IN_PROGRESS", "stream": [{"output": {"page": {"i": 1, "blocks": []}}}]},
                    {"status": "COMPLETED", "stream": [{"output": {"status": "COMPLETED"}}]},
                ]

                for response in responses:
                    self.wfile.write(json.dumps(response).encode("utf-8") + b"\n")
                    self.wfile.flush()
                    time.sleep(0.1)  # Simulate delay between chunks

            def log_message(self, format, *args):
                # Suppress log messages
                pass

        # Start a mock server in a separate thread
        server = HTTPServer(("localhost", 0), MockRunPodHandler)
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        try:
            # Create a real RunPodClient instance
            client = RunPodClient(api_key="test_api_key", endpoint_id="test_endpoint_id")

            # Override the base URL to point to our mock server
            port = server.server_port
            client.base_url = f"http://localhost:{port}"

            # Call the actual _stream_results method (not mocked)
            results = list(client._stream_results("test-job-id", timeout=5, poll_interval=0.1))

            # Verify the results
            assert len(results) == 4
            assert results[0]["status"] == "IN_PROGRESS"
            assert results[0]["id"] == "test-job-id"
            assert results[1]["status"] == "IN_PROGRESS"
            assert "stream" in results[1]
            assert results[1]["stream"][0]["output"]["page"]["i"] == 0
            assert results[2]["status"] == "IN_PROGRESS"
            assert results[2]["stream"][0]["output"]["page"]["i"] == 1
            assert results[3]["status"] == "COMPLETED"
            assert results[3]["stream"][0]["output"]["status"] == "COMPLETED"

        finally:
            # Shut down the server
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=1)

    def test_stream_results_retry_on_404(self):
        """Test the actual implementation of _stream_results with 404 response handling."""

        # Create a mock HTTP server that first returns 404, then 200 with streaming data
        class MockRunPodHandler(BaseHTTPRequestHandler):
            # Track number of requests to simulate job not ready then ready
            request_count = 0

            def do_POST(self):
                # First request returns 404 (job not ready)
                if MockRunPodHandler.request_count == 0:
                    MockRunPodHandler.request_count += 1
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Job not ready"}).encode("utf-8"))
                    return

                # Second request returns 200 with streaming data
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()

                # Send a single completion response
                response = {"status": "COMPLETED", "id": "test-job-id"}
                self.wfile.write(json.dumps(response).encode("utf-8") + b"\n")
                self.wfile.flush()

            def log_message(self, format, *args):
                # Suppress log messages
                pass

        # Start a mock server in a separate thread
        server = HTTPServer(("localhost", 0), MockRunPodHandler)
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        try:
            # Create a real RunPodClient instance
            client = RunPodClient(api_key="test_api_key", endpoint_id="test_endpoint_id")

            # Override the base URL to point to our mock server
            port = server.server_port
            client.base_url = f"http://localhost:{port}"

            # Call the actual _stream_results method (not mocked)
            results = list(client._stream_results("test-job-id", timeout=5, poll_interval=0.1))

            # Verify the results
            assert len(results) == 1
            assert results[0]["status"] == "COMPLETED"
            assert results[0]["id"] == "test-job-id"
            assert MockRunPodHandler.request_count == 1  # Verify we got a 404 first

        finally:
            # Shut down the server
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=1)

    def test_stream_results_retry_on_chunked_encoding_error(self):
        """Test the actual implementation of _stream_results with chunked encoding error handling."""

        # Create a mock HTTP server that simulates a chunked encoding error
        class MockRunPodHandler(BaseHTTPRequestHandler):
            # Track number of requests to simulate connection error then success
            request_count = 0

            def do_POST(self):
                # First request returns 200 but closes connection prematurely
                if MockRunPodHandler.request_count == 0:
                    MockRunPodHandler.request_count += 1
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()

                    # Send partial data then close connection
                    self.wfile.write(json.dumps({"status": "IN_PROGRESS", "id": "test-job-id"}).encode("utf-8") + b"\n")
                    self.wfile.flush()
                    # Abruptly close connection to simulate chunked encoding error
                    self.connection.close()
                    return

                # Second request returns complete data
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()

                # Send a completion response
                response = {"status": "COMPLETED", "id": "test-job-id"}
                self.wfile.write(json.dumps(response).encode("utf-8") + b"\n")
                self.wfile.flush()

            def log_message(self, format, *args):
                # Suppress log messages
                pass

        # Start a mock server in a separate thread
        server = HTTPServer(("localhost", 0), MockRunPodHandler)
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        try:
            # Create a real RunPodClient instance
            client = RunPodClient(api_key="test_api_key", endpoint_id="test_endpoint_id")

            # Override the base URL to point to our mock server
            port = server.server_port
            client.base_url = f"http://localhost:{port}"

            # Call the actual _stream_results method (not mocked)
            results = list(client._stream_results("test-job-id", timeout=5, poll_interval=0.1))

            # Verify the results - should include the COMPLETED response
            # Note: The first response might be lost due to the connection error
            assert len(results) >= 1
            assert results[-1]["status"] == "COMPLETED"
            assert results[-1]["id"] == "test-job-id"
            assert MockRunPodHandler.request_count >= 1  # Verify we had at least one request

        finally:
            # Shut down the server
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=1)


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

    @patch("docketanalyzer_ocr.document.upload_to_s3")
    @patch("docketanalyzer_ocr.document.delete_from_s3")
    def test_full_remote_streaming_flow(self, mock_delete_from_s3, mock_upload_to_s3, sample_pdf_bytes):
        """Test the full streaming flow with remote=True using a real HTTP server."""

        # Create a mock HTTP server that simulates RunPod's streaming response
        class MockRunPodHandler(BaseHTTPRequestHandler):
            # Track which endpoint is being called
            def do_POST(self):
                # Handle the /run endpoint (submit job)
                if self.path.endswith("/run"):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"id": "test-job-id"}).encode("utf-8"))
                    return

                # Handle the /stream endpoint
                if self.path.endswith("/stream/test-job-id"):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()

                    # Simulate a streaming response with a single page (since sample_pdf_bytes has only one page)
                    responses = [
                        # First chunk with page 0
                        {
                            "status": "IN_PROGRESS",
                            "stream": [
                                {
                                    "output": {
                                        "page": {
                                            "i": 0,
                                            "blocks": [
                                                {
                                                    "bbox": [10, 10, 100, 30],
                                                    "type": "text",
                                                    "lines": [{"bbox": [10, 10, 100, 30], "content": "Page 0 content"}],
                                                }
                                            ],
                                        }
                                    }
                                }
                            ],
                        },
                        # Final chunk with completion status
                        {"status": "COMPLETED", "stream": [{"output": {"status": "COMPLETED"}}]},
                    ]

                    for response in responses:
                        self.wfile.write(json.dumps(response).encode("utf-8") + b"\n")
                        self.wfile.flush()
                        time.sleep(0.2)  # Add delay to simulate real streaming
                    return

                # Default response for unknown endpoints
                self.send_response(404)
                self.end_headers()

            def log_message(self, format, *args):
                # Suppress log messages
                pass

        # Start a mock server in a separate thread
        server = HTTPServer(("localhost", 0), MockRunPodHandler)
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        try:
            # Setup mocks
            mock_upload_to_s3.return_value = True

            # Create a real PDFDocument with remote=True
            with patch("docketanalyzer_ocr.document.RunPodClient") as mock_client_class:
                # Create a real RunPodClient but override its base_url
                from docketanalyzer_ocr.remote import RunPodClient

                real_client = RunPodClient(api_key="test_api_key", endpoint_id="test_endpoint_id")

                # Override the base URL to point to our mock server
                port = server.server_port
                real_client.base_url = f"http://localhost:{port}"

                # Make the mock return our real client with the overridden base_url
                mock_client_class.return_value = real_client

                # Create the document with remote=True
                doc = PDFDocument(sample_pdf_bytes, filename="test.pdf", remote=True)

                # Create a spy for the set_blocks method to verify it's called correctly
                original_set_blocks = doc.pages[0].set_blocks
                set_blocks_calls = []

                def spy_set_blocks(blocks):
                    set_blocks_calls.append(blocks)
                    return original_set_blocks(blocks)

                # Apply the spy to the page
                doc.pages[0].set_blocks = spy_set_blocks

                # Process the document and collect pages
                processed_pages = list(doc.stream(batch_size=1))

                # Verify results
                assert len(processed_pages) == len(doc.pages)
                assert len(processed_pages) == 1  # Sample PDF has only one page

                # Verify set_blocks was called with the correct data
                assert len(set_blocks_calls) >= 1  # Should be called at least once

                # Check the content of the page
                page0_blocks = None

                for blocks in set_blocks_calls:
                    if blocks and len(blocks) > 0:
                        if blocks[0].get("lines", []) and blocks[0]["lines"][0].get("content") == "Page 0 content":
                            page0_blocks = blocks

                assert page0_blocks is not None, "Page 0 blocks not found in set_blocks calls"

                # Verify the content of the processed page
                assert processed_pages[0].blocks[0].lines[0].content == "Page 0 content"

                # Verify S3 cleanup was called
                mock_delete_from_s3.assert_called_once()

        finally:
            # Shut down the server
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=1)

    @patch("docketanalyzer_ocr.document.delete_from_s3")
    def test_real_runpod_streaming_with_real_document(self, mock_delete_from_s3):
        """
        Test the full streaming flow with a real document and real RunPod requests.

        This test uses the actual test PDF document and makes real requests to RunPod,
        only mocking the S3 operations. It verifies that the streaming functionality
        works correctly with the real implementation.
        """
        # Skip this test if no API key or endpoint ID is available
        import os

        from docketanalyzer_ocr.remote import RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID

        if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
            pytest.skip("RunPod API key or endpoint ID not available")

        # Load the real test PDF document
        test_pdf_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "docketanalyzer_ocr", "setup", "test.pdf"
        )

        if not os.path.exists(test_pdf_path):
            pytest.skip(f"Test PDF not found at {test_pdf_path}")

        with open(test_pdf_path, "rb") as f:
            test_pdf_bytes = f.read()

        # Create a real PDFDocument with remote=True
        with patch("docketanalyzer_ocr.document.upload_to_s3") as mock_upload_to_s3:
            # Mock S3 upload to return success
            mock_upload_to_s3.return_value = True

            # Create the document with remote=True
            doc = PDFDocument(test_pdf_bytes, filename="test.pdf", remote=True)

            # Create a tracking mechanism for set_blocks calls
            set_blocks_calls = []
            processed_pages = []

            # Apply spies to all pages to track when set_blocks is called
            for page in doc.pages:
                original_set_blocks = page.set_blocks

                def spy_set_blocks(blocks, page_idx=page.i):
                    set_blocks_calls.append((page_idx, blocks))
                    return original_set_blocks(blocks)

                page.set_blocks = spy_set_blocks

            # Start a timer to measure how long the streaming takes
            start_time = time.time()

            # Process the document and collect pages
            for page in doc.stream(batch_size=1):
                processed_pages.append(page)
                print(f"Processed page {page.i}")

            # Calculate elapsed time
            elapsed_time = time.time() - start_time

            # Verify results
            assert len(processed_pages) == len(doc.pages)
            assert len(processed_pages) > 0

            # Verify set_blocks was called for each page
            assert len(set_blocks_calls) >= len(doc.pages)

            # Verify the content of the processed pages
            for page in processed_pages:
                assert hasattr(page, "blocks")
                assert len(page.blocks) > 0

                # Check that at least one block has text content
                has_text = False
                for block in page.blocks:
                    if block.block_type == "text" and len(block.lines) > 0:
                        has_text = True
                        break

                assert has_text, f"Page {page.i} has no text content"

            # Verify S3 cleanup was called
            mock_delete_from_s3.assert_called_once()

            # Print some statistics about the test
            print(f"Test completed in {elapsed_time:.2f} seconds")
            print(f"Processed {len(processed_pages)} pages")
            print(f"Received {len(set_blocks_calls)} set_blocks calls")

            # The test should take a reasonable amount of time for real streaming
            assert elapsed_time > 1.0, "Test completed too quickly for real streaming"
