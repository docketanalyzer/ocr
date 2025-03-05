import base64
import json
import logging
import time

import pytest
import requests
import uvicorn
from fastapi.testclient import TestClient

from service import app, jobs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    force=True,  # Force reconfiguration of the root logger
)
logger = logging.getLogger(__name__)

# Ensure logs are displayed during test execution
logging.getLogger().setLevel(logging.INFO)


def log_timing(start_time, message):
    """Log elapsed time with a message."""
    elapsed = time.time() - start_time
    logger.info(f"{message} - Took {elapsed:.2f} seconds")
    return time.time()


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def clear_jobs():
    """Clear the jobs dictionary before and after each test."""
    jobs.clear()
    yield
    jobs.clear()


def run_server():
    """Run the FastAPI server in a separate process.

    This function is used by the service_process fixture to avoid
    pickling the FastAPI app directly.
    """
    import importlib
    import sys

    # Force reload the service module to ensure we get a fresh instance
    if "service" in sys.modules:
        importlib.reload(sys.modules["service"])

    # Import the app from the service module
    from service import app

    # Run the server
    logger.info("Starting FastAPI server")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")


@pytest.fixture(scope="function")
def service_process():
    """Start the FastAPI service in a separate process for testing."""
    # Create a process to run the service using the run_server function
    logger.info("Creating service process")

    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    process = mp.Process(target=run_server)

    # Start the service
    start_time = time.time()
    process.start()
    logger.info(f"Service process started with PID: {process.pid}")

    # Wait for the service to start
    time.sleep(2)
    log_timing(start_time, "Service startup wait completed")

    yield

    # Terminate the service process
    start_time = time.time()
    logger.info(f"Terminating service process with PID: {process.pid}")
    process.terminate()
    process.join()
    log_timing(start_time, "Service process terminated")


class TestOCRService:
    """Tests for the OCR service."""

    def test_service_api_endpoints(self, test_pdf_path, service_process, clear_jobs):
        """Test the service API endpoints directly.

        This test ensures that:
        1. The /run endpoint accepts a job and returns a job ID
        2. The /status endpoint returns the correct job status
        3. The /stream endpoint streams job results
        """
        logger.info("=" * 80)
        logger.info("STARTING TEST: test_service_api_endpoints")
        overall_start = time.time()

        # Read the test PDF file
        start_time = time.time()
        pdf_bytes = test_pdf_path.read_bytes()
        logger.info(f"Test PDF path: {test_pdf_path}, size: {len(pdf_bytes)} bytes")
        log_timing(start_time, "Read PDF file")

        # Base64 encode the PDF bytes for JSON serialization
        start_time = time.time()
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
        log_timing(start_time, "Base64 encode PDF")

        # Submit a job to the service
        start_time = time.time()
        logger.info("Submitting job to service")
        response = requests.post(
            "http://127.0.0.1:8000/run", json={"input": {"file": pdf_base64, "filename": "test.pdf", "batch_size": 1}}
        )
        # Check that the job was submitted successfully
        assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
        job_id = response.json()["id"]
        assert job_id, "Job ID should be returned"
        logger.info(f"Job submitted successfully with ID: {job_id}")
        log_timing(start_time, "Submit job")

        # Wait for the job to start processing (max 30 seconds)
        start_time = time.time()
        max_attempts = 30
        status = "PENDING"
        for attempt in range(max_attempts):
            logger.info(f"Checking job status (attempt {attempt + 1}/{max_attempts})")
            status_response = requests.post(f"http://127.0.0.1:8000/status/{job_id}")
            assert status_response.status_code == 200, f"Status endpoint failed with code {status_response.status_code}"

            status = status_response.json()["status"]
            logger.info(f"Job status: {status}")
            if status != "PENDING":
                break
            time.sleep(1)
        log_timing(start_time, f"Wait for job to start processing (status: {status})")

        # Check that the job status is either IN_PROGRESS or COMPLETED
        assert status in ["IN_PROGRESS", "COMPLETED"], f"Job status should be IN_PROGRESS or COMPLETED, got {status}"

        # Test the stream endpoint
        start_time = time.time()
        logger.info("Testing stream endpoint")
        stream_response = requests.post(f"http://127.0.0.1:8000/stream/{job_id}", stream=True)
        assert stream_response.status_code == 200, f"Stream endpoint failed with code {stream_response.status_code}"

        # Check that we get at least one result from the stream
        results = []
        page_count = 0
        logger.info("Reading complete stream response")
        for line in stream_response.iter_lines():
            if line:
                result = json.loads(line.decode("utf-8"))
                logger.info(f"Received stream result: {result.keys()}")
                results.append(result)
                if "stream" in result:
                    for item in result["stream"]:
                        if "output" in item and "page" in item["output"]:
                            page_count += 1
                            logger.info(f"Processed page {page_count}")

        assert len(results) > 0, "Should get at least one result from the stream"
        log_timing(start_time, f"Stream test completed with {len(results)} results")

        log_timing(overall_start, "COMPLETED TEST: test_service_api_endpoints")
