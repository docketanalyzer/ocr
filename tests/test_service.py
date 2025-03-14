import importlib
import multiprocessing
import sys
import time

import pytest
import uvicorn

from docketanalyzer_core import env
from docketanalyzer_ocr import pdf_document
from service import app

from .conftest import compare_docs


def run_server():
    """Run the FastAPI server in a separate process.

    This function is used by the service_process fixture to avoid
    pickling the FastAPI app directly.
    """
    if "service" in sys.modules:
        importlib.reload(sys.modules["service"])

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")


@pytest.fixture(scope="function")
def service_process():
    """Start the FastAPI service in a separate process for testing."""
    multiprocessing.set_start_method("spawn", force=True)
    process = multiprocessing.Process(target=run_server)

    process.start()
    time.sleep(2)

    yield

    time.sleep(0.5)
    process.terminate()
    process.join(timeout=3)

    if process.is_alive():
        print("Process didn't terminate gracefully, forcing kill")
        process.kill()
        process.join(timeout=2)

    assert not process.is_alive(), "Failed to terminate the service process"


def test_service_post(monkeypatch, sample_pdf_path, sample_pdf_json, service_process):
    """Test service with post, explicitly overriding endpoint."""
    time.sleep(5)  # Let service start
    # Do this to confirm override
    key_check = bool(env.RUNPOD_API_KEY)
    assert key_check, "RUNPOD_API_KEY is not set"
    key_check = bool(env.RUNPOD_OCR_ENDPOINT_ID)
    assert key_check, "RUNPOD_OCR_ENDPOINT_ID is not set"

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "disabled")
    key_check = env.AWS_ACCESS_KEY_ID == "disabled"
    assert key_check, "AWS_ACCESS_KEY_ID not disabled"

    doc1 = pdf_document(sample_pdf_path, load=sample_pdf_json)
    doc2 = pdf_document(
        sample_pdf_path,
        remote=True,
        use_s3=False,
        endpoint_url="http://localhost:8000/",
    )

    endpoint_check = "localhost" in doc2.remote_client.base_url
    assert endpoint_check, "Not using local endpoint"

    status = doc2.remote_client.get_health()
    assert status["workers"]["total"] == 1, "No active workers found"

    for _ in doc2.stream():
        pass

    assert len(doc1) == len(doc2), "Document lengths do not match"
    assert compare_docs(doc1, doc2), "Processed documents are not equal"


def test_service_s3(monkeypatch, sample_pdf_path, sample_pdf_json, service_process):
    """Test service with s3, with implicit local endpoint."""
    time.sleep(5)  # Let service start
    monkeypatch.setenv("RUNPOD_OCR_ENDPOINT_ID", "")
    key_check = bool(env.RUNPOD_OCR_ENDPOINT_ID)
    assert not key_check, "RUNPOD_OCR_ENDPOINT_ID is not disabled"

    doc1 = pdf_document(sample_pdf_path, load=sample_pdf_json)
    doc2 = pdf_document(
        sample_pdf_path,
        remote=True,
        use_s3=True,
        endpoint_url="http://localhost:8000/",
    )

    endpoint_check = "localhost" in doc2.remote_client.base_url
    assert endpoint_check, "Not using local endpoint"
    assert doc2.s3_available, "S3 availability check failed"

    status = doc2.remote_client.get_health()
    assert status["workers"]["total"] == 1, "No active workers found"

    doc2.process()

    assert len(doc1) == len(doc2), "Document lengths do not match"
    assert compare_docs(doc1, doc2), "Processed documents are not equal"
