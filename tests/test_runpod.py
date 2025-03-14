from docketanalyzer_core import env
from docketanalyzer_ocr import pdf_document

from .conftest import compare_docs


def test_runpod_post(monkeypatch, sample_pdf_path, sample_pdf_json):
    """Test process method, loading pdf from path."""
    key_check = bool(env.RUNPOD_API_KEY)
    assert key_check, "RUNPOD_API_KEY is not set"
    key_check = bool(env.RUNPOD_OCR_ENDPOINT_ID)
    assert key_check, "RUNPOD_OCR_ENDPOINT_ID is not set"

    doc1 = pdf_document(sample_pdf_path, load=sample_pdf_json)
    doc2 = pdf_document(sample_pdf_path, remote=True, use_s3=False)

    endpoint_check = env.RUNPOD_OCR_ENDPOINT_ID in doc2.remote_client.base_url
    assert endpoint_check, "Endpoint ID not found in remote client URL"

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "disabled")
    key_check = env.AWS_ACCESS_KEY_ID == "disabled"
    assert key_check, "AWS_ACCESS_KEY_ID not disabled"

    for _ in doc2.stream(batch_size=2):
        pass

    assert len(doc1) == len(doc2), "Document lengths do not match"
    assert compare_docs(doc1, doc2), "Processed documents are not equal"


def test_runpod_s3(sample_pdf_path, sample_pdf_json):
    """Test stream method, loading pdf from bytes."""
    key_check = bool(env.AWS_S3_BUCKET_NAME)
    assert key_check, "AWS_S3_BUCKET_NAME is not set"
    key_check = bool(env.AWS_ACCESS_KEY_ID)
    assert key_check, "AWS_ACCESS_KEY_ID is not set"
    key_check = bool(env.AWS_SECRET_ACCESS_KEY)
    assert key_check, "AWS_SECRET_ACCESS_KEY is not set"
    key_check = bool(env.AWS_S3_ENDPOINT_URL)
    assert key_check, "AWS_S3_ENDPOINT_URL is not set"

    doc1 = pdf_document(sample_pdf_path, load=sample_pdf_json)
    doc2 = pdf_document(sample_pdf_path, remote=True)

    assert doc2.s3_available, "S3 availability check failed"
    assert doc2.use_s3, "S3 setting didnt default to true"

    for _ in doc2.stream():
        pass

    assert len(doc1) == len(doc2), "Document lengths do not match"
    assert compare_docs(doc1, doc2), "Processed documents are not equal"
