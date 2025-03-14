from docketanalyzer_ocr import pdf_document

from .conftest import compare_docs


def test_load(sample_pdf_path, sample_pdf_json):
    """Test loading a PDF document from serialized data."""
    # Load from path
    sample_pdf_json_path = sample_pdf_path.with_suffix(".json")
    doc1 = pdf_document(sample_pdf_path, load=sample_pdf_json_path)

    # Load from dict
    doc2 = pdf_document(sample_pdf_path, load=sample_pdf_json)

    assert len(doc1) == len(doc2), "Document lengths do not match"
    assert compare_docs(doc1, doc2), "Processed documents are not equal"

    edited_pdf_json = sample_pdf_json.copy()
    edited_pdf_json["pages"][0]["blocks"][0]["lines"][0]["content"] = "Edited text"
    doc3 = pdf_document(sample_pdf_path, load=edited_pdf_json)
    assert not compare_docs(doc1, doc3), "Edited documents shoudl not be equal"


def test_local_process(sample_pdf_path, sample_pdf_json):
    """Test process method, loading pdf from path."""
    # Sample doc for comparison
    doc1 = pdf_document(sample_pdf_path, load=sample_pdf_json)

    # Doc to process from path
    doc2 = pdf_document(sample_pdf_path).process()

    assert len(doc1) == len(doc2), "Document lengths do not match"
    assert compare_docs(doc1, doc2), "Processed documents are not equal"


def test_local_stream(sample_pdf_path, sample_pdf_json):
    """Test stream method, loading pdf from bytes."""
    # Sample doc for comparison
    doc1 = pdf_document(sample_pdf_path, load=sample_pdf_json)

    # Doc to process from path
    doc2 = pdf_document(sample_pdf_path.read_bytes())
    for _ in doc2.stream():
        pass

    assert len(doc1) == len(doc2), "Document lengths do not match"
    assert compare_docs(doc1, doc2), "Processed documents are not equal"
