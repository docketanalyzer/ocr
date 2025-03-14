from pathlib import Path

import pytest
import simplejson as json


@pytest.fixture
def fixture_dir():
    """Path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_pdf_path(fixture_dir):
    """Path to the sample PDF file for testing."""
    return fixture_dir / "document.pdf"


@pytest.fixture
def sample_pdf_json(fixture_dir):
    """Path to the sample PDF file for testing."""
    return json.loads((fixture_dir / "document.json").read_text())


def compare_docs(doc1, doc2):
    """Compare two processed PDF documents for equality."""
    for page1, page2 in zip(doc1, doc2, strict=False):
        for block1, block2 in zip(page1, page2, strict=False):
            for line1, line2 in zip(block1, block2, strict=False):
                if line1.text != line2.text:
                    return False
            if block1.block_type != block2.block_type:
                return False
    return True
