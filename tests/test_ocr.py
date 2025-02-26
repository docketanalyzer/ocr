import time
from pathlib import Path

import fitz
import numpy as np

from docketanalyzer_ocr.document import page_to_image
from docketanalyzer_ocr.ocr import extract_ocr_text, load_model


class TestOCR:
    """Tests for OCR functionality."""

    def test_model_loading(self):
        """Test loading the OCR model."""
        # Reset global model to ensure it's loaded
        import docketanalyzer_ocr.ocr
        docketanalyzer_ocr.ocr.OCR_MODEL = None
        
        # Load the model
        model, device = load_model()
        
        # Verify model was loaded
        assert model is not None
        assert device in ["cpu", "cuda"]
        
        # Verify the model is cached
        model2, _ = load_model()
        assert model2 is model  # Should be the same instance
    
    def test_real_ocr_extraction(self, test_pdf_path):
        """Test actual OCR extraction on a real PDF document."""
        # Open the PDF and get the first page
        doc = fitz.open(test_pdf_path)
        page = doc[0]
        
        # Convert the page to an image
        image = page_to_image(page, dpi=150)  # Lower DPI for faster processing
        
        # Run OCR on the image
        ocr_start = time.time()
        ocr_results = extract_ocr_text(image)
        ocr_time = time.time() - ocr_start
        
        # Verify OCR results
        assert isinstance(ocr_results, list)
        assert len(ocr_results) > 0
        
        # Check structure of OCR results
        for result in ocr_results:
            assert "bbox" in result
            assert "content" in result
            assert isinstance(result["bbox"], list)
            assert len(result["bbox"]) == 4
            assert isinstance(result["content"], str)
            
        # Verify OCR actually ran (should take some time)
        assert ocr_time > 0.1
        
        doc.close()
