import fitz

from docketanalyzer_ocr.document import page_to_image
from docketanalyzer_ocr.layout import (
    boxes_overlap,
    load_model,
    merge_boxes,
    merge_overlapping_blocks,
    predict_layout,
)


class TestLayout:
    """Tests for layout analysis functionality."""

    def test_boxes_overlap(self):
        """Test detection of overlapping and non-overlapping boxes."""
        # Overlapping boxes
        box1 = (10, 10, 50, 50)
        box2 = (30, 30, 70, 70)
        assert boxes_overlap(box1, box2) is True
        
        # Box contained within another
        box3 = (20, 20, 40, 40)
        assert boxes_overlap(box1, box3) is True
        
        # Edge touching
        box4 = (50, 10, 90, 50)
        assert boxes_overlap(box1, box4) is True
        
        # Non-overlapping boxes
        box5 = (60, 60, 100, 100)
        assert boxes_overlap(box1, box5) is False

    def test_merge_boxes(self):
        """Test merging of bounding boxes."""
        # Overlapping boxes
        box1 = (10, 10, 50, 50)
        box2 = (30, 30, 70, 70)
        merged = merge_boxes(box1, box2)
        assert merged == (10, 10, 70, 70)
        
        # Non-overlapping boxes
        box3 = (100, 100, 150, 150)
        merged2 = merge_boxes(box1, box3)
        assert merged2 == (10, 10, 150, 150)

    def test_merge_overlapping_blocks(self):
        """Test merging overlapping layout blocks."""
        # Test with empty input
        assert merge_overlapping_blocks([]) == []
        
        # Test with blocks to merge
        blocks = [
            {"type": "text", "bbox": (10, 10, 50, 50)},
            {"type": "text", "bbox": (30, 30, 70, 70)},
            {"type": "figure", "bbox": (100, 100, 150, 150)},
            {"type": "title", "bbox": (20, 20, 40, 40)},  # Title has higher priority
        ]
        
        result = merge_overlapping_blocks(blocks)
        
        assert len(result) == 2  # Should merge the overlapping blocks
        
        # First block should be the merged one with type 'title' (highest priority)
        assert result[0]["type"] == "title"
        assert result[0]["bbox"] == (10, 10, 70, 70)
        
        # Second block should be the figure
        assert result[1]["type"] == "figure"
        assert result[1]["bbox"] == (100, 100, 150, 150)

    def test_model_loading(self):
        """Test loading the layout model."""
        # Reset global model to ensure it's loaded
        import docketanalyzer_ocr.layout
        docketanalyzer_ocr.layout.LAYOUT_MODEL = None
        
        # Load the model
        model, device = load_model()
        
        # Verify model was loaded
        assert model is not None
        assert device in ["cpu", "cuda"]
        
        # Verify the model is cached
        model2, _ = load_model()
        assert model2 is model  # Should be the same instance

    def test_layout_prediction(self, test_pdf_path):
        """Test layout prediction on a real PDF."""
        # Open the PDF and get the first page
        doc = fitz.open(test_pdf_path)
        page = doc[0]
        
        # Convert the page to an image
        image = page_to_image(page, dpi=150)
        
        # Run layout analysis
        layout_results = predict_layout([image])
        
        # Verify layout results
        assert isinstance(layout_results, list)
        assert len(layout_results) > 0
        
        # Check structure of layout results
        for block in layout_results[0]:
            assert "bbox" in block
            assert "type" in block
            assert isinstance(block["bbox"], tuple) or isinstance(block["bbox"], list)
            assert len(block["bbox"]) == 4
            assert block["type"] in ["text", "title", "list", "table", "figure"]
        
        doc.close()
