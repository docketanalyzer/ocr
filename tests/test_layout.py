from unittest.mock import MagicMock, patch

from docketanalyzer_ocr.layout import boxes_overlap, load_model, merge_boxes, merge_overlapping_blocks, predict_layout


class TestLayoutHelpers:
    """Tests for layout helper functions."""

    def test_boxes_overlap_true(self):
        """Test detection of overlapping boxes."""
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

    def test_boxes_overlap_false(self):
        """Test detection of non-overlapping boxes."""
        # Non-overlapping boxes
        box1 = (10, 10, 50, 50)
        box2 = (60, 60, 100, 100)

        assert boxes_overlap(box1, box2) is False

    def test_merge_boxes(self):
        """Test merging of bounding boxes."""
        box1 = (10, 10, 50, 50)
        box2 = (30, 30, 70, 70)

        merged = merge_boxes(box1, box2)

        assert merged == (10, 10, 70, 70)

        # Test with non-overlapping boxes
        box3 = (100, 100, 150, 150)
        merged2 = merge_boxes(box1, box3)

        assert merged2 == (10, 10, 150, 150)

    def test_merge_overlapping_blocks_empty(self):
        """Test merging with empty input."""
        result = merge_overlapping_blocks([])
        assert result == []

    def test_merge_overlapping_blocks(self):
        """Test merging overlapping layout blocks."""
        blocks = [
            {"type": "text", "bbox": (10, 10, 50, 50)},
            {"type": "text", "bbox": (30, 30, 70, 70)},
            {"type": "figure", "bbox": (100, 100, 150, 150)},
            {"type": "title", "bbox": (20, 20, 40, 40)},  # Title has higher priority
        ]

        result = merge_overlapping_blocks(blocks)

        assert len(result) == 2  # Should merge the first three blocks

        # First block should be the merged one with type 'title' (highest priority)
        assert result[0]["type"] == "title"
        assert result[0]["bbox"] == (10, 10, 70, 70)

        # Second block should be the figure
        assert result[1]["type"] == "figure"
        assert result[1]["bbox"] == (100, 100, 150, 150)


class TestLayoutModel:
    """Tests for layout model functions."""

    @patch("docketanalyzer_ocr.layout.YOLOv10")
    @patch("docketanalyzer_ocr.layout.torch.cuda.is_available")
    def test_load_model_cpu(self, mock_cuda_available, mock_yolo):
        """Test loading the layout model on CPU."""
        # Mock cuda not available
        mock_cuda_available.return_value = False
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # Reset global model to ensure it's loaded
        import docketanalyzer_ocr.layout

        docketanalyzer_ocr.layout.LAYOUT_MODEL = None

        model, device = load_model()

        assert model is mock_model
        assert device == "cpu"
        mock_yolo.assert_called_once()

    @patch("docketanalyzer_ocr.layout.YOLOv10")
    @patch("docketanalyzer_ocr.layout.torch.cuda.is_available")
    def test_load_model_gpu(self, mock_cuda_available, mock_yolo):
        """Test loading the layout model on GPU."""
        # Mock cuda available
        mock_cuda_available.return_value = True
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # Reset global model to ensure it's loaded
        import docketanalyzer_ocr.layout

        docketanalyzer_ocr.layout.LAYOUT_MODEL = None

        model, device = load_model()

        assert model is mock_model
        assert device == "cuda"
        mock_yolo.assert_called_once()

    @patch("docketanalyzer_ocr.layout.load_model")
    def test_predict_layout(self, mock_load_model, sample_image):
        """Test predicting layout on images."""
        # Create a batch of sample images
        images = [sample_image]  # Test with a single image

        # Mock layout model and results
        mock_model = MagicMock()

        # Create a mock prediction result that matches the expected format
        mock_pred = MagicMock()

        # Create tensor-like objects for xyxy, conf, and cls
        xyxy_tensor = MagicMock()
        xyxy_tensor.__iter__.return_value = [10, 10, 100, 50]

        conf_tensor = MagicMock()

        cls_tensor = MagicMock()
        cls_tensor.item.return_value = 0  # Class 0 = title

        # Create a second set for the second prediction
        xyxy_tensor2 = MagicMock()
        xyxy_tensor2.__iter__.return_value = [10, 60, 100, 100]

        cls_tensor2 = MagicMock()
        cls_tensor2.item.return_value = 1  # Class 1 = text

        # Set up the prediction to yield our mocked tensors
        mock_pred.__iter__.return_value = [
            [*xyxy_tensor, conf_tensor, cls_tensor],
            [*xyxy_tensor2, conf_tensor, cls_tensor2],
        ]

        # Set up the model to return our mock prediction
        mock_model.return_value = [mock_pred]

        mock_load_model.return_value = (mock_model, "cpu")

        results = predict_layout(images)

        assert isinstance(results, list)
        assert len(results) == 1  # One result for one image

        # Check first image results
        assert len(results[0]) == 2
        assert results[0][0]["type"] == "title"  # class 0 maps to 'title'
        assert results[0][0]["bbox"] == (10, 10, 100, 50)
        assert results[0][1]["type"] == "text"  # class 1 maps to 'text'

        # Verify the mock was called correctly
        mock_model.assert_called_once()
