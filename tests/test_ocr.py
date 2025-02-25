from unittest.mock import MagicMock, patch

import numpy as np

from docketanalyzer_ocr.ocr import extract_ocr_text, load_model


class TestOCR:
    """Tests for OCR functionality."""

    @patch("docketanalyzer_ocr.ocr.PaddleOCR")
    @patch("docketanalyzer_ocr.ocr.torch.cuda.is_available")
    def test_load_model_cpu(self, mock_cuda_available, mock_paddle_ocr):
        """Test loading the OCR model on CPU."""
        # Mock cuda not available
        mock_cuda_available.return_value = False
        mock_model = MagicMock()
        mock_paddle_ocr.return_value = mock_model

        # Reset global model to ensure it's loaded
        import docketanalyzer_ocr.ocr

        docketanalyzer_ocr.ocr.OCR_MODEL = None

        model, device = load_model()

        assert model is mock_model
        assert device == "cpu"
        mock_paddle_ocr.assert_called_once_with(
            lang="en",
            use_gpu=False,
            gpu_mem=5000,
            precision="bf16",
            show_log=False,
        )

    @patch("docketanalyzer_ocr.ocr.PaddleOCR")
    @patch("docketanalyzer_ocr.ocr.torch.cuda.is_available")
    def test_load_model_gpu(self, mock_cuda_available, mock_paddle_ocr):
        """Test loading the OCR model on GPU."""
        # Mock cuda available
        mock_cuda_available.return_value = True
        mock_model = MagicMock()
        mock_paddle_ocr.return_value = mock_model

        # Reset global model to ensure it's loaded
        import docketanalyzer_ocr.ocr

        docketanalyzer_ocr.ocr.OCR_MODEL = None

        model, device = load_model()

        assert model is mock_model
        assert device == "cuda"
        mock_paddle_ocr.assert_called_once_with(
            lang="en",
            use_gpu=True,
            gpu_mem=5000,
            precision="bf16",
            show_log=False,
        )

    @patch("docketanalyzer_ocr.ocr.load_model")
    def test_extract_ocr_text(self, mock_load_model, sample_image):
        """Test extracting text using OCR."""
        # Mock OCR model and results
        mock_model = MagicMock()
        mock_model.ocr.return_value = [
            [
                [[[10, 20], [100, 20], [100, 40], [10, 40]], ["Sample text", 0.99]],
                [[[10, 60], [150, 60], [150, 80], [10, 80]], ["Another line", 0.95]],
            ]
        ]
        mock_load_model.return_value = (mock_model, "cpu")

        result = extract_ocr_text(sample_image)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["bbox"] == [10, 20, 100, 40]
        assert result[0]["content"] == "Sample text"
        assert result[1]["bbox"] == [10, 60, 150, 80]
        assert result[1]["content"] == "Another line"

        mock_model.ocr.assert_called_once_with(sample_image, cls=False)

    @patch("docketanalyzer_ocr.ocr.load_model")
    def test_extract_ocr_text_empty_result(self, mock_load_model):
        """Test extracting text with empty OCR results."""
        # Mock OCR model with empty results
        mock_model = MagicMock()
        mock_model.ocr.return_value = [[]]
        mock_load_model.return_value = (mock_model, "cpu")

        result = extract_ocr_text(np.zeros((100, 100, 3), dtype=np.uint8))

        assert isinstance(result, list)
        assert len(result) == 0
