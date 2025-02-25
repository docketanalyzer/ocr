from paddleocr import PaddleOCR
import torch


OCR_MODEL = None


def load_model():
    global OCR_MODEL

    device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    if OCR_MODEL is None:
        OCR_MODEL = PaddleOCR(
            lang='en', use_gpu=device == 'cuda',
            gpu_mem=5000, precision='bf16',
            show_log=False,
        )
    
    return OCR_MODEL, device


def extract_ocr_text(image):
    model, _ = load_model()
    
    result = model.ocr(image, cls=False)
    data = []
    for idx in range(len(result)):
        res = result[idx]
        if res:
            for line in res:
                data.append({
                    'bbox': line[0][0] + line[0][2],
                    'content': line[1][0],
                })
    return data
