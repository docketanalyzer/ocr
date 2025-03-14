import atexit
import base64
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

FORCE_GPU = int(os.getenv("FORCE_GPU", 0))
SCRIPT_PATH = Path(__file__).resolve()
VENV_SCRIPT_PATH = (
    Path.home() / ".cache" / "docketanalyzer" / "ocr" / "venv" / SCRIPT_PATH.name
)
OCR_MODEL = None


class OCRService:
    """Service for running PaddleOCR in a separate process.

    We do gymnastics to avoid the Python 3.10 requirement for the PaddleOCR package.
    """

    def __init__(self, device: str | None = None):
        """Initialize the OCR service.

        Args:
            device: The device to use for OCR processing. Defaults to 'cuda' if
                available, otherwise 'cpu'.
        """
        import torch

        self.device = device or (
            "cpu" if not (torch.cuda.is_available() or FORCE_GPU) else "cuda"
        )

    def process_image(self, image: np.array) -> list[dict]:
        """Extracts text from an image using OCR.

        This function processes an image with the PaddleOCR model to extract text
        and bounding boxes for each detected text line.

        Args:
            image: The input image. Can be a file path, bytes, or a numpy array.

        Returns:
            list[dict]: A list of dictionaries, each containing:
                - 'bbox': The bounding box coordinates [x1, y1, x2, y2]
                - 'content': The extracted text content
        """
        global OCR_MODEL
        if OCR_MODEL is None:
            print("Loading OCR model...")
            from paddleocr import PaddleOCR

            OCR_MODEL = PaddleOCR(
                lang="en",
                use_gpu=self.device == "cuda",
                gpu_mem=5000,
                precision="bf16",
                show_log=True,
            )
            print("OCR model loaded.")

        result = OCR_MODEL.ocr(image, cls=False)
        data = []
        for idx in range(len(result)):
            res = result[idx]
            if res:
                for line in res:
                    data.append(
                        {
                            "bbox": line[0][0] + line[0][2],
                            "content": line[1][0],
                        }
                    )
        return data

    def run(self):
        """Main service loop listening for image input."""
        while True:
            try:
                input_line = sys.stdin.readline().strip()
                if not input_line:
                    continue

                request = json.loads(input_line)
                if "image" not in request:
                    continue

                print("Received image", flush=True)
                image = np.frombuffer(
                    base64.b64decode(request["image"]),
                    dtype=request["dtype"],
                ).reshape(request["shape"])

                data = self.process_image(image)
                print("OCR RESULT:" + json.dumps(data), flush=True)

            except Exception as e:
                print(f"ERROR: {e!s}", flush=True)


class OCRServiceClient:
    """Client for interacting with the OCR service.

    This class manages the subprocess for the OCR service and provides methods
    to send images for processing and receive results.
    """

    def __init__(self, verbose: bool = False):
        """Spawns the OCR service as a subprocess.

        Args:
            verbose: If True, prints debug information to stdout.
            Defaults to False.
        """
        self._process = None
        self.verbose = verbose
        self._lock = threading.Lock()

    def install(self):
        """Installs the PaddleOCR package in a virtual environment.

        This function checks for the existence of a virtual environment and
        the PaddleOCR package. If not found, it creates a new virtual environment
        and installs the necessary packages, including PaddleOCR and PyTorch.
        """
        import torch

        if (
            VENV_SCRIPT_PATH.exists()
            and VENV_SCRIPT_PATH.read_text() != SCRIPT_PATH.read_text()
        ):
            shutil.rmtree(VENV_SCRIPT_PATH.parent)
            VENV_SCRIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not VENV_SCRIPT_PATH.exists():
            print("Creating a virtual environment and installing PaddleOCR...")
            subprocess.check_call(
                ["uv", "venv", VENV_SCRIPT_PATH.parent, "--python", "3.10"]
            )
            VENV_SCRIPT_PATH.write_text(SCRIPT_PATH.read_text())
            venv_python = VENV_SCRIPT_PATH.parent / "bin" / "python"
            cmd = [
                "uv",
                "pip",
                "install",
                "--python",
                str(venv_python),
                "torch",
                "paddleocr",
                "setuptools",
            ]
            if torch.cuda.is_available() or FORCE_GPU:
                cmd.append("paddlepaddle-gpu==2.6.2")
            elif platform.system() == "Darwin":
                cmd.append("paddlepaddle==0.0.0")
                cmd.append("-f")
                cmd.append("https://www.paddlepaddle.org.cn/whl/mac/cpu/develop.html")
            else:
                cmd.append("paddlepaddle")

            subprocess.check_call(cmd)
            print("Installation complete. Downloading models...")

    @property
    def process(self) -> subprocess.Popen:
        """Returns the subprocess for the OCR service.

        If the subprocess is not running, it starts a new one and registers
        cleanup handlers to terminate the process on exit or signal.

        Returns:
            subprocess.Popen: The running OCR service process.

        Raises:
            RuntimeError: If the OCR service is not running or has failed.
        """
        if self._process is None:
            print("Starting OCR process...")
            self.install()
            venv_python = VENV_SCRIPT_PATH.parent / "bin" / "python"

            # Ensure process dies with main script
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
                preexec_fn = None
            else:
                creationflags = 0
                preexec_fn = os.setsid

            self._process = subprocess.Popen(
                [str(venv_python), str(VENV_SCRIPT_PATH)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
                preexec_fn=preexec_fn,
                creationflags=creationflags,
            )

            atexit.register(self.stop)
            signal.signal(signal.SIGTERM, self.stop)
            signal.signal(signal.SIGINT, self.stop)
        return self._process

    def process_image(self, image: np.array) -> list[dict]:
        """Sends an image to the OCR service and retrieves the results."""
        with self._lock:
            if self.process.poll() is not None:
                print("OCR Service terminated unexpectedly. Restarting...")
                self._process = None
                if self.process.poll() is not None:
                    raise RuntimeError("Failed to restart OCR Service")
                print("OCR Service restarted successfully")

            request = {
                "image": base64.b64encode(image.tobytes()).decode("utf-8"),
                "dtype": str(image.dtype),
                "shape": image.shape,
            }

            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()

            start = datetime.now()
            while (datetime.now() - start).total_seconds() < 200:
                response = self.process.stdout.readline().strip()
                if response and self.verbose:
                    print(response)
                if response.startswith("OCR RESULT:"):
                    return json.loads(response[11:])
                if response.startswith("ERROR:"):
                    raise RuntimeError(response[6:])
            raise TimeoutError("OCR service timed out")

    def stop(self, *args: Any):
        """Terminates the OCR service."""
        if self._process is not None:
            self._process.terminate()
            self._process.wait()
            self._process = None


OCR_CLIENT = OCRServiceClient()


def extract_ocr_text(image: str | bytes | Any) -> list[dict]:
    """Extracts text from an image using the OCR service.

    This function sends an image to the OCR service for processing and returns
    the extracted text and bounding boxes.

    Args:
        image: The input image. Can be a file path, bytes, or a numpy array.

    Returns:
        list[dict]: A list of dictionaries, each containing:
            - 'bbox': The bounding box coordinates [x1, y1, x2, y2]
            - 'content': The extracted text content
    """
    return OCR_CLIENT.process_image(image)


if __name__ == "__main__":
    OCRService().run()
