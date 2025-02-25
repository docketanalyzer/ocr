import os
import tempfile
from pathlib import Path
from typing import Optional, Union

import boto3
from botocore.client import Config
from dotenv import load_dotenv

load_dotenv(override=True)


BASE_DIR = Path(__file__).resolve().parent


RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")


AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")


s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    config=Config(signature_version="s3v4"),
)


def upload_to_s3(file_path: Union[str, Path], s3_key: str, overwrite: bool = False) -> bool:
    """Uploads a file to an S3 bucket.

    Args:
        file_path: Local path to the file to upload.
        s3_key: S3 key (path) where the file will be stored.
        overwrite: If True, overwrites existing file. Defaults to False.

    Returns:
        bool: True if upload was successful, False otherwise.
    """
    try:
        file_path = Path(file_path)
        if not overwrite:
            try:
                s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
                return False
            except Exception:
                pass

        s3_client.upload_file(str(file_path), S3_BUCKET_NAME, s3_key)
        return True
    except Exception as e:
        print(f"Error uploading file to S3: {str(e)}")
        return False


def download_from_s3(s3_key: str, local_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Downloads a file from an S3 bucket.

    Args:
        s3_key: S3 key (path) of the file to download.
        local_path: Local path where to save the file. If None, saves to the same name as s3_key.

    Returns:
        Optional[Path]: Path to the downloaded file if successful, None otherwise.
    """
    try:
        if local_path is None:
            local_path = Path(Path(s3_key).name)
        else:
            local_path = Path(local_path)

        s3_client.download_file(S3_BUCKET_NAME, s3_key, str(local_path))
        return local_path
    except Exception as e:
        print(f"Error downloading file from S3: {str(e)}")
        return None


def load_pdf(
    file: Optional[bytes] = None,
    s3_key: Optional[str] = None,
    filename: Optional[str] = None,
) -> tuple[bytes, str]:
    """Loads a PDF file either from binary content or S3.

    This function handles loading a PDF file from either binary content or from an S3 bucket.
    It returns the binary content of the PDF file and the filename.

    Args:
        file: PDF file content as bytes. Defaults to None.
        s3_key: S3 key if the PDF should be fetched from S3. Defaults to None.
        filename: Optional filename to use. If not provided, will be derived from s3_key or set to a default.

    Returns:
        tuple[bytes, str]: A tuple containing:
            - The binary content of the PDF file
            - The filename of the PDF

    Raises:
        ValueError: If neither file nor s3_key is provided.
    """
    if file is None and s3_key is None:
        raise ValueError("Either file or s3_key must be provided")

    if filename is None:
        if s3_key:
            filename = Path(s3_key).name
        else:
            filename = "document.pdf"

    # If we already have the file content, just return it
    if file is not None:
        return file, filename

    # Otherwise, we need to download from S3
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_path = Path(temp_file.name)
        download_from_s3(s3_key, temp_path)
        return temp_path.read_bytes(), filename
