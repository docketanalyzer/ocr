import os
import shutil
import tempfile
from typing import Optional, Tuple
import urllib
from dotenv import load_dotenv
import boto3
from botocore.client import Config
from pathlib import Path


load_dotenv(override=True)


BASE_DIR = Path(__file__).resolve().parent


RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY')
RUNPOD_ENDPOINT_ID = os.getenv('RUNPOD_ENDPOINT_ID')


AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')


s3_client = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    config=Config(signature_version='s3v4')
)


def upload_to_s3(file_path: str | Path, s3_key: str, overwrite: bool = False) -> bool:
    """
    Upload a file to S3 bucket
    
    Args:
        file_path (str | Path): Local path to the file
        s3_key (str): S3 key (path) where the file will be stored
        overwrite (bool): If True, overwrites existing file
        
    Returns:
        bool: True if upload successful, False otherwise
    """
    try:
        file_path = Path(file_path)
        if not overwrite:
            try:
                s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
                return False
            except:
                pass
                
        s3_client.upload_file(str(file_path), S3_BUCKET_NAME, s3_key)
        return True
    except Exception as e:
        print(f"Error uploading file to S3: {str(e)}")
        return False


def download_from_s3(s3_key: str, local_path: Optional[str | Path] = None) -> Optional[Path]:
    """
    Download a file from S3 bucket
    
    Args:
        s3_key (str): S3 key (path) of the file to download
        local_path (str | Path, optional): Local path where to save the file. 
                                         If None, saves to the same name as s3_key
                                  
    Returns:
        Path: Path to the downloaded file if successful, None otherwise
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


def load_pdf(file: Optional[bytes] = None, s3_key: Optional[str] = None, filename: Optional[str] = None) -> Tuple[Path, str]:
    """
    Load a PDF file either from binary content or S3, placing it in a temporary directory
    
    Args:
        file: PDF file content as bytes
        s3_key: S3 key if the PDF should be fetched from S3
        
    Returns:
        Tuple containing:
        - Path to temporary directory
        - Original filename
        
    Raises:
        ValueError: If neither PDF content nor S3 key is provided
    """
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        if s3_key:
            original_filename = Path(s3_key).name
            pdf_path = temp_dir / "doc.pdf"
            download_from_s3(s3_key, pdf_path)
            file = pdf_path.read_bytes()
            
        elif file is not None:
            original_filename = "uploaded.pdf"
        else:
            raise ValueError("Must provide either PDF content or S3 key")
            
        return file, filename or original_filename
        
    except Exception as e:
        shutil.rmtree(temp_dir)
        raise e


def download_file(url, output_file, chunk_size=1024):
    """
    Downloads a large file from the given URL in chunks.
    
    Args:
        url (str): The URL of the file to download.
        output_file (str): The path where the file will be saved.
        chunk_size (int): The size of each chunk (in bytes). Default is 1 KB.
    """
    try:
        with urllib.request.urlopen(url) as response, open(output_file, 'wb') as out_file:
            # Get the total file size from the headers (if available)
            file_size = int(response.headers.get('content-length', 0))
            print(f"File size: {file_size / (1024 * 1024):.2f} MB")

            # Read the file in chunks
            downloaded = 0
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)

                # Print progress
                progress = (downloaded / file_size) * 100 if file_size else 0
                print(f"Downloaded {downloaded} bytes ({progress:.2f}%)", end='\r')
    except Exception as e:
        print(f"An error occurred: {e}")
