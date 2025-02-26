import json
import time
from typing import Any, Dict, Generator, List, Optional, Union
import pandas as pd
import requests

from .utils import RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID


class RunPodClient:
    """Client for making API calls to RunPod endpoints.

    This class handles communication with RunPod serverless endpoints, including
    authentication, request formatting, and streaming response handling.
    """

    def __init__(self, api_key: Optional[str] = None, endpoint_id: Optional[str] = None):
        """Initialize the RunPod client.

        Args:
            api_key: RunPod API key. If None, uses RUNPOD_API_KEY from environment.
            endpoint_id: RunPod endpoint ID. If None, uses RUNPOD_ENDPOINT_ID from environment.

        Raises:
            ValueError: If API key or endpoint ID is not provided and not in environment.
        """
        self.api_key = api_key or RUNPOD_API_KEY
        self.endpoint_id = endpoint_id or RUNPOD_ENDPOINT_ID

        if not self.api_key:
            raise ValueError("RunPod API key not provided and not found in environment")
        if not self.endpoint_id:
            raise ValueError("RunPod endpoint ID not provided and not found in environment")

        self.base_url = f"https://api.runpod.ai/v2/{self.endpoint_id}"
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def __call__(
        self,
        s3_key: Optional[str] = None,
        file: Optional[bytes] = None,
        filename: Optional[str] = None,
        batch_size: int = 1,
        stream: bool = True,
        timeout: int = 600,
        poll_interval: float = 1.0,
        **extra_params,
    ) -> Union[List[Dict[str, Any]], Generator[Dict[str, Any], None, None]]:
        """Make a request to the RunPod endpoint.

        Args:
            s3_key: S3 key to the PDF file. Either s3_key or file must be provided.
            file: Binary PDF data. Either s3_key or file must be provided.
            filename: Optional filename for the PDF.
            batch_size: Batch size for processing. Defaults to 1.
            stream: Whether to stream the response. Defaults to True.
            timeout: Request timeout in seconds. Defaults to 600 (10 minutes).
            poll_interval: Interval in seconds between status checks. Defaults to 1.0.
            **extra_params: Additional parameters to include in the input payload.

        Returns:
            If stream=True, returns a generator yielding response chunks.
            If stream=False, returns a list of all response chunks.

        Raises:
            requests.RequestException: If the request fails.
            ValueError: If the response format is invalid or neither s3_key nor file is provided.
            TimeoutError: If the request times out.
        """
        if not s3_key and not file:
            raise ValueError("Either s3_key or file must be provided")

        # Construct the payload
        input_data = {"batch_size": batch_size}

        if s3_key:
            input_data["s3_key"] = s3_key
        if file:
            input_data["file"] = file
        if filename:
            input_data["filename"] = filename

        # Add any extra parameters
        input_data.update(extra_params)

        payload = {"input": input_data}

        # Submit job and get job_id
        job_id = self._submit_job(payload, timeout)

        if stream:
            # Stream results for the job
            return self._stream_results(job_id, timeout, poll_interval)
        else:
            # Accumulate all streaming results and return as a list
            results = []
            for chunk in self._stream_results(job_id, timeout, poll_interval):
                results.append(chunk)
                # Check if this is the final chunk with COMPLETED status
                if chunk.get("status") == "COMPLETED":
                    break
            return results

    def _submit_job(self, payload: Dict[str, Any], timeout: int) -> str:
        """Submit a job to the RunPod endpoint.

        Args:
            payload: The request payload.
            timeout: Request timeout in seconds.

        Returns:
            str: The job ID.

        Raises:
            requests.RequestException: If the request fails.
            ValueError: If the response format is invalid.
        """
        url = f"{self.base_url}/run"

        response = requests.post(url, headers=self.headers, json=payload, timeout=timeout)
        response.raise_for_status()

        try:
            result = response.json()
            if "id" not in result:
                raise ValueError(f"Invalid response format, missing 'id': {result}")
            return result["id"]
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse response: {e}")

    def _stream_results(self, job_id: str, timeout: int, poll_interval: float) -> Generator[Dict[str, Any], None, None]:
        """Stream results from a job.

        Args:
            job_id: The job ID.
            timeout: Maximum time to wait for results in seconds.
            poll_interval: Interval between status checks in seconds.

        Yields:
            Dict[str, Any]: Each chunk of the streaming response.

        Raises:
            requests.RequestException: If the request fails.
            ValueError: If the response format is invalid.
            TimeoutError: If the request times out.
        """
        url = f"{self.base_url}/stream/{job_id}"
        start_time = time.time()
        completed = False

        while not completed and time.time() - start_time < timeout:
            try:
                with requests.post(url, headers=self.headers, stream=True, timeout=timeout) as response:
                    if response.status_code == 200:
                        # Process the streaming response
                        for line in response.iter_lines():
                            if not line:
                                continue

                            try:
                                data = json.loads(line.decode("utf-8"))

                                # Check if this is the final chunk
                                if data.get("status") == "COMPLETED":
                                    completed = True

                                yield data

                                # If we got a completion or error status, we're done
                                if data.get("status") in ["COMPLETED", "FAILED", "CANCELLED"]:
                                    return

                            except json.JSONDecodeError as e:
                                raise ValueError(f"Failed to parse response: {e}")

                    elif response.status_code == 404:
                        # Job not ready yet, wait and retry
                        time.sleep(poll_interval)
                    else:
                        response.raise_for_status()

            except requests.exceptions.ChunkedEncodingError:
                # Connection was closed prematurely, retry
                time.sleep(poll_interval)
                continue

        if not completed and time.time() - start_time >= timeout:
            raise TimeoutError(f"Streaming results timed out after {timeout} seconds")

    def get_status(self, job_id: str, timeout: int = 30) -> Dict[str, Any]:
        """Get the status of a job.

        Args:
            job_id: The job ID.
            timeout: Request timeout in seconds.

        Returns:
            Dict[str, Any]: The job status.

        Raises:
            requests.RequestException: If the request fails.
            ValueError: If the response format is invalid.
        """
        url = f"{self.base_url}/status/{job_id}"

        response = requests.post(url, headers=self.headers, timeout=timeout)
        response.raise_for_status()

        try:
            return response.json()
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse response: {e}")

    def cancel_job(self, job_id: str, timeout: int = 30) -> Dict[str, Any]:
        """Cancel a job.

        Args:
            job_id: The job ID.
            timeout: Request timeout in seconds.

        Returns:
            Dict[str, Any]: The cancellation response.

        Raises:
            requests.RequestException: If the request fails.
            ValueError: If the response format is invalid.
        """
        url = f"{self.base_url}/cancel/{job_id}"

        response = requests.post(url, headers=self.headers, timeout=timeout)
        response.raise_for_status()

        try:
            return response.json()
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse response: {e}")

    def purge_queue(self, timeout: int = 30) -> Dict[str, Any]:
        """Purge all queued jobs.

        Args:
            timeout: Request timeout in seconds.

        Returns:
            Dict[str, Any]: The purge response.

        Raises:
            requests.RequestException: If the request fails.
            ValueError: If the response format is invalid.
        """
        url = f"{self.base_url}/purge-queue"

        response = requests.post(url, headers=self.headers, timeout=timeout)
        response.raise_for_status()

        try:
            return response.json()
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse response: {e}")

    def get_health(self, timeout: int = 30) -> Dict[str, Any]:
        """Get endpoint health information.

        Args:
            timeout: Request timeout in seconds.

        Returns:
            Dict[str, Any]: The health information.

        Raises:
            requests.RequestException: If the request fails.
            ValueError: If the response format is invalid.
        """
        url = f"{self.base_url}/health"

        response = requests.get(url, headers=self.headers, timeout=timeout)
        response.raise_for_status()

        try:
            return response.json()
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse response: {e}")
