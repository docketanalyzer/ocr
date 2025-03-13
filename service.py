import asyncio
import base64
import json
import uuid
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from docketanalyzer_ocr import load_pdf, pdf_document
from docketanalyzer_ocr.utils import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

# Dictionary to store job information
jobs = {}

# Cleanup task reference
cleanup_task = None


async def cleanup_old_jobs():
    """Periodically clean up old jobs to prevent memory leaks."""
    while True:
        try:
            # Wait for 1 hour between cleanups
            await asyncio.sleep(3600)

            # Get current time
            now = datetime.now()

            # Find jobs older than 24 hours
            old_jobs = []
            for job_id, job in jobs.items():
                created_at = datetime.fromisoformat(job["created_at"])
                if (now - created_at).total_seconds() > 86400:  # 24 hours
                    old_jobs.append(job_id)

            # Remove old jobs
            for job_id in old_jobs:
                del jobs[job_id]

        except asyncio.CancelledError:
            break
        except Exception:
            # Log error and continue
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI app."""
    global cleanup_task
    cleanup_task = asyncio.create_task(cleanup_old_jobs())

    yield

    if cleanup_task:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task


app = FastAPI(
    title="OCR Service",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class JobInput(BaseModel):
    """Input model for job submission."""

    s3_key: str | None = None
    file: str | None = None  # Base64 encoded file content
    filename: str | None = None
    batch_size: int = 1


class JobRequest(BaseModel):
    """Request model for job submission."""

    input: JobInput


class JobResponse(BaseModel):
    """Response model for job submission."""

    id: str


class JobStatus(BaseModel):
    """Status model for job status."""

    status: str
    stream: list[dict[str, Any]] | None = None


async def process_document(job_id: str, input_data: JobInput):
    """Process a document asynchronously.

    Args:
        job_id: The job ID.
        input_data: The input data for the job.
    """
    start = datetime.now()
    jobs[job_id]["status"] = "IN_PROGRESS"
    jobs[job_id]["stream"] = []

    try:
        # Load the PDF data
        if input_data.s3_key:
            if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
                raise ValueError(
                    "Run `da configure s3` to set AWS_ACCESS_KEY_ID and "
                    "AWS_SECRET_ACCESS_KEY"
                )
            pdf_data, filename = load_pdf(
                s3_key=input_data.s3_key, filename=input_data.filename
            )
        elif input_data.file:
            # Decode base64 file content
            pdf_bytes = base64.b64decode(input_data.file)
            pdf_data, filename = load_pdf(file=pdf_bytes, filename=input_data.filename)
        else:
            raise ValueError("Neither 's3_key' nor 'file' provided in input")

        # Process the PDF
        doc = pdf_document(pdf_data, filename=filename)

        # Stream the results
        for i, page in enumerate(doc.stream(batch_size=input_data.batch_size)):
            duration = (datetime.now() - start).total_seconds()

            # Create a stream item with the page data for the RemoteClient
            stream_item = {
                "output": {
                    "page": page.data,
                    "seconds_elapsed": duration,
                    "progress": i / len(doc),
                    "status": "success",
                }
            }

            # Add to job stream
            jobs[job_id]["stream"].append(stream_item)

            # Small delay to prevent CPU hogging
            await asyncio.sleep(0.1)

        # Mark job as completed
        jobs[job_id]["status"] = "COMPLETED"

    except Exception as e:
        # Handle errors
        error_result = {
            "output": {
                "error": str(e),
                "status": "failed",
            }
        }
        jobs[job_id]["stream"].append(error_result)
        jobs[job_id]["status"] = "FAILED"


@app.post("/run", response_model=JobResponse)
async def run_job(request: JobRequest, background_tasks: BackgroundTasks):
    """Submit a job for processing.

    Args:
        request: The job request.
        background_tasks: FastAPI background tasks.

    Returns:
        JobResponse: The job response with the job ID.
    """
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "PENDING",
        "stream": [],
        "created_at": datetime.now().isoformat(),
    }

    # Start processing in the background
    background_tasks.add_task(process_document, job_id, request.input)

    return {"id": job_id}


@app.post("/stream/{job_id}")
async def stream_job(job_id: str):
    """Stream job results.

    Args:
        job_id: The job ID.

    Returns:
        StreamingResponse: A streaming response with job results.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # If job is still pending, return 404 to mimic RunPod behavior
    if jobs[job_id]["status"] == "PENDING":
        raise HTTPException(status_code=404, detail="Job not ready yet")

    # Get the current stream position
    stream_position = 0

    async def generate():
        nonlocal stream_position

        while True:
            if stream_position < len(jobs[job_id]["stream"]):
                new_items = jobs[job_id]["stream"][stream_position:]
                yield json.dumps({"stream": new_items}) + "\n"
                stream_position = len(jobs[job_id]["stream"])

            if jobs[job_id]["status"] in ["COMPLETED", "FAILED", "CANCELLED"]:
                yield json.dumps({"status": jobs[job_id]["status"]}) + "\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/status/{job_id}", response_model=JobStatus)
async def job_status(job_id: str):
    """Get job status.

    Args:
        job_id: The job ID.

    Returns:
        JobStatus: The job status.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return {
        "status": jobs[job_id]["status"],
        "stream": jobs[job_id]["stream"]
        if jobs[job_id]["status"] != "PENDING"
        else None,
    }


@app.post("/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a job.

    Args:
        job_id: The job ID.

    Returns:
        dict: A success message.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if jobs[job_id]["status"] in ["COMPLETED", "FAILED", "CANCELLED"]:
        return {"status": "Job already finished"}

    jobs[job_id]["status"] = "CANCELLED"
    return {"status": "CANCELLED"}


@app.get("/health")
async def health_check():
    """Get service health information.

    Returns:
        dict: Health information.
    """
    # Count active jobs
    active_jobs = sum(
        1 for job in jobs.values() if job["status"] in ["PENDING", "IN_PROGRESS"]
    )

    return {
        "workers": {
            "active": active_jobs,
            "total": 1,
        },
        "jobs": {
            status: sum(1 for job in jobs.values() if job["status"] == status)
            for status in ["PENDING", "IN_PROGRESS", "COMPLETED", "FAILED", "CANCELLED"]
            if sum(1 for job in jobs.values() if job["status"] == status) > 0
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
