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
from docketanalyzer_ocr.ocr import OCR_CLIENT

process = OCR_CLIENT.process

jobs = {}

cleanup_task = None

ocr_semaphore = asyncio.Semaphore(1)


async def cleanup_old_jobs():
    """Periodically clean up old jobs to prevent memory leaks."""
    while True:
        try:
            await asyncio.sleep(3600)
            now = datetime.now()
            old_jobs = [
                job_id
                for job_id, job in jobs.items()
                if (now - datetime.fromisoformat(job["created_at"])).total_seconds()
                > 86400
            ]
            for job_id in old_jobs:
                del jobs[job_id]
        except asyncio.CancelledError:
            break
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI app."""
    global cleanup_task
    cleanup_task = asyncio.create_task(cleanup_old_jobs())

    yield

    print("Stopping OCR service...", flush=True)
    OCR_CLIENT.stop()

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
    file: str | None = None
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
        if input_data.s3_key:
            pdf_data, filename = load_pdf(
                s3_key=input_data.s3_key, filename=input_data.filename
            )
        elif input_data.file:
            pdf_bytes = base64.b64decode(input_data.file)
            pdf_data, filename = load_pdf(file=pdf_bytes, filename=input_data.filename)
        else:
            raise ValueError("Neither 's3_key' nor 'file' provided in input")

        async with ocr_semaphore:
            doc = pdf_document(pdf_data, filename=filename)
            pages = list(doc.stream(batch_size=input_data.batch_size))

        for i, page in enumerate(pages):
            duration = (datetime.now() - start).total_seconds()

            stream_item = {
                "output": {
                    "page": page.data,
                    "seconds_elapsed": duration,
                    "progress": i / len(doc),
                    "status": "success",
                }
            }

            jobs[job_id]["stream"].append(stream_item)

            await asyncio.sleep(0.1)

        jobs[job_id]["status"] = "COMPLETED"
        doc.close()

    except Exception as e:
        print(f"Error processing job {job_id}: {e}", flush=True)
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

    if jobs[job_id]["status"] == "PENDING":
        raise HTTPException(status_code=404, detail="Job not ready yet")

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
