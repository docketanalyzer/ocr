from datetime import datetime
import runpod
from docketanalyzer_ocr import load_pdf, process_pdf


def handler(event):
    start = datetime.now()
    inputs = event.pop('input')
    filename = inputs.get('filename')
    batch_size = inputs.get('batch_size', 1)
    if inputs.get('s3_key'):
        file, filename = load_pdf(s3_key=inputs.pop('s3_key'), filename=filename)
    elif inputs.get('file'):
        file, filename = load_pdf(file=inputs.pop('file'), filename=filename)
    else:
        raise ValueError("Neither 's3_key' nor 'file' provided in input")

    try:
        doc = process_pdf(file, filename=filename)
        completed = 0
        for page in doc.stream(batch_size=batch_size):
            completed += 1
            duration = (datetime.now() - start).total_seconds()
            yield {
                'page': page.data,
                'seconds_elapsed': duration,
                'progress': len(doc) / completed,
                'status': 'success',
            }
    except Exception as e:
        yield {
            'error': str(e),
            'status': 'failed',
        }


runpod.serverless.start(
    {
        "handler": handler,
        "return_aggregate_stream": False,
    }
)

