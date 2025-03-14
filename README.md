# Docket Analyzer OCR

## Installation

```bash
pip install 'docketanalyzer[ocr]'
```

## Local Usage

Process a document:

```python
from docketanalyzer.ocr import pdf_document

path = 'path/to/doc.pdf
doc = pdf_document(path) # the input can also be raw bytes
doc.process()

for page in doc:
    for block in page:
        for line in block:
            pass
```

You can also stream pages as they are processed:

```python
doc = pdf_document(path)

for page in doc.stream():
    print(page.text)
```

Pages, blocks, and lines have common attributes:

```python
# where item is a page, block, or line

item.data # A dictionary representation of the item and it's children
item.text # The item's text content
item.page_num # The page the item appears on
item.i # The item-level index
item.id # A unique id constructed from the item and it's parents index (e.g. 3-2-1 for the first line in the second block on the third page).
item.bbox # Bounding box (blocks and lines only)
item.clip() # Extract element as an image from the original pdf
```

Blocks also have a block type attribute:

```python
print(block.block_type) # 'title', 'text', 'figure', etc.
```

Save and load data:

```python
# Saving a document
doc.save('doc.json')

# Loading a document
doc = pdf_document(path, load='doc.json')
```

# Remote Usage

You can also serve this tool with Docker.

```bash
docker pull nadahlberg/docketanalyzer-ocr:latest
docker run --gpus all -p 8000:8000 nadahlberg/docketanalyzer-ocr:latest
```

And then use process the document in remote mode:

```python
doc = pdf_document(path, remote=True) # pass endpoint_url if not using localhost

for page in doc.stream():
    print(page.text)
```

# S3 Support

When using the remote service, if you want to avoid sending the file in a POST request, configure your S3 credentials. Your document will be temporarily pushed to your bucket to be retrieved by the service.

To configure your S3 credentials run:

```
da configure s3
```

Or set the following in your env:

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_S3_BUCKET_NAME
AWS_S3_ENDPOINT_URL
```

Usage is identical. We default to using S3 if credentials are available. You can control this explicitly by passing `use_s3=False` to `pdf_document`.

# Serverless Support

For serverless usage you can deploy this to RunPod. To get set up:

1. Create a serverless worker on RunPod using the docker container.

```bash
nadahlberg/docketanalyzer-ocr:latest
```

2. Add the following custom run command.

```
python -u handler.py
```

3. Add your S3 credentials to the RunPod worker.

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_S3_BUCKET_NAME
AWS_S3_ENDPOINT_URL
```

4. On your local machine, configure your RunPod key and the worker id.

You can run:

```
da configure runpod
```

Or set the following in your env:

```
RUNPOD_API_KEY
RUNPOD_OCR_ENDPOINT_ID
```

Usage is otherwise identical, just use `remote=True` with `pdf_document` 
