# Docket Analyzer OCR

## Installation

Requires Python 3.10

```bash
pip install git+https://github.com/docketanalyzer/ocr
```

To install with GPU support (much faster):

```
pip install 'git+https://github.com/docketanalyzer/ocr[gpu]'
```

## Local Usage

Process a document:

```python
from docketanalyzer_ocr import pdf_document

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
with open('saved.json', 'w') as f:
    f.write(json.dumps(doc.data))

# Loading a document
with open('saved.json', 'r') as f:
    data = json.loads(f.read())
doc = pdf_document(path, load=data)
```

# Remote Usage

You can also serve this tool with Docker.

```
docker build -t docketanalyzer-ocr .
docker run --gpus all -p 8000:8000 docketanalyzer-ocr
```

And then use process the document in remote mode:

```python
doc = pdf_document(path, remote=True) # pass endpoint_url if not using localhost

for page in doc.stream():
    print(page.text)
```

# S3 Support

When using the remote service, if you want to avoid sending the file in a POST request, configure your S3 credentials. Your document will be temporarily pushed to your bucket to be retrieved by the service.

Set the following in your environment (both for the client and service):

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
S3_BUCKET_NAME=...
S3_ENDPOINT_URL=...
```

Usage is identical. We default to using S3 if credentials are available. You can disable this by passing `s3=False` to `process` or `stream`.

# Serverless Support

For serverless usage you can deploy this to RunPod. Just include a custom run command:

```
python -u handler.py
```

On the client side, add the following variables to your env:

```
RUNPOD_API_KEY=...
RUNPOD_OCR_ENDPOINT_ID=...
```

Usage is otherwise identical, just use the remote flag.


[![codecov](https://codecov.io/gh/docketanalyzer/ocr/graph/badge.svg?token=XRATNOME24)](https://codecov.io/gh/docketanalyzer/ocr)
