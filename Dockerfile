FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y libreoffice && \
    apt-get -y install cudnn9-cuda-11 && \
    apt-get remove -y python3-blinker && \
    rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install .

RUN python test.py

CMD [ "python", "-u", "/app/test.py" ]
