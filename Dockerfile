FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8901

CMD ["python3", "app.py"]
