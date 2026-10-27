FROM python:3.12-slim


WORKDIR /ocr

COPY pyproject.toml .
COPY main_ocr.py main_API.py utils.py ./
COPY mineru/ ./mineru/

# system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# For Debian/Ubuntu-based images it is needed libreoffice
RUN apt-get update && apt-get install -y libreoffice && rm -rf /var/lib/apt/lists/*
 
RUN pip install --upgrade pip setuptools wheel \
    && pip install -e .[core]

EXPOSE 8084
CMD ["uvicorn", "main_API:app", "--host", "0.0.0.0", "--port", "8084", "--workers", "1"]

