FROM python:3.10-slim

# dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# pip
RUN pip install --no-cache-dir --upgrade pip

# dependências python
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY . /app

# expõe a porta da API
EXPOSE 9000

# comando para iniciar
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000", "--workers", "1"]
