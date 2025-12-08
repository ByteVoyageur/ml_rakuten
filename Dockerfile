FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl unzip libgl1 libc6 libstdc++6 tree && \
    rm -rf /var/lib/apt/lists/*


# create a non-root user
RUN useradd -m -u 1000 appuser
WORKDIR /workspace
RUN chown -R 1000:1000 /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /workspace/requirements.txt && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

EXPOSE 9000

USER 1000:1000
CMD ["bash","-lc","jupyter lab \
  --ip=0.0.0.0 --port=9000 --no-browser \
  --ServerApp.token='' \
  --ServerApp.password='' \
  --ServerApp.allow_origin='*' \
  --ServerApp.allow_remote_access=True"]
