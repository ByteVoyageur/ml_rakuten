FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl unzip libgl1 libc6 libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

# 在原有 apt 安装行里加上 tree
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl unzip libgl1 libc6 libstdc++6 tree && \
    rm -rf /var/lib/apt/lists/*


# 建非 root 用户（UID=1000）
RUN useradd -m -u 1000 appuser
WORKDIR /workspace
RUN chown -R 1000:1000 /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /workspace/requirements.txt

EXPOSE 9000

USER 1000:1000
CMD ["bash","-lc","jupyter lab \
  --ip=0.0.0.0 --port=9000 --no-browser \
  --ServerApp.token='' \
  --ServerApp.password='' \
  --ServerApp.allow_origin='*' \
  --ServerApp.allow_remote_access=True"]
