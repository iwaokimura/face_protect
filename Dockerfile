# Dockerfile
# GitHub Actions でビルドされ GHCR に push されるイメージ
# リポジトリ: https://github.com/iwaokimura/face_protect

FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/iwaokimura/face_protect"
LABEL org.opencontainers.image.description="Adversarial perturbation for face recognition evasion"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    PYTHONPATH=/opt/arcface_torch \
    GLOG_minloglevel=2 \
    TF_CPP_MIN_LOG_LEVEL=3

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        libgl1-mesa-glx libglib2.0-0 \
        libsm6 libxext6 libxrender-dev \
        libgomp1 libjpeg-turbo8 libpng16-16 \
        wget curl ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN python3 -m pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

RUN python3 -m pip install --no-cache-dir \
    "insightface==0.7.3" \
    "onnxruntime-gpu==1.20.1" \
    "facenet-pytorch==2.6.0"

RUN python3 -m pip install --no-cache-dir \
    opencv-python-headless Pillow numpy scipy lpips tqdm rich && \
    python3 -m pip cache purge

RUN git clone --depth=1 --filter=blob:none --sparse \
        https://github.com/deepinsight/insightface.git /tmp/insightface_src && \
    cd /tmp/insightface_src && \
    git sparse-checkout set recognition/arcface_torch && \
    cp -r recognition/arcface_torch /opt/arcface_torch && \
    cd / && rm -rf /tmp/insightface_src

COPY face_protect.py /opt/face_protect.py

RUN python3 -c "\
import torch, insightface; \
from facenet_pytorch import InceptionResnetV1; \
import cv2; \
print(f'PyTorch {torch.__version__} | InsightFace {insightface.__version__}'); \
print('Docker build OK')"

ENTRYPOINT ["python3", "/opt/face_protect.py"]
