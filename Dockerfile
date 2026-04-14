# Dockerfile  (v3: cmake/cython 追加, facenet-pytorch --no-deps)
# Base: nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/iwaokimura/face_protect"
LABEL org.opencontainers.image.description="Adversarial perturbation for face recognition evasion"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    PYTHONPATH=/opt/arcface_torch \
    GLOG_minloglevel=2 \
    TF_CPP_MIN_LOG_LEVEL=3

# ─── システムパッケージ ─────────────────────────────────────────
# cmake: insightface==0.7.3 のビルドに必須
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        libgl1-mesa-glx libglib2.0-0 \
        libsm6 libxext6 libxrender-dev \
        libgomp1 libjpeg-turbo8 libpng16-16 \
        cmake \
	build-essential \
        wget curl ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

# ─── pip / ビルドツール ─────────────────────────────────────────
# cython: insightface のネイティブ拡張ビルドに必要
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel cython

# ─── PyTorch CUDA 12.4 ─────────────────────────────────────────
RUN python3 -m pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ─── InsightFace + ONNX Runtime ────────────────────────────────
# insightface は cmake/cython が揃った後にインストール
RUN python3 -m pip install --no-cache-dir \
    "insightface==0.7.3" \
    "onnxruntime-gpu==1.20.1"

# ─── facenet-pytorch ───────────────────────────────────────────
# 2.6.0 は torch<2.3.0 を要求するが実際のコードは 2.5.x と互換。
# --no-deps でバージョン制約チェックをスキップし、
# 依存パッケージ (numpy / Pillow / requests / tqdm) は別途インストール。
RUN python3 -m pip install --no-cache-dir \
    "numpy<2.0" Pillow requests tqdm && \
    python3 -m pip install --no-cache-dir --no-deps \
    "facenet-pytorch==2.6.0"

# ─── 画像処理・ユーティリティ ──────────────────────────────────
RUN python3 -m pip install --no-cache-dir \
    opencv-python-headless scipy lpips rich && \
    python3 -m pip cache purge

# ─── ArcFace PyTorch バックボーン ──────────────────────────────
RUN git clone --depth=1 --filter=blob:none --sparse \
        https://github.com/deepinsight/insightface.git /tmp/insightface_src && \
    cd /tmp/insightface_src && \
    git sparse-checkout set recognition/arcface_torch && \
    cp -r recognition/arcface_torch /opt/arcface_torch && \
    cd / && rm -rf /tmp/insightface_src

# ─── パイプライン ──────────────────────────────────────────────
COPY face_protect.py /opt/face_protect.py

# ─── ビルド確認 ────────────────────────────────────────────────
RUN python3 -c "\
import torch, insightface; \
from facenet_pytorch import InceptionResnetV1; \
import cv2; \
print(f'PyTorch {torch.__version__} | InsightFace {insightface.__version__}'); \
print('Docker build OK')"

ENTRYPOINT ["python3", "/opt/face_protect.py"]
