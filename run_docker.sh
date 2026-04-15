#!/usr/bin/env bash
# run_docker.sh — face_protect Docker 実行ラッパー
# GitHub: https://github.com/iwaokimura/face_protect
# GHCR image: ghcr.io/iwaokimura/face_protect:latest
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="ghcr.io/iwaokimura/face_protect:latest"
MODELS_DIR="${SCRIPT_DIR}/models"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

# ─── イメージがなければ自動 pull ────────────────────────────
if ! docker image inspect "${IMAGE}" > /dev/null 2>&1; then
    echo -e "${YELLOW}[info]${NC} ローカルにイメージが見つかりません。"
    echo -e "${YELLOW}[pull]${NC} GHCR から取得します..."
    echo "  イメージ: ${IMAGE}"
    docker pull "${IMAGE}"
    echo -e "${GREEN}[pull 完了]${NC} ${IMAGE}"
fi

mkdir -p "${MODELS_DIR}"

MODEL_VOLUMES=(-v "${MODELS_DIR}:/models")

# ─── --setup : モデルダウンロード ───────────────────────────
if [[ "${1:-}" == "--setup" ]]; then
    echo -e "${GREEN}[setup]${NC} モデルをダウンロードしています..."
    docker run --rm --gpus all \
        "${MODEL_VOLUMES[@]}" \
        "${IMAGE}" \
        --download-models --models /models
    echo -e "${GREEN}[setup 完了]${NC} 保存先: ${MODELS_DIR}"
    exit 0
fi

# ─── --test : 動作確認 ──────────────────────────────────────
if [[ "${1:-}" == "--test" ]]; then
    docker run --rm -i --gpus all \
        --entrypoint python3 \
        "${IMAGE}" - <<'PYEOF'
import sys, torch, insightface
from facenet_pytorch import InceptionResnetV1
import cv2, numpy as np

print(f"PyTorch     : {torch.__version__}")
print(f"CUDA 利用可 : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
print(f"InsightFace : {insightface.__version__}")
print(f"OpenCV      : {cv2.__version__}")
print(f"numpy       : {np.__version__}")
_ = InceptionResnetV1(pretrained=None)   # モデル構造だけ確認（重みは不要）
print("facenet-pytorch : OK")
print("─────────────────────────────")
print("ALL OK")
PYEOF
    exit 0
fi

# ─── --pull : イメージを強制再取得 ─────────────────────────
if [[ "${1:-}" == "--pull" ]]; then
    echo -e "${YELLOW}[pull]${NC} 最新イメージを取得..."
    docker pull "${IMAGE}"
    echo -e "${GREEN}[完了]${NC} ${IMAGE}"
    exit 0
fi

# ─── --exec : コンテナ内で任意コマンドを実行 ────────────────
# 使用例:
#   ./run_docker.sh --exec python3 /opt/cosine_similarity.py orig.jpg prot.png
#   ./run_docker.sh --exec bash
#   ./run_docker.sh --exec python3 --version
if [[ "${1:-}" == "--exec" ]]; then
    shift   # --exec を除く
    EXEC_BIN="${1}"; shift   # 最初の引数を entrypoint に
    docker run --rm -i --gpus all \
        --entrypoint "${EXEC_BIN}" \
        "${MODEL_VOLUMES[@]}" \
        "${IMAGE}" "$@"
    exit 0
fi

# ─── 通常実行 ───────────────────────────────────────────────
INPUT_PATH="${1:-}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/output}"

if [[ -z "${INPUT_PATH}" ]]; then
    echo "使用方法:"
    echo "  ./run_docker.sh --pull                          # イメージを最新版に更新"
    echo "  ./run_docker.sh --setup                         # モデルのダウンロード（初回）"
    echo "  ./run_docker.sh --test                          # 動作確認"
    echo "  ./run_docker.sh --exec <cmd> [args...]          # コンテナ内で任意コマンド実行"
    echo "  ./run_docker.sh <入力パス> [出力ディレクトリ]  # 処理実行"
    echo ""
    echo "処理オプション例:"
    echo "  --iterations 150  # PGD 反復回数"
    echo "  --format png      # 出力形式: png / jpeg"
    echo "  --epsilon 0.039   # L∞ 摂動上限"
    echo "  --verbose         # 詳細ログ"
    echo ""
    echo "GHCR イメージ: ${IMAGE}"
    echo "※ GPU 使用には NVIDIA Container Toolkit が必要です"
    exit 1
fi

INPUT_ABS="$(realpath "${INPUT_PATH}")"
if [[ -d "${INPUT_ABS}" ]]; then
    BIND_INPUT="${INPUT_ABS}:/data/input:ro"
    CONTAINER_INPUT="/data/input"
else
    BIND_INPUT="$(dirname "${INPUT_ABS}"):/data/input:ro"
    CONTAINER_INPUT="/data/input/$(basename "${INPUT_ABS}")"
fi

mkdir -p "${OUTPUT_DIR}"
OUTPUT_ABS="$(realpath "${OUTPUT_DIR}")"

echo -e "${GREEN}[run]${NC} 処理開始"
echo "  入力  : ${INPUT_ABS}"
echo "  出力  : ${OUTPUT_ABS}"
echo "  モデル: ${MODELS_DIR}"
echo ""

docker run --rm --gpus all \
    "${MODEL_VOLUMES[@]}" \
    -v "${BIND_INPUT}" \
    -v "${OUTPUT_ABS}:/data/output" \
    "${IMAGE}" \
    --input  "${CONTAINER_INPUT}" \
    --output /data/output \
    --models /models \
    "${@:3}"

echo -e "\n${GREEN}[done]${NC} 出力先: ${OUTPUT_ABS}"
