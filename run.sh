#!/usr/bin/env bash
# run.sh — face_protect SingularityCE 実行ラッパー
# GitHub: https://github.com/iwaokimura/face_protect
# GHCR image: ghcr.io/iwaokimura/face_protect:latest
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF="${SCRIPT_DIR}/face_protect.sif"
MODELS_DIR="${SCRIPT_DIR}/models"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

# ─── SIF がなければ自動 pull ────────────────────────────────
if [[ ! -f "${SIF}" ]]; then
    echo -e "${YELLOW}[info]${NC} face_protect.sif が見つかりません。"
    echo -e "${YELLOW}[pull]${NC} GHCR から取得します..."
    echo "  イメージ: ghcr.io/iwaokimura/face_protect:latest"
    singularity pull "${SIF}" docker://ghcr.io/iwaokimura/face_protect:latest
    echo -e "${GREEN}[pull 完了]${NC} ${SIF}"
fi

mkdir -p "${MODELS_DIR}"

# ─── --setup : モデルダウンロード ───────────────────────────
if [[ "${1:-}" == "--setup" ]]; then
    echo -e "${GREEN}[setup]${NC} モデルをダウンロードしています..."
    singularity exec --nv \
        --bind "${MODELS_DIR}:/models" \
        "${SIF}" \
        python3 /opt/face_protect.py --download-models --models /models
    echo -e "${GREEN}[setup 完了]${NC} 保存先: ${MODELS_DIR}"
    exit 0
fi

# ─── --test : 動作確認 ──────────────────────────────────────
if [[ "${1:-}" == "--test" ]]; then
    singularity test "${SIF}"
    exit 0
fi

# ─── --pull : SIF を強制再取得 ──────────────────────────────
if [[ "${1:-}" == "--pull" ]]; then
    echo -e "${YELLOW}[pull]${NC} 最新イメージを取得..."
    singularity pull --force "${SIF}" docker://ghcr.io/iwaokimura/face_protect:latest
    echo -e "${GREEN}[完了]${NC} ${SIF}"
    exit 0
fi

# ─── 通常実行 ───────────────────────────────────────────────
INPUT_PATH="${1:-}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/output}"

if [[ -z "${INPUT_PATH}" ]]; then
    echo "使用方法:"
    echo "  ./run.sh --pull                          # SIF を最新版に更新"
    echo "  ./run.sh --setup                         # モデルのダウンロード（初回）"
    echo "  ./run.sh --test                          # 動作確認"
    echo "  ./run.sh <入力パス> [出力ディレクトリ]  # 処理実行"
    echo ""
    echo "処理オプション例:"
    echo "  --iterations 150  # PGD 反復回数"
    echo "  --format png      # 出力形式: png / jpeg"
    echo "  --epsilon 0.039   # L∞ 摂動上限"
    echo "  --verbose         # 詳細ログ"
    echo ""
    echo "GHCR イメージ: ghcr.io/iwaokimura/face_protect:latest"
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

singularity run --nv \
    --bind "${MODELS_DIR}:/models" \
    --bind "${BIND_INPUT}" \
    --bind "${OUTPUT_ABS}:/data/output" \
    "${SIF}" \
    --input  "${CONTAINER_INPUT}" \
    --output /data/output \
    --models /models \
    "${@:3}"

echo -e "\n${GREEN}[done]${NC} 出力先: ${OUTPUT_ABS}"
