#!/usr/bin/env python3
"""
cosine_similarity.py
────────────────────
元画像と保護済み画像の顔埋め込み間のコサイン類似度を計測し、
顔認識回避の有効性を評価するスクリプト。

使用例:
  # Singularity コンテナ内で直接実行
  python3 cosine_similarity.py original.jpg protected.png

  # run.sh 経由で実行
  ./run.sh --exec python3 /opt/cosine_similarity.py original.jpg protected.png

  # ディレクトリ内の *_protected.png を一括評価
  python3 cosine_similarity.py --batch ./photos/ ./protected/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import cv2

# ─── 埋め込み抽出 ────────────────────────────────────────────────────────────

def load_app():
    """InsightFace の FaceAnalysis（RetinaFace + ArcFace R100）を初期化する。"""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def get_embedding(app, img_path: Path) -> np.ndarray | None:
    """画像から最大顔の 512 次元埋め込みを返す。顔未検出時は None。"""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[error] 読み込み失敗: {img_path}", file=sys.stderr)
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = app.get(img_rgb)
    if not faces:
        print(f"[warn]  顔未検出: {img_path}", file=sys.stderr)
        return None
    # 最大面積の顔を選択
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    emb = face.normed_embedding          # L2 正規化済み (512,)
    return emb.astype(np.float32)


# ─── コサイン類似度 ────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """L2 正規化済みベクトル間のコサイン類似度（= 内積）を返す。"""
    return float(np.dot(a, b))


# ─── 評価ラベル ───────────────────────────────────────────────────────────

def verdict(sim: float) -> str:
    """
    ArcFace の典型的な閾値（同一人物: sim > 0.28、insightface 推奨 0.4〜0.5）
    に基づく判定文字列を返す。
    """
    if sim >= 0.50:
        return "❌  回避失敗（同一人物と強く判定される可能性が高い）"
    elif sim >= 0.28:
        return "⚠️  境界域（システムによっては同一人物と判定される可能性あり）"
    else:
        return "✅  回避成功（同一人物と判定されにくい）"


# ─── 単一ペア評価 ─────────────────────────────────────────────────────────

def evaluate_pair(app, orig_path: Path, prot_path: Path, verbose: bool = True):
    emb_orig = get_embedding(app, orig_path)
    emb_prot = get_embedding(app, prot_path)

    if emb_orig is None or emb_prot is None:
        return None

    sim = cosine_sim(emb_orig, emb_prot)

    if verbose:
        bar_len = 40
        filled  = int((sim + 1) / 2 * bar_len)   # [-1,1] → [0,40]
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\n元画像   : {orig_path.name}")
        print(f"保護済み : {prot_path.name}")
        print(f"コサイン類似度 : {sim:+.4f}  [{bar}]")
        print(f"判定           : {verdict(sim)}\n")

    return sim


# ─── バッチ評価 ───────────────────────────────────────────────────────────

def evaluate_batch(app, orig_dir: Path, prot_dir: Path):
    pairs = []
    for orig in sorted(orig_dir.glob("*")):
        if orig.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        stem = orig.stem
        for ext in [".png", ".jpg", ".jpeg"]:
            cand = prot_dir / f"{stem}_protected{ext}"
            if cand.exists():
                pairs.append((orig, cand))
                break

    if not pairs:
        print(f"[error] {prot_dir} に対応する _protected ファイルが見つかりません",
              file=sys.stderr)
        sys.exit(1)

    sims = []
    print(f"\n{'─'*60}")
    print(f" バッチ評価: {len(pairs)} ペア")
    print(f"{'─'*60}")
    for orig, prot in pairs:
        sim = evaluate_pair(app, orig, prot, verbose=True)
        if sim is not None:
            sims.append(sim)

    if sims:
        print(f"{'─'*60}")
        print(f" 平均類似度 : {np.mean(sims):+.4f}")
        print(f" 最大類似度 : {np.max(sims):+.4f}  （回避に最も失敗）")
        print(f" 最小類似度 : {np.min(sims):+.4f}  （回避に最も成功）")
        success = sum(1 for s in sims if s < 0.28)
        print(f" 回避成功数 : {success}/{len(sims)}")
        print(f"{'─'*60}\n")


# ─── メイン ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="顔認識回避の有効性をコサイン類似度で評価する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input1",  help="元画像のパス（または --batch 時は元画像ディレクトリ）")
    parser.add_argument("input2",  help="保護済み画像のパス（または --batch 時は出力ディレクトリ）")
    parser.add_argument("--batch", action="store_true",
                        help="input1/input2 をディレクトリとして一括評価")
    args = parser.parse_args()

    print("[info] InsightFace（ArcFace R100）初期化中...")
    app = load_app()
    print("[info] 準備完了\n")

    if args.batch:
        evaluate_batch(app, Path(args.input1), Path(args.input2))
    else:
        sim = evaluate_pair(app, Path(args.input1), Path(args.input2))
        if sim is None:
            sys.exit(1)


if __name__ == "__main__":
    main()
