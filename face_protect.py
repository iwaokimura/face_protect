#!/usr/bin/env python3
"""
face_protect.py
高解像度顔写真への不可視敵対的摂動付与ツール（顔認識回避）
対象 : コンパクトデジカメ撮影 JPEG (12MP 前後)
GPU  : RTX 4090 / CUDA 12.4
"""
import argparse, glob, os, sys, time, warnings
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─── 定数 ─────────────────────────────────────────────────────
FACE_SIZE   = 112          # ArcFace / FaceNet 入力サイズ
DEFAULT_EPS  = 10 / 255   # L∞ 摂動上限（高解像度向け）
DEFAULT_ITER = 150         # PGD 反復回数
DEFAULT_STEP = 1.5 / 255  # PGD ステップサイズ
DEFAULT_PAD  = 0.40        # 顔クロップパディング率
DETECT_SIZE  = (1280, 1280)  # 高解像度対応の検出サイズ
MIN_FACE_PX  = 60          # 処理対象とする最小顔幅 (px)
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
ARCFACE_ONNX_NAMES = ("w600k_r50.onnx", "glintr100.onnx")


# ─── モデル管理 ────────────────────────────────────────────────
class ModelManager:
    def __init__(self, models_dir: str, device: torch.device):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self._detector = None
        self._attack_models: List[Tuple[str, torch.nn.Module]] = []

    # ── 顔検出器 (InsightFace RetinaFace) ─────────────────────
    def get_detector(self):
        if self._detector is None:
            from insightface.app import FaceAnalysis
            root = str(self.models_dir / "insightface")
            self._detector = FaceAnalysis(
                name="buffalo_l", root=root,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._detector.prepare(
                ctx_id=0 if self.device.type == "cuda" else -1,
                det_size=DETECT_SIZE,
            )
        return self._detector

    # ── 攻撃モデル (アンサンブル) ──────────────────────────────
    def get_attack_models(self) -> List[Tuple[str, torch.nn.Module]]:
        if not self._attack_models:
            m3 = self._load_arcface_iresnet()
            if m3 is not None:
                self._attack_models.append(("arcface_r50", m3))
            for ds in ("vggface2", "casia-webface"):
                m = self._load_facenet(ds)
                if m is not None:
                    self._attack_models.append((f"facenet_{ds[:3]}", m))
            if not self._attack_models:
                raise RuntimeError(
                    "攻撃モデルが1つもロードできません。"
                    " --download-models を先に実行してください。"
                )
            names = [n for n, _ in self._attack_models]
            print(f"[model] アンサンブル構成: {names}")
        return self._attack_models

    def _load_facenet(self, dataset: str):
        from facenet_pytorch import InceptionResnetV1
        cache = self.models_dir / f"facenet_{dataset.replace('-','_')}.pth"
        try:
            if cache.exists():
                model = InceptionResnetV1(classify=False)
                # strict=False: 保存時に含まれる logits.weight/bias を無視
                # (pretrained モデルは classify=True で保存されているため)
                model.load_state_dict(
                    torch.load(str(cache), map_location="cpu", weights_only=True),
                    strict=False,
                )
            else:
                print(f"[download] facenet-pytorch ({dataset}) ...")
                model = InceptionResnetV1(pretrained=dataset)
                torch.save(model.state_dict(), str(cache))
            return model.eval().to(self.device)
        except Exception as e:
            print(f"[warn] facenet ({dataset}) ロード失敗: {e}")
            return None

    def _load_arcface_iresnet(self):
        try:
            from onnx import load as onnx_load
            from onnx2torch import convert

            onnx_path = self._find_arcface_onnx()
            model = convert(onnx_load(str(onnx_path)))
            return model.eval().to(self.device)
        except Exception as e:
            print(f"[warn] ArcFace IResNet ロード失敗: {e}")
            return None

    def _find_arcface_onnx(self) -> Path:
        self.get_detector()
        search_root = self.models_dir / "insightface"
        for name in ARCFACE_ONNX_NAMES:
            matches = list(search_root.rglob(name))
            if matches:
                return matches[0]
        raise RuntimeError(
            f"ArcFace ONNX が見つかりません: names={ARCFACE_ONNX_NAMES} root={search_root}"
        )

    def download_all(self):
        print("=== モデルのダウンロード開始 ===")
        print("[1/4] InsightFace buffalo_l ...")
        self.get_detector()
        print("      → 完了")
        print("[2/4] arcface_r50 (buffalo_l ONNX) ...")
        model = self._load_arcface_iresnet()
        if model is None:
            raise RuntimeError("ArcFace ONNX の取得またはロードに失敗しました。")
        print("      → 完了")
        for i, ds in enumerate(("vggface2", "casia-webface"), start=3):
            print(f"[{i}/4] facenet-pytorch ({ds}) ...")
            self._load_facenet(ds)
            print("      → 完了")
        print(f"\n保存先: {self.models_dir}")
        print("=== ダウンロード完了 ===")


# ─── 前処理 ────────────────────────────────────────────────────
def preprocess(model_name: str, x: torch.Tensor) -> torch.Tensor:
    """クロップテンソル [1,C,H,W] → モデル入力 [1,C,112,112]"""
    t = F.interpolate(x, size=(FACE_SIZE, FACE_SIZE),
                      mode="bilinear", align_corners=False)
    return (t - 0.5) / 0.5   # [-1,1] 正規化 (共通)


def embed(model_name: str, model: torch.nn.Module,
          x: torch.Tensor) -> torch.Tensor:
    e = model(preprocess(model_name, x))
    if isinstance(e, (tuple, list)):
        e = e[0]
    return F.normalize(e, p=2, dim=1)


def gaussian_blur(x: torch.Tensor, ks: int = 3) -> torch.Tensor:
    k = (torch.ones(1, 1, ks, ks, device=x.device) / ks ** 2).expand(3, 1, -1, -1)
    return F.conv2d(x, k, padding=ks // 2, groups=3)


# ─── PGD コア ──────────────────────────────────────────────────
def pgd_attack(
    crop: torch.Tensor,
    models: List[Tuple[str, torch.nn.Module]],
    epsilon: float = DEFAULT_EPS,
    n_iter:  int   = DEFAULT_ITER,
    step:    float = DEFAULT_STEP,
    verbose: bool  = False,
) -> torch.Tensor:
    """
    L∞-PGD (アンサンブル + Gaussian-blur 版, LowKey 方式)
    入力  : crop [1,C,H,W] float32 [0,1]
    出力  : 保護済み crop [1,C,H,W] float32 [0,1]
    """
    ct = crop.detach().clone()
    with torch.no_grad():
        refs = {n: embed(n, m, ct) for n, m in models}

    delta = torch.zeros_like(ct)
    bar = tqdm(range(n_iter), desc="  PGD", leave=False,
               disable=not verbose, ncols=60)

    for _ in bar:
        delta = delta.detach().requires_grad_(True)
        adv  = (ct + delta).clamp(0.0, 1.0)
        ablur = gaussian_blur(adv)

        losses = [
            (F.cosine_similarity(embed(n, m, adv), refs[n]).mean() +
             F.cosine_similarity(embed(n, m, ablur), refs[n]).mean()) * 0.5
            for n, m in models
        ]
        loss = torch.stack(losses).mean()

        loss.backward()
        with torch.no_grad():
            delta = delta - step * delta.grad.sign()
            delta = delta.clamp(-epsilon, epsilon)

    with torch.no_grad():
        return (ct + delta).clamp(0.0, 1.0)


# ─── 1枚処理 ───────────────────────────────────────────────────
def protect_image(
    img_rgb: np.ndarray,
    mgr: ModelManager,
    epsilon: float = DEFAULT_EPS,
    n_iter:  int   = DEFAULT_ITER,
    step:    float = DEFAULT_STEP,
    padding: float = DEFAULT_PAD,
    verbose: bool  = False,
) -> Tuple[np.ndarray, int]:
    H, W = img_rgb.shape[:2]
    detector = mgr.get_detector()
    atk_models = mgr.get_attack_models()
    dev = mgr.device

    faces = detector.get(img_rgb)
    faces = [f for f in faces if (f.bbox[2] - f.bbox[0]) >= MIN_FACE_PX]
    if not faces:
        return img_rgb.copy(), 0

    result = img_rgb.copy().astype(np.float32) / 255.0

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        pad = int(padding * max(x2 - x1, y2 - y1))
        cx1 = max(0, x1 - pad);  cy1 = max(0, y1 - pad)
        cx2 = min(W, x2 + pad);  cy2 = min(H, y2 + pad)
        cw, ch = cx2 - cx1, cy2 - cy1

        if verbose:
            print(f"  [face] bbox=({x1},{y1})-({x2},{y2}) | "
                  f"crop={cw}×{ch}px | score={face.det_score:.2f}")

        crop_t = (
            torch.from_numpy(result[cy1:cy2, cx1:cx2])
            .float().permute(2, 0, 1).unsqueeze(0).to(dev)
        )
        prot_t = pgd_attack(crop_t, atk_models,
                            epsilon=epsilon, n_iter=n_iter,
                            step=step, verbose=verbose)
        result[cy1:cy2, cx1:cx2] = (
            prot_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        )

    return (result * 255.0).clip(0, 255).astype(np.uint8), len(faces)


# ─── バッチ処理 ────────────────────────────────────────────────
def collect_images(path: str) -> List[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    elif p.is_dir():
        files = []
        for ext in SUPPORTED_EXT:
            files += list(p.glob(f"*{ext}")) + list(p.glob(f"*{ext.upper()}"))
        return sorted(str(f) for f in files)
    return sorted(glob.glob(path))


def process_batch(
    input_path:   str,
    output_dir:   str,
    mgr:          ModelManager,
    epsilon:      float = DEFAULT_EPS,
    n_iter:       int   = DEFAULT_ITER,
    step:         float = DEFAULT_STEP,
    padding:      float = DEFAULT_PAD,
    save_format:  str   = "png",
    jpeg_quality: int   = 95,
    verbose:      bool  = False,
):
    files = collect_images(input_path)
    if not files:
        print(f"[error] 画像が見つかりません: {input_path}")
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[batch] {len(files)} 枚処理開始")
    print(f"  eps={epsilon:.4f}  iter={n_iter}  "
          f"step={step:.5f}  pad={padding:.0%}  fmt={save_format}")

    t_total, n_faces_total, failed = 0.0, 0, []

    for fpath in tqdm(files, desc="処理中", ncols=72, unit="枚"):
        t0 = time.perf_counter()
        try:
            bgr = cv2.imread(fpath, cv2.IMREAD_COLOR)
            if bgr is None:
                raise IOError("読み込み失敗")
            h, w = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            if verbose:
                print(f"\n[proc] {Path(fpath).name}  ({w}×{h}, {w*h/1e6:.1f}MP)")

            prot_rgb, n_faces = protect_image(
                rgb, mgr, epsilon=epsilon, n_iter=n_iter,
                step=step, padding=padding, verbose=verbose,
            )
            elapsed = time.perf_counter() - t0
            t_total += elapsed
            n_faces_total += n_faces

            stem = Path(fpath).stem
            prot_bgr = cv2.cvtColor(prot_rgb, cv2.COLOR_RGB2BGR)
            if save_format == "png":
                out_path = out_dir / f"{stem}_protected.png"
                # IMWRITE_PNG_COMPRESSION=1 → 高速保存（可逆）
                cv2.imwrite(str(out_path), prot_bgr,
                            [cv2.IMWRITE_PNG_COMPRESSION, 1])
            else:
                out_path = out_dir / f"{stem}_protected.jpg"
                cv2.imwrite(str(out_path), prot_bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

            tqdm.write(f"  {Path(fpath).name}: {n_faces}人  "
                       f"{elapsed:.1f}s → {out_path.name}")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            failed.append((fpath, str(e)))
            tqdm.write(f"  [FAIL] {Path(fpath).name}: {e}")

    n_ok = len(files) - len(failed)
    print(f"\n{'='*56}")
    print(f"完了: {n_ok}/{len(files)} 枚成功 | 顔数: {n_faces_total}")
    avg = t_total / max(len(files), 1)
    print(f"合計時間: {t_total:.1f}s  平均: {avg:.1f}s/枚")
    if failed:
        print(f"失敗 {len(failed)} 件:")
        for fp, e in failed:
            print(f"  {fp}: {e}")


# ─── CLI ───────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="顔写真に不可視の敵対的摂動を付与して顔認識を回避するツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 初回: モデルダウンロード
  python3 face_protect.py --download-models --models /models

  # 1枚処理
  python3 face_protect.py -i photo.jpg -o ./out --models /models

  # ディレクトリ一括 (推奨設定)
  python3 face_protect.py -i ./photos -o ./protected \\
      --models /models --iterations 150 --format png

  # 高速モード (ASR やや低下)
  python3 face_protect.py -i ./photos -o ./protected \\
      --models /models --iterations 50 --format jpeg
""",
    )
    p.add_argument("--input",  "-i", help="入力ファイルまたはディレクトリ")
    p.add_argument("--output", "-o", help="出力ディレクトリ")
    p.add_argument("--models", "-m", default="./models",
                   help="モデルキャッシュディレクトリ (default: ./models)")
    p.add_argument("--download-models", action="store_true",
                   help="モデルをダウンロードして終了")
    p.add_argument("--epsilon",    type=float, default=DEFAULT_EPS,
                   help=f"L∞ 摂動上限 (default: {DEFAULT_EPS:.4f} = 10/255)")
    p.add_argument("--iterations", type=int,   default=DEFAULT_ITER,
                   help=f"PGD 反復回数 (default: {DEFAULT_ITER})")
    p.add_argument("--step-size",  type=float, default=DEFAULT_STEP,
                   help=f"PGD ステップサイズ (default: {DEFAULT_STEP:.5f})")
    p.add_argument("--padding",    type=float, default=DEFAULT_PAD,
                   help=f"顔クロップパディング率 (default: {DEFAULT_PAD})")
    p.add_argument("--format",     choices=["png", "jpeg"], default="png",
                   help="出力形式: png (推奨・可逆) / jpeg (高速)")
    p.add_argument("--jpeg-quality", type=int, default=95,
                   help="JPEG 品質 (default: 95)")
    p.add_argument("--gpu",  type=int, default=0,
                   help="GPU インデックス (default: 0)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="詳細ログを表示")
    return p


def main():
    args = build_parser().parse_args()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"[device] {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        print("[device] CPU モード（低速）")

    mgr = ModelManager(args.models, device)

    if args.download_models:
        mgr.download_all()
        return

    if not args.input or not args.output:
        print("[error] --input と --output は必須です。")
        sys.exit(1)

    print("[init] モデルをロード中 ...")
    mgr.get_detector()
    attack_models = mgr.get_attack_models()
    attack_names = [name for name, _ in attack_models]
    print(f"[init] 攻撃モデル: {attack_names}")
    print("[init] 評価モデル: insightface/buffalo_l (ArcFace 系) を想定")
    print("[init] 準備完了。\n")

    process_batch(
        input_path=args.input,
        output_dir=args.output,
        mgr=mgr,
        epsilon=args.epsilon,
        n_iter=args.iterations,
        step=args.step_size,
        padding=args.padding,
        save_format=args.format,
        jpeg_quality=args.jpeg_quality,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
