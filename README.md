# face_protect

顔写真に不可視の敵対的摂動 ($L^\infty$-PGD) を付与し，顔認識システムを回避するツール．
SingularityCE または Docker 上で **root 権限なし** で動作します．

- **GitHub**: https://github.com/iwaokimura/face_protect
- **GHCR**: `ghcr.io/iwaokimura/face_protect:latest`

---

## ファイル構成

```
face_protect/
├── .github/workflows/
│   └── build_container.yml  # GitHub Actions: Docker → GHCR 自動 push
├── Dockerfile                # OCI イメージ定義
├── face_protect.def          # Singularity 定義（Sylabs Remote Build 用）
├── face_protect.py           # Python パイプライン本体
├── run.sh                    # 実行ラッパー（SingularityCE，SIF 自動 pull 機能付き）
├── run_docker.sh             # 実行ラッパー（Docker，イメージ自動 pull 機能付き）
└── requirements.txt          # pip 依存（参照用）
```

---

## Docker での実行

> SingularityCE を使わず，ローカルの Docker 環境で実行したい場合はこちら．
> GPU を使用するには [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) が必要です．

### Step 1: イメージの取得（初回・または --pull で最新化）

```bash
# 自動: run_docker.sh がローカルにイメージがなければ自動 pull します
./run_docker.sh --setup

# 手動で取得したい場合:
docker pull ghcr.io/iwaokimura/face_protect:latest

# Private パッケージの場合（PAT が必要）:
echo <GitHub PAT with read:packages> | docker login ghcr.io -u iwaokimura --password-stdin
docker pull ghcr.io/iwaokimura/face_protect:latest
```

### Step 2: 動作確認

```bash
./run_docker.sh --test
# 期待出力: PyTorch 2.5.1 | CUDA=True | GPU: NVIDIA GeForce RTX 4090 | ALL OK
```

### Step 3: モデルのダウンロード（初回のみ，ネット接続必須）

```bash
./run_docker.sh --setup
# → ./models/ に InsightFace buffalo_l, FaceNet 重みを保存（合計 ~700MB）
```

### 使用方法（Docker）

```bash
# 1枚処理
./run_docker.sh ./photos/DSC_0001.JPG ./output/

# ディレクトリ一括処理（推奨設定）
./run_docker.sh ./photos/ ./protected/ --iterations 150 --format png

# 高速モード
./run_docker.sh ./photos/ ./protected/ --iterations 50 --format jpeg

# イメージを最新版に更新
./run_docker.sh --pull

# コンテナ内で任意コマンドを実行
./run_docker.sh --exec python3 /opt/cosine_similarity.py orig.jpg prot.png
```

---

## セットアップ（root 不要・SingularityCE）

### Step 1: GitHub リポジトリを作成して push

```bash
# GitHub で face_protect リポジトリを新規作成してから:
git init && git add .
git commit -m "Initial commit"
git remote add origin https://github.com/iwaokimura/face_protect.git
git push -u origin main
```

GitHub Actions が自動起動します（約 20〜30 分）．
進捗確認: https://github.com/iwaokimura/face_protect/actions

### Step 2: GHCR パッケージを Public に設定

プライベートリポジトリの場合，パッケージはデフォルト private です．
認証なしで pull したい場合は Public に変更してください:

```
https://github.com/iwaokimura?tab=packages
→ face_protect → Package settings → Change visibility → Public
```

### Step 3: SIF を取得（初回・または --pull で最新化）

```bash
# 自動: run.sh が SIF のなければ自動 pull します
./run.sh --setup

# 手動で取得したい場合:
singularity pull face_protect.sif \
    docker://ghcr.io/iwaokimura/face_protect:latest

# Private パッケージの場合（PAT が必要）:
export SINGULARITY_DOCKER_USERNAME=iwaokimura
export SINGULARITY_DOCKER_PASSWORD=<GitHub PAT with read:packages>
singularity pull face_protect.sif \
    docker://ghcr.io/iwaokimura/face_protect:latest
```

### Step 4: 動作確認

```bash
./run.sh --test
# 期待出力: PyTorch 2.5.1 | CUDA=True | GPU: NVIDIA GeForce RTX 4090 | ALL OK
```

### Step 5: モデルのダウンロード（初回のみ，ネット接続必須）

```bash
./run.sh --setup
# → ./models/ に InsightFace buffalo_l, FaceNet 重みを保存（合計 ~700MB）
```

---

## 使用方法

```bash
# 1枚処理
./run.sh ./photos/DSC_0001.JPG ./output/

# ディレクトリ一括処理（推奨設定）
./run.sh ./photos/ ./protected/ --iterations 150 --format png

# 高速モード
./run.sh ./photos/ ./protected/ --iterations 50 --format jpeg

# SIF を最新版に更新
./run.sh --pull
```

---

## パラメータ

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `--epsilon` | `0.0392` (10/255) | $L^\infty$ 摂動上限 |
| `--iterations` | `150` | PGD 反復回数 |
| `--step-size` | `0.00588` | PGD ステップサイズ |
| `--padding` | `0.40` | 顔クロップパディング率 |
| `--format` | `png` | 出力形式 (png 推奨 / jpeg 高速) |
| `--verbose` | off | 詳細ログ |

---

## 処理時間の目安（RTX 4090, 12MP JPEG）

| シナリオ | 推定時間 |
|---|---|
| ポートレート 1人 / iterations=150 / PNG | 5〜8 秒/枚 |
| ポートレート 1人 / iterations=50  / JPEG | 2〜4 秒/枚 |
| 集合写真 3人   / iterations=150 / PNG | 10〜21 秒/枚 |

---

## ArcFace 重みの追加（オプション・保護強度向上）

InsightFace model_zoo から `ms1mv3_arcface_r50_fp16.pth` を取得し，
`models/arcface_r50_ms1mv3.pth` として配置するとアンサンブルに追加されます．
https://github.com/deepinsight/insightface/tree/master/model_zoo
