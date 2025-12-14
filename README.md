# マンデルブロ集合ビューア

Python と Rust で実装したインタラクティブなマンデルブロ集合ビューア。

## 実装

| ディレクトリ | 内容 |
|-------------|------|
| [python/](./python/) | マンデルブロ (Python) - シンプル実装 |
| [python_and_rust/](./python_and_rust/) | マンデルブロ (Python) - NumPy + Matplotlib、Rust拡張で高速化対応 |
| [rust/](./rust/) | マンデルブロ (Rust) - GPU/CPU/任意精度の3モード自動切替 |
| [mandelbulb_3d/](./mandelbulb_3d/) | マンデルバルブ 3D - CPU/GPU版、リアルタイム3Dレンダリング |

## 共通操作

| 操作 | 機能 |
|------|------|
| マウスホイール | 拡大 / 縮小 |
| 左クリック | クリック位置を中心に移動 |
| 右クリック | ズームイン |
| `R` キー | リセット |
| `S` キー | 画像保存 |
| `Q` / `Escape` | 終了 |

## クイックスタート

### Python版 (シンプル)

```bash
cd python
pip install -r requirements.txt
python mandelbrot.py
```

### Python + Rust版 (高速化対応)

```bash
cd python_and_rust
pip install -r requirements.txt
python mandelbrot.py
```

**Rust拡張で高速化する場合（64倍高速）:**

```bash
cd python_and_rust
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib maturin
cd rust_ext && maturin develop --release && cd ..
python mandelbrot.py
```

### Rust版

```bash
cd rust
brew install gmp  # 初回のみ
cargo run --release                        # CPU版
cargo run --release --bin mandelbrot-gpu   # GPU版（推奨）
```

### Mandelbulb 3D

```bash
cd mandelbulb_3d
cargo run --release       # CPU版
cd gpu && cargo run --release  # GPU版
```

## ライセンス

MIT License
