# マンデルブロ集合ビューア

Python と Rust で実装したインタラクティブなマンデルブロ集合ビューア。

## 実装

| 言語 | 特徴 |
|------|------|
| [Python](./python/) | NumPy + Matplotlib、シンプル実装 |
| [Rust](./rust/) | 高速 + 任意精度、無限ズーム対応 |

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

### Python版
```bash
cd python
pip install -r requirements.txt
python mandelbrot.py
```

### Rust版
```bash
cd rust
cargo run --release
```

## ライセンス

MIT License
