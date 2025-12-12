# マンデルブロ集合ビューア

M1 Mac (Apple Silicon) 対応の Python 製インタラクティブ・マンデルブロ集合描画ツール。

![マンデルブロ集合](mandelbrot.png)

## 必要環境

- Python 3.10+
- macOS (M1/M2 Mac 対応)

## インストール

```bash
pip install -r requirements.txt
```

## 使い方

```bash
python mandelbrot.py
```

## 操作方法

| 操作 | 機能 |
|------|------|
| マウスホイール | 拡大 / 縮小 |
| 左クリック | クリック位置を中心に移動（パン） |
| 右クリック | クリック位置を中心にズームイン |
| `r` キー | 初期表示にリセット |
| `s` キー | 現在の表示を画像として保存 |
| `q` キー | 終了 |

## ファイル構成

```text
python/
├── mandelbrot.py      # メインスクリプト
├── requirements.txt   # 依存関係
└── README.md          # このファイル
```

## 技術的な特徴

- **NumPy ベクトル化計算**: 高速な並列計算
- **カスタムカラーマップ**: 美しいグラデーション表現
- **日本語フォント対応**: macOS 標準フォント使用

## ライセンス

MIT License
