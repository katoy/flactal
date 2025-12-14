# マンデルブロ集合ビューア

M1 Mac (Apple Silicon) 対応の Python 製インタラクティブ・マンデルブロ集合描画ツール。

![マンデルブロ集合](mandelbrot.png)

## 目次

- [必要環境](#必要環境)
- [インストール](#インストール)
- [使い方](#使い方)
- [操作方法](#操作方法)
- [ファイル構成](#ファイル構成)
- [コード構成](#コード構成)
- [技術的な特徴](#技術的な特徴)
- [設定のカスタマイズ](#設定のカスタマイズ)
- [ライセンス](#ライセンス)

## 必要環境

- Python 3.10+
- macOS (M1/M2/M3 Mac 対応)

## インストール

```bash
pip install -r requirements.txt
```

### 依存ライブラリ

- `numpy` - 高速なベクトル化計算
- `matplotlib` - グラフ描画・GUI

## 使い方

```bash
python mandelbrot.py
```

## 操作方法

| 操作 | 機能 |
|------|------|
| マウスホイール上 | ズームイン (0.8倍) |
| マウスホイール下 | ズームアウト (1.25倍) |
| 左クリック | クリック位置を中心に移動（パン） |
| 右クリック | クリック位置を中心にズームイン (0.8倍) |
| `r` キー | 初期表示にリセット |
| `s` キー | 現在の表示を画像として保存 (`mandelbrot_001.png` 等) |
| `q` キー | 終了 |

## ファイル構成

```text
python/
├── mandelbrot.py      # メインスクリプト
├── requirements.txt   # 依存関係
├── mandelbrot.png     # サンプル画像
└── README.md          # このファイル
```

## コード構成

```text
mandelbrot.py
├── ViewerConfig       # 設定定数 (dataclass)
├── ViewBounds         # 表示範囲管理 (dataclass)
├── MandelbrotViewer   # メインビューアクラス
└── ユーティリティ関数
    ├── create_colormap()           # カラーマップ生成
    ├── print_progress_bar()        # プログレスバー表示
    ├── format_usage_for_console()  # コンソール用操作説明
    ├── format_usage_for_gui()      # GUI用操作説明
    └── mandelbrot_set_vectorized() # マンデルブロ集合計算
```

## 技術的な特徴

- **NumPy ベクトル化計算**: 高速な並列計算でマンデルブロ集合を描画
- **カスタムカラーマップ**: 美しいグラデーション表現 (深い青 → 水色 → 黄 → 赤 → 黒)
- **日本語フォント対応**: macOS 標準フォント (Hiragino Sans) 使用
- **プログレスバー表示**: コンソールに計算進捗を表示
- **インタラクティブ操作**: マウスとキーボードでリアルタイム探索
- **型ヒント完備**: 全関数・クラスに型アノテーション付き
- **Google style docstring**: 統一されたドキュメント形式

## 設定のカスタマイズ

`mandelbrot.py` 内の `ViewerConfig` クラスで各種設定を変更できます：

| 設定項目 | デフォルト値 | 説明 |
|----------|-------------|------|
| `INITIAL_BOUNDS` | (-2.5, 1.0, -1.5, 1.5) | 初期表示範囲 |
| `DEFAULT_WIDTH` | 800 | 画像幅 (ピクセル) |
| `DEFAULT_HEIGHT` | 600 | 画像高さ (ピクセル) |
| `DEFAULT_MAX_ITER` | 256 | 最大反復回数 |
| `ZOOM_FACTOR_SCROLL_IN` | 0.8 | スクロールズームイン倍率 |
| `ZOOM_FACTOR_SCROLL_OUT` | 1.25 | スクロールズームアウト倍率 |
| `ZOOM_FACTOR_RIGHT_CLICK` | 0.8 | 右クリックズーム倍率 |
| `FIGURE_SIZE` | (12, 9) | ウィンドウサイズ |
| `SAVE_DPI` | 150 | 保存画像の解像度 |
| `COLORMAP_COLORS` | (RGB タプル) | カラーマップ用色定義 |

## ライセンス

MIT License
