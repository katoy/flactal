#!/usr/bin/env python3
"""
マンデルブロ集合を描画するスクリプト (インタラクティブ版)
M1 Mac (Apple Silicon) 対応

操作方法:
- マウスホイール: 拡大/縮小
- 左クリック: クリック位置を中心に移動（パン）
- 右クリック: クリック位置を中心にズームイン
- 'r' キー: 初期表示にリセット
- 's' キー: 現在の表示を画像として保存
- 'q' キー: 終了
"""

import sys
from dataclasses import dataclass
from typing import Tuple, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import AxesImage
from matplotlib.text import Text

# 定数
INITIAL_X_MIN = -2.5
INITIAL_X_MAX = 1.0
INITIAL_Y_MIN = -1.5
INITIAL_Y_MAX = 1.5

ZOOM_FACTOR_SCROLL_UP = 0.8  # ズームアウト
ZOOM_FACTOR_SCROLL_DOWN = 1.25  # ズームイン
ZOOM_FACTOR_RIGHT_CLICK = 0.8  # 右クリック時のズーム率（縮小範囲＝拡大）

# macOS用の日本語フォント設定
mpl.rcParams['font.family'] = [
    'Hiragino Sans', 'Hiragino Maru Gothic Pro', 'sans-serif'
]


@dataclass
class ViewPort:
    """描画範囲（ビューポート）を管理するクラス"""
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @classmethod
    def default(cls) -> 'ViewPort':
        return cls(INITIAL_X_MIN, INITIAL_X_MAX, INITIAL_Y_MIN, INITIAL_Y_MAX)

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def center(self) -> complex:
        return complex((self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2)

    @property
    def extent(self) -> List[float]:
        return [self.xmin, self.xmax, self.ymin, self.ymax]

    def pan(self, x_center: float, y_center: float) -> None:
        """指定した中心座標に移動する"""
        w = self.width
        h = self.height
        self.xmin = x_center - w / 2
        self.xmax = x_center + w / 2
        self.ymin = y_center - h / 2
        self.ymax = y_center + h / 2

    def zoom(self, factor: float, center: Optional[Tuple[float, float]] = None) -> None:
        """
        ズームを行う
        :param factor: ズーム倍率 (1.0未満で拡大、1.0以上で縮小...としたいが、
                       現状のロジックは範囲にかける係数なので、
                       factor < 1.0 -> 範囲が狭くなる -> ズームイン
                       factor > 1.0 -> 範囲が広くなる -> ズームアウト
                       となっている点に注意)
        :param center: ズームの中心座標 (Noneの場合は現在の中心)
        """
        if center:
            cx, cy = center
        else:
            cx, cy = self.center.real, self.center.imag

        new_width = self.width * factor
        new_height = self.height * factor

        self.xmin = cx - new_width / 2
        self.xmax = cx + new_width / 2
        self.ymin = cy - new_height / 2
        self.ymax = cy + new_height / 2


def mandelbrot_set_vectorized(
    viewport: ViewPort,
    width: int,
    height: int,
    max_iter: int
) -> np.ndarray:
    """
    マンデルブロ集合をベクトル化して高速に計算する
    """
    x = np.linspace(viewport.xmin, viewport.xmax, width)
    y = np.linspace(viewport.ymin, viewport.ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    Z = np.zeros_like(C)
    M = np.zeros_like(C, dtype=float)

    # ベクトル化された計算
    # 多少メモリを食うがPythonループより圧倒的に速い
    for i in range(max_iter):
        mask = np.abs(Z) <= 2
        if not np.any(mask):
            break
        
        # マスクされた部分（発散していない部分）のみ計算
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        M[mask] = i + 1

    return M


def create_colormap() -> LinearSegmentedColormap:
    """美しいカラーマップを作成する"""
    colors = [
        (0.0, 0.0, 0.2),
        (0.1, 0.2, 0.5),
        (0.2, 0.5, 0.8),
        (0.5, 0.8, 0.9),
        (1.0, 1.0, 0.8),
        (1.0, 0.8, 0.3),
        (1.0, 0.5, 0.1),
        (0.8, 0.2, 0.1),
        (0.5, 0.0, 0.2),
        (0.0, 0.0, 0.0),
    ]
    return LinearSegmentedColormap.from_list("mandelbrot", colors, N=256)


class MandelbrotViewer:
    """インタラクティブなマンデルブロ集合ビューア"""

    def __init__(self, width: int = 800, height: int = 600, max_iter: int = 256):
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.cmap = create_colormap()

        # 表示範囲管理
        self.viewport = ViewPort.default()
        
        # 画像保存カウンタ
        self.save_counter = 0

        # Matplotlib オブジェクト
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.im: Optional[AxesImage] = None
        self.status_text: Optional[Text] = None

        self._setup_plot()
        self._update_image()

    def _setup_plot(self):
        """プロットの初期設定"""
        self.fig, self.ax = plt.subplots(figsize=(12, 9), dpi=100)
        
        if self.fig.canvas.manager:
            self.fig.canvas.manager.set_window_title('マンデルブロ集合ビューア')

        # 初期画像（ダミー）
        self.im = self.ax.imshow(
            np.zeros((self.height, self.width)),
            extent=self.viewport.extent,
            cmap=self.cmap,
            origin='lower',
            aspect='equal'
        )

        self.ax.set_title('マンデルブロ集合 (Mandelbrot Set)', fontsize=16)
        self.ax.set_xlabel('Re(c) - 実部', fontsize=12)
        self.ax.set_ylabel('Im(c) - 虚部', fontsize=12)

        # カラーバー
        cbar = plt.colorbar(self.im, ax=self.ax, shrink=0.8)
        cbar.set_label('反復回数', fontsize=12)

        # ステータステキスト
        self.status_text = self.fig.text(
            0.02, 0.02, '', fontsize=10,
            transform=self.fig.transFigure,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        # 操作説明
        help_text = (
            "操作:\n"
            "  - ホイール: 拡大/縮小\n"
            "  - 左クリック: クリック位置を中心に移動（パン）\n"
            "  - 右クリック: クリック位置を中心にズームイン\n"
            "  - r: リセット, s: 保存, q: 終了"
        )
        self.fig.text(
            0.5, 0.02, help_text,
            fontsize=9,
            ha='center',
            transform=self.fig.transFigure,
            bbox=dict(
                boxstyle='round',
                facecolor='lightblue',
                alpha=0.8
            )
        )

        # イベントハンドラ接続
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)

    def _update_image(self):
        """画像を更新"""
        if self.status_text:
            self.status_text.set_text('計算中...')
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        # マンデルブロ集合を計算
        M = mandelbrot_set_vectorized(
            self.viewport,
            self.width, self.height, self.max_iter
        )

        # 画像を更新
        self.im.set_data(M)
        self.im.set_extent(self.viewport.extent)
        self.im.set_clim(0, self.max_iter)

        # ステータス更新
        zoom_level = 3.5 / self.viewport.width
        c = self.viewport.center
        
        if self.status_text:
            self.status_text.set_text(
                f'中心: ({c.real:.6f}, {c.imag:.6f}i) | ズーム: ×{zoom_level:.2f}'
            )

        self.fig.canvas.draw_idle()

    def _on_scroll(self, event):
        """マウスホイールによるズーム"""
        if event.inaxes != self.ax:
            return

        # ズーム倍率 (範囲にかける係数)
        # up (スクロールアップ) -> ズームアウト -> 範囲を広げる -> factor > 1 ?
        # 元のコード:
        # ZOOM_FACTOR_SCROLL_UP = 0.8  (上方向) -> 範囲が狭くなる -> ズームイン？
        # 一般的にGoogleMaps等はスクロールアップでズームイン。
        # 元のコードの定数名と値を確認:
        # ZOOM_FACTOR_SCROLL_UP = 0.8  -> 範囲 * 0.8 -> 範囲縮小 -> ズームイン
        # ZOOM_FACTOR_SCROLL_DOWN = 1.25 -> 範囲 * 1.25 -> 範囲拡大 -> ズームアウト
        # 変数名と挙動が逆っぽいが、元のコードの挙動（値）を尊重する。
        # ただし、操作感としてはスクロールアップで拡大（値0.8）が直感的。
        
        factor = (
            ZOOM_FACTOR_SCROLL_UP if event.button == 'up'
            else ZOOM_FACTOR_SCROLL_DOWN
        )

        self.viewport.zoom(factor, center=(event.xdata, event.ydata))
        self._update_image()

    def _on_press(self, event):
        """マウスボタン押下"""
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # 左クリック: パン
            self.viewport.pan(event.xdata, event.ydata)
            self._update_image()

        elif event.button == 3:  # 右クリック: ズームイン
            # 右クリックでズームイン（範囲を狭める）
            self.viewport.zoom(ZOOM_FACTOR_RIGHT_CLICK, center=(event.xdata, event.ydata))
            self._update_image()

    def _on_key(self, event):
        """キー入力"""
        if event.key == 'r':  # リセット
            self.viewport = ViewPort.default()
            self._update_image()
        elif event.key == 's':  # 保存
            self.save_counter += 1
            filename = f'mandelbrot_{self.save_counter:03d}.png'
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"画像を保存しました: {filename}")
        elif event.key == 'q':  # 終了
            plt.close(self.fig)

    def show(self):
        """ビューアを表示"""
        plt.show()


def main():
    print("マンデルブロ集合ビューアを起動中...")
    print("\n操作方法:")
    print("  - マウスホイール: 拡大/縮小")
    print("  - 左クリック: クリック位置を中心に移動（パン）")
    print("  - 右クリック: クリック位置を中心にズームイン")
    print("  - 'r' キー: 初期表示にリセット")
    print("  - 's' キー: 現在の表示を画像として保存")
    print("  - 'q' キー: 終了\n")

    viewer = MandelbrotViewer(
        width=800,
        height=600,
        max_iter=256
    )
    viewer.show()


if __name__ == "__main__":
    main()
