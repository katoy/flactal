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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import sys

# 定数
INITIAL_BOUNDS = (-2.5, 1.0, -1.5, 1.5)  # 初期表示範囲 (xmin, xmax, ymin, ymax)
ZOOM_FACTOR_SCROLL_UP = 0.8  # マウスホイール上方向（ズームアウト）
ZOOM_FACTOR_SCROLL_DOWN = 1.25  # マウスホイール下方向（ズームイン）
ZOOM_FACTOR_RIGHT_CLICK = 0.8  # 右クリック（ズームイン）

# macOS用の日本語フォント設定
mpl.rcParams['font.family'] = [
    'Hiragino Sans', 'Hiragino Maru Gothic Pro', 'sans-serif'
]


def mandelbrot_set_vectorized(
    xmin: float, xmax: float,
    ymin: float, ymax: float,
    width: int, height: int,
    max_iter: int,
    show_progress: bool = True
) -> np.ndarray:
    """
    マンデルブロ集合をベクトル化して高速に計算する
    """
    import sys

    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    Z = np.zeros_like(C)
    M = np.zeros_like(C, dtype=float)

    bar_width = 30
    # プログレスバー更新頻度を調整: max_iter の1%ごとに更新 (ただし最低1回は更新)
    update_interval = max(1, max_iter // 100)

    for i in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        M[mask] = i + 1

        # プログレスバー表示
        if show_progress and (i % update_interval == 0 or i == max_iter - 1):
            progress = (i + 1) / max_iter
            filled = int(progress * bar_width)
            empty = bar_width - filled
            percent = int(progress * 100)
            sys.stdout.write(
                f"\r🔄 計算中: [{'█' * filled}{'░' * empty}] {percent:>3}%"
            )
            sys.stdout.flush()

    if show_progress:
        print(" 完了!")

    return M


def create_colormap():
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

    def __init__(self, width=800, height=600, max_iter=256):
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.cmap = create_colormap()

        # 初期表示範囲
        self.initial_bounds = INITIAL_BOUNDS
        self.xmin, self.xmax, self.ymin, self.ymax = self.initial_bounds

        # 画像保存カウンタ
        self.save_counter = 0

        self._setup_plot()
        self._update_image()

    def _setup_plot(self):
        """プロットの初期設定"""
        self.fig, self.ax = plt.subplots(figsize=(12, 9), dpi=100)
        self.fig.canvas.manager.set_window_title('マンデルブロ集合ビューア')

        # 初期画像（ダミー）
        self.im = self.ax.imshow(
            np.zeros((self.height, self.width)),
            extent=[self.xmin, self.xmax, self.ymin, self.ymax],
            cmap=self.cmap,
            origin='lower',
            aspect='equal'
        )

        self.ax.set_title('マンデルブロ集合 (Mandelbrot Set)', fontsize=16)
        self.ax.set_xlabel('Re(c) - 実部', fontsize=12)
        self.ax.set_ylabel('Im(c) - 虚部', fontsize=12)

        # カラーバー
        self.cbar = plt.colorbar(self.im, ax=self.ax, shrink=0.8)
        self.cbar.set_label('反復回数', fontsize=12)

        # ステータステキスト
        self.status_text = self.fig.text(
            0.02, 0.02, '', fontsize=10,
            transform=self.fig.transFigure,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        # 操作説明
        help_text = (
            "操作: ホイール=拡大/縮小, 左クリック=移動, "
            "右クリック=ズームイン, r=リセット, s=保存, q=終了"
        )
        self.fig.text(
            0.5, 0.02, help_text, fontsize=9,
            ha='center', transform=self.fig.transFigure,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        )

        # イベントハンドラ接続
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)

    def _update_image(self):
        """画像を更新"""
        self.status_text.set_text('計算中...')
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # マンデルブロ集合を計算
        M = mandelbrot_set_vectorized(
            self.xmin, self.xmax, self.ymin, self.ymax,
            self.width, self.height, self.max_iter
        )

        # 画像を更新
        self.im.set_data(M)
        self.im.set_extent([self.xmin, self.xmax, self.ymin, self.ymax])
        self.im.set_clim(0, self.max_iter)

        # ステータス更新
        zoom_level = 3.5 / (self.xmax - self.xmin)
        center_x = (self.xmin + self.xmax) / 2
        center_y = (self.ymin + self.ymax) / 2
        self.status_text.set_text(
            f'中心: ({center_x:.6f}, {center_y:.6f}i) | ズーム: ×{zoom_level:.2f}'
        )

        self.fig.canvas.draw_idle()

    def _on_scroll(self, event):
        """マウスホイールによるズーム"""
        if event.inaxes != self.ax:
            return

        # ズーム倍率
        zoom_factor = ZOOM_FACTOR_SCROLL_UP if event.button == 'up' else ZOOM_FACTOR_SCROLL_DOWN

        # マウス位置を中心にズーム
        x_center = event.xdata
        y_center = event.ydata

        x_range = (self.xmax - self.xmin) * zoom_factor
        y_range = (self.ymax - self.ymin) * zoom_factor

        self.xmin = x_center - x_range / 2
        self.xmax = x_center + x_range / 2
        self.ymin = y_center - y_range / 2
        self.ymax = y_center + y_range / 2

        self._update_image()

    def _on_press(self, event):
        """マウスボタン押下"""
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # 左クリック: クリック位置を中心に移動
            x_center = event.xdata
            y_center = event.ydata

            x_range = self.xmax - self.xmin
            y_range = self.ymax - self.ymin

            self.xmin = x_center - x_range / 2
            self.xmax = x_center + x_range / 2
            self.ymin = y_center - y_range / 2
            self.ymax = y_center + y_range / 2

            self._update_image()
        elif event.button == 3:  # 右クリック: ズームイン (1.25x)
            x_center = event.xdata
            y_center = event.ydata
            zoom_factor = ZOOM_FACTOR_RIGHT_CLICK  # 定数を使用

            x_range = (self.xmax - self.xmin) * zoom_factor
            y_range = (self.ymax - self.ymin) * zoom_factor

            self.xmin = x_center - x_range / 2
            self.xmax = x_center + x_range / 2
            self.ymin = y_center - y_range / 2
            self.ymax = y_center + y_range / 2

            self._update_image()

    def _on_key(self, event):
        """キー入力"""
        if event.key == 'r':  # リセット
            self.xmin, self.xmax, self.ymin, self.ymax = self.initial_bounds
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
