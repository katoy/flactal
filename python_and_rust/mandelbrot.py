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

import copy
import sys
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Callable, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backend_bases import MouseEvent, KeyEvent, MouseButton
from matplotlib.text import Text
import matplotlib as mpl

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.image import AxesImage
    from matplotlib.colorbar import Colorbar


# =============================================================================
# 定数
# =============================================================================
@dataclass(frozen=True)
class ViewerConfig:
    """ビューア設定の定数。

    Attributes:
        INITIAL_BOUNDS: 初期表示範囲 (xmin, xmax, ymin, ymax)
        ZOOM_FACTOR_SCROLL_IN: スクロールズームイン倍率
        ZOOM_FACTOR_SCROLL_OUT: スクロールズームアウト倍率
        ZOOM_FACTOR_RIGHT_CLICK: 右クリックズーム倍率
        DEFAULT_WIDTH: デフォルト画像幅 (ピクセル)
        DEFAULT_HEIGHT: デフォルト画像高さ (ピクセル)
        DEFAULT_MAX_ITER: デフォルト最大反復回数
        FIGURE_SIZE: ウィンドウサイズ (幅, 高さ)
        FIGURE_DPI: 表示DPI
        SAVE_DPI: 保存DPI
        INITIAL_X_RANGE: 初期x軸範囲 (ズームレベル計算用)
        PROGRESS_BAR_WIDTH: プログレスバー文字幅
        COLORMAP_COLORS: カラーマップ用RGB色リスト
    """
    # 初期表示範囲 (xmin, xmax, ymin, ymax)
    INITIAL_BOUNDS: Tuple[float, float, float, float] = (-2.5, 1.0, -1.5, 1.5)

    # ズーム倍率
    ZOOM_FACTOR_SCROLL_IN: float = 0.8
    ZOOM_FACTOR_SCROLL_OUT: float = 1.25
    ZOOM_FACTOR_RIGHT_CLICK: float = 0.8

    # 表示設定
    DEFAULT_WIDTH: int = 800
    DEFAULT_HEIGHT: int = 600
    DEFAULT_MAX_ITER: int = 256
    FIGURE_SIZE: Tuple[int, int] = (12, 9)
    FIGURE_DPI: int = 100
    SAVE_DPI: int = 150

    # 初期x軸範囲 (ズームレベル計算用)
    INITIAL_X_RANGE: float = 3.5

    # プログレスバー
    PROGRESS_BAR_WIDTH: int = 30

    # カラーマップ用の色定義 (RGB タプルのリスト)
    COLORMAP_COLORS: Tuple[Tuple[float, float, float], ...] = (
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
    )


CONFIG = ViewerConfig()

# 操作説明 (コンソール出力とGUI表示で共有)
USAGE_LINES: Tuple[Tuple[str, str], ...] = (
    ("マウスホイール", "拡大/縮小"),
    ("左クリック", "クリック位置を中心に移動（パン）"),
    ("右クリック", "クリック位置を中心にズームイン"),
    ("'r' キー", "初期表示にリセット"),
    ("'s' キー", "現在の表示を画像として保存"),
    ("'q' キー", "終了"),
)


# =============================================================================
# macOS用の日本語フォント設定
# =============================================================================
mpl.rcParams['font.family'] = [
    'Hiragino Sans', 'Hiragino Maru Gothic Pro', 'sans-serif'
]


# =============================================================================
# ユーティリティ関数
# =============================================================================
def create_colormap(colors: Tuple[Tuple[float, float, float], ...] = CONFIG.COLORMAP_COLORS) -> LinearSegmentedColormap:
    """美しいカラーマップを作成する。

    Args:
        colors: RGB色のタプル (各要素は0.0-1.0)

    Returns:
        作成したカラーマップ
    """
    return LinearSegmentedColormap.from_list("mandelbrot", list(colors), N=256)


def print_progress_bar(
    progress: float,
    bar_width: int = CONFIG.PROGRESS_BAR_WIDTH
) -> None:
    """プログレスバーを表示する。

    Args:
        progress: 進捗率 (0.0-1.0)
        bar_width: バーの文字幅
    """
    filled = int(progress * bar_width)
    empty = bar_width - filled
    percent = int(progress * 100)
    sys.stdout.write(
        f"\r🔄 計算中(py): [{'█' * filled}{'░' * empty}] {percent:>3}%"
    )
    sys.stdout.flush()


def format_usage_for_console() -> str:
    """コンソール出力用の操作説明を生成する。

    Returns:
        整形された操作説明文字列
    """
    lines = ["マンデルブロ集合ビューアを起動中...", "", "操作方法:"]
    for key, desc in USAGE_LINES:
        lines.append(f"  - {key}: {desc}")
    lines.append("")
    return "\n".join(lines)


def format_usage_for_gui() -> str:
    """GUI表示用の操作説明を生成する。

    Returns:
        整形された操作説明文字列 (短縮版)
    """
    # GUIでは短縮表記を使用
    short_descs = {
        "マウスホイール": "ホイール",
        "'r' キー": "r",
        "'s' キー": "s",
        "'q' キー": "q",
    }
    lines = ["操作:"]
    for key, desc in USAGE_LINES:
        short_key = short_descs.get(key, key)
        # 最後の3つはまとめて表示
        if key in ("'r' キー", "'s' キー", "'q' キー"):
            continue
        lines.append(f"  - {short_key}: {desc}")
    lines.append("  - r: リセット, s: 保存, q: 終了")
    return "\n".join(lines)


# =============================================================================
# マンデルブロ集合計算
# =============================================================================

# Rust拡張の読み込みを試行
try:
    import mandelbrot_rs
    _USE_RUST = True
except ImportError:
    _USE_RUST = False


def _mandelbrot_python(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    width: int,
    height: int,
    max_iter: int,
    show_progress: bool = True
) -> np.ndarray:
    """Pure Python版マンデルブロ集合計算 (フォールバック用)。"""
    # 複素平面上のグリッドを生成
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    Z = np.zeros_like(C)
    M = np.zeros_like(C, dtype=float)

    # プログレスバー更新頻度: max_iter の1%ごと (最低1回)
    update_interval = max(1, max_iter // 100)

    for i in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        M[mask] = i + 1

        # プログレスバー表示
        if show_progress and (i % update_interval == 0 or i == max_iter - 1):
            progress = (i + 1) / max_iter
            print_progress_bar(progress)

    if show_progress:
        print(" 完了!")

    return M


def mandelbrot_set_vectorized(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    width: int,
    height: int,
    max_iter: int,
    show_progress: bool = True
) -> np.ndarray:
    """マンデルブロ集合を計算する。

    Rust拡張が利用可能な場合は高速なRust版を使用し、
    そうでない場合はPure Python版にフォールバックする。

    Args:
        xmin: x軸の最小値
        xmax: x軸の最大値
        ymin: y軸の最小値
        ymax: y軸の最大値
        width: 画像幅 (ピクセル)
        height: 画像高さ (ピクセル)
        max_iter: 最大反復回数
        show_progress: プログレスバーを表示するか

    Returns:
        反復回数を格納した2次元配列 (height x width)
    """
    if _USE_RUST:
        if show_progress:
            sys.stdout.write("🚀 Rust版で計算中...")
            sys.stdout.flush()
        result = mandelbrot_rs.mandelbrot_set_vectorized(
            xmin, xmax, ymin, ymax, width, height, max_iter
        )
        if show_progress:
            print(" 完了!")
        return result
    else:
        return _mandelbrot_python(
            xmin, xmax, ymin, ymax, width, height, max_iter, show_progress
        )



# =============================================================================
# ViewBounds クラス
# =============================================================================
@dataclass
class ViewBounds:
    """表示範囲を管理するデータクラス。

    Attributes:
        xmin: x軸の最小値
        xmax: x軸の最大値
        ymin: y軸の最小値
        ymax: y軸の最大値
    """
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @classmethod
    def from_tuple(cls, bounds: Tuple[float, float, float, float]) -> 'ViewBounds':
        """タプルからインスタンスを生成する。

        Args:
            bounds: (xmin, xmax, ymin, ymax) のタプル

        Returns:
            新しい ViewBounds インスタンス
        """
        return cls(*bounds)

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """タプルに変換する。

        Returns:
            (xmin, xmax, ymin, ymax) のタプル
        """
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    @property
    def x_range(self) -> float:
        """x軸の範囲を取得する。"""
        return self.xmax - self.xmin

    @property
    def y_range(self) -> float:
        """y軸の範囲を取得する。"""
        return self.ymax - self.ymin

    @property
    def center(self) -> Tuple[float, float]:
        """中心座標を取得する。"""
        return ((self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2)

    def zoom_to(
        self,
        x_center: float,
        y_center: float,
        factor: float
    ) -> 'ViewBounds':
        """指定した中心点に向かってズームする。

        Args:
            x_center: ズーム中心のx座標
            y_center: ズーム中心のy座標
            factor: ズーム倍率 (< 1でズームイン、> 1でズームアウト)

        Returns:
            新しい ViewBounds インスタンス
        """
        new_x_range = self.x_range * factor
        new_y_range = self.y_range * factor

        return ViewBounds(
            xmin=x_center - new_x_range / 2,
            xmax=x_center + new_x_range / 2,
            ymin=y_center - new_y_range / 2,
            ymax=y_center + new_y_range / 2
        )

    def pan_to(self, x_center: float, y_center: float) -> 'ViewBounds':
        """指定した点を中心に移動する。

        Args:
            x_center: 新しい中心のx座標
            y_center: 新しい中心のy座標

        Returns:
            新しい ViewBounds インスタンス
        """
        return self.zoom_to(x_center, y_center, 1.0)


# =============================================================================
# MandelbrotViewer クラス
# =============================================================================
class MandelbrotViewer:
    """インタラクティブなマンデルブロ集合ビューア。

    Attributes:
        width: 画像幅 (ピクセル)
        height: 画像高さ (ピクセル)
        max_iter: 最大反復回数
        cmap: カラーマップ
        initial_bounds: 初期表示範囲
        bounds: 現在の表示範囲
        save_counter: 画像保存カウンタ
    """

    def __init__(
        self,
        width: int = CONFIG.DEFAULT_WIDTH,
        height: int = CONFIG.DEFAULT_HEIGHT,
        max_iter: int = CONFIG.DEFAULT_MAX_ITER
    ) -> None:
        """ビューアを初期化する。

        Args:
            width: 画像幅 (ピクセル)
            height: 画像高さ (ピクセル)
            max_iter: 最大反復回数
        """
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.cmap = create_colormap()

        # 表示範囲
        self.initial_bounds = ViewBounds.from_tuple(CONFIG.INITIAL_BOUNDS)
        self.bounds = ViewBounds.from_tuple(CONFIG.INITIAL_BOUNDS)

        # 画像保存カウンタ
        self.save_counter = 0

        # プロット要素 (型ヒントのためにOptionalで初期化)
        self.fig: Optional['Figure'] = None
        self.ax: Optional['Axes'] = None
        self.im: Optional['AxesImage'] = None
        self.cbar: Optional['Colorbar'] = None
        self.status_text: Optional[Text] = None

        # キーハンドラマッピング
        self._key_handlers: Dict[str, Callable[[], None]] = {
            'r': self._reset_view,
            's': self._save_image,
            'q': self._quit,
        }

        self._setup_plot()
        self._update_image()

    def _setup_plot(self) -> None:
        """プロットの初期設定を行う。"""
        self.fig, self.ax = plt.subplots(
            figsize=CONFIG.FIGURE_SIZE,
            dpi=CONFIG.FIGURE_DPI
        )
        self.fig.canvas.manager.set_window_title('マンデルブロ集合ビューア')

        # 初期画像（ダミー）
        self.im = self.ax.imshow(
            np.zeros((self.height, self.width)),
            extent=list(self.bounds.to_tuple()),
            cmap=self.cmap,
            origin='lower',
            aspect='equal'
        )

        self._setup_labels()
        self._setup_colorbar()
        self._setup_status_text()
        self._setup_help_text()
        self._connect_events()

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)

    def _setup_labels(self) -> None:
        """タイトルとラベルを設定する。"""
        self.ax.set_title('マンデルブロ集合 (Mandelbrot Set)', fontsize=16)
        self.ax.set_xlabel('Re(c) - 実部', fontsize=12)
        self.ax.set_ylabel('Im(c) - 虚部', fontsize=12)

    def _setup_colorbar(self) -> None:
        """カラーバーを設定する。"""
        self.cbar = plt.colorbar(self.im, ax=self.ax, shrink=0.8)
        self.cbar.set_label('反復回数', fontsize=12)

    def _setup_status_text(self) -> None:
        """ステータステキストを設定する。"""
        self.status_text = self.fig.text(
            0.02, 0.02, '',
            fontsize=10,
            transform=self.fig.transFigure,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

    def _setup_help_text(self) -> None:
        """操作説明テキストを設定する。"""
        self.fig.text(
            0.5, 0.02, format_usage_for_gui(),
            fontsize=9,
            ha='center',
            transform=self.fig.transFigure,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        )

    def _connect_events(self) -> None:
        """イベントハンドラを接続する。"""
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _update_image(self) -> None:
        """画像を更新する。"""
        self._set_status('計算中...')
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # マンデルブロ集合を計算
        M = mandelbrot_set_vectorized(
            self.bounds.xmin, self.bounds.xmax,
            self.bounds.ymin, self.bounds.ymax,
            self.width, self.height, self.max_iter
        )

        # 画像を更新
        self.im.set_data(M)
        self.im.set_extent(list(self.bounds.to_tuple()))
        self.im.set_clim(0, self.max_iter)

        # ステータス更新
        self._update_status_display()
        self.fig.canvas.draw_idle()

    def _set_status(self, text: str) -> None:
        """ステータステキストを設定する。

        Args:
            text: 表示するテキスト
        """
        self.status_text.set_text(text)

    def _update_status_display(self) -> None:
        """ステータス表示を更新する。"""
        zoom_level = CONFIG.INITIAL_X_RANGE / self.bounds.x_range
        center_x, center_y = self.bounds.center
        self._set_status(
            f'中心: ({center_x:.6f}, {center_y:.6f}i) | ズーム: ×{zoom_level:.2f}'
        )

    def _on_scroll(self, event: MouseEvent) -> None:
        """マウスホイールによるズームを処理する。

        Args:
            event: マウスイベント
        """
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        # ズーム倍率を決定
        zoom_factor = (
            CONFIG.ZOOM_FACTOR_SCROLL_IN if event.button == 'up'
            else CONFIG.ZOOM_FACTOR_SCROLL_OUT
        )

        # マウス位置を中心にズーム
        self.bounds = self.bounds.zoom_to(event.xdata, event.ydata, zoom_factor)
        self._update_image()

    def _on_press(self, event: MouseEvent) -> None:
        """マウスボタン押下を処理する。

        Args:
            event: マウスイベント
        """
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        new_bounds = None
        if event.button == MouseButton.LEFT:
            # 左クリック: クリック位置を中心に移動
            new_bounds = self.bounds.pan_to(event.xdata, event.ydata)
        elif event.button == MouseButton.RIGHT:
            # 右クリック: ズームイン
            new_bounds = self.bounds.zoom_to(
                event.xdata, event.ydata,
                CONFIG.ZOOM_FACTOR_RIGHT_CLICK
            )

        if new_bounds is not None:
            self.bounds = new_bounds
            self._update_image()

    def _on_key(self, event: KeyEvent) -> None:
        """キー入力を処理する。

        Args:
            event: キーイベント
        """
        handler = self._key_handlers.get(event.key)
        if handler:
            handler()

    def _reset_view(self) -> None:
        """表示を初期状態にリセットする。"""
        self.bounds = copy.copy(self.initial_bounds)
        self._update_image()

    def _save_image(self) -> None:
        """現在の表示を画像として保存する。"""
        self.save_counter += 1
        filename = f'mandelbrot_{self.save_counter:03d}.png'
        self.fig.savefig(filename, dpi=CONFIG.SAVE_DPI, bbox_inches='tight')
        print(f"画像を保存しました: {filename}")

    def _quit(self) -> None:
        """ビューアを終了する。"""
        plt.close(self.fig)

    def show(self) -> None:
        """ビューアを表示する。"""
        plt.show()


# =============================================================================
# エントリーポイント
# =============================================================================
def main() -> None:
    """メイン関数。"""
    print(format_usage_for_console())

    viewer = MandelbrotViewer(
        width=CONFIG.DEFAULT_WIDTH,
        height=CONFIG.DEFAULT_HEIGHT,
        max_iter=CONFIG.DEFAULT_MAX_ITER
    )
    viewer.show()


if __name__ == "__main__":
    main()
