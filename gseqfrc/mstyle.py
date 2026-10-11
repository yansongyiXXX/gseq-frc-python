"""MATLAB 默认绘图样式的复刻：色图、颜色循环、字体、刻度、图例。

`mplot` 原来是"能把图真的画出来"就行（见 `mplot` 的模块说明与 `PORTING.md` §3），
渲染用的是 matplotlib 的出厂值——色图 viridis、字体 DejaVu Sans、线宽 1.5、
刻度朝外。原论文的图是 MATLAB 画的，两边摆在一起差别一眼可见。这里把 MATLAB
那套默认值搬过来，让移植版出的图和 MATLAB 出的图对得上。

下面这些数不是猜的：能用 `get(0,'default...')` 量的就在本机 MATLAB R2023a 上量过，
能直接从论文附带的 `.fig` 里读的就读出来（`untitled.fig` 是第二篇 Fig.4 的 MATLAB
原图）。量到的值：

* 屏幕 1536 x 864 像素（`get(0,'ScreenSize')`）；
* 坐标区出厂默认 FontSize 10、LineWidth 0.5、TickDir 'in'、
  TickLength [0.01, 0.025]、LabelFontSizeMultiplier 1.1、颜色 [0.15 0.15 0.15]；
* 图例默认 FontSize = 0.9 x 坐标区 FontSize（实测 10 -> 9，25 -> 22.5），
  字重跟着坐标区走；
* ColorOrder 出厂 7 色；
* **显式给了颜色的线照样占用 ColorOrder 的下标**（实测：先 `plot(...,'m-')`
  再 `plot(...,'--')`，第二条拿到的是第 2 号色橙而不是第 1 号蓝，
  ColorOrderIndex 停在 3）。这一点很关键——论文 Fig.4 的 (d)(e) 两格正是
  洋红实线 + 橙色虚线，不照抄这条就画成别的颜色。
"""

from __future__ import annotations

import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler

#: MATLAB parula 色图。取自 `untitled.fig` 里那份 256 级 figure Colormap，
#: 这里等间隔抽 64 级建表（再插值回 256 级视觉上无差别）。
_PARULA_64 = [
    (0.2422, 0.1504, 0.6603),
    (0.2503, 0.1648, 0.7071),
    (0.2576, 0.1814, 0.7501),
    (0.2645, 0.1972, 0.7937),
    (0.2704, 0.2138, 0.8346),
    (0.2749, 0.2330, 0.8692),
    (0.2781, 0.2543, 0.8973),
    (0.2802, 0.2764, 0.9204),
    (0.2813, 0.2985, 0.9397),
    (0.2811, 0.3204, 0.9563),
    (0.2798, 0.3421, 0.9702),
    (0.2754, 0.3693, 0.9840),
    (0.2691, 0.3916, 0.9912),
    (0.2591, 0.4145, 0.9955),
    (0.2424, 0.4378, 0.9991),
    (0.2189, 0.4620, 0.9968),
    (0.1950, 0.4861, 0.9887),
    (0.1831, 0.5084, 0.9793),
    (0.1785, 0.5296, 0.9677),
    (0.1764, 0.5504, 0.9516),
    (0.1686, 0.5705, 0.9357),
    (0.1540, 0.5902, 0.9218),
    (0.1461, 0.6089, 0.9080),
    (0.1383, 0.6272, 0.8975),
    (0.1253, 0.6453, 0.8887),
    (0.1119, 0.6627, 0.8770),
    (0.0964, 0.6789, 0.8609),
    (0.0713, 0.6938, 0.8409),
    (0.0328, 0.7071, 0.8183),
    (0.0046, 0.7192, 0.7941),
    (0.0046, 0.7301, 0.7688),
    (0.0369, 0.7400, 0.7426),
    (0.1023, 0.7510, 0.7088),
    (0.1448, 0.7593, 0.6812),
    (0.1741, 0.7678, 0.6527),
    (0.1954, 0.7765, 0.6225),
    (0.2178, 0.7849, 0.5899),
    (0.2491, 0.7922, 0.5546),
    (0.2929, 0.7975, 0.5170),
    (0.3424, 0.8009, 0.4774),
    (0.3921, 0.8029, 0.4344),
    (0.4463, 0.8024, 0.3904),
    (0.5044, 0.7993, 0.3480),
    (0.5609, 0.7943, 0.3050),
    (0.6161, 0.7878, 0.2622),
    (0.6701, 0.7796, 0.2239),
    (0.7218, 0.7703, 0.1924),
    (0.7710, 0.7604, 0.1658),
    (0.8172, 0.7505, 0.1535),
    (0.8600, 0.7413, 0.1587),
    (0.9000, 0.7336, 0.1749),
    (0.9357, 0.7290, 0.2061),
    (0.9689, 0.7292, 0.2373),
    (0.9966, 0.7458, 0.2351),
    (0.9969, 0.7683, 0.2181),
    (0.9949, 0.7915, 0.2012),
    (0.9885, 0.8156, 0.1875),
    (0.9778, 0.8404, 0.1757),
    (0.9671, 0.8654, 0.1636),
    (0.9608, 0.8902, 0.1532),
    (0.9597, 0.9143, 0.1418),
    (0.9629, 0.9379, 0.1261),
    (0.9692, 0.9609, 0.1061),
    (0.9769, 0.9839, 0.0805),
]

PARULA = LinearSegmentedColormap.from_list("parula", _PARULA_64, N=256)

#: MATLAB 出厂 ColorOrder（`untitled.fig` 里读到的就是这一组）
COLOR_ORDER = [
    (0.0000, 0.4470, 0.7410),
    (0.8500, 0.3250, 0.0980),
    (0.9290, 0.6940, 0.1250),
    (0.4940, 0.1840, 0.5560),
    (0.4660, 0.6740, 0.1880),
    (0.3010, 0.7450, 0.9330),
    (0.6350, 0.0780, 0.1840),
]

#: MATLAB 的坐标区出厂默认值
FACTORY_AXES_FONTSIZE = 10.0
FACTORY_AXES_LINEWIDTH = 0.5

#: 坐标区文字/边框的颜色（MATLAB 的 XColor/YColor 出厂值）
AXES_COLOR = (0.15, 0.15, 0.15)

#: 刻度长度（点）。MATLAB 的 TickLength 是"最长轴长的 0.01"，换算成点大约 4~5 pt。
TICK_LENGTH = 4.5

#: MATLAB 的屏幕。`get(0,'ScreenSize')` 给的是 [1 1 1536 864]，100 dpi 折成英寸。
SCREEN_INCHES = (15.36, 8.64)

#: MATLAB 自带字体的首选。Helvetica 在 Windows 上一般没有，退到 Arial（外观几乎一样）；
#: 末尾缀上中文字体，`tools/` 里那些中文标题才不至于画成方框（matplotlib 3.6+
#: 会按这个列表逐字回退，拉丁字仍然用 Arial）。
FONT_STACK = ["Helvetica", "Arial", "Helvetica Neue", "DejaVu Sans",
              "Microsoft YaHei", "SimHei"]


def resolve_font_stack() -> list:
    """把 `FONT_STACK` 里**这台机器上真装了**的字体挑出来。

    没装的留在 `rcParams` 里，matplotlib 每找一次字体就往 stderr 打一行
    `findfont: Font family 'Helvetica' not found.`——图上没影响，日志很吵，
    所以装的时候先筛一遍。
    """
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    names = [name for name in FONT_STACK if name in available]
    return names or ["DejaVu Sans"]

_INSTALLED = False

#: MATLAB 的线型说明里允许出现的字符
MARKER_CHARS = "o+*.xsd^v><ph"
LINESTYLE_CHARS = {"-": "-", "--": "--", "-.": "-.", ":": ":"}
#: MATLAB 的单字符颜色（注意 g 是**纯绿** [0,1,0]，matplotlib 的 'g' 是暗绿）
COLOR_CHARS = {
    "r": (1.0, 0.0, 0.0),
    "g": (0.0, 1.0, 0.0),
    "b": (0.0, 0.0, 1.0),
    "c": (0.0, 1.0, 1.0),
    "m": (1.0, 0.0, 1.0),
    "y": (1.0, 1.0, 0.0),
    "k": (0.0, 0.0, 0.0),
    "w": (1.0, 1.0, 1.0),
}


def parse_linespec(spec):
    """把 MATLAB 的线型说明拆成 matplotlib 认的样式参数。

    `'k--'` -> 黑 + 虚线，`'mx'` -> 洋红的 x 标记（MATLAB 里没给线型说明就是不
    画线，matplotlib 那边用 `linestyle='None'` 表达），`':'` -> 只给线型。
    三段（颜色 / 线型 / 标记）在 MATLAB 里可以任意顺序，这里按同样的规则扫。
    """
    if not isinstance(spec, str) or not spec:
        return {}
    token = spec
    result: dict = {}
    for style in ("--", "-.", "-", ":"):
        if style in token:
            result["linestyle"] = style
            token = token.replace(style, "", 1)
            break
    for char in token:
        if char in MARKER_CHARS:
            result["marker"] = char
            token = token.replace(char, "", 1)
            break
    for char in token:
        if char in COLOR_CHARS:
            result["color"] = COLOR_CHARS[char]
            break
    # MATLAB 的 `plot(x,y,'mx')`：给了标记但没给线型 -> 只画标记，不连线。
    if "marker" in result and "linestyle" not in result:
        result["linestyle"] = "None"
    return result


def install() -> None:
    """把 MATLAB 的绘图默认值装进 matplotlib 的 rcParams（幂等）。"""
    global _INSTALLED
    if "parula" not in matplotlib.colormaps:
        matplotlib.colormaps.register(PARULA)
    if _INSTALLED:
        return
    _INSTALLED = True
    family = resolve_font_stack()
    matplotlib.rcParams.update({
        # ---- 字体：MATLAB 用 Helvetica；数学部分用 Computer Modern（LaTeX 解释器）
        # `font.family` 直接给**具体字体名列表**：matplotlib 只有在这种写法下才会
        # 逐字回退（写 'sans-serif' 别名时只解析成一个字体，中文会变方框）。
        "font.family": family,
        "font.sans-serif": family,
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        # ---- 坐标区：MATLAB 的箱子、朝内的刻度、四周都有刻度
        "axes.linewidth": FACTORY_AXES_LINEWIDTH,
        "axes.edgecolor": AXES_COLOR,
        "axes.labelcolor": AXES_COLOR,
        "axes.labelsize": FACTORY_AXES_FONTSIZE * 1.1,
        "axes.labelweight": "normal",
        "axes.titlesize": FACTORY_AXES_FONTSIZE * 1.1,
        "axes.titleweight": "bold",
        "axes.prop_cycle": cycler(color=[matplotlib.colors.to_hex(c)
                                         for c in COLOR_ORDER]),
        "xtick.color": AXES_COLOR,
        "ytick.color": AXES_COLOR,
        "xtick.labelsize": FACTORY_AXES_FONTSIZE,
        "ytick.labelsize": FACTORY_AXES_FONTSIZE,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": TICK_LENGTH,
        "ytick.major.size": TICK_LENGTH,
        "xtick.major.width": FACTORY_AXES_LINEWIDTH,
        "ytick.major.width": FACTORY_AXES_LINEWIDTH,
        # ---- 网格：MATLAB 的 `grid on` 默认是实线；脚本再按需改成点线
        "grid.color": AXES_COLOR,
        "grid.linewidth": 0.8,
        "grid.linestyle": ":",
        "grid.alpha": 0.25,
        # ---- 线：MATLAB 出厂 LineWidth 是 0.5，脚本几乎都会显式给 2~3
        "lines.linewidth": 0.5,
        "lines.solid_capstyle": "butt",
        # ---- 色图：MATLAB 从 R2014b 起默认 parula
        "image.cmap": "parula",
        # ---- 图窗：白底（MATLAB 出厂是浅灰，脚本都显式设成白的）
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 100,
        # ---- 图例：MATLAB 的图例白底、深灰边；字号在 `mplot.legend` 里按 0.9x 给
        "legend.framealpha": 1.0,
        "legend.edgecolor": AXES_COLOR,
        "legend.facecolor": "white",
        "legend.fancybox": False,
        "legend.borderpad": 0.3,
        "legend.handlelength": 1.6,
    })


def style_axes(axes, fontsize=None, fontweight="normal", linewidth=None,
               color=AXES_COLOR, box=True, grid=None, force=False):
    """把 MATLAB 的坐标区样式落到一个 matplotlib 的 `Axes` 上。

    对应 MATLAB 的 FontSize/FontWeight/LineWidth/XColor/YColor/Box/TickDir/
    TickLength：字号同时管刻度标签与轴标签（轴标签再乘 1.1），边框与刻度线用同一
    个线宽，刻度朝里、四周都画。

    同一套参数重复调用时直接返回：二维求解器每 100 轮画一次图、每轮里十几次
    `set(gca,...)`，每次都把刻度和文字重排一遍在实测里能占到总时间的 10%
    （`Axes.cla()` 之后必须重来，那时候 `_mplot_style_key` 已经被清掉了）。

    `grid` 给的是网格线的样式（`{"linestyle": ":", ...}`）。给 `None` 就完全不碰
    网格——**开不开网格由调用方决定**，这里只改已有的网格线长什么样。
    `force=True` 时跳过那道短路（动图每帧都会 `ax.clear()`，样式必须重来一遍）。
    """
    size = FACTORY_AXES_FONTSIZE if fontsize is None else float(np_scalar(fontsize))
    width = (FACTORY_AXES_LINEWIDTH if linewidth is None
             else float(np_scalar(linewidth)))
    key = (size, str(fontweight), width, _color_key(color), bool(box))
    if not force and getattr(axes, "_mplot_style_key", None) == key:
        return axes
    axes._mplot_style_key = key
    axes.tick_params(axis="both", which="both", direction="in",
                     top=box, right=box, labelsize=size,
                     width=width, length=TICK_LENGTH, colors=color)
    for spine in axes.spines.values():
        spine.set_linewidth(width)
        spine.set_color(color)
    axes.spines["top"].set_visible(box)
    axes.spines["right"].set_visible(box)
    for label in (axes.xaxis.label, axes.yaxis.label, axes.title):
        label.set_fontsize(size * 1.1)
        label.set_fontweight(fontweight)
        label.set_color(color)
    for label in list(axes.get_xticklabels()) + list(axes.get_yticklabels()):
        label.set_fontweight(fontweight)
    if grid is not None:
        # 不传 visible：只更新网格线的样式，不改"开不开"
        axes.grid(**grid)
    return axes


def np_scalar(value):
    """把可能是 1x1 数组的 MATLAB 标量取成 python 浮点。"""
    import numpy as np

    return float(np.asarray(value).ravel()[0])


def _color_key(color):
    """把颜色规格变成可比较、可哈希的键（字符串原样，RGB 序列转元组）。"""
    if isinstance(color, str):
        return color
    try:
        import numpy as np

        return tuple(float(v) for v in np.asarray(color).ravel())
    except Exception:  # noqa: BLE001
        return str(color)


def style_figure(fig, fontsize=13.0, fontweight="bold", linewidth=1.2,
                 grid=None, color=AXES_COLOR, box=True, legend_frame=False):
    """把一张普通 matplotlib 图（`plt.subplots` 那种）按 MATLAB 样式整体过一遍。

    `tools/` 下那些正向演化脚本是自己用 `plt.subplots` 画的（不是走 `mplot` 的移植
    脚本），这里给它们一个统一入口：坐标区（含色条那根轴）、图例、suptitle 一起按
    MATLAB 的规矩来。
    """
    for axes in list(fig.axes):
        style_axes(axes, fontsize=fontsize, fontweight=fontweight,
                   linewidth=linewidth, color=color, box=box, grid=grid,
                   force=True)
        handle = axes.get_legend()
        if handle is not None:
            handle.set_frame_on(legend_frame)
            for text in handle.get_texts():
                text.set_fontsize(fontsize * 0.9)
                text.set_fontweight(fontweight)
    for text in fig.texts:                      # suptitle 之类
        text.set_fontweight(fontweight)
    return fig


#: MATLAB 那个"点线网格"的样子（脚本里 `set(gca,'GridLineStyle',':','GridColor','k',
#: 'GridAlpha',0.25)` 就是这一组）
GRID_STYLE = {"linestyle": ":", "color": AXES_COLOR, "alpha": 0.25}

#: MATLAB 的单字符颜色，`tools/` 里想直接用 `'r-'` 那种写法时取这里
RED = COLOR_CHARS["r"]
GREEN = COLOR_CHARS["g"]
BLUE = COLOR_CHARS["b"]
CYAN = COLOR_CHARS["c"]
MAGENTA = COLOR_CHARS["m"]
BLACK = COLOR_CHARS["k"]
