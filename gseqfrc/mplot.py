"""MATLAB 绘图的极简替代：保持调用结构，把图真的画出来。

移植版照抄原文的绘图调用（`figure` / `subplot` / `plot` / `xlabel` / `title` /
`print`），一是保持脚本结构，二是让程序跑完真的能得到能看的图。`num2str` 生成的
图名也照 MATLAB 的格式拼。

**渲染风格**照着 MATLAB 走：色图 parula、颜色循环、字体（Helvetica/Arial +
LaTeX 数学）、刻度朝里、四周带刻度、字号/线宽/图例/色条的默认倍数，都在
`gseqfrc/mstyle.py` 里按本机 MATLAB R2023a 实测（或从论文附带的 `.fig` 里读出）
的值定好，导入时一次装进 rcParams。线型说明（`'r--'`、`'mx'`、`'ko-'`）也
真的解析成颜色/线型/标记，不再像原来那样"认出来就丢掉"。渲染器毕竟还是
matplotlib，做不到逐像素相同，但配色、字号、版式与 MATLAB 出的图是一致的。

**不弹窗**：后端固定成 Agg，`figure()` 只在内存里建图，`print()` 把图写进产出目录
（`mio.set_output_dir()` 指定的那个）。所以整套流程可以无人值守地跑完。
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # 不弹窗，直接出图
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.legend import Legend  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from . import mio  # noqa: E402
from . import mstyle  # noqa: E402

#: MATLAB 那套默认样式（字体、色图 parula、颜色循环、刻度朝里…）在导入时一次装好。
#: 具体的取值与依据见 `mstyle` 的模块说明。
mstyle.install()

# MATLAB 的坐标区**不留边距**（自动范围就是数据范围本身），matplotlib 默认四周留
# 5%——`figure_4_2.m` 面板 (a) 的数据到 0.153，MATLAB 的 ylim 就是 [0, 0.153]，
# matplotlib 会画成 [0, 0.1606]（=0.153×1.05），整张图看着"缩"了一圈。这里全局关掉，
# 与 MATLAB 对齐（只影响渲染，数值一字不动）。
matplotlib.rcParams["axes.xmargin"] = 0
matplotlib.rcParams["axes.ymargin"] = 0

_figures: list = []
_current = None
_current_axes = None
#: `hold off` 之后、下一次绘图之前要清空坐标区（MATLAB 的 NextPlot='replace'）
_pending_clear = False
#: 跨格子的子图（`subplot(2,3,1:3)`）要缓存：matplotlib 每调一次
#: `add_gridspec` 就新建一套坐标区，1000 次绘图段下来会白占几个 GB。
_spanning: dict[tuple, tuple] = {}
#: 普通子图也要缓存：**matplotlib 3.10 的 `add_subplot(r,c,i)` 不复用**（同一格调
#: 两次会得到两个坐标区），而 MATLAB 的 `subplot` 是复用的。不缓存的话，绘图循环里
#: 每来一次 `subplot(2,3,1)` 就多一套坐标区：图会叠成好几层（看得见的内容还是最后
#: 那一层，但底下全留着），内存也跟着涨。
_subplots: dict[tuple, tuple] = {}


def _options(args) -> dict:
    """把 MATLAB 的成对参数（'名字', 值, ...）收成字典，跳过线型说明。

    MATLAB 允许把 LineSpec（`'r--'`、`'k-'`、`':'`）放在参数表的任意位置，
    它**不占用**一个值。不跳过它的话，后面整串名值对都会错位一格——比如
    `contour(...,'r--','linewidth',3)` 会解析成 `{'r--': 'linewidth'}`，
    结果线宽（唯一真正影响图样的那个参数）被丢掉。
    """
    options = {}
    items = list(args)
    while len(items) >= 2:
        key = items.pop(0)
        if _is_linespec(key):
            continue
        options[str(key).lower()] = items.pop(0)
    return options


#: MATLAB 的线型说明里允许出现的字符：三种成分（线型 `-` `--` `-.` `:`、
#: 颜色 `rgbcmykw`、标记 `o + * . x s d ^ v > < p h`）里的任意组合。
_LINESPEC_CHARS = set("-.:krgbcmyw") | set(mstyle.MARKER_CHARS)


def _is_linespec(item) -> bool:
    """判断参数是不是 MATLAB 的 LineSpec（`'r--'`、`'k'`、`':'`、`'ko-'`、`'mx'`）。

    规则：全部字符都在线型/颜色/标记的字符集里。这样 `'k'`（纯颜色）、`'--'`、
    `'r--'`、`'ko-'`、`'mx'`（洋红 x 标记，不连线）都算，而 `'linewidth'`、
    `'markersize'` 这种真正的选项名不算（它们含有集合外的字母 `l`/`i`/`e`…）。
    """
    if not isinstance(item, str) or not item:
        return False
    return bool(set(item) <= _LINESPEC_CHARS)


def _figure_defaults(fig) -> dict:
    """图窗上挂着的那套 MATLAB `Default*` 属性（新建坐标区时照着它来）。"""
    return getattr(fig, "_mplot_defaults", {})


def figure(*args):
    """`figure('position',[...], ...)`：新建一个图窗并设为当前图窗。

    `Position` 的单位按 MATLAB 的规矩分两种（原程序两种都用）：

    * `'unit','normalized'` 时是**屏幕比例**。用户看 MATLAB 图窗时基本都会最大化
      （论文里那些图就是最大化窗口截的），所以移植版直接按整屏出画布
      （`mstyle.SCREEN_INCHES`）：坐标区的归一化位置照抄不用动，字号是绝对点数，
      与画布的相对关系就和 MATLAB 最大化时一致——不然 25 pt 的字挤在 1/4 屏的
      画布上，标签会互相压住（这正是改之前的样子）。
    * 不给 `unit`（默认 `pixels`）时是像素（`figure('position',[50 100 600 300])`），
      按 100 dpi 折算成英寸。

    `DefaultAxesFontSize` / `DefaultAxesFontWeight` / `DefaultAxesLineWidth` /
    `DefaultTextInterpreter` 这几个 MATLAB 的图窗默认值也照收：脚本全靠它们一次性
    定下整张图的字号与字重（`DefaultAxesFontSize`,25 这种），原来移植版是把它们
    丢掉的，所以字号一直被 matplotlib 的 10 pt 顶着。
    """
    global _current, _current_axes
    options = _options(args)
    defaults = {
        "fontsize": mstyle.FACTORY_AXES_FONTSIZE,
        "fontweight": "normal",
        "linewidth": mstyle.FACTORY_AXES_LINEWIDTH,
        "interpreter": "tex",
    }
    fig = plt.figure(figsize=(8, 4.8), dpi=100)
    position = options.get("position")
    if position is not None:
        values = [float(v) for v in np.asarray(position).ravel()]
        if str(options.get("unit", "pixels")).startswith("normal"):
            fig.set_size_inches(*mstyle.SCREEN_INCHES)
        else:
            fig.set_size_inches(values[2] / 100.0, values[3] / 100.0)
    if "defaultaxesfontsize" in options:
        defaults["fontsize"] = float(
            np.asarray(options["defaultaxesfontsize"]).ravel()[0])
    if "defaultaxesfontweight" in options:
        defaults["fontweight"] = str(options["defaultaxesfontweight"]).lower()
    if "defaultaxeslinewidth" in options:
        defaults["linewidth"] = float(
            np.asarray(options["defaultaxeslinewidth"]).ravel()[0])
    if "defaulttextinterpreter" in options:
        defaults["interpreter"] = str(options["defaulttextinterpreter"]).lower()
    color = options.get("color")
    if color is not None:
        face = color if isinstance(color, str) else np.asarray(color, dtype=float).ravel()
        fig.set_facecolor(face)
    fig._mplot_defaults = defaults
    _figures.append(fig)
    _current = fig
    _current_axes = None
    return fig


def gcf():
    """MATLAB 的 `gcf`。"""
    global _current
    if _current is None:
        return figure()
    return _current


def open_figures() -> list:
    """当前打开的图窗，按创建顺序（MATLAB 里 `findall(groot,'Type','figure')` 的替代）。

    原文的绘图脚本大多只是"把图窗画出来给人看"，不写文件；`run.py` 靠这个函数
    在每一步跑完之后把图存进产出目录，这样每个流程都有能看的产物。
    """
    return list(_figures)


def subplot(*spec):
    """`subplot(221)`：按行优先选子图，与 MATLAB 一致。

    也支持 `subplot(2,3,1:3)` 这种"占一整行"的写法：位置按行优先编号，
    取这些位置的包围盒做成一个跨格子的坐标区（MATLAB 的语义）。
    """
    global _current_axes
    if len(spec) == 1:
        digits = str(int(spec[0]))
        rows, cols, index = int(digits[0]), int(digits[1]), int(digits[2])
    else:
        rows, cols, index = int(spec[0]), int(spec[1]), spec[2]
    if hasattr(index, "ravel") and not isinstance(index, str):
        positions = [int(v) for v in np.asarray(index).ravel()]
        cells = [((p - 1) // cols, (p - 1) % cols) for p in positions]
        row0 = min(cell[0] for cell in cells)
        row1 = max(cell[0] for cell in cells)
        col0 = min(cell[1] for cell in cells)
        col1 = max(cell[1] for cell in cells)
        key = (rows, cols, row0, row1, col0, col1)
        cached = _spanning.get(key)
        if cached is not None and cached[0] is gcf():
            axes = cached[1]
        else:
            grid = gcf().add_gridspec(rows, cols)
            axes = gcf().add_subplot(grid[row0:row1 + 1, col0:col1 + 1])
            _init_axes(axes)
            _spanning[key] = (gcf(), axes)
    else:
        index = int(index)
        key = (rows, cols, index)
        cached = _subplots.get(key)
        if cached is not None and cached[0] is gcf():
            axes = cached[1]
        else:
            axes = gcf().add_subplot(rows, cols, index)
            _init_axes(axes)
            _subplots[key] = (gcf(), axes)
    _current_axes = axes
    return axes


def _axes():
    if _current_axes is None:
        return subplot(111)
    return _current_axes


def _init_axes(axes_obj):
    """新坐标区按图窗的 `Default*` 初始化（MATLAB 的 `axes`/`subplot` 就是这个行为）。

    MATLAB 里 `figure('DefaultAxesFontSize',25,'DefaultAxesFontWeight','bold')`
    之后建的每个坐标区都会带上 25 pt 粗体的刻度与轴标签；移植版原来只把这两个默认值
    丢掉，于是所有图的字号都是 matplotlib 的 10 pt。这里补上，并且顺手把 MATLAB 的
    "刻度朝里、四周都画、颜色 [0.15 0.15 0.15]、网格压在数据下面" 一并定好。
    """
    defaults = _figure_defaults(gcf())
    mstyle.style_axes(axes_obj,
                      fontsize=defaults.get("fontsize"),
                      fontweight=defaults.get("fontweight"),
                      linewidth=defaults.get("linewidth"))
    axes_obj.set_axisbelow(True)          # MATLAB 的 Layer 'bottom'
    axes_obj._mplot_grid_on = False
    axes_obj._mplot_grid = {"linestyle": ":", "color": mstyle.AXES_COLOR,
                            "alpha": 0.25}
    axes_obj._mplot_box = True
    axes_obj._mplot_color_index = 0
    axes_obj._mplot_legend_artists = []
    axes_obj._mplot_fontsize = defaults.get("fontsize")
    axes_obj._mplot_fontweight = defaults.get("fontweight", "normal")
    axes_obj._mplot_linewidth = defaults.get("linewidth")
    return axes_obj


def _restyle_axes(axes_obj):
    """把记下来的 MATLAB 坐标区样式重新落一遍（`cla` 之后要用）。

    matplotlib 的 `Axes.cla()` 会把坐标区几乎所有属性按 rcParams 重置回去——边框
    线宽、刻度字号都回到出厂值，MATLAB 的 `cla` 不会（它只清内容）。二维求解器
    每 100 轮一段绘图、循环里反复 `cla`，不补这一刀最后只剩最后一帧的样子是对的。
    """
    mstyle.style_axes(axes_obj,
                      fontsize=getattr(axes_obj, "_mplot_fontsize", None),
                      fontweight=getattr(axes_obj, "_mplot_fontweight", "normal"),
                      linewidth=getattr(axes_obj, "_mplot_linewidth", None),
                      box=getattr(axes_obj, "_mplot_box", True))
    if getattr(axes_obj, "_mplot_grid_on", False):
        axes_obj.grid(True, **getattr(axes_obj, "_mplot_grid", {}))
    return axes_obj


def _remember_legend_artist(axes_obj, artist):
    """记下这个坐标区按顺序画出来的图元（`legend` 的标签就是按这个顺序对上去的）。"""
    artists = getattr(axes_obj, "_mplot_legend_artists", None)
    if artists is None:
        artists = axes_obj._mplot_legend_artists = []
    artists.append(artist)
    return artist


def _apply_axes_props(axes_obj, options) -> None:
    """把 MATLAB 的坐标区属性落到 matplotlib 的 `Axes` 上（`set(gca, ...)`）。

    `FontSize`/`FontWeight`/`LineWidth`/`GridColor`/`GridAlpha`/`GridLineStyle`/
    `Box`/`XColor`/`YColor`/`TickDir` 这些原来只被"记下来"、不生效（见 `set_` 的
    说明），于是脚本里 `set(gca,'Fontsize',25)`、`set(gca,'GridLineStyle',':')`
    全都白写。这里让它们真正落到图上。
    """
    defaults = _figure_defaults(gcf())
    fontsize = defaults.get("fontsize")
    fontweight = defaults.get("fontweight")
    linewidth = defaults.get("linewidth")
    if "fontsize" in options:
        fontsize = mstyle.np_scalar(options["fontsize"])
    if "fontweight" in options:
        fontweight = str(options["fontweight"]).lower()
    if "linewidth" in options:
        linewidth = mstyle.np_scalar(options["linewidth"])
    grid = dict(getattr(axes_obj, "_mplot_grid", {}))
    if "gridlinestyle" in options:
        grid["linestyle"] = str(options["gridlinestyle"])
    if "gridcolor" in options:
        grid["color"] = options["gridcolor"]
    if "gridalpha" in options:
        grid["alpha"] = mstyle.np_scalar(options["gridalpha"])
    if "box" in options:
        axes_obj._mplot_box = str(options["box"]).lower() != "off"
    axes_obj._mplot_grid = grid
    color = mstyle.AXES_COLOR
    if "xcolor" in options:
        color = options["xcolor"]
    axes_obj._mplot_fontsize = fontsize
    axes_obj._mplot_fontweight = fontweight
    axes_obj._mplot_linewidth = linewidth
    mstyle.style_axes(axes_obj, fontsize=fontsize, fontweight=fontweight,
                      linewidth=linewidth, color=color,
                      box=getattr(axes_obj, "_mplot_box", True))
    if getattr(axes_obj, "_mplot_grid_on", False) and grid:
        axes_obj.grid(True, **grid)
    legend_obj = axes_obj.get_legend()
    if legend_obj is not None:
        _style_legend(legend_obj, axes_obj)


def _prepare_draw():
    """画图前处理 `hold off`：MATLAB 里它是"下一次绘图前清空"，不是立刻清空。

    二维求解器每 100 轮跑一次绘图段，一次要画 4 张图，1000 次下来如果不丢旧的，
    内存会堆到 GB 级（实测 4.7 GB）；但也不能立刻清——那样最后一帧图会变空，
    而 MATLAB 的图窗里留着最后一次绘图段的内容。
    """
    global _pending_clear
    if not _pending_clear:
        return
    _pending_clear = False
    axes = _axes()
    for other in list(gcf().axes):
        if other is not axes and other.get_label() == "<colorbar>":
            other.remove()   # 上一轮的色条也跟着丢掉
    axes.cla()
    axes._mplot_style_key = None          # 同上
    axes._mplot_color_index = 0
    axes._mplot_legend_artists = []
    _restyle_axes(axes)


def plot(*args, **kwargs):
    """`plot(x,y)` 或 `plot(x1,y1,x2,y2,'--','linewidth',2)`。

    MATLAB 的线型说明（`'--'`、`'k--'`）在 matplotlib 里用关键字表达，
    这里把它们认出来并忽略；成对的 `'名字', 值` 照常解析。

    还有两条 MATLAB 的规则要照抄：

    * **x 是向量、y 是矩阵时按 y 的列逐条画**（`appendix_A/figure_A_3.m` 里
      `Be_2 + sum_br_curave` 就是这种情况：1×129 的行加上 129×1 的列，逐元素
      展开成 129×129，于是那一句画的是 129 条线）。本机 MATLAB 实测：
      `plot(1:5, 5×5)` 给出 5 条线。
    * **线型说明挂在它前面那对 (x, y) 上**，而且**显式给了颜色的线照样占用
      ColorOrder 的下一个下标**（本机实测：先 `plot(...,'m-')` 再 `plot(...,'--')`，
      第二条拿到的是第 2 号色橙）。论文 Fig.4 的 (d)(e) 两格正是洋红实线 + 橙色
      虚线，靠的就是这一条；原来这些说明被整串丢掉（见 `_options`），画出来
      全变成了 matplotlib 的蓝橙黄。
    """
    pairs = []          # 每项是 [x, y, 线型说明] —— 线型说明跟随它前面那对
    options = {}
    index = 0
    while index < len(args):
        item = args[index]
        if isinstance(item, str):
            if _is_linespec(item):
                if pairs and pairs[-1][2] is None:
                    pairs[-1][2] = item
                index += 1
                continue
            if index + 1 < len(args):
                options[item.lower()] = args[index + 1]
                index += 2
                continue
            index += 1
            continue
        if pairs and pairs[-1][1] is None:
            pairs[-1][1] = item
        else:
            pairs.append([item, None, None])
        index += 1

    _prepare_draw()
    width = mstyle.np_scalar(options.get("linewidth", 2))
    order = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
    axes = _axes()
    drawn = []          # (x, y, 样式)：一次 plot 调用的所有线
    for x_item, y_item, linespec in pairs:
        if y_item is None:
            continue
        # MATLAB 的 `plot` 遇到复数数据只画实部（不是实部对虚部，那是 plot(z) 单独
        # 一个参数的写法）；绘图脚本里 MRR 的 ψ 有时是复数，照这个语义取实部。
        x = np.real(np.asarray(x_item))
        y = np.real(np.asarray(y_item))
        style = mstyle.parse_linespec(linespec)
        if _is_vector(x) and y.ndim == 2 and y.shape[1] > 1 and x.size == y.shape[0]:
            for column in range(y.shape[1]):       # 向量 × 矩阵：一列一条线
                drawn.append((x.ravel(), y[:, column], dict(style)))
        elif _is_vector(y) and x.ndim == 2 and x.shape[1] > 1 and y.size == x.shape[0]:
            for column in range(x.shape[1]):
                drawn.append((x[:, column], y.ravel(), dict(style)))
        else:
            drawn.append((x.ravel(), y.ravel(), dict(style)))

    for x, y, style in drawn:
        color = style.pop("color", None)
        if color is None:
            index_color = getattr(axes, "_mplot_color_index", 0)
            color = order[index_color % len(order)]
        axes._mplot_color_index = getattr(axes, "_mplot_color_index", 0) + 1
        axes.plot(x, y, color=color, linewidth=width, **style, **kwargs)
        _remember_legend_artist(axes, axes.lines[-1])
    return axes.lines[-1]


def _is_vector(array: np.ndarray) -> bool:
    """MATLAB 意义上的向量：一维数组，或者 1×n / n×1 的二维数组。"""
    return array.ndim <= 1 or (array.ndim == 2 and 1 in array.shape)


def xlabel(text, *args):
    """MATLAB 的 `xlabel`：字号默认是坐标区字号的 1.1 倍，字重跟着坐标区走。"""
    axes = _axes()
    axes.set_xlabel(_label(text), **_label_font(axes, args))


def ylabel(text, *args):
    axes = _axes()
    axes.set_ylabel(_label(text), **_label_font(axes, args))


def title(text, *args):
    axes = _axes()
    axes.set_title(_label(text), **_label_font(axes, args))


def ylim(limits):
    values = [float(v) for v in np.asarray(limits).ravel()]
    if len(values) != 2 or values[0] >= values[1]:
        # MATLAB 的 ylim 要求"递增的二元素向量"，否则直接报错。
        # `gen_psib.m` 的 icase==4（figure8 副本）里 psiw 是负数，
        # 于是 ylim([0,2*psiw]) 报错、脚本中止、psib.mat 根本没写出来——
        # 基准把这个算例记成 expectError，移植版必须同样报错。
        raise ValueError("范围必须为包含递增的数值的二元素向量")
    _axes().set_ylim(values[0], values[1])


def xlim(limits):
    values = [float(v) for v in np.asarray(limits).ravel()]
    _axes().set_xlim(values[0], values[1])


def gca():
    """MATLAB 的 `gca`。"""
    return _axes()


def hold_on():
    """MATLAB 的 `hold on`：matplotlib 默认就是叠加画线，这里只标记一下。

    顺带把网格压到数据下面（MATLAB 的 `Layer` 默认是 `'bottom'`，网格线不盖住
    曲线）；原来这里设的是 `axisbelow(False)`，网格会画在曲线上面。
    """
    global _pending_clear
    _pending_clear = False
    _axes().set_axisbelow(True)


def hold_off():
    """MATLAB 的 `hold off`：把坐标区设成"下一次绘图前清空"（见 `_prepare_draw`）。"""
    global _pending_clear
    _pending_clear = True


def cla():
    """MATLAB 的 `cla`：清空**当前**坐标区的内容（坐标区本身留着，仍是当前区）。

    色条在 matplotlib 里是**另一个坐标区**，`Axes.cla()` 不会动它；MATLAB 的
    `cla` 会把色条一并清掉，所以这里把挂在当前坐标区上的色条也删掉
    （`colorbar()` 建的时候记了账）。
    """
    global _pending_clear
    axes_obj = _axes()
    for colorbar_axes in getattr(axes_obj, "_mplot_colorbars", []):
        colorbar_axes.remove()
    axes_obj._mplot_colorbars = []
    axes_obj.cla()
    axes_obj._mplot_style_key = None      # cla 之后刻度/文字都重建了，样式得重来
    # MATLAB 的 cla 会把 ColorOrderIndex 拨回 1，下一批线从第 1 号色重新开始
    axes_obj._mplot_color_index = 0
    axes_obj._mplot_legend_artists = []
    _restyle_axes(axes_obj)
    _pending_clear = False
    return axes_obj


def yyaxis(side):
    """MATLAB 的 `yyaxis left|right`：在同一个格子里切左右两条 y 轴。

    MATLAB 的 `yyaxis right` 会在右边加一条独立的 y 轴，左右各有自己的量程；
    matplotlib 里对应的是 `twinx()`（新建一个共享 x 轴的孪生坐标区）。切回
    `left` 时回到原来那个。`plt_psi_r.m` 靠它在同一格里同时画 \(B_z\)（左）与
    \(B_r\)（右）——两者的量级差很多，共用一条轴是画不出来的。

    孪生坐标区上打了 `_mplot_yyaxis_right` 标记，测试用它区分"左边那套"和"右边那套"。
    """
    global _current_axes
    name = str(side).lower()
    axes = _axes()
    if name.startswith("left"):
        left = getattr(axes, "_mplot_yyaxis_left", axes)
        _current_axes = left
        return left
    right = getattr(axes, "_mplot_yyaxis_right", None)
    if right is None:
        left = getattr(axes, "_mplot_yyaxis_left", axes)
        right = left.twinx()
        left._mplot_yyaxis_left = left
        left._mplot_yyaxis_right = right
        right._mplot_yyaxis_left = left
        right._mplot_yyaxis_right = right
        right._mplot_yyaxis_is_right = True
    _current_axes = right
    return right


def grid_on():
    """MATLAB 的 `grid on`：网格线用当前坐标区的 `GridColor`/`GridAlpha`/`GridLineStyle`。"""
    axes = _axes()
    axes._mplot_grid_on = True
    axes.grid(True, **getattr(axes, "_mplot_grid", {}))
    return axes


def box(mode="on"):
    """MATLAB 的 `box on` / `box off`：控制坐标区的边框。

    MATLAB 的 `box off` 只去掉**上、右**两条边（下、左留着，刻度也只在下面和左边
    出现），不是把整个框都拆掉——matplotlib 的 `set_frame_on(False)` 是后者，
    所以这里自己按 MATLAB 的语义来。
    """
    state = str(mode).lower() != "off"
    axes = _axes()
    axes._mplot_box = state
    mstyle.style_axes(axes, fontsize=getattr(axes, "_mplot_fontsize", None),
                      fontweight=getattr(axes, "_mplot_fontweight", "normal"),
                      box=state)
    return axes


def text(x, y, label, *args):
    """MATLAB 的 `text`：两种写法都支持。

    * `text(x, y, '...', 'fontsize', 13)`——位置在前；
    * `text('Interpreter','latex','String','(c)','Position',[x y],'FontSize',25)`
      ——全部走名值对（仓库里的绘图脚本大量用这一种）。
    """
    if isinstance(x, str):
        options = _options((x, y, label, *args))
        position = [float(v) for v in np.asarray(options["position"]).ravel()]
        x, y = position[0], position[1]
        label = options.get("string", "")
    else:
        options = _options(args)
    kwargs = {}
    if "fontsize" in options:
        # 绘图脚本常常把字号写成 1×1 的矩阵（`fontsize=25` 再传给 text()），
        # 直接 float() 会踩 numpy 的"数组转标量"弃用告警，这里先取标量。
        kwargs["fontsize"] = mstyle.np_scalar(options["fontsize"])
    if "color" in options:
        kwargs["color"] = options["color"]
    if "fontweight" in options:
        kwargs["fontweight"] = str(options["fontweight"]).lower()
    if "horizontalalignment" in options:
        kwargs["ha"] = str(options["horizontalalignment"]).lower()
    if "verticalalignment" in options:
        kwargs["va"] = str(options["verticalalignment"]).lower()
    if "rotation" in options:
        kwargs["rotation"] = mstyle.np_scalar(options["rotation"])
    axes = _axes()
    # MATLAB 的 latex/tex 解释器能吃的写法（`$\psi$`）matplotlib 的 mathtext 也认得；
    # `_label` 把裸的 `\psi` 包成 `$...$`，带 `$` 的原样放行。
    return axes.text(float(np.asarray(x).ravel()[0]),
                     float(np.asarray(y).ravel()[0]), str(_label(str(label))), **kwargs)


def colorbar(*args):
    """MATLAB 的 `colorbar`：给当前坐标区最近画的那张图上色条。

    返回值包一层 `_ColorbarHandle`：原程序里有脚本会拿它设 `TickLabels`
    （`appendix_B/Figure_B.m` 把刻度标签换成 \(10^{-5}\) 那套）和 `Label.String`
    （色条标题），这两条要能落到位。

    ⚠️ 这里有两个 matplotlib 的坑，都得绕开（MATLAB 的 colorbar 是幂等的，
    重复调用既不会越缩越小、也不会改父坐标区的框）：

    1. **父坐标区来自 `add_gridspec`**（`mplot.subplot(2,3,1:3)` 就是），
       matplotlib 会走 `make_axes_gridspec` 那条路：它按格子算色条位置，而格子的
       位置**每调一次就再缩一次**（实测色条宽度按 \(0.8^n\) 衰减，第 166 轮父框
       只剩 \(6\times10^{-17}\) 宽），接着 `transAxes.inverted()` 里
       `np.linalg.inv` 抛 `LinAlgError: Singular matrix`，整个算例中止。
       传 `use_gridspec=False` 让它走普通的 `make_axes` 就与调用次数无关了。
    2. 即便走 `make_axes`，它也会顺手把父坐标区缩掉一块给色条腾地方，所以先记下
       父框当"基准"：每次调用都先回到基准再让 matplotlib 缩（MATLAB 的 colorbar
       也是幂等的，重复调用不会越缩越小）。**缩完之后不再放回去**——放回去会让色条
       压在坐标区上（matplotlib 的色条是放在"腾出来的那条带子"里的，MATLAB 的
       `colorbar` 同样如此）。原程序里真要恢复坐标区框的（`figure7.m` 的
       `ax.Position=axpos`）会自己调 `set_`，那一步照抄。

    `'Location'` 也照 MATLAB 认：`northOutside`/`north` → 上边、`south*` → 下边、
    `east*`（默认）→ 右边、`west*` → 左边。
    """
    axes = _axes()
    options = _options(args)
    # MATLAB 允许只写一个位置串：`colorbar('northOutside')`、`colorbar('east')`。
    # 参数个数是奇数时，第一个字符串不可能是"名字,值"对里的名字，就当位置处理。
    if len(args) % 2 == 1 and isinstance(args[0], str):
        options.setdefault("location", args[0])
    location = str(options.get("location", "east")).lower()
    if location.startswith("north"):
        place = "top"
    elif location.startswith("south"):
        place = "bottom"
    elif location.startswith("west"):
        place = "left"
    else:
        place = "right"
    # 只有带 cmap 的图元（contour/contourf/image）能上色条；折线不算。
    # 单一层级的等值线（原程序用 `[0,0]` 强调分界面）会让 matplotlib 的
    # 色条分档退化成一格而报错，所以跳过它、退到上一层多层级等值线上。
    for artist in reversed(list(axes.collections) + list(axes.images)):
        levels = getattr(artist, "levels", None)
        if hasattr(artist, "cmap") and (levels is None or np.size(levels) >= 2):
            base = getattr(axes, "_mplot_colorbar_base", None)
            if base is None:
                base = axes.get_position(original=True)
                axes._mplot_colorbar_base = base
            else:
                axes.set_position(base)
            handle = gcf().colorbar(artist, ax=axes, use_gridspec=False,
                                    location=place)
            # 记下来，`cla` 时一并清掉（matplotlib 的色条是独立坐标区）
            axes._mplot_colorbars = (getattr(axes, "_mplot_colorbars", [])
                                     + [handle.ax])
            _style_colorbar(handle, axes)
            wrapper = _ColorbarHandle(handle)
            if "ticks" in options:
                # `colorbar('Location','east','Ticks',[2,8,14])`：把色条刻度点到给定值
                wrapper.TickLabels = options["ticks"]
            return wrapper
    return None


def _style_colorbar(handle, axes_obj):
    """MATLAB 的色条：字号是坐标区字号的 0.9 倍，字重跟着坐标区走。"""
    size = getattr(axes_obj, "_mplot_fontsize", None)
    size = mstyle.FACTORY_AXES_FONTSIZE if size is None else size
    handle.ax.tick_params(labelsize=float(size) * 0.9)
    for label in list(handle.ax.get_yticklabels()) + list(handle.ax.get_xticklabels()):
        label.set_fontweight(getattr(axes_obj, "_mplot_fontweight", "normal"))
    return handle


class _ColorbarLabel:
    """`h.Label`：MATLAB 里色条的标题（脚本只用它的 `String`）。"""

    def __init__(self, colorbar_axes):
        self._axes = colorbar_axes
        self.String = ""

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        if name == "String":
            super().__setattr__(name, value)
            self._axes.set_ylabel(_label(value))
            return
        super().__setattr__(name, value)


class _PositionProxy:
    """MATLAB 的 `h.Position`：支持 `h.Position(3) = ...` 这种"按元素改"。

    `figure7.m` 用 `c.Position(3)=0.3*c.Position(3)` 把色条改窄、再把父坐标区的框
    `ax.Position=axpos` 放回去。MATLAB 的 Position 是 `[left bottom width height]`，
    这里把下标写回坐标区的位置。
    """

    def __init__(self, axes_obj) -> None:
        self._axes = axes_obj

    def _values(self) -> list:
        box = self._axes.get_position()
        return [float(box.x0), float(box.y0), float(box.width), float(box.height)]

    def __getitem__(self, index):
        return self._values()[int(np.asarray(index).ravel()[0]) - 1]

    def __setitem__(self, index, value):
        key = int(np.asarray(index).ravel()[0]) - 1
        values = self._values()
        values[key] = float(np.asarray(value).ravel()[0])
        self._axes.set_position(values)

    def __array__(self, dtype=None):
        return np.asarray(self._values(), dtype=dtype)

    def __repr__(self) -> str:
        return f"Position{tuple(round(v, 6) for v in self._values())}"


class _ColorbarHandle:
    """`colorbar(...)` 的返回值：先认 MATLAB 的属性名，其余转发给 matplotlib 的色条。"""

    def __init__(self, colorbar) -> None:
        super().__setattr__("_colorbar", colorbar)
        super().__setattr__("_label", _ColorbarLabel(colorbar.ax))

    def __getattr__(self, name):
        if name == "Label":
            return self._label
        if name == "Position":
            return _PositionProxy(self._colorbar.ax)
        return getattr(self._colorbar, name)

    def __setattr__(self, name, value):
        if name == "TickLabels":
            values = np.asarray(value, dtype=float).ravel()
            self._colorbar.set_ticks(values)
            self._colorbar.set_ticklabels([f"{v:g}" for v in values])
            return
        setattr(self._colorbar, name, value)


def colormap(name):
    """MATLAB 的 `colormap(hot)`：把色图应用到当前坐标区已有的图上。

    渲染结果不参与验收，这里只保证调用结构照抄、不报错。
    """
    axes = _axes()
    cmap = matplotlib.colormaps[str(name)]
    for artist in list(axes.collections) + list(axes.images):
        artist.set_cmap(cmap)
    return cmap


def pcolor(x, y, c, *args, **kwargs):
    """MATLAB 的 `pcolor(X,Y,C)`（等价于 matplotlib 的 `pcolormesh`）。

    原程序后面常跟一句 `shading interp`；matplotlib 的着色方式建图时就定了，
    所以那位只在 `shading()` 里记录一下（渲染不参与验收，见 §3）。
    """
    _prepare_draw()
    axes = _axes()
    return axes.pcolormesh(np.asarray(x), np.asarray(y), np.asarray(c), **kwargs)


def shading(mode="flat"):
    """MATLAB 的 `shading interp|flat|faceted`：只影响渲染，这里记一下就好。"""
    return mode


def semilogy(*args, **kwargs):
    """MATLAB 的 `semilogy`：与 `plot` 同一套解析，纵轴取对数。"""
    handle = plot(*args, **kwargs)
    _axes().set_yscale("log")
    return handle


class _Handle:
    """MATLAB 图形句柄的极简替身：只提供原脚本真正会读写的属性。

    原程序把 `rectangle`/`uicontrol` 的返回值当结构体用（`h.LineWidth`、
    `h.Callback`、`h.String`…），所以这里按 MATLAB 的属性名（大写开头）转发到
    matplotlib 对象上。
    """

    def __init__(self, artist, **fields):
        self._artist = artist
        self._fields = dict(fields)

    def __getattr__(self, name):
        if name in self._fields:
            return self._fields[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        self._fields[name] = value
        artist, mapping = self._artist, {
            "LineWidth": ("set_linewidth", value),
            "EdgeColor": ("set_edgecolor", value),
            "FaceColor": ("set_facecolor", value),
            "Color": ("set_color", value),
            "LineStyle": ("set_linestyle", value),
        }
        if name in mapping:
            setter, arg = mapping[name]
            getattr(artist, setter)(arg)

    def __getitem__(self, name):
        return self._fields[name]


def rectangle(*args, **kwargs):
    """MATLAB 的 `rectangle('Position',[x y w h],'FaceColor',c,'linewidth',n)`。

    原程序用它在剖面图上画线圈截面（`Position` 是左下角 + 宽高，数据坐标）。
    MATLAB 的 rectangle **参与坐标区的自动范围**（matplotlib 的 `add_patch` 不参与），
    所以这里手工把它的两个角点也算进数据范围——`figure_4_2.m` 面板 (a) 的
    `ylim` 就是靠线圈上沿（0.153）定出来的。
    """
    options = _options(args)
    pos = [float(v) for v in np.asarray(options.get("position")).ravel()]
    face = options.get("facecolor", options.get("edgecolor", "none"))
    if not isinstance(face, str):
        face = np.asarray(face, dtype=float).ravel()
    patch = matplotlib.patches.Rectangle(
        (pos[0], pos[1]), pos[2], pos[3],
        facecolor=face if not isinstance(face, str) or face != "none" else "none",
        edgecolor="none", linewidth=float(options.get("linewidth", 0.5)))
    axes_obj = _axes()
    axes_obj.add_patch(patch)
    axes_obj.update_datalim([(pos[0], pos[1]),
                             (pos[0] + pos[2], pos[1] + pos[3])])
    return _Handle(patch, LineWidth=float(options.get("linewidth", 0.5)),
                   LineStyle=options.get("linestyle", "-"),
                   FaceColor=face, EdgeColor=options.get("edgecolor", "none"))


def axes(*args, **kwargs):
    """MATLAB 的 `axes`：`axes(h)` 切换当前坐标区；`axes('Position',[...])` 新建一个。"""
    global _current_axes
    if args and not isinstance(args[0], str):
        _current_axes = args[0]
        return args[0]
    options = _options(args)
    figure_obj = gcf()
    position = options.get("position")
    if position is None:
        axes_obj = figure_obj.add_axes([0.13, 0.11, 0.775, 0.815])
    else:
        axes_obj = figure_obj.add_axes([float(v) for v in np.asarray(position).ravel()])
    _init_axes(axes_obj)
    if "fontsize" in options:
        _apply_axes_props(axes_obj, {"fontsize": options["fontsize"]})
    _current_axes = axes_obj
    return axes_obj


def caxis(limits):
    """MATLAB 的 `caxis([lo,hi])`（新版叫 `clim`）：设色标范围。"""
    values = [float(v) for v in np.asarray(limits).ravel()]
    for artist in list(_axes().collections) + list(_axes().images):
        if hasattr(artist, "set_clim"):
            artist.set_clim(values[0], values[1])
    return values


def axis(*args):
    """MATLAB 的 `axis equal|tight|off|([xmin xmax ymin ymax])`。"""
    axes_obj = _axes()
    if not args:
        return None
    spec = args[0]
    if isinstance(spec, str):
        if spec == "equal":
            axes_obj.set_aspect("equal", adjustable="box")
        elif spec == "tight":
            axes_obj.autoscale(tight=True)
        elif spec == "off":
            axes_obj.set_axis_off()
        elif spec == "square":
            axes_obj.set_aspect(1.0, adjustable="box")
        return spec
    values = [float(v) for v in np.asarray(spec).ravel()]
    if len(values) == 4:
        axes_obj.axis(values)
    return values


def drawnow(*args):
    """MATLAB 的 `drawnow`：Agg 后端下无事可做。"""
    return None


def uicontrol(*args, **kwargs):
    """MATLAB 的 `uicontrol`：GUI 控件的极简替身。

    交互不参与验收（§3），这里只**记录**控件的属性（Style/String/Callback/
    Position），好让移植后的 GUI 脚本能照原样建出来、并且以后想接真控件时
    有地方挂。不创建真的 matplotlib 控件——无界面环境下点不动，反而会误导。

    第一个参数可以是**父容器句柄**（`uicontrol(hf,'Style','edit',...)`），它不参与
    名值对，先跳过再解析（不跳的话后面整串属性都会错位一格）。
    """
    items = list(args)
    if items and not isinstance(items[0], str):
        items.pop(0)
    options = _options(items)
    key = id(options)
    _uicontrols[key] = options
    return _Handle(None, **{k.title().replace("_", ""): v for k, v in options.items()})


def uigetfile(*args, **kwargs):
    """MATLAB 的 `uigetfile`：无界面环境下按"用户取消"返回 `(None, 0)`。

    原程序的回调里正是这么判的（`if filename==0, return, end`），所以取消分支
    的行为与 MATLAB 一致。
    """
    return None, 0


def waitforbuttonpress(*args, **kwargs):
    """MATLAB 的 `waitforbuttonpress`：无界面环境下直接返回（当作按了一下）。"""
    return 1


#: `warndlg` 弹过的警告（无界面环境下只记下文本，供测试查看）
_warnings: list[str] = []


def warndlg(message, *args, **kwargs):
    """MATLAB 的 `warndlg(msg)`：警告对话框。

    MATLAB 里它是**阻塞**的（等用户点确定）；无界面环境下移植版只把文本记下来、
    立刻返回，好让 `fun_frc_coil.m` 里"线圈表长度不一致就 `warndlg` + `return`"
    那条分支能照抄而不卡住（原脚本就是"提示完就退出"）。
    """
    _warnings.append(str(message))
    warnings.warn(str(message), RuntimeWarning, stacklevel=2)
    return None


#: 建过的 GUI 控件（属性字典），供测试/调试查看
_uicontrols: dict[int, dict] = {}


def _contour_style(args) -> dict:
    """等值线的线型说明（`contour(...,'r--','linewidth',3)` 里的 `'r--'`）。

    MATLAB 里这几个字符是**真的在线型说明的位置**上、不占名值对，`_options` 会把它
    跳过；原来这一跳就把颜色和线型一起丢了，于是论文里那条红色虚线的分界面
    （`contour(Z,R,psi,[0,0],'r--','linewidth',3)`）画成了别的颜色。
    """
    spec = next((item for item in args if _is_linespec(item)), None)
    parsed = mstyle.parse_linespec(spec)
    style = {}
    if "color" in parsed:
        style["colors"] = parsed["color"]
    if "linestyle" in parsed and parsed["linestyle"] != "None":
        style["linestyles"] = parsed["linestyle"]
    return style


def contour(x, y, z, levels=None, *args, return_matrix=False, **kwargs):
    """MATLAB 的 `contour(X,Y,Z,levels,...)`。

    两套语义要分清（都照 MATLAB）：

    * `contour(Z,n)`——**标量** n 表示"自动取 n 条等值线"，matplotlib 的
      `levels=<int>` 与它同义，所以标量原样传下去；
    * `contour(Z,v)`——**向量** v 表示"就在这些层值上画"（原程序用 `[0,0]`
      强调分界面），matplotlib 要求递增，所以只在传给它时去重。

    `return_matrix=True` 时模仿旧式 `[ch,hh]=contour(...)`：返回轮廓矩阵 `ch`
    与句柄。调用方只用 `ch(1,:)`（各条等值线的层值），所以这里按层值重复填充，
    形状与 MATLAB 的 (2×N) 一致。
    """
    _prepare_draw()
    options = _options(args)
    axes = _axes()
    passed_levels = None
    if levels is not None:
        spec = np.asarray(levels)
        if spec.size == 1:
            # 标量（包括 MATLAB 的 1×1 数组，如 `No=30`）→ "自动取 n 条等值线"
            passed_levels = int(spec.ravel()[0])
        else:
            values = spec.ravel()
            passed_levels = np.unique(values).tolist() if values.size else None
    handle = axes.contour(np.asarray(x), np.asarray(y), np.asarray(z),
                          levels=passed_levels,
                          linewidths=mstyle.np_scalar(options.get("linewidth", 2)),
                          **_contour_style(args), **kwargs)
    if not return_matrix:
        return handle
    # 按 MATLAB 的轮廓矩阵格式构造：对每一段，
    #   第 1 行 = [层值, x1..xn]，第 2 行 = [顶点数, y1..yn]
    # 所以 `max(c(1,:))-min(c(1,:))` 实际是"所有顶点 x 坐标的跨度"——
    # 原程序正是用这一条算分界面的伸长比 E。
    first, second = [], []
    for level, segments in zip(handle.levels, handle.allsegs):
        for segment in segments:
            if len(segment) == 0:
                continue
            first.extend([float(level), *segment[:, 0].tolist()])
            second.extend([float(len(segment)), *segment[:, 1].tolist()])
    return np.array([first, second]), handle


def contourf(x, y, z, levels=None, *args, return_matrix=False, **kwargs):
    """MATLAB 的 `contourf`（填色等值线）。

    `levels` 的语义与 `contour` 一致：标量 n = "自动取 n 条"，向量 = 给定层值。
    `return_matrix=True` 按 `[c,h]=contourf(...)` 返回（等值线矩阵, 句柄）——
    `figure7.m` 拿这个句柄读 `h.Parent`（= 坐标区）再 `set(h.Parent,'YScale','log')`，
    所以句柄用 `_Handle` 包一层、把自己的坐标区挂在 `Parent` 上。
    """
    _prepare_draw()
    options = _options(args)
    axes = _axes()
    passed_levels = None
    if levels is not None:
        spec = np.asarray(levels)
        if spec.size == 1:
            passed_levels = int(spec.ravel()[0])   # 标量（含 1×1 数组）= 层数
        else:
            values = spec.ravel()
            passed_levels = np.unique(values).tolist() if values.size else None
    handle = axes.contourf(np.asarray(x), np.asarray(y), np.asarray(z),
                           levels=passed_levels, **_contour_style(args), **kwargs)
    if not return_matrix:
        return handle
    first, second = [], []
    for level, segments in zip(handle.levels, handle.allsegs):
        for segment in segments:
            if len(segment) == 0:
                continue
            first.extend([float(level), *segment[:, 0].tolist()])
            second.extend([float(len(segment)), *segment[:, 1].tolist()])
    return np.array([first, second]), _Handle(handle, Parent=axes)


def pause(seconds: float) -> None:
    """MATLAB 的 `pause`：绘图段里用来让图窗刷新，这里无事可做。"""
    return None


#: `legend` 认的名值对（其余字符串参数都是标签）
_LEGEND_OPTIONS = {"location", "orientation", "interpreter", "fontsize",
                   "textcolor", "edgecolor", "box", "numcolumns", "position",
                   "autoupdate", "outside", "plot"}

#: `set(gca, ...)` 里真会改变坐标区外观的那几个名字（其余只影响刻度/范围/位置，
#: 由 `set_` 自己的那几段处理，不必再走一遍坐标区重排）
_AXES_STYLE_KEYS = {"fontsize", "fontweight", "linewidth", "gridcolor",
                    "gridalpha", "gridlinestyle", "box", "xcolor", "ycolor",
                    "tickdir", "ticklength", "colororder"}

#: MATLAB 的图例位置 → matplotlib 的图例位置
_LEGEND_LOCATIONS = {
    "north": "upper center", "south": "lower center",
    "east": "center right", "west": "center left",
    "northeast": "upper right", "northwest": "upper left",
    "southeast": "lower right", "southwest": "lower left",
    "best": "best", "northeastoutside": "upper left",
}


def _style_legend(handle, axes_obj):
    """MATLAB 的图例默认值：字号 = 0.9 × 坐标区字号，字重跟着坐标区走。

    本机实测：坐标区 10 pt 时图例 9 pt、坐标区 25 pt 时图例 22.5 pt，字重与坐标区
    一致（`DefaultAxesFontWeight='bold'` 时图例也是粗体）。脚本若自己
    `set(h,'FontSize',25)` 过就按脚本的来，不再被这里覆盖。
    """
    if not getattr(handle, "_mplot_explicit_fontsize", False):
        size = getattr(axes_obj, "_mplot_fontsize", None)
        size = mstyle.FACTORY_AXES_FONTSIZE if size is None else size
        for text in handle.get_texts():
            text.set_fontsize(float(size) * 0.9)
    weight = getattr(axes_obj, "_mplot_fontweight", "normal")
    for text in handle.get_texts():
        text.set_fontweight(weight)
    return handle


def legend(labels=None, *args):
    """`legend('a','b',...)` / `legend(str,'location','best')` / `legend('boxoff')`。

    MATLAB 的标签可以一次给好几个（`legend('$2D$','$1D$')`），也可以给一个元胞
    字符串数组（`valid1d.m` 的 `legend(str,'location','best')`），名值对混在后面。
    matplotlib 的 `legend` 只能一次收一串标签，所以这里按 MATLAB 的规则把标签收齐
    ——只取第一个参数的话，多标签的图例会只写出第一个名字。
    """
    axes = _axes()
    options = {}
    items = [] if labels is None else [labels]
    items.extend(args)
    texts = []
    index = 0
    while index < len(items):
        item = items[index]
        if (isinstance(item, str) and item.lower() in _LEGEND_OPTIONS
                and index + 1 < len(items)):
            options[item.lower()] = items[index + 1]
            index += 2
            continue
        texts.append(item)
        index += 1

    if not texts:
        return None
    if len(texts) == 1 and isinstance(texts[0], str) and texts[0].lower() in ("boxoff", "boxon"):
        # `legend('boxoff')` / `legend('boxon')`：只改图例边框
        if axes.get_legend() is not None:
            axes.get_legend().set_frame_on(texts[0].lower() == "boxon")
        return None

    names = []
    for item in texts:
        if hasattr(item, "ravel"):
            names.extend(_cell_text(value) for value in np.asarray(item).ravel())
        else:
            names.append(str(item))
    # MATLAB 的图例按"画出来的顺序"取图元；`_mplot_legend_artists` 就是按这个顺序
    # 记的（`plot` 每画一条记一条）。没有记录时退回坐标区自己的线列表。
    handles = list(getattr(axes, "_mplot_legend_artists", []))
    if not handles:
        handles = list(axes.get_lines())
    count = min(len(names), len(handles))
    location = str(options.get("location", "best")).lower().replace("_", "")
    handle = axes.legend(handles=handles[:count], labels=names[:count],
                         loc=_LEGEND_LOCATIONS.get(location, location))
    # 标签里的 `\psi` 这类 TeX 写法统一转成 matplotlib 认得的样子（`_label` 对已经
    # 带 `$...$` 的放行）。原脚本多数是画完再 `set(h,'interpreter','latex')`，
    # 那时已经来不及改标签文本，所以这里在建立图例时就转好。
    for text in handle.get_texts():
        text.set_text(_label(text.get_text()))
    _style_legend(handle, axes)
    return handle


def tight_subplot(cols, rows, gap, margins_h, margins_w):
    """对应 `tight_subplot.m`：按给定边距/间隙排布子图，返回轴句柄列表。

    只影响绘图版式（返回的句柄在原程序快照里也是被跳过的对象），给一个语义
    等价的实现即可。
    """
    global _current_axes
    cols, rows = int(cols), int(rows)
    gap_h, gap_w = [float(v) for v in np.asarray(gap).ravel()][:2]
    lower, upper = [float(v) for v in np.asarray(margins_h).ravel()][:2]
    left, right = [float(v) for v in np.asarray(margins_w).ravel()][:2]
    width = (1.0 - left - right - (cols - 1) * gap_w) / cols
    height = (1.0 - lower - upper - (rows - 1) * gap_h) / rows
    handles = []
    for column in range(cols):
        for row in range(rows):
            handles.append(gcf().add_axes([
                left + column * (width + gap_w),
                lower + row * (height + gap_h),
                width,
                height,
            ]))
            _init_axes(handles[-1])
    _current_axes = handles[-1]
    return handles


def set_(target, *args):
    """`set(h, '名字', 值, ...)`：能对应的就应用到 matplotlib 对象上，其余只记下来。

    会应用到图上的：**刻度**（`xtick`/`ytick`）、**刻度标签**（`xticklabel`/
    `yticklabel`）、**轴范围**（`xlim`/`ylim`）、**轴的尺度**（`xscale`/`yscale`）、
    **位置**（`position`），以及坐标区的 `FontSize`/`FontWeight`/`LineWidth`/
    `GridColor`/`GridAlpha`/`GridLineStyle`/`Box`/`XColor`（见 `_apply_axes_props`）；
    作用在图例/文本句柄上时，改的是它们的字号、字重、解释器、边框（见
    `_apply_legend_props` / `_apply_text_props`）。剩下认不出来的名字只记下来。
    """
    options = _options(args)
    for key, setter in (("xtick", "set_xticks"), ("ytick", "set_yticks"),
                        ("xlim", "set_xlim"), ("ylim", "set_ylim")):
        if key in options and hasattr(target, setter):
            values = np.asarray(options[key]).ravel()
            # matplotlib 3.10 的 `set_xticks/set_yticks` 会把**刻度的范围**也算进
            # 坐标轴范围（实测：数据到 5.007、`set_yticks(0:2:10)` 之后 ylim 变成
            # [0,10]）；MATLAB 不会——刻度超出数据范围的部分直接不画。所以设完刻度
            # 把范围与自动缩放状态放回去。
            saved = None
            if key in ("xtick", "ytick") and hasattr(target, "get_xlim"):
                saved = (target.get_xlim(), target.get_ylim(),
                         target.get_autoscalex_on(), target.get_autoscaley_on())
            if values.size == 0:
                # `set(gca,'xtick',[])`：MATLAB 把刻度和标签一起清掉
                getattr(target, setter)([])
                labels = "set_xticklabels" if key.startswith("x") else "set_yticklabels"
                if hasattr(target, labels):
                    getattr(target, labels)([])
            else:
                getattr(target, setter)([float(v) for v in values])
            if saved is not None:
                target.set_xlim(saved[0])
                target.set_ylim(saved[1])
                target.set_autoscalex_on(saved[2])
                target.set_autoscaley_on(saved[3])
    for key, setter in (("xticklabel", "set_xticklabels"),
                        ("yticklabel", "set_yticklabels")):
        if key in options and hasattr(target, setter):
            values = np.asarray(options[key]).ravel()
            getattr(target, setter)([_tick_text(v) for v in values])
    if "xscale" in options and hasattr(target, "set_xscale"):
        target.set_xscale(str(options["xscale"]))
    if "yscale" in options and hasattr(target, "set_yscale"):
        target.set_yscale(str(options["yscale"]))
    if "position" in options and hasattr(target, "set_position"):
        target.set_position([float(v) for v in np.asarray(options["position"]).ravel()])
    if "currentaxes" in options and hasattr(target, "add_subplot"):
        # `set(gcf,'currentaxes',h)`：切换到指定的坐标区（GUI 回调里反复用）
        global _current_axes
        _current_axes = options["currentaxes"]
    if hasattr(target, "set_name_value"):
        target.set_name_value(options)
    if isinstance(target, Axes):
        # `set(gca,'Fontsize',25)`、`set(gca,'GridLineStyle',':')` 这类：真正落到图上。
        # 只设刻度/范围/位置的调用（`set(gca,'xtick',...)` 这种最多）不走重排那套，
        # 否则二维求解器里上千轮绘图会白白多花几分钟。
        if _AXES_STYLE_KEYS & set(options):
            _apply_axes_props(target, options)
    elif isinstance(target, Legend):
        _apply_legend_props(target, options)
    elif isinstance(target, Text):
        _apply_text_props(target, options)
    return target


def _apply_legend_props(handle, options):
    """`set(h,'interpreter','latex')` / `set(h,'FontSize',25)`：改图例的文字。"""
    if "interpreter" in options:
        # 标签在建立图例时已经转好了（见 `legend`），这里只是认下这个调用。
        for text in handle.get_texts():
            text.set_text(_label(text.get_text()))
    if "fontsize" in options:
        # matplotlib 3.9 起 `Legend.set_fontsize` 没了，改逐个文字对象设
        for text in handle.get_texts():
            text.set_fontsize(mstyle.np_scalar(options["fontsize"]))
        handle._mplot_explicit_fontsize = True
    if "fontweight" in options:
        for text in handle.get_texts():
            text.set_fontweight(str(options["fontweight"]).lower())
    if "textcolor" in options:
        for text in handle.get_texts():
            text.set_color(options["textcolor"])
    if "edgecolor" in options:
        handle.get_frame().set_edgecolor(options["edgecolor"])
    if "box" in options:
        handle.set_frame_on(str(options["box"]).lower() != "off")
    if "location" in options:
        location = str(options["location"]).lower().replace("_", "")
        handle.set_loc(_LEGEND_LOCATIONS.get(location, location))
    return handle


def _apply_text_props(handle, options):
    """`set(h,'FontSize',25)` 之类的对文本对象生效的那几条。"""
    if "fontsize" in options:
        handle.set_fontsize(mstyle.np_scalar(options["fontsize"]))
    if "fontweight" in options:
        handle.set_fontweight(str(options["fontweight"]).lower())
    if "color" in options:
        handle.set_color(options["color"])
    if "string" in options:
        handle.set_text(str(_label(str(options["string"]))))
    return handle


def _tick_text(value) -> str:
    """`set(gca,'xticklabel',get(gca,'xtick'))` 里给的是数值，标签要转成文本。"""
    return f"{float(np.asarray(value).ravel()[0]):g}"


def get_(target, name):
    """`get(h,'名字')`：返回现在生效的刻度 / 范围 / 位置。

    * `get(gcf,'Position')` 只用于设置纸张大小，给一个等价的方框即可；
    * `get(gca,'Position')` 给 MATLAB 的 `[left bottom width height]`——
      `figure7.m` 用它把（被色条改窄的）坐标区框记下来再放回去；
    * `get(gca,'xtick')`/`'ytick'`/`'xlim'`/`'ylim'` 给现在的刻度与范围
      （`figure7.m` 把刻度数值直接当标签用）。
    """
    key = str(name).lower()
    if key == "position":
        if hasattr(target, "get_size_inches"):        # 图窗
            width, height = gcf().get_size_inches()
            return [0.0, 0.0, float(width), float(height)]
        box = target.get_position()
        return [float(box.x0), float(box.y0), float(box.width), float(box.height)]
    if key in ("xtick", "ytick"):
        ticks = target.get_xticks() if key == "xtick" else target.get_yticks()
        return np.asarray(ticks, dtype=float).reshape(1, -1)
    if key in ("xlim", "ylim"):
        limits = target.get_xlim() if key == "xlim" else target.get_ylim()
        return np.asarray(limits, dtype=float).reshape(1, -1)
    raise ValueError(f"mplot.get_: 不支持 {name}")


def print_fig(target, *args):
    """`print(gcf,'-dpng',name)`：把图存进算例的产出目录。"""
    name = str(args[-1])
    path = Path(mio.PRODUCED.directory) / name
    target.savefig(path)
    return path


#: `imwrite(...,'WriteMode','append')` 累积的帧（MATLAB 是往文件里追加，这里最后
#: 一次性写出，键用产出目录里的完整路径，免得跨算例串味）
_gif_frames: dict[str, list] = {}


def getframe(figure_obj=None):
    """MATLAB 的 `getframe(gcf)`：抓当前图窗的一帧（这里是 RGBA 像素数组）。"""
    figure_obj = gcf() if figure_obj is None else figure_obj
    figure_obj.canvas.draw()
    return np.asarray(figure_obj.canvas.buffer_rgba())


def frame2im(frame):
    """MATLAB 的 `frame2im(frame)`：把帧变成图像数据。

    MATLAB 的 `getframe` 返回的是带 `cdata` 字段的结构体，这里 `getframe` 直接给
    像素数组，所以这一步是恒等映射。
    """
    return np.asarray(frame)


def rgb2ind(image, n=256):
    """MATLAB 的 `[ind,cm]=rgb2ind(im,256)`：量化成索引图 + 调色板（给 GIF 用）。"""
    from PIL import Image

    array = np.asarray(image)
    rgb = array[..., :3] if array.ndim == 3 else np.repeat(array[..., None], 3, axis=-1)
    img = Image.fromarray(rgb.astype("uint8")).convert(
        "P", palette=Image.ADAPTIVE, colors=int(n))
    indices = np.asarray(img)
    palette = np.asarray(img.getpalette()[:3 * int(n)], dtype=float).reshape(-1, 3)
    return indices, palette


def imwrite(indices, palette, filename, loop_count=None, delay_time=None,
            write_mode=None):
    """MATLAB 的 `imwrite(ind,cm,name,'gif','WriteMode','append','DelayTime',t)`。

    只在 `appendix_B/grass_ft/plot_data.m` 里用（那一段要攒 41 帧再写成一个 GIF）。
    GIF 的帧延时以 10 ms 为单位，MATLAB 那个 `1e-4` 秒的延时落到这里取 1 ms 为下限。
    """
    from PIL import Image

    path = Path(mio.PRODUCED.directory) / str(filename)
    image = Image.fromarray(np.asarray(indices).astype("uint8"), mode="P")
    image.putpalette(np.asarray(palette, dtype="uint8").ravel().tolist())
    frames = _gif_frames.setdefault(str(path), [])
    frames.append(image)
    duration = max(1, int(round(float(delay_time) * 1000))) if delay_time is not None else 100
    frames[0].save(path, format="GIF", save_all=True, append_images=frames[1:],
                   duration=duration, loop=0)
    return path


def close_all():
    """MATLAB 的 `close all`。"""
    global _current, _current_axes, _pending_clear
    plt.close("all")
    _figures.clear()
    _spanning.clear()
    _subplots.clear()
    # 关掉图窗，挂在它上面的控件也就不存在了（MATLAB 的 close all 同样如此）
    _uicontrols.clear()
    _current = None
    _current_axes = None
    _pending_clear = False


def _label(text) -> str:
    """把 MATLAB 的 TeX 标签（`\\psi`、`\\beta_s`）换成 matplotlib 能认的写法。

    文本里**已经有 `$`** 的（绘图脚本大量写 `'$\\psi$'`）原样放行——再包一层会
    变成 `$J_$\\theta$$` 这种坏掉的写法。
    """
    if not isinstance(text, str):
        return text
    if "$" in text:
        return text
    return _LATEX.sub(lambda match: f"${match.group(0)}$", text)


def _font(args) -> dict:
    """把 `'Fontsize',25` 之类的成对参数换成 matplotlib 的字体参数。"""
    options = _options(args)
    kwargs = {}
    if "fontsize" in options:
        kwargs["fontsize"] = mstyle.np_scalar(options["fontsize"])
    if "fontweight" in options:
        kwargs["fontweight"] = str(options["fontweight"]).lower()
    if "color" in options:
        kwargs["color"] = options["color"]
    return kwargs


def _label_font(axes_obj, args) -> dict:
    """轴标签的字体参数：脚本没给字号/字重时按 MATLAB 的默认倍数补上。"""
    kwargs = _font(args)
    size = getattr(axes_obj, "_mplot_fontsize", None)
    size = mstyle.FACTORY_AXES_FONTSIZE if size is None else size
    kwargs.setdefault("fontsize", float(size) * 1.1)
    kwargs.setdefault("fontweight", getattr(axes_obj, "_mplot_fontweight", "normal"))
    return kwargs


def _cell_text(value) -> str:
    array = np.asarray(value)
    if array.dtype.kind in "US":
        return "".join(array.ravel().tolist())
    return str(value)


_LATEX = re.compile(r"\\[A-Za-z]+(?:_\{?[A-Za-z0-9]+\}?)?")
