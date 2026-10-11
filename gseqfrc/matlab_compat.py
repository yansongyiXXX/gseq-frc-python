"""MATLAB 语义垫片。

逐行移植里唯一允许改写的就是 MATLAB 特有的语义：1 基索引、列优先的线性索引、
数组作 `if` 条件时的隐含 `all()`、`size` 的返回形状、`length` 的定义等等。
它们全部集中在这里，每条都注明对应的 MATLAB 行为。

约定：移植文件里凡是从 MATLAB 数组语义转换过来的地方都显式调用本模块，
不去偷偷用 numpy 的等价写法——这样 review 时一眼能看出哪些是原文、哪些是语言差异。
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np

from . import fastmath


def scalar(value) -> np.ndarray:
    """MATLAB 的 1×1 double 矩阵。

    MATLAB 没有真正的标量：`n = numel(x)` 得到的是 1×1 矩阵，快照里就存成
    shape=(1,1)。Python 的 float 是 0 维，形状对不上会被比对工具判为不等，
    所以脚本里每个 MATLAB 标量都要过一遍这个函数。
    """
    return np.array([[scalar_value(value)]])


def scalar_value(value) -> float:
    """把一个 MATLAB 的 1×1 矩阵取成 Python 浮点（相当于 `double(x)`）。

    `float(数组)` 在 numpy 1.25 起会被弃用，所有取标量值的地方都走这里，
    顺便把"传入的不是标量"这种错误早点暴露出来。
    """
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"期望标量（1×1），得到形状 {array.shape}")
    return float(array.reshape(-1)[0])


def numel(value) -> int:
    """MATLAB `numel`。"""
    return int(np.asarray(value).size)


def length(value) -> int:
    """MATLAB `length`：各维尺寸的最大值，空数组为 0。"""
    shape = np.asarray(value).shape
    return int(max(shape)) if shape else 1


def size(value) -> np.ndarray:
    """MATLAB `size`：返回 1×2 的 double 行向量（`[nrows ncols]`）。"""
    arr = np.asarray(value)
    if arr.ndim == 0:
        dims = (1, 1)
    elif arr.ndim == 1:
        # MATLAB 没有一维数组：长度为 n 的向量被当作 1×n
        dims = (1, arr.shape[0])
    else:
        dims = arr.shape[:2]
    return np.array([[float(dims[0]), float(dims[1])]])


def zeros(*dims: int) -> np.ndarray:
    """MATLAB `zeros(m, n)`。"""
    return np.zeros(_as_shape(dims), dtype=float)


def ones(*dims: int) -> np.ndarray:
    """MATLAB `ones(m, n)`。"""
    return np.ones(_as_shape(dims), dtype=float)


def _as_shape(dims) -> tuple:
    dims = tuple(int(d) for d in dims)
    return dims if len(dims) > 1 else (dims[0], dims[0])


def mrange(start: int, stop: int | None = None) -> list[int]:
    """MATLAB 的 `a:b`（两端都含）。`mrange(n)` 等价于 `1:n`。"""
    if stop is None:
        start, stop = 1, start
    return list(range(int(start), int(stop) + 1))


def colon(start: float, step: float | None = None, stop: float | None = None) -> np.ndarray:
    """MATLAB 的浮点冒号 `a:b:c`。

    MATLAB 的末端规则和 `np.arange` 不一样：`(c-a)/b` 只要在最后几位浮点上等于整数，
    它就把端点取进来；而 `np.arange` 会少取一个点。`fun_Bfield.m` 方法 2 里的
    `rr=rmin:dr:rmax` 正是这种情形（`dr=(rmax-rmin)/(nr-1)`），少一个点会让
    后面按 `rr(jr)` 取值整体错位。

    返回值一律是行向量（1×n），与 MATLAB 一致。
    """
    start = float(np.asarray(start).ravel()[0])
    if stop is None:
        stop = float(np.asarray(step).ravel()[0])
        step = 1.0
    step = float(np.asarray(step).ravel()[0])
    stop = float(np.asarray(stop).ravel()[0])
    if step == 0.0:
        raise ZeroDivisionError("colon 的步长不能为 0")

    raw = (stop - start) / step
    nearest = round(raw)
    if abs(raw - nearest) <= 8.0 * np.finfo(float).eps * max(1.0, abs(raw)):
        n = int(nearest)
    else:
        n = int(np.floor(raw))
    if n < 0:
        return np.zeros((0,))
    return (start + np.arange(n + 1) * step).reshape(1, -1)


def linspace(start: float, stop: float, n: int) -> np.ndarray:
    """MATLAB 的 `linspace(a,b,n)`，返回 1×n 行向量。

    照抄 MATLAB 的构造方式：前 n-1 个点由 `(b-a)/(n-1)` 累加得到，
    **最后一个点直接取 b**。这样端点不会有累加误差，逐位比对才过得去。
    """
    start = float(np.asarray(start).ravel()[0])
    stop = float(np.asarray(stop).ravel()[0])
    n = int(np.asarray(n).ravel()[0])
    if n <= 0:
        return np.zeros((1, 0))
    if n == 1:
        return np.array([[stop]])
    values = start + np.arange(n - 1) * (stop - start) / (n - 1)
    return np.concatenate([values, [stop]]).reshape(1, -1)


def ndgrid(*vectors) -> list:
    """MATLAB 的 `[R,Z]=ndgrid(rr,zz)`：`R(i,j)=rr(i)`、`Z(i,j)=zz(j)`。

    注意这与 `meshgrid` 的默认序相反，所以必须显式用 `indexing='ij'`。
    """
    grids = np.meshgrid(*[np.asarray(v).ravel() for v in vectors], indexing="ij")
    return list(grids)


def sub2ind(dims, *subscripts) -> np.ndarray:
    """MATLAB 的 `sub2ind`：把行列下标换成 1 基的线性下标（列优先）。"""
    dims = [int(d) for d in np.asarray(dims).ravel()]
    subs = np.broadcast_arrays(*[np.asarray(s, dtype=int) for s in subscripts])
    linear = np.zeros(subs[0].shape, dtype=int)
    stride = 1
    for size, index in zip(dims, subs):
        linear = linear + (index - 1) * stride
        stride *= size
    return linear + 1


def take(array, index) -> np.ndarray:
    """MATLAB 的线性索引取值 `a(index)`（列优先，1 基）。

    有一条 MATLAB 特有的形状规则要照抄：**如果被索引的是向量（1×n 或 n×1），
    结果就跟随被索引对象的朝向，而不是下标的朝向**。`u_psib_coil` 里
    `jrb2([1 64 ...])` 得到的是列向量而不是行向量，正是靠这条规则。
    """
    source = np.asarray(array)
    index = np.asarray(index)
    flat = source.reshape(-1, order="F")
    values = flat[index.astype(int).ravel() - 1].reshape(-1)
    source_is_column = source.ndim == 2 and source.shape[1] == 1
    source_is_row = source.ndim == 2 and source.shape[0] == 1
    index_is_vector = index.ndim <= 1 or (index.ndim == 2 and 1 in index.shape)
    if index_is_vector and source_is_column:
        return values.reshape(-1, 1)
    if index_is_vector and source_is_row:
        return values.reshape(1, -1)
    return values.reshape(index.shape)


def set_at(array: np.ndarray, index, value) -> None:
    """MATLAB 的索引赋值 `x(ind) = value`（线性索引，1 基）。

    右值是标量时按 MATLAB 的标量扩展填满这组下标，是数组时按下标顺序逐个赋。

    还有一条相关语义：`x(ind)=v` 在 `x` 还不存在时会**自动创建**它，大小取到能
    容纳最大的下标（行向量的下标给出行向量，其余元素为 0）。`gen_psib.m` 里的
    `psib(ind1)=...; psib(ind2)=...` 正是靠这条——移植时要先建好同形状的零数组。
    """
    flat = np.asarray(array).reshape(-1, order="F")
    positions = np.asarray(index).astype(int).ravel() - 1
    values = np.asarray(value)
    if values.size == 1:
        flat[positions] = values.reshape(-1)[0]
    else:
        flat[positions] = values.reshape(-1)


def cumsum(array) -> np.ndarray:
    """MATLAB 的 `cumsum`：沿**第一个非单例维**累加，形状不变。

    numpy 的 `cumsum(v)` 不带 axis 会先展平，行向量会变成一维数组——形状对不上，
    `psi=cumsum(rr.*Bz*dr)` 这类写法必须按 MATLAB 的规则走。
    """
    arr = np.asarray(array)
    axis = 0
    for index, size in enumerate(arr.shape):
        if size != 1:
            axis = index
            break
    return np.cumsum(arr, axis=axis)


def mmin(array):
    """MATLAB 的 `min(X)`：**忽略 NaN**（numpy 的 `np.min` 会把 NaN 传出来）。

    绘图脚本里的 ψ 常带 NaN 掩膜（壁面以外置 NaN，见 `figure_8_1.m`），
    MATLAB 的 `min(min(psi))` 照样给有限值里的最小值，而 `np.min` 给 NaN——
    直接用 numpy 的话 `psi==min(min(psi))` 一个都匹配不上，`find` 返回空，
    脚本随后在下标处崩掉。`plt_psi_r.m` 里那句只打印不赋值的
    `min(psi(:,floor(Zs*nZ)))` 也是同一条规矩。
    """
    return float(np.nanmin(np.asarray(array, dtype=float)))


def mmax(array):
    """MATLAB 的 `max(X)`：**忽略 NaN**（见 `mmin`）。"""
    return float(np.nanmax(np.asarray(array, dtype=float)))


def diff(array, n: int = 1, dim: int = 1):
    r"""MATLAB 的 `diff(X,n,dim)`：沿第 `dim` 维做 n 阶前向差分。

    仓库里的用处只有一个：`figure_A_3.m` 的 `diff(Br,1,2)`，Br 是 nR×nZ，
    结果沿 z 方向少一列（nR×(nZ-1)），之后取第 `Z0` 列当 \((\partial_zB_r)\)。
    一维输入按 MATLAB 的"第一个非单例维"处理（也就是直接给相邻差）。
    """
    values = np.asarray(array, dtype=float)
    if values.ndim == 1:
        return np.diff(values, n=n)
    return np.diff(values, n=n, axis=dim - 1)


def cumtrapz(x, y=None) -> np.ndarray:
    r"""MATLAB 的 `cumtrapz(X,Y)`：梯形法**累积**积分，形状与 Y 相同。

    `figure_A_3.m` 用它算 \(\int_0^r \frac{B_z}{\mu_0}\frac{\partial B_r}{\partial z}dr\)：
    X 是一维坐标向量（`rr`，1×nR），Y 是沿**第一个非单例维**排列的数据
    （那里是 nR×1 的列向量）。X 给的是真实坐标、不是等间距步长，所以按
    `0.5*(y_k+y_{k+1})*(x_{k+1}-x_k)` 逐段累加（MATLAB 对非等间距 X 正是这么算的）。
    """
    values = np.asarray(y if y is not None else x, dtype=float)
    axis = 0
    for index, size in enumerate(values.shape):
        if size != 1:
            axis = index
            break
    if y is None:
        steps = np.ones(values.shape[axis] - 1)
    else:
        steps = np.diff(np.asarray(x, dtype=float).ravel())
    moved = np.moveaxis(values, axis, 0)
    increments = (0.5 * (moved[1:] + moved[:-1])
                  * steps.reshape((-1,) + (1,) * (moved.ndim - 1)))
    head = np.zeros((1,) + moved.shape[1:], dtype=float)
    cumulative = np.concatenate([head, np.cumsum(increments, axis=0)], axis=0)
    return np.moveaxis(cumulative, 0, axis)


def trapz(x, y=None) -> float:
    """MATLAB 的 `trapz(X,Y)`：梯形法积分（X 是坐标、可以不等间距）。

    `plt_psi_r.m` 用它算加权平均：`trapz(rrs,rrs.*PP)/trapz(rrs,rrs)`。
    实现上就是 `cumtrapz` 的最后一段（两者共用同一套梯形规则）。
    """
    cumulative = cumtrapz(x, y)
    return float(np.asarray(cumulative).ravel(order="F")[-1])


def cellstr(values) -> np.ndarray:
    """MATLAB 的元胞字符串数组（`str{jd}=...`），返回 1×n 的 object 数组。

    每个格子存的是 MATLAB 的字符行向量，scipy 读出来是 `array(['文本'])`
    （shape (1,)、dtype '<Un'）；这里按同样的表示构造，方便与基准比对。
    """
    texts = list(values)
    cells = np.empty((1, len(texts)), dtype=object)
    for index, text in enumerate(texts):
        cells[0, index] = np.array([str(text)])
    return cells


def interp1(x, v, xq, method: str = "linear"):
    """MATLAB 的 `interp1(x,v,xq,'linear'|'spline')`（一维插值）。

    * `linear`：分段线性，等价于 `np.interp`（区间外的点按端点取值）；
    * `spline`：**not-a-knot** 三次样条，这正是 MATLAB `interp1(...,'spline')`
      用的边界条件，所以用 `scipy.interpolate.CubicSpline(..., bc_type='not-a-knot')`
      而不是默认的自然样条。

    x/v 可能是行向量也可能是列向量；`xq` 的形状跟随 MATLAB 的规则（这里用于
    网格上的插值，返回与 xq 同形状）。
    """
    from scipy.interpolate import CubicSpline

    xs = np.asarray(x, dtype=float).ravel()
    vs = np.asarray(v, dtype=float).ravel()
    query = np.asarray(xq, dtype=float)
    if method == "linear":
        values = np.interp(query, xs, vs)
    elif method == "spline":
        values = CubicSpline(xs, vs, bc_type="not-a-knot")(query)
    else:
        raise ValueError(f"interp1: 不支持的方法 {method!r}")
    return values


def interp2(x, y, v, xq, yq, method: str = "linear"):
    r"""MATLAB 的 `interp2(X,Y,V,Xq,Yq,'nearest'|'linear'|'spline')`。

    采样网格的朝向不固定：`ndgrid` 出来的是 X 沿轴 1、Y 沿轴 0，但仓库里存的
    `.mat` 也有相反朝向的。这里按"哪条轴在变"自动判断，再按 values 的维度顺序
    交给 scipy。返回值形状跟随查询网格（与 MATLAB 一致）。

    三种方法分别对应：

    * `nearest`/`linear`：`RegularGridInterpolator`（二维求解器用 `nearest` 打初值，
      那条路径有冻结算例覆盖）；
    * `spline`：MATLAB 的 `'spline'` 是**张量积三次样条**（先按行、再按列各做一次
      not-a-knot 三次样条），对应 scipy 的 `RectBivariateSpline(kx=ky=3, s=0)`
      ——默认边界条件同样是 not-a-knot。绘图脚本（`figure_4_1`、`Fig8-9`、
      `appendix_A` 那几份）都是 `interp2(...,'spline')`，在此之前这里会把它**静默
      退化成** `linear`。就随论文提供的数据而言两者数值上没差别（查询网格与数据
      网格重合，插值点正好落在采样点上：实测 ψ 的相对偏差 \(6\times10^{-16}\)、
      由它中心差分出的 \(B_z\) 是 \(2\times10^{-14}\)），但网格一旦不重合，
      "线性"和"三次样条"就是两张不同的图，所以按原文补上。
    """
    from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator

    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    # values(i,j) 对应 (X(i,j), Y(i,j))，所以 scipy 的轴顺序就是 (Y轴, X轴)
    if xa.shape[1] > 1 and xa[0, 1] != xa[0, 0]:
        axes = (ya[:, 0], xa[0, :])   # X 沿轴 1、Y 沿轴 0
    else:
        axes = (xa[:, 0], ya[0, :])   # X 沿轴 0、Y 沿轴 1
    values = np.asarray(v, dtype=float)
    xq_arr = np.asarray(xq, dtype=float)
    yq_arr = np.asarray(yq, dtype=float)
    if method == "spline":
        spline = RectBivariateSpline(axes[0], axes[1], values, kx=3, ky=3, s=0)
        return spline(yq_arr.ravel(order="F"), xq_arr.ravel(order="F"),
                      grid=False).reshape(xq_arr.shape, order="F")
    interpolator = RegularGridInterpolator(
        axes, values, method="nearest" if method == "nearest" else "linear",
        bounds_error=False, fill_value=None,  # 与 MATLAB 一样允许外推
    )
    query = np.stack([yq_arr.ravel(order="F"), xq_arr.ravel(order="F")], axis=-1)
    return interpolator(query).reshape(xq_arr.shape, order="F")


def exp(value):
    """MATLAB 的 `exp`——**逐位**对齐（FDLIBM 的 `__ieee754_exp`）。

    为什么要这么较真：numpy/glibc 的 `exp` 与 MATLAB 在约 **10%** 的输入上差
    1 ULP。原程序里 `F1=exp(-alpha*psim)*(sigma*psim+1)^n-1/betas` 是两个 O(1)
    的数相减，1 ULP 的 `exp` 差异放大成 2 ULP 的 `F1`；`F1` 又决定内层 `psim`，
    而外层被积函数在 `psim` 处有平方根奇点（积分对 `psim` 的灵敏度是 \(10^6\)
    量级），于是 1 ULP 一路放大成 \(10^{-10}\) 的函数值差异。外层 `fsolve` 用的是
    **前向差分**雅可比（步长 \(10^{-7}\)），函数值上 \(10^{-10}\) 的噪声就是雅可比
    上百分之几的误差——多根问题（`s_solveMRR_f4/f5`）由此走到另一个根上。

    实测（`tools/exp_probe.m` + `tools/mexp.py`）：MATLAB R2023a 的 `exp` 与
    FDLIBM 的 `__ieee754_exp` 在 402 个随机样本上**逐位相同**，而 glibc 有 41 个
    不同。所以这里照抄 FDLIBM 的算法，而不是"更准的实现"——要的是**一样**。
    """
    array = np.asarray(value)
    if array.ndim == 0:
        return _exp_scalar(complex(array) if np.iscomplexobj(array)
                           else float(array))
    if np.iscomplexobj(array):
        # MATLAB 的 exp 对复数用 exp(a)*(cos b + i sin b)；原程序里没走到
        return np.exp(array)
    # 同一条 FDLIBM 算法，标量循环版（numba）比向量化版快 ~30 倍：向量化版每个
    # 表达式都要分配一整片 1 MB 的临时量，131841 点上 21 ms，标量循环 0.57 ms。
    # 逐位相同（`tools/exp_numba_probe.py` 与 tests 都在盯这件事）；numba 不在
    # 环境里时返回 None，自动退回下面的 `_exp_array`。
    fast = fastmath.exp_array(array)
    if fast is not None:
        return fast
    return _exp_array(array)


# --- FDLIBM e_exp.c 的常数（一个字符都不改） ---------------------------------
_EXP_HALF = (0.5, -0.5)
_EXP_HUGE = 1.0e300
_EXP_TWOM1000 = 9.33263618503218878990e-302
_EXP_O_THRESHOLD = 7.09782712893383973096e02
_EXP_U_THRESHOLD = -7.45133219101941108420e02
_EXP_LN2HI = (6.93147180369123816490e-01, -6.93147180369123816490e-01)
_EXP_LN2LO = (1.90821492927058770002e-10, -1.90821492927058770002e-10)
_EXP_INVLN2 = 1.44269504088896338700e00
_EXP_P1 = 1.66666666666666019037e-01
_EXP_P2 = -2.77777777770155933842e-03
_EXP_P3 = 6.61375632143793436117e-05
_EXP_P4 = -1.65339022054652515390e-06
_EXP_P5 = 4.13813679705723846039e-08


def _exp_scalar(x: float) -> float:
    """标量版：与 `e_exp.c` 的分支一一对应。"""
    import struct

    bits = struct.unpack("<Q", struct.pack("<d", x))[0]
    hx = bits >> 32
    xsb = (hx >> 31) & 1
    hx &= 0x7FFFFFFF

    if hx >= 0x40862E42:                    # |x| 很大
        if hx >= 0x7FF00000:
            if ((hx & 0xFFFFF) | (bits & 0xFFFFFFFF)) != 0:
                return x + x                # NaN
            return x if xsb == 0 else 0.0    # ±Inf
        if x > _EXP_O_THRESHOLD:
            return _EXP_HUGE * _EXP_HUGE     # 上溢 → Inf
        if x < _EXP_U_THRESHOLD:
            return _EXP_TWOM1000 * _EXP_TWOM1000

    k = 0
    hi = 0.0
    lo = 0.0
    if hx > 0x3FD62E42:                     # |x| > 0.5*ln2
        if hx < 0x3FF0A2B2:                 # |x| < 1.5*ln2
            hi = x - _EXP_LN2HI[xsb]
            lo = _EXP_LN2LO[xsb]
            k = 1 - xsb - xsb
        else:
            k = int(_EXP_INVLN2 * x + _EXP_HALF[xsb])
            t = float(k)
            hi = x - t * _EXP_LN2HI[0]
            lo = t * _EXP_LN2LO[0]
        x = hi - lo
    elif hx < 0x3E300000:                   # |x| < 2**-28
        if _EXP_HUGE + x > 1.0:
            return 1.0 + x

    t = x * x
    c = x - t * (_EXP_P1 + t * (_EXP_P2 + t * (_EXP_P3
                                              + t * (_EXP_P4 + t * _EXP_P5))))
    if k == 0:
        return 1.0 - ((x * c) / (c - 2.0) - x)
    y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi)
    ybits = struct.unpack("<Q", struct.pack("<d", y))[0]

    def with_high_word(bits: int, highword: int) -> float:
        # SET_HIGH_WORD：只替换高 32 位（指数+符号），低 32 位不动
        return struct.unpack("<d", struct.pack(
            "<Q", (bits & 0xFFFFFFFF) | ((highword & 0xFFFFFFFF) << 32)))[0]

    high = ybits >> 32
    if k >= -1021:
        return with_high_word(ybits, high + (k << 20))
    return with_high_word(ybits, high + ((k + 1000) << 20)) * _EXP_TWOM1000


def _exp_array(value: np.ndarray) -> np.ndarray:
    """数组版：同样的算法，用 numpy 一次算完整段（供 `integral` 的节点数组用）。"""
    with np.errstate(all="ignore"):
        return _exp_array_raw(value)


def _exp_array_raw(value: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(value, dtype=np.float64)
    bits = a.view(np.uint64)
    high = (bits >> np.uint64(32)).astype(np.int64)
    xsb = ((high >> 31) & 1).astype(np.int64)
    hx = high & 0x7FFFFFFF

    ln2hi = np.where(xsb == 0, _EXP_LN2HI[0], -_EXP_LN2HI[0])
    ln2lo = np.where(xsb == 0, _EXP_LN2LO[0], -_EXP_LN2LO[0])

    hi = np.zeros(a.shape, dtype=np.float64)
    lo = np.zeros(a.shape, dtype=np.float64)
    k = np.zeros(a.shape, dtype=np.int64)

    m1 = (hx > 0x3FD62E42) & (hx < 0x3FF0A2B2)
    # 注意：C 里这个 else 分支**没有上界**（|x| 很大的那些值由上面的提前 return 处理，
    # 落不到这里的才继续往下走）——加个上界会把次正规结果算成 1.0。
    m2 = hx >= 0x3FF0A2B2
    # 加速：原来是 np.where(cond, A, B) 链，每次都要为 A、B 各分配一整片数组；
    # 改成"先算候选值、再掩码写入"，参与的算式与元素完全一样（逐位不变），
    # 但省掉了大量临时量。下面的 `hi/lo/k/x/y` 同理。
    if m1.any():
        hi[m1] = a[m1] - ln2hi[m1]
        lo[m1] = ln2lo[m1]
        k[m1] = 1 - 2 * xsb[m1]
    if m2.any():
        kk = np.trunc(_EXP_INVLN2 * a + np.where(xsb == 0, 0.5, -0.5)
                      ).astype(np.int64)
        t = kk.astype(np.float64)
        hi = np.where(m2, a - t * _EXP_LN2HI[0], hi)
        lo = np.where(m2, t * _EXP_LN2LO[0], lo)
        k = np.where(m2, kk, k)
    x = np.where(m1 | m2, hi - lo, a)

    t = x * x
    c = x - t * (_EXP_P1 + t * (_EXP_P2 + t * (_EXP_P3
                                              + t * (_EXP_P4 + t * _EXP_P5))))
    with np.errstate(all="ignore"):
        y0 = 1.0 - ((x * c) / (c - 2.0) - x)
        y1 = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi)
    y = np.where(k == 0, y0, y1)

    tiny = hx < 0x3E300000
    if tiny.any():
        y = np.where(tiny, 1.0 + a, y)

    # 乘 2^k 用"直接改指数位"实现（FDLIBM 的 SET_HIGH_WORD）
    ybits = y.view(np.uint64)
    big = k >= -1021
    # 注意：FDLIBM 是把 k<<20 加到**高 32 位**上，在 64 位模式里就是 k<<52
    delta = (np.where(big, k, k + 1000).astype(np.int64) << 52).astype(np.uint64)
    scaled = (ybits + delta).view(np.float64)
    y = np.where(big, scaled, scaled * _EXP_TWOM1000)

    special = hx >= 0x40862E42
    if special.any():
        nan_mask = np.zeros(a.shape, dtype=bool)
        if (hx >= 0x7FF00000).any():
            mantissa = ((high & 0xFFFFF) << 32) | (bits & 0xFFFFFFFF).astype(np.int64)
            nan_mask = (hx >= 0x7FF00000) & (mantissa != 0)
            inf_mask = (hx >= 0x7FF00000) & ~nan_mask
            y = np.where(inf_mask, np.where(xsb == 0, a, 0.0), y)
        y = np.where(special & (a > _EXP_O_THRESHOLD),
                     _EXP_HUGE * _EXP_HUGE, y)
        y = np.where(special & (a < _EXP_U_THRESHOLD),
                     _EXP_TWOM1000 * _EXP_TWOM1000, y)
        y = np.where(nan_mask, a + a, y)
    return y


def sqrt(value):
    """MATLAB 的 `sqrt`：**负数开方返回复数**，而 numpy 给的是 nan。

    这条差别会改变整个求解路径：原程序里 `1./sqrt(1.0-fP(psi))` 一旦遇到负的
    被开方数，MATLAB 得到虚数，被积函数随之变成复数，`integral` 与 `fsolve`
    都在复数上继续算——`s_3pe` 与 `s_mrr2` 的冻结值正是 complex128（虚部全 0
    或极小）。用 numpy 的 sqrt 会得到 nan，求根会跑偏。
    """
    array = np.asarray(value)
    if np.iscomplexobj(array):
        with np.errstate(all="ignore"):
            return np.sqrt(array)
    with np.errstate(invalid="ignore", divide="ignore"):
        if np.any(array < 0):
            return np.sqrt(array.astype(complex))
        return np.sqrt(array)


def el(array, index: int):
    """MATLAB 的线性索引 `x(index)`，按**列优先**展开。"""
    return np.asarray(array).reshape(-1, order="F")[int(index) - 1]


def set_el(array: np.ndarray, index: int, value) -> None:
    """MATLAB 的线性索引赋值 `x(index) = value`，按列优先展开。

    1×1 的右值会做标量扩展，与原程序一致。
    """
    flat = np.asarray(array).reshape(-1, order="F")
    flat[int(index) - 1] = _unwrap_scalar(value)


def ij(array, i: int, j: int):
    """MATLAB 的二维索引 `A(i, j)`。"""
    return np.asarray(array)[int(i) - 1, int(j) - 1]


def set_ij(array: np.ndarray, i: int, j: int, value) -> None:
    """MATLAB 的二维索引赋值 `A(i, j) = value`。"""
    np.asarray(array)[int(i) - 1, int(j) - 1] = _unwrap_scalar(value)


def ijk(array, i: int, j: int, k: int):
    """MATLAB 的三维索引 `A(i, j, k)`。"""
    return np.asarray(array)[int(i) - 1, int(j) - 1, int(k) - 1]


def set_ijk(array: np.ndarray, i: int, j: int, k: int, value) -> None:
    """MATLAB 的三维索引赋值 `A(i, j, k) = value`。"""
    np.asarray(array)[int(i) - 1, int(j) - 1, int(k) - 1] = _unwrap_scalar(value)


def row(*values) -> np.ndarray:
    """MATLAB 的水平拼接 `[a b c]`（1×n）。参数本身是向量时按 MATLAB 展开拼接。"""
    parts = [np.asarray(value, dtype=float).ravel() for value in values]
    if not parts:
        return np.zeros((1, 0))
    return np.concatenate(parts).reshape(1, -1)


def col(*values) -> np.ndarray:
    """MATLAB 的垂直拼接 `[a; b; c]`（n×1）。"""
    parts = [np.asarray(value, dtype=float).ravel() for value in values]
    if not parts:
        return np.zeros((0, 1))
    return np.concatenate(parts).reshape(-1, 1)


def struct(**fields) -> np.ndarray:
    """MATLAB 的 1×1 struct，例如 `struct('betas', betas, 'ls', ls)`。

    返回的形状与 `scipy.io.loadmat` 读 MATLAB struct 的形状一致（1×1 的结构化数组），
    这样比对时才不需要额外的转换。
    """
    array = np.empty((1, 1), dtype=[(name, object) for name in fields])
    for name, value in fields.items():
        array[name][0, 0] = value
    return array


def truth(condition) -> bool:
    """MATLAB `if (cond)` 的语义：对数组条件隐含 `all()`。

    例：`if ( k2 == 1.0 )` 在 `k2` 是向量时，只有当**每个**元素都等于 1 才成立。
    直接写成 `if np.any(...)` 会走到不同的分支。
    """
    return bool(np.all(np.asarray(condition, dtype=bool)))


def find(mask) -> np.ndarray:
    """MATLAB 的 `find(mask)`：返回 1 基下标，**朝向跟随被索引的数组**。

    行向量的 `find` 给行向量，列向量给列向量——`ind=find(rr<=1.0)` 之后
    还要用 `rr(ind)` 取值，朝向错了后面就会错位。
    """
    source = np.asarray(mask)
    indices = np.flatnonzero(source.reshape(-1, order="F")) + 1
    if source.ndim == 2 and 1 in source.shape:
        return indices.reshape(1, -1) if source.shape[0] == 1 else indices.reshape(-1, 1)
    return indices.reshape(-1, 1)


def num2str(value, digits: int | None = None) -> str:
    """MATLAB 的 `num2str`：默认 5 位有效数字，`num2str(x,n)` 给 n 位。

    只用在标题和图名里（`title(['\\beta_s=',num2str(betas)])`、
    `print(...,'RR_betas=0.6_ls=0.26511_alpha=5.964.png')`），
    但格式要对得上——图名是这些脚本的产出之一。

    第二个参数还可以是 printf 风格的格式串：`grass_ft/plot_data.m` 用
    `num2str(i-1,'%06d')` 拼数据文件名（`000000_pse.DAT`、`000050_pse.DAT`…）。
    """
    array = np.asarray(value).ravel()
    if array.size == 0:
        return ""
    if array.size > 1:
        # 数组情形：MATLAB 排成多行
        return "\n".join(num2str(x, digits) for x in array)
    number = float(array[0])
    if isinstance(digits, str):
        return digits % number
    if digits is None:
        text = f"{number:.5g}"
    else:
        text = f"{number:.{int(digits)}g}"
    return text.replace("e+0", "e+").replace("e-0", "e-")


def _unwrap_scalar(value):
    """把 MATLAB 的 1×1 矩阵还原成 Python 数值，供赋值使用。"""
    arr = np.asarray(value)
    return arr.item() if arr.size == 1 else value


def ellipticK(m):
    """MATLAB 内建的 `ellipticK(m)`：第一类完全椭圆积分。

    原程序 `fun_ellipEK.m` 的 `apx==0` 分支调它（`u_ellipEK_alt` 覆盖到）。
    数值上与 `ellipke` 的第一类完全积分一致，所以直接复用同一套 AGM。
    """
    return ellipke(m)[0]


def ellipticE(m):
    """MATLAB 内建的 `ellipticE(m)`：第二类完全椭圆积分。"""
    return ellipke(m)[1]


def ellipke(m, tol: float = 2.2204e-16):
    """MATLAB `ellipke(m, tol)`：AGM 求第一、二类完全椭圆积分。

    MATLAB 的 `ellipke(m)` 默认容差是 `eps`，`ellipke(m, tol)` 允许放宽。
    scipy 的 `ellipk/ellipe` 是另一套实现（机器精度，但**不保证逐位相同**），
    所以这里按 AGM 重写，好让 `k2 -> 1` 附近也和原程序一致。

    目前没有任何算例覆盖到这条分支（见 PORTING.md §5），先按最朴素的形式放这里。
    """
    m = np.asarray(m, dtype=float)
    scalar_input = m.ndim == 0
    a = np.ones_like(m)
    b = np.sqrt(1.0 - m)
    c = np.sqrt(m)
    # 标准 AGM：E = K·(1 - Σ 2^{n-1} c_n²)，第一项是 c_0²/2 = m/2
    s = 0.5 * m
    factor = 1.0
    for _ in range(100):
        a_next = 0.5 * (a + b)
        b_next = np.sqrt(a * b)
        c_next = 0.5 * (a - b)
        s = s + factor * c_next ** 2
        factor *= 2.0
        a, b, c = a_next, b_next, c_next
        if np.all(np.abs(c) <= tol * np.abs(a)):
            break
    k = np.pi / (2.0 * a)
    e = k * (1.0 - s)
    if scalar_input:
        # MATLAB 里标量进、1×1 出
        return np.array([[float(k)]]), np.array([[float(e)]])
    return k, e


# ---------------------------------------------------------------------------
# 绘图/诊断类脚本（`figure*.m`、`plt_*.m` 那一批）会用到的几种："数学库垫片"
# 之外的小工具。它们不影响任何有冻结算例的路径，但缺了那批脚本跑不起来。
# ---------------------------------------------------------------------------


def meshgrid(x, y):
    """MATLAB 的 `[X,Y]=meshgrid(x,y)`。

    注意 MATLAB 的 `meshgrid` 与它自己的 `ndgrid` **朝向相反**：`meshgrid` 出来的
    `X` 沿轴 1 变化（行是 y）。这正好是 numpy 默认的 `indexing='xy'`，所以直接转。
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    return np.meshgrid(x, y)


def str2num(value) -> float:
    """MATLAB 的 `str2num`：把字符串当**数值表达式**读出来。

    两种用法都要支持：读标量（`num2str` 的逆），以及读 GUI 输入框里
    "逗号分隔的一串数"——`frc_coil_gui.m` 的回调就是
    `rl=str2num(get(heditrl,'String'))`，那些字符串形如
    `'0.205,0.205,0.034,0.034,0.205,0.205'`（MATLAB 里逗号是行连接 → 1×6 行向量）。
    """
    if isinstance(value, (int, float, np.number)):
        return float(value)
    if hasattr(value, "dtype") and np.asarray(value).dtype.kind in "fiu":
        return float(np.asarray(value).ravel()[0])
    text = "".join(str(part) for part in np.asarray(value).ravel()) \
        if hasattr(value, "dtype") else str(value)
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        parts = [part for part in re.split(r"[,\s]+", text) if part]
        try:
            numbers = np.array([float(part) for part in parts], dtype=float)
        except ValueError as exc:
            raise ValueError(f"str2num: 解析不了 {value!r}") from exc
        return numbers.reshape(1, -1) if numbers.size > 1 else float(numbers[0])


def char(code) -> str:
    """MATLAB 的 `char(96+j)`：把编码变成字符。"""
    return chr(int(np.asarray(code).ravel()[0]))


def strcat(*parts) -> str:
    """MATLAB 的 `strcat`：拼接字符串，并**去掉每个输入尾部的空白**
    （这是 MATLAB 的语义，和直接相加不同）。
    """
    return "".join(str(p).rstrip() for p in parts)


def find2(mask):
    """MATLAB 的 `[i,j]=find(A)`：返回两个下标（1 基），按列优先。

    与 `find(mask)` 不同：这个是二维下标，原程序用它取 `R(i(1),j(1))` 这类。
    """
    index = find(mask)
    n_rows = np.shape(mask)[0]
    flat = np.asarray(index).ravel() - 1
    return flat % n_rows + 1, flat // n_rows + 1


def griddata(x, y, v, xq, yq):
    """MATLAB 的 `griddata(x,y,v,xq,yq)`：散点线性插值到规则网格。

    MATLAB 默认走 `'linear'`（Delaunay 三角剖分 + 线性插值），与
    `scipy.interpolate.griddata(..., method="linear")` 是同一个方法，但三角剖分
    与线性代数的实现不同，结果不会逐位相同——**只用在绘图脚本里**（那批脚本没有
    冻结基准，见 PORTING.md §7 的"绘图脚本"一节）。
    """
    from scipy.interpolate import griddata as _griddata

    points = np.column_stack([np.asarray(x).ravel(), np.asarray(y).ravel()])
    return _griddata(points, np.asarray(v).ravel(),
                     (np.asarray(xq), np.asarray(yq)), method="linear")


def importdata(name, delimiter=None):
    """MATLAB 的 `importdata`：这里只覆盖仓库里用到的两种情形。

    * 读 `.mat`（MATLAB 5 格式）：MATLAB 在这个用法下（`importdata('x.mat',' ')`）
      返回的是**矩阵**而不是结构体——本机实测：`rr_data.mat` 给 18×2、
      `f.mat` 给 75×2 的 double。所以这里：文件里**只有一个变量**时返回那个矩阵，
      多个变量时退化成字典（那种情形仓库里没用到）。
    * 读普通文本：按空白分隔的数值表返回。

    相对路径按 MATLAB 的"先当前目录、再搜索路径"找：`grass_ft/plot_grass_ft_f90.m`
    里的 `importdata('data/ps_t.dat')` 就是相对**脚本所在目录**的。
    """
    from scipy.io import loadmat

    path = data_path(name)
    if path.suffix == ".mat":
        data = {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
        if len(data) == 1:
            return next(iter(data.values()))
        return data
    return np.loadtxt(path)


def data_path(name) -> Path:
    r"""把一个（相对）数据文件名解析成磁盘上的路径。

    MATLAB 的规矩是"先当前目录、再搜索路径"。移植版沿搜索路径找，随论文提供的
    数据文件就放在脚本旁边（见 `mpath.original_dir`），所以
    `data/ps_t.dat`、`./RR/rr_datq_i09.mat`
    这类相对路径都能找到（绘图脚本大量这么写）。
    """
    from . import mpath

    path = Path(name)
    if path.is_file():
        return path
    for directory in mpath.DEFAULT_PATH.dirs:
        candidates = [directory / name]
        original = mpath.DEFAULT_PATH.original_dir(directory)
        if original is not None:
            candidates.append(original / name)
        hit = next((candidate for candidate in candidates if candidate.is_file()), None)
        if hit is not None:
            return hit
    return path


class _FileId:
    r"""MATLAB `fopen` 返回的文件句柄：只实现 `fscanf` 用到的那点东西。

    `figure7.m` 的读法是 `fscanf(fid,'%f',1)` 一个一个数地读 13×210 个数
    （按**空白**分词，跨行连着读），所以这里把整份文本切好、逐个吐出去。
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        text = path.read_text(encoding="utf-8", errors="replace")
        self._tokens = text.split()
        self._index = 0

    def next_token(self) -> str:
        if self._index >= len(self._tokens):
            raise ValueError(f"fscanf: {self.path} 已经读到结尾了")
        token = self._tokens[self._index]
        self._index += 1
        return token


def fopen(name, mode: str = "r"):
    """MATLAB 的 `fopen(name,mode)`：只支持读（`figure7.m` 就是读表）。"""
    path = data_path(name)
    if "r" not in str(mode):
        raise NotImplementedError(f"fopen: 移植版只实现了读模式（收到 {mode!r}）")
    if not path.is_file():
        raise FileNotFoundError(f"fopen: 找不到 {name}（解析成 {path}）")
    return _FileId(path)


def fscanf(fid, format_string: str, size=None):
    """MATLAB 的 `fscanf(fid,format,size)`：按格式串读 `size` 个数。

    只覆盖仓库里用到的 `%f`/`%d`/`%g`/`%e`/`%s`；`size` 为 1 时返回标量
    （与 `betas(i)=fscanf(fid,'%f',1)` 的用法一致），否则返回 1×n 的行向量。
    """
    conversion = str(format_string).replace("%", "").strip()[-1]
    count = None if size is None else int(np.asarray(size).ravel()[0])
    tokens = []
    while count is None or len(tokens) < count:
        if count is None and not fid._tokens[fid._index:]:
            break
        tokens.append(fid.next_token())
    if conversion == "s":
        values = np.array(tokens, dtype=object)
    else:
        values = np.array([float(token) for token in tokens], dtype=float)
    if len(tokens) == 1:
        return values[0] if conversion == "s" else float(values[0])
    return values.reshape(1, -1)


def fclose(fid) -> int:
    """MATLAB 的 `fclose(fid)`：这里无事可做（文件是整份读进来的）。"""
    return 0


def disp(text) -> None:
    """MATLAB 的 `disp(x)`：把内容打到命令窗口（GUI 脚本用它报告开始/结束）。"""
    print(str(text))


def cputime() -> float:
    """MATLAB 的 `cputime`：进程用掉的 CPU 秒数（`fun_frc_coil.m` 用它报耗时）。"""
    return time.process_time()
