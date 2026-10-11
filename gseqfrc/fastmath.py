"""numba 加速内核——**逐位**与 numpy 版相同。

## 为什么

移植版为了跟 MATLAB R2023a 逐位对齐，`exp` 照抄了 FDLIBM 的 `__ieee754_exp`
（见 `matlab_compat.exp` 的长注释）。纯 numpy 的向量化写法算法没错，但一条表达式
一个临时数组，40 多个 1 MB 的中间量来回搬：131841 个点上实测 **21 ms**。
同样的算术放进标量循环只要 **0.57 ms（37×）**——瓶颈是分配不是算法
（`x*x` 分配 0.252 ms，`np.multiply(x,x,out=buf)` 只要 0.049 ms）。

二维定边界求解器的内层一轮里，`exp` 占 62%，其余 numpy 表达式（`dP`/`P`/`Jzeta`/
`rhs`/三个电流积分）又占 30% 强，且都是同一类开销，所以这里把整个内层循环也搬进
numba。

## 正确性

* numba 用 `fastmath=False` 编译：不开 FMA 收缩、不重排浮点运算；整数位运算本来
  就精确。所以与 numpy 版、进而与 MATLAB 逐位相同。
* `tools/exp_numba_probe.py` 逐位盯这件事（含次正规数、±0、极值、NaN/Inf、
  以及真实算例上的整片数组）。
* `exp` 之外的内核（`sources` / `rhs_relax` / `dpsim_copy`）保持**运算顺序、
  累加次序、循环次序**与镜像文件里那段 numpy 代码一一对应，注释逐条标了。

numba 不可用时 `exp_array` 返回 `None`、各内核不被调用，调用方自动退回 numpy 路径。
"""

from __future__ import annotations

import math

import numpy as np

try:  # pragma: no cover - 环境相关
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # noqa: BLE001 pragma: no cover
    NUMBA_AVAILABLE = False


# --- FDLIBM e_exp.c 的常数（与 matlab_compat 里那几个一一对应） ----------------
_LN2HI0 = 6.93147180369123816490e-01
_LN2LO0 = 1.90821492927058770002e-10
_INVLN2 = 1.44269504088896338700e00
_P1 = 1.66666666666666019037e-01
_P2 = -2.77777777770155933842e-03
_P3 = 6.61375632143793436117e-05
_P4 = -1.65339022054652515390e-06
_P5 = 4.13813679705723846039e-08
_HUGE = 1.0e300
_TWOM1000 = 9.33263618503218878990e-302
_O_THRESHOLD = 7.09782712893383973096e02
_U_THRESHOLD = -7.45133219101941108420e+02


if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=False)
    def _exp_c(x, box, box_u):
        """FDLIBM `__ieee754_exp` 的单点版；分支与 C 源码一一对应。

        `box`/`box_u` 是一块 1×1 float64 缓冲和它的 uint64 视图，用来做
        "浮点 ↔ 位模式"转换（numba 里没有 struct；位运算本身是精确的）。
        """
        box[0] = x
        b = box_u[0]
        high = np.int64(b >> np.uint64(32))
        xsb = (high >> 31) & 1
        hx = high & 0x7FFFFFFF
        if hx >= 0x40862E42:                    # |x| 很大
            if hx >= 0x7FF00000:
                if ((high & 0xFFFFF) != 0) or ((b & np.uint64(0xFFFFFFFF)) != 0):
                    return x + x                # NaN
                return x if xsb == 0 else 0.0    # ±Inf
            if x > _O_THRESHOLD:
                return _HUGE * _HUGE             # 上溢
            if x < _U_THRESHOLD:
                return _TWOM1000 * _TWOM1000     # 下溢

        k = 0
        hi = 0.0
        lo = 0.0
        if hx > 0x3FD62E42:                     # |x| > 0.5*ln2
            if hx < 0x3FF0A2B2:                 # |x| < 1.5*ln2
                sgn = 1.0 if xsb == 0 else -1.0
                hi = x - sgn * _LN2HI0
                lo = sgn * _LN2LO0
                k = 1 - xsb - xsb
            else:
                half = 0.5 if xsb == 0 else -0.5
                k = np.int64(np.trunc(_INVLN2 * x + half))
                t = np.float64(k)
                hi = x - t * _LN2HI0
                lo = t * _LN2LO0
            xx = hi - lo
        elif hx < 0x3E300000:                   # |x| < 2**-28
            if _HUGE + x > 1.0:
                return 1.0 + x
            xx = x
        else:
            xx = x

        tt = xx * xx
        c = xx - tt * (_P1 + tt * (_P2 + tt * (_P3 + tt * (_P4 + tt * _P5))))
        if k == 0:
            return 1.0 - ((xx * c) / (c - 2.0) - xx)
        y = 1.0 - ((lo - (xx * c) / (2.0 - c)) - hi)
        # FDLIBM 的 SET_HIGH_WORD：把 k 加到指数位上。等价于乘 2^k，但逐位与
        # 原码一致（包括 k < -1021 时先加 1000 再乘 2^-1000 那一段）。
        box[0] = y
        kk = k if k >= -1021 else k + 1000
        box_u[0] = np.uint64(np.int64(box_u[0]) + (kk << 52))
        y = box[0]
        if k < -1021:
            y = y * _TWOM1000
        return y

    @njit(cache=True, fastmath=False)
    def _exp_loop(a, out, box, box_u):
        for i in range(a.shape[0]):
            out[i] = _exp_c(a[i], box, box_u)

    @njit(cache=True, fastmath=False)
    def sources(psi, R, jrb, jz0, alpha, BeRs2, sigma, nexp, nn, k1, k2, base,
                twopi, dR, dZ, dP, Jzeta, P, box, box_u):
        """一轮的 `dP`/`Jzeta`/`P` 与三个电流积分，一次扫描算完。

        与镜像文件里那段 numpy 逐条对应：

        * `x = -alpha*psi/(Be*Rs^2)`、`ex = exp(x)`、`u = sigma*psi/(Be*Rs^2)+1`
        * `dP = C*Pm0*betas * ex * u^(n-1) * (-alpha/(Be*Rs^2)*u + n*sigma/(Be*Rs^2))`
        * `Jzeta = -R*dP`、`P = C*Pm0*betas * ex * u^n`
        * 三个累加（`Jztotal`/`Jztotal1`/`Jztotal2`）按原文的
          `for jz → for jr=1:(jrb(jz)-1)` 次序**逐个相加**，与 cumsum 的末项一致。

        注意 `math.pow`：镜像里 `n` 是从 `.mat` 读出来的 **1×1 数组**，
        `uu ** n` 走的是 numpy 的"数组指数"通路，直接调 libm 的 `pow`；
        而 numpy 的"标量指数"通路会把 `y=2.0` 特判成 `x*x`（`y=0.5` 特判成
        `sqrt`）。两条通路在约 0.05% 的点上差 1 ULP，所以这里必须用
        `math.pow`（而且指数是运行期参数，numba 才不会把它折叠掉）。
        `tools/gfix_loopbench.py` 里量过：这一步差一点，`P` 就会有几十个元素漂。
        """
        nR = psi.shape[0]
        nZ = psi.shape[1]
        jzt = 0.0
        jzt1 = 0.0
        jzt2 = 0.0
        for j in range(nZ):
            ncol = jrb[j] - 1                  # 原文 jr=1:(jrb(jz)-1)
            for i in range(nR):
                p = psi[i, j]
                ex = _exp_c((-alpha * p) / BeRs2, box, box_u)
                u = (sigma * p) / BeRs2 + 1.0
                # nexp==1.0 时 libm 的 pow(u,1.0) 就是 u（C99 要求），省一次调用；
                # 本算例 n=2，走的正是这一支。nn 那一次不能省（pow(u,2.0) != u*u）。
                if nexp == 1.0:
                    dp = base * ex * u * (k1 * u + k2)
                else:
                    dp = base * ex * math.pow(u, nexp) * (k1 * u + k2)
                jz = -R[i, j] * dp
                dP[i, j] = dp
                Jzeta[i, j] = jz
                P[i, j] = base * ex * math.pow(u, nn)
                if i < ncol:
                    if j == jz0:
                        jzt = jzt + twopi * jz * R[i, j] * dR
                    jzt1 = jzt1 + jz * dR * dZ
                    jzt2 = jzt2 + twopi * jz * R[i, j] * dR * dZ
        return jzt, jzt1, jzt2

    @njit(cache=True, fastmath=False)
    def rhs_relax(psi, psi0, dP, rr, jrb, mu0, dR, dZ, dR2, dZ2, w, w1, rhs):
        """`rhs` 段（ifd==2）+ 松弛更新，一次扫描算完。

        与 numpy 版的运算次序一致：`coef * (((a-b)+c)+d)`、`w*psi0 + rhs*(1-w)`。
        """
        nZ = psi.shape[1]
        coef = 1.0 / (2.0 / dR2 + 2.0 / dZ2)
        for j in range(1, nZ - 1):
            jend = jrb[j] - 1                  # 原文 i=2:(jrb(j)-1)
            for i in range(1, jend):
                r_i = rr[i]
                val = coef * (
                    (psi0[i + 1, j] + psi0[i - 1, j]) / dR2
                    - (psi0[i + 1, j] - psi0[i - 1, j]) / (2.0 * dR * r_i)
                    + (psi0[i, j + 1] + psi0[i, j - 1]) / dZ2
                    # 注意括号：numpy 那边是 `mu0 * rr**2 * dP`，先算 rr*rr 再乘 mu0
                    + mu0 * (r_i * r_i) * dP[i, j])
                rhs[i, j] = val
                psi[i, j] = w * psi0[i, j] + val * w1

    @njit(cache=True, fastmath=False)
    def dpsim_copy(psi, psi0):
        """`dpsim = max(max(abs(psi-psi0)))` 且 `psi0 = psi`，一趟走完。

        逐元素取最大值；NaN 一旦出现就保持 NaN（和 MATLAB 的 `max` 一样）。
        """
        m = 0.0
        for j in range(psi.shape[1]):
            for i in range(psi.shape[0]):
                d = abs(psi[i, j] - psi0[i, j])
                if d != d:
                    m = d
                elif d > m:
                    m = d
                psi0[i, j] = psi[i, j]
        return m


#: 只保留一块 1×1 的"浮点/位模式"转换缓冲（numba 里没有 struct）
_BOX: np.ndarray | None = None


def box() -> np.ndarray:
    """给 numba 内核用的 1×1 位转换缓冲。"""
    global _BOX
    if _BOX is None:
        _BOX = np.zeros(1, dtype=np.float64)
    return _BOX


def exp_array(value) -> np.ndarray | None:
    """与 `matlab_compat._exp_array` 逐位相同的 `exp`；numba 不可用时返回 None。"""
    if not NUMBA_AVAILABLE:
        return None
    a = np.ascontiguousarray(value, dtype=np.float64)
    flat = a.reshape(-1)
    out = np.empty_like(flat)
    one = box()
    _exp_loop(flat, out, one, one.view(np.uint64))
    return out.reshape(a.shape)
