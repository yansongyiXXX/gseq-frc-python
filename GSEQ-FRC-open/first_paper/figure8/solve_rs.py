"""solve_rs.m 的逐行移植：从 psi(r) 剖面里插值出分界面半径 rs（psi 过零处）。

这是五个二维定边界求解器唯一依赖的仓库内函数（其他依赖都是 MATLAB 内建），
而且它只有四行——用 `find` 定位 psi 的最小值，再从那里往后用 `interp1` 的
**线性**模式找 psi=0 的位置。注意区间取到 `end-1`（原程序如此），照抄。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat


def solve_rs(psi, r):
    """[rs]=solve_rs(psi,r)"""
    ind_min = mat.find(np.asarray(psi) == np.min(psi))
    start = int(mat.el(ind_min, 1))
    stop = mat.length(psi) - 1
    return mat.scalar(mat.interp1(
        mat.take(psi, mat.mrange(start, stop)),
        mat.take(r, mat.mrange(start, stop)),
        0,
        "linear",
    ))
