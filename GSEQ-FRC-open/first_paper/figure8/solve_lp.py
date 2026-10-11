"""`solve_lp(P,psi,r)`：分界面处的**压强梯度标长** \(L_p=P/(\mathrm{d}P/\mathrm{d}r)\)。

**修改版新增**——原文仓库里没有这个文件，而 `gseq.m` 第 2 轮会调它
（`lp=solve_lp(P_m,psi_m,rr'); Lntemp=-(eta+1)*lp;`）。原文跑到这一行就中止，
所以 `Ln` 这条轨迹从来没被算出来过。

定义直接取自仓库里**同一个量**的既有算法：`plt_psi_r.m` 里

    N_rs=floor(rs/rw*nR);  ps=(Pr(N_rs)+Pr(N_rs+1))/2;
    dpdr=(Pr(N_rs+1)-Pr(N_rs))/(rr(N_rs)-rr(N_rs+1));  lp=ps/dpdr;

也就是"在分界面两侧取中点压强、用两侧差商当斜率的倒数"。这里唯一的差别是
分界面下标的定位方式：`plt_psi_r` 由 `rs` 反推下标，而 `solve_lp` 手上只有
`psi` 剖面，所以按 `solve_rs.m` 的同一条判据（从 ψ 的最小点往后第一次跨过 0）
定出那个内点。

返回标量。`psi` 在内部为负、分界面处为 0，所以 `P` 递减、`dpdr > 0`、
`lp > 0`。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat


def _crossing_index(psi) -> int:
    """从 ψ 的最小点往后找第一次跨过 0 的内点（1 基下标），与 `solve_rs.m` 同一判据。"""
    values = np.asarray(psi, dtype=float).reshape(-1, order="F")
    start = int(np.argmin(values))
    for k in range(start, values.size - 1):
        if values[k] <= 0.0 <= values[k + 1]:
            return k + 1                      # 1 基、取靠里的那一个点
    return values.size - 1


def solve_lp(P, psi, r):
    """[lp]=solve_lp(P,psi,r)"""
    n = mat.numel(psi)
    j = _crossing_index(psi)
    if j >= n:
        j = n - 1
    ps = (mat.el(P, j) + mat.el(P, j + 1)) / 2
    dpdr = (mat.el(P, j + 1) - mat.el(P, j)) / (mat.el(r, j) - mat.el(r, j + 1))
    return mat.scalar(ps / dpdr)
