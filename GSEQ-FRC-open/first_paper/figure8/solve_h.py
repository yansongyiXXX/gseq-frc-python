"""`solve_h(jr,r,psi)`：分界面处的**电流密度梯度标长** \(h=j_\zeta/(\mathrm{d}j_\zeta/\mathrm{d}r)\)。

**修改版新增**——原文仓库里同样没有这个文件，而 `gseq.m` 第 2 轮会调它
（`htemp=solve_h(jr_m,rr,psi_m); fprintf('h =%5.4f',htemp);`）。

⚠️ 与 `solve_lp` 不同，这个量在仓库里**找不到任何定义**：没有别的脚本算它，
`gseq.m` 里唯一消费它的两行（`C2=wc*C2+(1-wc)*C2tmp*htemp/h;`）也是注释掉的。
所以这里按与 `solve_lp` 完全对称的写法补一版（同样的分界面下标、同样的中点 +
差商），保证脚本能按作者的迭代结构跑下去；它目前只影响打印值和 `psiq1.mat`
里存的 `htemps`，不参与任何求解。

返回标量。
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


def solve_h(jr, r, psi):
    """[h]=solve_h(jr,r,psi)"""
    n = mat.numel(psi)
    j = _crossing_index(psi)
    if j >= n:
        j = n - 1
    js = (mat.el(jr, j) + mat.el(jr, j + 1)) / 2
    djdr = (mat.el(jr, j + 1) - mat.el(jr, j)) / (mat.el(r, j) - mat.el(r, j + 1))
    return mat.scalar(js / djdr)
