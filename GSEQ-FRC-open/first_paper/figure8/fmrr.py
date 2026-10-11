"""fmrr.m 的移植：由 (β_s, r_s, L_p) 反解 MRR 参数 (α, σ, ψ_min) 的方程组。

源文件：`gseq-main/GSEQ-FRC-open/first_paper/figure8/fmrr.m`

    F(1) = (n·σ − α)·sqrt(1−β_s) + r_s/L_p
    F(2) = exp(−α·ψ_min)·(σ·ψ_min+1)^n − 1/β_s
    F(3) = −∫₀^{ψ_min} 1/sqrt(1−β_s·exp(−αψ)·(σψ+1)^n) dψ − 1/4

⚠️ 这是个**函数**（不是脚本），参数走 MATLAB 的 `global betas rs Lp`——仓库里
只有 `MRR1/solve_MRR1.m`、`MRR2/solve_MRR2.m` 里**注释掉的行**提到它
（`ff(ja)=fmrr(aa(ja));`），没有任何冻结算例覆盖它。所以它的靶子只能是
"手工设定全局量之后与原版 MATLAB 逐点比 F 值"（见测试里的
`FMRR_REFERENCE`）。积分用的是 MATLAB `integral` 的**默认容差**
（`RelTol=1e-6`、`AbsTol=1e-10`）——与 §5 里"容差受限的核函数"是同一类情形。
`moptim.integral` 是同一套 quadva 算法的逐行移植，所以两边**逐位相同**；
测试里按 \(10^{-9}\) 相对容差盯住（比算法自身的容差紧得多，但留了一点余量）。

还有一处 MATLAB/numpy 的语义差要照抄：被积函数在区间内可能取到
\(1-\beta_s e^{-\alpha\psi}(\sigma\psi+1)^n<0\)（本机实测 `x=[10,0.5,-0.1]`
就是），MATLAB 的 `sqrt(负数)` 给**复数**、`integral` 也照复数积分
（实测 `-0.166018007674+0.134312725959i`），而 numpy 的 `np.sqrt` 给 NaN。
所以这里用 `np.emath.sqrt`（非负实数给实数、负数给复数，与 MATLAB 的 `sqrt` 一致）。

验收：§7「绘图/GUI 脚本」那一节的口径 + 上面这条数值对照。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mglobals
from gseqfrc import moptim


def fmrr(x):
    """F=fmrr(x)：`x = [alpha, sigma, psim]`，返回 3×1（与 MATLAB 同形）。"""
    G = mglobals.G
    # global betas rs Lp;
    G.declare("betas", "rs", "Lp")

    x = np.asarray(x, dtype=float).reshape(-1)
    alpha = x[0]
    sigma = x[1]
    psim = x[2]

    n = 2

    F1 = (n * sigma - alpha) * np.sqrt(1 - G.betas) + G.rs / G.Lp
    F2 = mat.exp(-alpha * psim) * (sigma * psim + 1) ** n - 1 / G.betas

    def fpsi(psi):
        # `np.emath.sqrt`：非负实数给实数、负数给复数 —— 与 MATLAB 的 sqrt 一致
        return 1.0 / np.emath.sqrt(
            1 - G.betas * mat.exp(-alpha * psi) * (sigma * psi + 1) ** n
        )

    # 原文 `integral(fpsi,0,psim+1e-10)`：不给容差 → 用 MATLAB 的默认值
    F3 = -moptim.integral(fpsi, 0, psim + 1e-10) - 1 / 4

    # MATLAB 里 `F=0.*x; F(3)=<复数>` 会把**整个数组提升为复数**；numpy 往 float
    # 数组里写复数只会截掉虚部（还只给个告警）。这里用同一套语义：全实数时是
    # float、F3 是复数时整个数组提升为 complex。
    F = np.array([F1, F2, F3])
    return F.reshape(-1, 1)      # MATLAB 的 3×1 列向量
