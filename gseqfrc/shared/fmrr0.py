"""fmrr0.m 的逐行移植（**四份副本合并后**的唯一实现）。

源文件（四份副本逐字节相同，SHA256 一致）：

    gseq-main/GSEQ-FRC-open/second_paper/Fig4/fmrr0.m
    gseq-main/GSEQ-FRC-open/second_paper/Fig5/fmrr0.m
    gseq-main/GSEQ-FRC-open/second_paper/Fig6/fmrr0.m
    gseq-main/GSEQ-FRC-open/second_paper/Fig7/Bz_free_072/fmrr0.m

2026-09-17 副本合并：实现搬到 `gseqfrc/shared/`，程序树里那四个原位置改成薄壳
（内容只是 `from gseqfrc.shared.fmrr0 import fmrr0`），保证 `mpath` 按路径解析的
行为与合并前一致。合并前先核对过四份内容逐字节相同，合并是纯去重。

MRR 参数方程：由 (betas, rs, Lp) 定出 (alpha, sigma, psim)。参数全部走全局变量
（见 `mglobals`）：先用 `fsolve` 解出根的 psim，再沿磁通积分。

**这个函数的冻结值只精确到原程序自己给的容差**（`u_fmrr0` 里 tol=1e-10）：
用独立的高精度方法算出的真值与冻结值相差 \(1.8\times10^{-10}\)。所以验收判据是
相对容差而不是 ULP，理由写在 PORTING.md §5。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mglobals
from gseqfrc import moptim


def fmrr0(x):
    """F=fmrr0(x)：MRR 参数方程在 sigma=x 处的值。"""
    G = mglobals.G
    # global betas ls n tol alpha0 ls0 psim0 psig;
    G.declare("betas", "ls", "n", "tol", "alpha0", "ls0", "psim0", "psig")

    sigma = x

    # % n=2;
    alpha = ((1 / G.ls) / np.sqrt(1 - G.betas) + G.n * sigma)

    def F1(psim):
        return mat.exp(-alpha * psim) * (sigma * psim + 1) ** G.n - 1 / G.betas

    # F1=@(psim)betas*exp(-alpha*psim).*(sigma*psim+1).^n-1-1e-6;
    # options2 = optimset('TolFun',tol,'Tolx',tol,'Display','off');
    psim = moptim.fsolve(F1, G.psig, tol_fun=G.tol, tol_x=G.tol)
    # psim=fzero(F1,-0.005,options2);

    psim = np.real(psim)

    def fpsi(psi):
        return 1.0 / np.sqrt(
            1.0 - G.betas * mat.exp(-alpha * psi) * (sigma * psi + 1) ** G.n
        )

    F = moptim.integral(fpsi, 0, psim, G.tol, G.tol) + 1 / 4

    # F=real(F);  %%
    return F
