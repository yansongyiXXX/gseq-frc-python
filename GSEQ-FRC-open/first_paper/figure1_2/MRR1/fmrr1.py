"""fmrr1.m 的逐行移植（q>1 的 MRR1 参数方程）。

源文件：gseq-main/GSEQ-FRC-open/first_paper/figure1_2/MRR1/fmrr1.m

与 `fmrr0` 的差别只有两处幂次：方程与积分里的 `(sigma*psim+1)` 变成
`(sigma*psim^q+1)`；`alpha` 不再由 `ls` 现算，而是直接用全局里的 `alpha`。
验收判据同样是相对容差而不是 ULP，理由见 PORTING.md §5。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mglobals
from gseqfrc import moptim


def fmrr1(x):
    """F=fmrr1(x)：MRR1 参数方程在 sigma=x 处的值。"""
    G = mglobals.G
    # global betas ls alpha n q tol alpha0 ls0 psim0 psig;
    G.declare("betas", "ls", "alpha", "n", "q", "tol", "alpha0", "ls0",
              "psim0", "psig")

    sigma = x

    alpha = G.alpha

    def F1(psim):
        return mat.exp(-alpha * psim) * (sigma * psim ** G.q + 1) ** G.n - 1 / G.betas

    # options2 = optimset('TolFun',tol,'Tolx',tol,'Display','off');
    psim = moptim.fsolve(F1, G.psig, tol_fun=G.tol, tol_x=G.tol)
    # psim=fzero(F1,-0.005,options2);

    psim = np.real(psim)

    def fpsi(psi):
        return 1.0 / np.sqrt(
            1.0 - G.betas * mat.exp(-alpha * psi) * (sigma * psi ** G.q + 1) ** G.n
        )

    F = moptim.integral(fpsi, 0, psim, G.tol, G.tol) + 1 / 4

    # F=real(F);  %%
    return F
