"""fmrr2.m 的逐行移植（n>1 的 MRR2 参数方程）。

源文件：gseq-main/GSEQ-FRC-open/first_paper/figure1_2/MRR2/fmrr2.m

这一份把 `(sigma*psim+1)^n` 换成了 `exp(-sigma*psim^n)` 的形式：

    F1 = exp(-alpha*psim - sigma*psim^n) - 1/betas

验收判据同样是相对容差而不是 ULP，理由见 PORTING.md §5。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mglobals
from gseqfrc import moptim


def fmrr2(x):
    """F=fmrr2(x)：MRR2 参数方程在 sigma=x 处的值。"""
    G = mglobals.G
    # global betas ls alpha n tol alpha0 ls0 psim0 psig;
    G.declare("betas", "ls", "alpha", "n", "tol", "alpha0", "ls0", "psim0",
              "psig")

    sigma = x

    alpha = G.alpha

    def F1(psim):
        return mat.exp(-alpha * psim - sigma * psim ** G.n) - 1 / G.betas

    # options2 = optimset('TolFun',tol,'Tolx',tol,'Display','off');
    psim = moptim.fsolve(F1, G.psig, tol_fun=G.tol, tol_x=G.tol)
    # psim=fzero(F1,-0.005,options2);

    psim = np.real(psim)

    def fpsi(psi):
        # MATLAB 的 sqrt 对负数给复数（numpy 给 nan），见 matlab_compat.sqrt
        return 1.0 / mat.sqrt(
            1.0 - G.betas * mat.exp(-alpha * psi - sigma * psi ** G.n)
        )

    F = moptim.integral(fpsi, 0, psim, G.tol, G.tol) + 1 / 4

    # F=real(F);  %%
    return F
