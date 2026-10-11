r"""f3pe.m 的逐行移植（三点压力剖面 3PE 的参数方程）。

源文件：gseq-main/GSEQ-FRC-open/first_paper/figure1_2/3PE/f3pe.m

三处需要照抄的原程序写法：

1. `alpha` 由 `sigma` 现算，并且**写回全局** `alpha`（所以 `u_f3pe` 的
   快照里 `alpha` 是最后一次调用留下的值）。
2. `fP` 的两个分支 `(psi<0).*(...) + (psi>=0).*(...)` 是"两个分支都算一遍、
   再乘 0/1 选一个"—— MATLAB 的写法，不能改写成 if/else，否则数值上不等价。
3. `F1` 用的是 `fP(psim)-1+1e-6`，比 1 小了一个 `1e-6`，所以积分上限处的被积
   函数是 \(1/\\sqrt{10^{-6}}\) 而不是无穷——**这一条没有端点奇异性**，
   和 `fmrr0/1/2` 不同。

验收判据是相对容差，理由见 PORTING.md §5。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mglobals
from gseqfrc import moptim


def f3pe(x):
    """F=f3pe(x)：3PE 参数方程在 sigma=x 处的值。"""
    G = mglobals.G
    # global betas ls alpha pm tol alpha0 ls0 psim0 psig;
    G.declare("betas", "ls", "alpha", "pm", "tol", "alpha0", "ls0", "psim0",
              "psig")

    sigma = x

    # % pm=1;
    G.alpha = (
        (16 * sigma * G.ls * np.sqrt(1 - G.betas) + 7)
        + G.pm * np.sqrt(
            (16 * sigma * G.ls * np.sqrt(1 - G.betas) + 7) ** 2
            - 4 * sigma * 48 * G.ls * np.sqrt(1 - G.betas)
        )
    ) / (2 * 48 * G.ls * np.sqrt(1 - G.betas))

    alpha = G.alpha

    def fP(psi):
        # 逐元素选择分支（MATLAB 的 (psi<0).*(...) 语义）
        negative = np.asarray(psi) < 0
        non_negative = np.asarray(psi) >= 0
        # MATLAB 对溢出不发警告，`0*Inf` 同样是安安静静地变成 NaN，这里照抄
        with np.errstate(all="ignore"):
            return G.betas - 8 * G.betas * alpha ** 2 / (7 * alpha - sigma) * (
                negative * (
                    8 * psi * (1 + 4 * sigma * psi)
                    + (sigma + alpha)
                    * ((1 - mat.exp(16 * alpha * psi)) / (8 * alpha ** 2))
                )
                + non_negative * (
                    (1 - mat.exp(-8 * alpha * psi)) / alpha
                    - (sigma + alpha)
                    * (1 - mat.exp(-16 * alpha * psi)) / (8 * alpha ** 2)
                )
            )

    def F1(psim):
        return fP(psim) - 1 + 1e-6  # 20-08-01 21:25

    # F11=@(psim)(F1(psim)).^2;
    # options2 = optimset('TolFun',tol,'Tolx',tol,'Display','off');
    psim = moptim.fsolve(F1, G.psig, tol_fun=G.tol, tol_x=G.tol)
    # psim=fzero(F1,psig,options2);
    # psim=fminsearch(F11,psig,options2);

    # % psim=real(psim);   —— 这一份没有取实部

    def fpsi(psi):
        # MATLAB 的 sqrt 对负数给复数、numpy 给 nan——用 mat.sqrt 保持原行为
        return 1.0 / mat.sqrt(1.0 - fP(psi))

    F = moptim.integral(fpsi, 0, psim, G.tol, G.tol) + 1 / 4

    # F=real(F);  %%
    return F
