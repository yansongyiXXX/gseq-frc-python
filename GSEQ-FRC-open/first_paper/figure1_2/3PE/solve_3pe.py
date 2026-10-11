"""solve_3pe.m 的逐行移植：三点压力剖面（3PE）参数 + 一维径向平衡。

这一份最长，也最能体现原程序的写法：

* 先用 `fsolve(@f3pe, x0)` 解 sigma，中间还把 `f3pe` 在 74 个点上扫一遍画图；
* 然后由 sigma 现算 `alpha`（同时写回全局）；
* `fP`/`fdP` 两个匿名函数都用 `(psi<0).*(...) + (psi>=0).*(...)` 选分支
  ——两个分支都会被求值，照抄；
* `fpsi` 的被积函数在上限处接近 \(1/\\sqrt{10^{-6}}\) 的尖峰（不是奇点），
  所以这一步的求积比 `fmrr0` 那类慢得多；
* 末尾写出 `3pe_datq.mat`。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mglobals
from gseqfrc import mio
from gseqfrc import moptim
from gseqfrc import mpath
from gseqfrc import mplot


def run(ws) -> None:
    mplot.close_all()  # close all;clear;clc;
    G = mglobals.G

    # global betas ls alpha pm tol alpha0 ls0 psim0 psig;
    G.declare("betas", "ls", "alpha", "pm", "tol", "alpha0", "ls0", "psim0",
              "psig")
    G.pm = mat.scalar(1)
    G.tol = mat.scalar(1e-8)

    G.betas = mat.scalar(0.6)
    # % betas=0.6655;
    G.alpha0 = 8 * np.log((np.sqrt(1 - G.betas) + 1) / np.sqrt(G.betas))
    G.ls0 = 1 / (G.alpha0 * np.sqrt(1 - G.betas))
    G.psim0 = np.log(G.betas) / G.alpha0
    # % ls=0.75;
    G.ls = mat.scalar(0.1)
    # % ls=ls0;

    G.psig = mat.scalar(-0.035)

    options = moptim.optimset(TolFun=G.tol, Tolx=G.tol, Maxfuneval=10000,
                              MaxIter=5000)

    x0 = mat.scalar(0.8)
    x = moptim.fsolve(lambda s: mpath.call("f3pe", s), x0, options)
    F = mpath.call("f3pe", x)
    sigma = np.real(x)

    if 1 == 1:
        aa = mat.colon(-5.0, 0.11, 3.1)
        ff = 0.0 * aa
        for ja in mat.mrange(1, mat.length(aa)):
            mat.set_el(ff, ja, mpath.call("f3pe", mat.el(aa, ja)))
        mplot.plot(aa, np.real(ff), aa, np.imag(ff), "--", "linewidth", 2)
        mplot.xlabel("\\sigma")
        mplot.ylabel("F_3(\\sigma)")
        mplot.legend("Re[F]", "Im[F]", "location", "best")
        mplot.legend("boxoff")
        mplot.title(f"\\beta_s={mat.num2str(G.betas)}, l_s={mat.num2str(G.ls)}, "
                    f"l_{{s0}}^{{RR}}={mat.num2str(G.ls0)}, pm={mat.num2str(G.pm)}")

    # if(1==0) 那一块没有被执行

    G.alpha = (
        (16 * sigma * G.ls * np.sqrt(1 - G.betas) + 7)
        + G.pm * np.sqrt(
            (16 * sigma * G.ls * np.sqrt(1 - G.betas) + 7) ** 2
            - 4 * sigma * 48 * G.ls * np.sqrt(1 - G.betas)
        )
    ) / (2 * 48 * G.ls * np.sqrt(1 - G.betas))
    alpha = G.alpha

    def F1(psim):
        return (
            G.betas - 8 * G.betas * alpha ** 2 / (7 * alpha - sigma) * (
                8 * psim * (1 + 4 * sigma * psim)
                + (sigma + alpha) * (1 - mat.exp(16 * alpha * psim)) / (8 * alpha ** 2)
            )
        ) - 1 + 1e-6

    # % F11=@(psim)(F1(psim)).^2;
    options2 = moptim.optimset(TolFun=G.tol, Tolx=G.tol, Display="off")
    psim = moptim.fsolve(F1, G.psig, options2)
    ans = F1(psim)  # 裸表达式 → ans

    # %%
    mplot.close_all()

    def fP(psi):
        # 分支标志必须逐元素——MATLAB 的 (psi<0).*(...) 是对整个数组做选择的，
        # 取成标量会让整条 psi<0 的区段都走错分支（这个 bug 让 Pr/Jt 差了 10~100%）。
        negative = np.asarray(psi) < 0
        non_negative = np.asarray(psi) >= 0
        with np.errstate(all="ignore"):
            return G.betas - 8 * G.betas * alpha ** 2 / (7 * alpha - sigma) * (
                negative * (
                    8 * psi * (1 + 4 * sigma * psi)
                    + (sigma + alpha) * ((1 - mat.exp(16 * alpha * psi)) / (8 * alpha ** 2))
                )
                + non_negative * (
                    (1 - mat.exp(-8 * alpha * psi)) / alpha
                    - (sigma + alpha) * (1 - mat.exp(-16 * alpha * psi)) / (8 * alpha ** 2)
                )
            )

    def fdP(psi):
        negative = np.asarray(psi) < 0
        non_negative = np.asarray(psi) >= 0
        with np.errstate(all="ignore"):
            return -8 * G.betas * alpha ** 2 / (7 * alpha - sigma) * (
                negative * (
                    8 * (1 + 8 * sigma * psi)
                    + (sigma + alpha) * (-2 * mat.exp(16 * alpha * psi)) / alpha
                )
                + non_negative * (
                    8 * mat.exp(-8 * alpha * psi)
                    + (sigma + alpha) * (-2 * mat.exp(-16 * alpha * psi)) / alpha
                )
            )

    def fpsi(psi_):
        return 1.0 / np.sqrt(1.0 - fP(psi_) + 0e-4)

    dr = mat.scalar(0.001)
    rmax = mat.scalar(1.5)
    rr = mat.colon(0, dr, rmax)
    nr = mat.length(rr)
    psi = 0.0 * rr

    for jr in mat.mrange(2, nr):
        r = mat.el(rr, jr)
        if r <= 1 / np.sqrt(2):
            def FF1(psi_):
                return (moptim.integral(fpsi, 0, psi_, float(G.tol), float(G.tol))
                        + r ** 2 / 2)

            mat.set_el(psi, jr, moptim.fsolve(FF1, np.real(mat.el(psi, jr - 1)),
                                              options2))
        else:
            def FF2(psi_):
                return (moptim.integral(fpsi, psim, psi_, float(G.tol), float(G.tol))
                        - (r ** 2 - 1 / 2) / 2)

            mat.set_el(psi, jr, moptim.fsolve(FF2, 0.99 * np.real(mat.el(psi, jr - 1)),
                                              options2))

    # %%
    mplot.close_all()

    mplot.figure("unit", "normalized", "DefaultAxesFontSize", 16,
                 "DefaultAxesFontWeight", "bold", "DefaultAxesLineWidth", 2,
                 "position", mat.row(0.02, 0.2, 0.5, 0.6))

    # to update r>rs
    Pr = fP(psi)

    dP = fdP(psi)
    Jt = -rr * dP
    Bz = (-np.sqrt(1 - Pr) * (rr <= 1 / np.sqrt(2))
          + np.sqrt(1 - Pr) * (rr > 1 / np.sqrt(2)))

    ids = mat.el(mat.find(np.abs(rr - 1) == np.min(np.abs(rr - 1))), 1)
    lsp = mat.scalar(np.real(
        -(mat.el(Pr, ids) / ((mat.el(Pr, ids + 1) - mat.el(Pr, ids - 1))
                             / (mat.el(rr, ids + 1) - mat.el(rr, ids - 1))))
    ))

    ind = mat.find(rr <= 1.0)
    h = np.real(fdP(psim) / np.sum(2 * mat.take(dP, ind) * mat.take(rr, ind) * dr))

    mplot.subplot(221)
    mplot.plot(rr, psi, "linewidth", 2)
    mplot.xlabel("r")
    mplot.ylabel("\\psi")
    mplot.title(f"\\beta_s={mat.num2str(G.betas)}, l_s={mat.num2str(G.ls)}, "
                f"l_{{sp}}={mat.num2str(lsp, 3)}")
    mplot.subplot(222)
    mplot.plot(rr, Pr, "linewidth", 2)
    mplot.xlabel("r")
    mplot.ylabel("P")
    mplot.title(f"l_{{s0}}={mat.num2str(G.ls0)}, \\alpha={mat.num2str(alpha)}, "
                f"pm={mat.num2str(G.pm)}")
    mplot.subplot(223)
    mplot.plot(rr, Jt, "linewidth", 2)
    mplot.xlabel("r")
    mplot.ylabel("J_\\theta")
    mplot.title(f"\\sigma={mat.num2str(sigma)}, h={mat.num2str(h)}, "
                f"\\psi_g={mat.num2str(G.psig)}")
    mplot.subplot(224)
    mplot.plot(rr, Bz, "linewidth", 2)
    mplot.xlabel("r")
    mplot.ylabel("B_z")
    mplot.title(f"\\psi_m={mat.num2str(psim)}, \\alpha_0={mat.num2str(G.alpha0)}, "
                f"x_0={mat.num2str(x0)}")

    mplot.set_(mplot.gcf(), "Units", "inches")
    screenposition = mplot.get_(mplot.gcf(), "Position")
    mplot.set_(mplot.gcf(), "PaperPosition",
               mat.row(0, 0, screenposition[2], screenposition[3]),
               "PaperSize", mat.row(screenposition[2], screenposition[3]))
    mplot.print_fig(
        mplot.gcf(), "-dpng",
        f"3PE_betas={mat.num2str(G.betas)}_ls={mat.num2str(G.ls)}"
        f"_pm={mat.num2str(G.pm)}_alpha={mat.num2str(alpha)}"
        f"_tol={mat.num2str(G.tol)}.png",
    )

    # %% if(1==0) 那一块没有被执行

    datq_3pe = np.vstack([rr, Pr, psi, Bz, Jt]).T
    mio.save("3pe_datq.mat", datq_3pe=datq_3pe, betas=G.betas, ls=G.ls)

    for name in ("betas", "ls", "alpha", "pm", "tol", "alpha0", "ls0", "psim0",
                 "psig"):
        ws[name] = getattr(G, name)
    ws.options = options
    ws.options2 = options2
    ws.x0 = x0
    ws.x = x
    ws.F = mat.scalar(F)
    ws.sigma = sigma
    ws.psim = psim
    ws.ans = ans
    ws.aa = aa
    ws.ff = ff
    ws.ja = mat.scalar(ja)
    ws.dr = dr
    ws.rmax = rmax
    ws.rr = rr
    ws.nr = mat.scalar(nr)
    ws.psi = psi
    ws.r = mat.scalar(r)
    ws.jr = mat.scalar(jr)
    ws.Pr = Pr
    ws.dP = dP
    ws.Jt = Jt
    ws.Bz = Bz
    ws.ids = mat.scalar(ids)
    ws.lsp = lsp
    ws.ind = ind
    ws.h = h
    ws.screenposition = mat.row(*screenposition)
    ws.datq_3pe = datq_3pe
