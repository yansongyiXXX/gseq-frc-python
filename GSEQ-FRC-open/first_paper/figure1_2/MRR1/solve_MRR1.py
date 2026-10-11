"""solve_MRR1.m 的逐行移植：MRR1（q=1）参数 + 一维径向平衡。

先用 `fsolve(@fmrr1,...)` 从 sigma 里解出参数，再解 psim，然后沿 r 逐点解一维平衡，
最后画四张图（这一份的 `print` 被注释掉了，所以 s_mrr1 不产出图片）。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mglobals
from gseqfrc import moptim
from gseqfrc import mpath
from gseqfrc import mplot


def run(ws) -> None:
    mplot.close_all()  # close all;clear;clc;
    G = mglobals.G

    # global betas ls alpha n q tol alpha0 ls0 psim0 psig;
    G.declare("betas", "ls", "alpha", "n", "q", "tol", "alpha0", "ls0",
              "psim0", "psig")
    G.n = mat.scalar(2)
    G.q = mat.scalar(1)
    G.tol = mat.scalar(1e-8)

    G.betas = mat.scalar(0.6)
    # % betas=0.6655;
    G.alpha0 = 8 * np.log((np.sqrt(1 - G.betas) + 1) / np.sqrt(G.betas))
    G.ls0 = 1 / (G.alpha0 * np.sqrt(1 - G.betas))
    G.psim0 = np.log(G.betas) / G.alpha0
    # % ls=0.55;
    # % ls=0.148;
    G.ls = G.ls0 * 0.4

    G.alpha = 1 / (G.ls * np.sqrt(1 - G.betas))

    # % psig=-0.056;
    G.psig = mat.scalar(-0.054)

    options = moptim.optimset(TolFun=G.tol, Tolx=G.tol, Maxfuneval=10000,
                              MaxIter=5000)

    x0 = mat.scalar(2.4)
    # % x0=5.2;
    x = moptim.fsolve(lambda s: mpath.call("fmrr1", s), x0, options)
    F = mat.scalar(mpath.call("fmrr1", x))
    sigma = np.real(x)

    def F1(psim):
        return (mat.exp(-G.alpha * psim) * (sigma * psim ** G.q + 1) ** G.n
                - 1 / G.betas)

    # % options2 = optimset('TolFun',1e-8,'Tolx',1e-8,'Display','iter');
    options2 = moptim.optimset(TolFun=G.tol, Tolx=G.tol, Display="off")
    psim = moptim.fsolve(F1, G.psig, options2)
    # % psim=fzero(F1,-0.005,options2);
    ans = F1(psim)  # 原脚本里这一行是裸表达式，结果落在 ans 里

    # % aa=-10:0.1:50;ff=0.*aa;

    # %%
    mplot.close_all()

    def fpsi(psi_):
        return 1.0 / np.sqrt(1.0 - G.betas * mat.exp(-G.alpha * psi_)
                             * (sigma * psi_ ** G.q + 1) ** G.n)

    dr = mat.scalar(0.02)
    rmax = mat.scalar(3.0 / 1)
    rr = mat.colon(0, dr, rmax)
    nr = mat.length(rr)
    psi = 0.0 * rr

    for jr in mat.mrange(2, nr):
        r = mat.el(rr, jr)
        if r <= 1 / np.sqrt(2):
            def FF1(psi_):
                return moptim.integral(fpsi, 0, psi_, 1e-8, 1e-8) + r ** 2 / 2

            mat.set_el(psi, jr, moptim.fsolve(FF1, np.real(mat.el(psi, jr - 1)),
                                              options2))
        else:
            def FF2(psi_):
                return (moptim.integral(fpsi, psim, psi_, 1e-8, 1e-8)
                        - (r ** 2 - 1 / 2) / 2)

            mat.set_el(psi, jr, moptim.fsolve(FF2, 0.99 * np.real(mat.el(psi, jr - 1)),
                                              options2))

    # %%
    mplot.close_all()

    mplot.figure("unit", "normalized", "DefaultAxesFontSize", 16,
                 "DefaultAxesFontWeight", "bold", "DefaultAxesLineWidth", 2,
                 "position", mat.row(0.5, 0.2, 0.5, 0.6))
    fP = lambda psi_: (G.betas * mat.exp(-G.alpha * psi_)
                       * (sigma * psi_ ** G.q + 1) ** G.n)
    Pr = G.betas * mat.exp(-G.alpha * psi) * (sigma * psi ** G.q + 1) ** G.n
    fdP = lambda psi_: (G.betas * mat.exp(-G.alpha * psi_)
                        * (sigma * psi_ ** G.q + 1) ** (G.n - 1)
                        * (-G.alpha * (sigma * psi_ ** G.q + 1)
                           + G.n * G.q * sigma * psi_ ** (G.q - 1)))
    dP = (G.betas * mat.exp(-G.alpha * psi) * (sigma * psi ** G.q + 1) ** (G.n - 1)
          * (-G.alpha * (sigma * psi ** G.q + 1)
             + G.n * G.q * sigma * psi ** (G.q - 1)))
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
    mplot.title(f"l_{{s0}}={mat.num2str(G.ls0)}, \\alpha={mat.num2str(G.alpha)}, "
                f"q={mat.num2str(G.q)}")
    mplot.subplot(223)
    mplot.plot(rr, Jt, "linewidth", 2)
    mplot.xlabel("r")
    mplot.ylabel("J_\\theta")
    mplot.title(f"\\sigma={mat.num2str(sigma)}, h={mat.num2str(h)}, "
                f"n={mat.num2str(G.n)}")
    mplot.subplot(224)
    mplot.plot(rr, Bz, "linewidth", 2)
    mplot.xlabel("r")
    mplot.ylabel("B_z")
    mplot.title(f"\\psi_m={mat.num2str(psim)}, \\alpha_0={mat.num2str(G.alpha0)}")

    mplot.set_(mplot.gcf(), "Units", "inches")
    screenposition = mplot.get_(mplot.gcf(), "Position")
    mplot.set_(mplot.gcf(), "PaperPosition",
               mat.row(0, 0, screenposition[2], screenposition[3]),
               "PaperSize", mat.row(screenposition[2], screenposition[3]))
    # % print(gcf,'-dpng',['MRR1_betas=',...]);

    for name in ("betas", "ls", "alpha", "n", "q", "tol", "alpha0", "ls0",
                 "psim0", "psig"):
        ws[name] = getattr(G, name)
    ws.options = options
    ws.options2 = options2
    ws.x0 = x0
    ws.x = x
    ws.F = F
    ws.sigma = sigma
    ws.psim = psim
    ws.ans = ans
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
