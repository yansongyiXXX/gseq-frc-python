"""solve_RR.m 的逐行移植（刚性转子 RR 剖面的一维径向平衡）。

这是一个**脚本**而不是函数：没有函数头，直接在基本工作区里
跑（`close all;clear;clc` 开头），靠 global 传参，末尾画图并写出 `rr_datq.mat`。
所以移植版对应 `run(ws)`，由 `run.py` 在把源目录加进搜索路径之后调用它。

`clear` 与 `clc` 在这里是空操作：每个算例本来就从干净的工作区开始（见
`cases.run_case`），这与 harness 给每个算例单开一个 MATLAB 进程是等价的。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mglobals
from gseqfrc import mio
from gseqfrc import moptim
from gseqfrc import mpath
from gseqfrc import mplot

SRC = ["first_paper/figure1_2/RR"]


def run(ws) -> None:
    mpath.addpath(*SRC)
    G = mglobals.G

    mplot.close_all()  # close all;clear;clc;

    # global betas ls alpha;
    G.declare("betas", "ls", "alpha")
    tol = 1e-8

    # % betas=0.6655;
    G.betas = mat.scalar(0.6)
    G.alpha = 8 * np.log((np.sqrt(1 - G.betas) + 1) / np.sqrt(G.betas))
    G.ls = 1 / (G.alpha * np.sqrt(1 - G.betas))
    psim = np.log(G.betas) / G.alpha

    options2 = moptim.optimset(TolFun=tol, Tolx=tol, Display="off")

    # %%
    mplot.close_all()

    def fpsi(psi):
        return 1.0 / np.sqrt(1.0 - G.betas * mat.exp(-G.alpha * psi))

    dr = mat.scalar(0.02)
    rmax = mat.scalar(1.5 / 1)
    rr = mat.colon(0, dr, rmax)
    # % nr=rmax/dr;
    nr = mat.length(rr)
    psi = 0.0 * rr

    for jr in mat.mrange(2, nr):
        r = mat.el(rr, jr)
        if r <= 1 / np.sqrt(2):
            def FF1(psi_):
                return moptim.integral(fpsi, 0, psi_, 1e-8, 1e-8) + r ** 2 / 2

            # % psi(jr)=fsolve(FF1,-0.01,options2);
            mat.set_el(psi, jr, moptim.fsolve(FF1, np.real(mat.el(psi, jr - 1)),
                                              options2))
        else:
            def FF2(psi_):
                return (moptim.integral(fpsi, psim, psi_, 1e-8, 1e-8)
                        - (r ** 2 - 1 / 2) / 2)

            # % psi(jr)=fsolve(FF2,psim0,options2);
            mat.set_el(psi, jr, moptim.fsolve(FF2, 0.99 * np.real(mat.el(psi, jr - 1)),
                                              options2))

    # %%
    # % psi(abs(imag(psi))>=1e-6)=NaN;
    mplot.close_all()

    mplot.figure("unit", "normalized", "DefaultAxesFontSize", 16,
                 "DefaultAxesFontWeight", "bold", "DefaultAxesLineWidth", 2,
                 "position", mat.row(0.02, 0.2, 0.5, 0.6))
    fP = lambda psi_: G.betas * mat.exp(-G.alpha * psi_)
    Pr = G.betas * mat.exp(-G.alpha * psi)
    fdP = lambda psi_: -G.betas * mat.exp(-G.alpha * psi_) * (G.alpha)
    dP = fdP(psi)
    Jt = -rr * dP
    Bz = (-np.sqrt(1 - Pr) * (rr <= 1 / np.sqrt(2))
          + np.sqrt(1 - Pr) * (rr > 1 / np.sqrt(2)))

    ids = mat.find(np.abs(rr - 1) == np.min(np.abs(rr - 1)))
    ids = mat.el(ids, 1)
    lsp = mat.scalar(np.real(
        -(mat.el(Pr, ids)
          / ((mat.el(Pr, ids + 1) - mat.el(Pr, ids - 1))
             / (mat.el(rr, ids + 1) - mat.el(rr, ids - 1))))
    ))

    ind = mat.find(rr <= 1.0)
    h = np.real(fdP(psim) / np.sum(2 * mat.take(dP, ind) * mat.take(rr, ind) * dr))

    mplot.subplot(221)
    mplot.plot(rr, psi, "linewidth", 2)
    mplot.xlabel("r")
    mplot.ylabel("\\psi")
    mplot.title(f"\\beta_s={mat.num2str(G.betas)}, l_s={mat.num2str(G.ls)}")
    mplot.subplot(222)
    mplot.plot(rr, Pr, "linewidth", 2)
    mplot.xlabel("r")
    mplot.ylabel("P")
    mplot.title(f"\\alpha={mat.num2str(G.alpha)}, l_{{sp}}={mat.num2str(lsp, 3)}")
    mplot.subplot(223)
    mplot.plot(rr, Jt, "linewidth", 2)
    mplot.xlabel("r")
    mplot.ylabel("J_\\theta")
    mplot.title(f"h={mat.num2str(h)}")
    mplot.subplot(224)
    mplot.plot(rr, Bz, "linewidth", 2)
    mplot.xlabel("r")
    mplot.ylabel("B_z")
    mplot.title(f"\\psi_m={mat.num2str(psim)}")

    mplot.set_(mplot.gcf(), "Units", "inches")
    screenposition = mplot.get_(mplot.gcf(), "Position")
    mplot.set_(mplot.gcf(), "PaperPosition",
               mat.row(0, 0, screenposition[2], screenposition[3]),
               "PaperSize", mat.row(screenposition[2], screenposition[3]))
    mplot.print_fig(
        mplot.gcf(), "-dpng",
        f"RR_betas={mat.num2str(G.betas)}_ls={mat.num2str(G.ls)}"
        f"_alpha={mat.num2str(G.alpha)}.png",
    )

    datq_rr = np.vstack([rr, Pr, psi, Bz, Jt]).T
    mio.save("rr_datq.mat", datq_rr=datq_rr, betas=G.betas, ls=G.ls)

    for name in ("betas", "ls", "alpha"):
        ws[name] = getattr(G, name)
    ws.tol = mat.scalar(tol)
    ws.psim = psim
    ws.options2 = options2
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
    ws.datq_rr = datq_rr
