"""solve_2PE09.m 的逐行移植（Steinhauer 2009 版两点压力剖面，含 r>rs 段）。

与 2008 版的差别在于 r>rs 那一段用 `sigma2` 的指数尾巴接上；脚本里 `fb`、`fp`、
`dbdu` 三个匿名函数都用 `(条件).*(值)` 的写法选分支——**两个分支都会被求值**，
这是原程序的写法，照抄。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mglobals
from gseqfrc import mio
from gseqfrc import mpath
from gseqfrc import mplot


def run(ws) -> None:
    mplot.close_all()  # close all;clear;clc;
    G = mglobals.G

    # global betas ls alpha0 ls0 psim0;
    G.declare("betas", "ls", "alpha0", "ls0", "psim0")

    G.betas = mat.scalar(0.6655)
    G.alpha0 = 8 * np.log((np.sqrt(1 - G.betas) + 1) / np.sqrt(G.betas))
    G.ls0 = 1 / (G.alpha0 * np.sqrt(1 - G.betas))
    G.psim0 = np.log(G.betas) / G.alpha0
    G.ls = mat.scalar(0.148)

    sigma = (G.betas / (4 * (1 - G.betas) * G.ls) - 1) / 4
    bs = np.sqrt(1 - G.betas)
    sigma2 = (1 + 4 * sigma) * bs / (1 - bs)
    # % psim=sqrt(1-betas)/(8*sigma)*(exp(-sigma)-1);

    def fb(u):
        return ((np.abs(u) <= 1) * (bs * u * mat.exp(sigma * (u ** 4 - 1)))
                + (u > 1) * (1 - (1 - bs) * mat.exp(-sigma2 * (u - 1))))

    def fp(u):
        return 1 - (fb(u)) ** 2

    def dbdu(u):
        return ((np.abs(u) <= 1) * (bs * mat.exp(sigma * (u ** 4 - 1))
                                    * (1 + 4 * sigma * u ** 4))
                + (u > 1) * ((1 - bs) * sigma2 * mat.exp(-sigma2 * (u - 1))))

    dr = mat.scalar(0.005)
    rr = mat.colon(0, dr, 1.5)
    u = 2 * rr ** 2 - 1
    Bz = fb(u)
    psi = mat.cumsum(rr * Bz * dr)
    Pr = fp(u)
    Jt = 4 * rr * dbdu(u)  # to check

    psim = mat.scalar(np.min(psi))

    ids = mat.find(np.abs(rr - 1) == np.min(np.abs(rr - 1)))
    ids = mat.el(ids, 1)
    lsp = mat.scalar(np.real(-(mat.el(Pr, ids)
                               / ((mat.el(Pr, ids + 1) - mat.el(Pr, ids - 1))
                                  / (mat.el(rr, ids + 1) - mat.el(rr, ids - 1))))))

    ind = mat.find(rr <= 1.0)
    h = np.real(dbdu(mat.scalar(0)) / np.sum(2 * mat.take(dbdu(u), ind) * dr))

    mplot.close_all()

    mplot.figure("unit", "normalized", "DefaultAxesFontSize", 16,
                 "DefaultAxesFontWeight", "bold", "DefaultAxesLineWidth", 2,
                 "position", mat.row(0.02, 0.2, 0.5, 0.6))

    mplot.subplot(221)
    mplot.plot(rr, psi, "linewidth", 2)
    mplot.xlabel("r")
    mplot.ylabel("\\psi")
    mplot.title(f"\\beta_s={mat.num2str(G.betas)}, l_s={mat.num2str(G.ls)}, "
                f"l_{{sp}}={mat.num2str(lsp, 3)}")
    mplot.subplot(222)
    mplot.plot(rr, Pr, "linewidth", 2)
    mplot.ylim(mat.row(0, 1))
    mplot.xlabel("r")
    mplot.ylabel("P")
    mplot.title(f"l_{{s0}}={mat.num2str(G.ls0)}")
    mplot.subplot(223)
    mplot.plot(rr, Jt, "linewidth", 2)
    mplot.xlabel("r")
    mplot.ylabel("J_\\theta")
    mplot.title(f"\\sigma={mat.num2str(sigma)}, h={mat.num2str(h)}")
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
    mplot.print_fig(
        mplot.gcf(), "-dpng",
        f"2PE09_betas={mat.num2str(G.betas)}_ls={mat.num2str(G.ls)}"
        f"_sigma={mat.num2str(sigma)}_psim={mat.num2str(psim)}.png",
    )

    datq_2pe = np.vstack([rr, Pr, psi, Bz, Jt]).T
    mio.save("2pe_datq_i09.mat", datq_2pe=datq_2pe, betas=G.betas, ls=G.ls)

    for name in ("betas", "ls", "alpha0", "ls0", "psim0"):
        ws[name] = getattr(G, name)
    ws.sigma = sigma
    ws.bs = bs
    ws.sigma2 = sigma2
    ws.psim = psim
    ws.dr = dr
    ws.rr = rr
    ws.u = u
    ws.Bz = Bz
    ws.psi = psi
    ws.Pr = Pr
    ws.Jt = Jt
    ws.ids = mat.scalar(ids)
    ws.lsp = lsp
    ws.ind = ind
    ws.h = h
    ws.screenposition = mat.row(*screenposition)
    ws.datq_2pe = datq_2pe
