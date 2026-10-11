"""solve_2PE08.m 的逐行移植（Steinhauer 2008 版两点压力剖面）。

这份是纯代数求解：没有求根、没有积分，`sigma`/`psim` 都由 betas 和 ls 直接算出，
后面只是沿 r 把 `psi`、`Pr`、`Bz`、`Jt` 算出来。所以它与冻结值的偏差应该只有
几个 ULP 量级——与 `solve_RR.m` 那种"先解积分方程"的脚本形成对照。
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

    G.betas = mat.scalar(0.6)
    G.alpha0 = 8 * np.log((np.sqrt(1 - G.betas) + 1) / np.sqrt(G.betas))
    G.ls0 = 1 / (G.alpha0 * np.sqrt(1 - G.betas))
    G.psim0 = np.log(G.betas) / G.alpha0
    G.ls = mat.scalar(0.1)

    sigma = (G.betas / (8 * (1 - G.betas) * G.ls) - 1) / 2
    psim = np.sqrt(1 - G.betas) / (8 * sigma) * (mat.exp(-sigma) - 1)

    dr = mat.scalar(0.01)
    rr = mat.colon(0, dr, 1.02)
    u = 2 * rr ** 2 - 1
    psi = np.sqrt(1 - G.betas) / (8 * sigma) * (mat.exp(sigma * (u ** 2 - 1)) - 1)
    Pr = 1 - (1 - G.betas) * u ** 2 * mat.exp(2 * sigma * (u ** 2 - 1))

    def fdP(psi_):
        return (-8 * np.sqrt(1 - G.betas) * sigma
                * (1 / sigma + 2 / sigma * np.log(8 * sigma / np.sqrt(1 - G.betas) * psi_ + 1) + 2)
                * (8 * sigma / np.sqrt(1 - G.betas) * psi_ + 1))

    dP = (-8 * np.sqrt(1 - G.betas) * sigma
          * (1 / sigma + 2 / sigma * np.log(8 * sigma / np.sqrt(1 - G.betas) * psi + 1) + 2)
          * (8 * sigma / np.sqrt(1 - G.betas) * psi + 1))
    Jt = -rr * dP
    Bz = np.sqrt(1 - G.betas) * u * mat.exp(sigma * (u ** 2 - 1))

    ids = mat.find(np.abs(rr - 1) == np.min(np.abs(rr - 1)))
    ids = mat.el(ids, 1)
    lsp = mat.scalar(np.real(-(mat.el(Pr, ids)
                               / ((mat.el(Pr, ids + 1) - mat.el(Pr, ids - 1))
                                  / (mat.el(rr, ids + 1) - mat.el(rr, ids - 1))))))

    ind = mat.find(rr <= 1.0)
    h = np.real(fdP(psim) / np.sum(2 * mat.take(dP, ind) * mat.take(rr, ind) * dr))

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
        f"2PE_betas={mat.num2str(G.betas)}_ls={mat.num2str(G.ls)}"
        f"_sigma={mat.num2str(sigma)}_psim={mat.num2str(psim)}.png",
    )

    datq_2pe = np.vstack([rr, Pr, psi, Bz, Jt]).T
    mio.save("2pe_datq.mat", datq_2pe=datq_2pe, betas=G.betas, ls=G.ls)

    for name in ("betas", "ls", "alpha0", "ls0", "psim0"):
        ws[name] = getattr(G, name)
    ws.sigma = sigma
    ws.psim = psim
    ws.dr = dr
    ws.rr = rr
    ws.u = u
    ws.psi = psi
    ws.Pr = Pr
    ws.dP = dP
    ws.Jt = Jt
    ws.Bz = Bz
    ws.ids = mat.scalar(ids)
    ws.lsp = lsp
    ws.ind = ind
    ws.h = h
    ws.screenposition = mat.row(*screenposition)
    ws.datq_2pe = datq_2pe
