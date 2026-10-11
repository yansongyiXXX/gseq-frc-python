"""tst_betas_ls.m 的移植：把理论关系 l_s(β_s)、α(β_s)、ψ_m(β_s) 画成三联图。

源文件：`gseq-main/GSEQ-FRC-open/first_paper/figure1_2/RR/tst_betas_ls.m`

**自包含**：没有外部数据，只有三条解析式

    α₀ = 8·ln((sqrt(1−β_s)+1)/sqrt(β_s))
    l_s = 1/(α₀·sqrt(1−β_s))
    ψ_m = ln(β_s)/α₀

在 `β_s = 0.02:0.02:0.95` 上算出来，用 `subplot(131/132/133)` 画三格
（第一格纵轴取对数）。脚本还 `global` 了 `betas ls n tol alpha0 ls0 psim0`——
这些是给同一目录下的 `solve_RR.m`/`fmrr*.m` 用的全局量，移植版照抄写进 `mglobals`。

仓库里有它当年的成品图 `RR/tst_betas_ls.png`，可以人眼对照。

验收：§7「绘图脚本」那一节的三条（能跑通 / 结构对 / 人眼比对论文图）。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mglobals
from gseqfrc import mplot

#: `.m` 开头那句 `global betas ls n tol alpha0 ls0 psim0`
GLOBALS = ("betas", "ls", "n", "tol", "alpha0", "ls0", "psim0")


def run(ws) -> None:
    mplot.close_all()  # close all;clear;clc;

    G = mglobals.G
    G.declare(*GLOBALS)
    G.n = mat.scalar(2)
    G.tol = mat.scalar(1e-8)

    bb = mat.colon(0.02, 0.02, 0.95)
    ll = 0.0 * bb
    aa = 0.0 * bb
    pp = 0.0 * bb
    for jb in mat.mrange(1, int(np.size(bb, 1))):
        betas = float(mat.el(bb, jb))
        G.betas = mat.scalar(betas)
        G.alpha0 = 8 * np.log((np.sqrt(1 - betas) + 1) / np.sqrt(betas))
        G.ls0 = 1 / (G.alpha0 * np.sqrt(1 - betas))
        G.psim0 = np.log(betas) / G.alpha0
        mat.set_el(aa, jb, G.alpha0)
        mat.set_el(pp, jb, G.psim0)
        mat.set_el(ll, jb, G.ls0)

    mplot.figure("unit", "normalized", "DefaultAxesFontSize", 16,
                 "DefaultAxesFontWeight", "bold", "DefaultAxesLineWidth", 2,
                 "position", mat.row(0.02, 0.6, 0.8, 0.3))

    # % plot(bb,log10(ll),'linewidth',2);
    mplot.subplot(131)
    mplot.semilogy(bb, ll, "linewidth", 2)
    mplot.grid_on()
    mplot.box("on")
    mplot.xlabel("\\beta_s")
    mplot.ylabel("l_s")

    mplot.subplot(132)
    mplot.plot(bb, aa, "linewidth", 2)
    mplot.grid_on()
    mplot.box("on")
    mplot.xlabel("\\beta_s")
    mplot.ylabel("\\alpha")

    mplot.subplot(133)
    mplot.plot(bb, pp, "linewidth", 2)
    mplot.grid_on()
    mplot.box("on")
    mplot.xlabel("\\beta_s")
    mplot.ylabel("\\psi_m")

    ws["bb"] = bb
    ws["ll"] = ll
    ws["aa"] = aa
    ws["pp"] = pp
    ws["jb"] = mat.scalar(jb)
    for name in GLOBALS:
        ws[name] = getattr(G, name)
