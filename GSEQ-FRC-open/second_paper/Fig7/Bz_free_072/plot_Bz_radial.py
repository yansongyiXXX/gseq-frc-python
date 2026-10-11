"""plot_Bz_radial.m（Bz_free_072 副本）的移植——与 `Bz_fixed_072` 那份同骨架。

三处差别（逐条照抄）：线型 `m-` → `k-`、**不设 `ylim`**、图例文字
`fixed-boundary` → `free-boundary`。同样是**会话链的末端**（需要上游诊断图脚本
留在工作区里的二维 `Z`/`Bz`），单跑会缺变量——见同目录那份的说明。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mio
from gseqfrc import mplot

#: 同 `Bz_fixed_072` 那份
NEEDS_WORKSPACE = ("Z", "Bz", "R", "psi")


def run(ws) -> None:
    missing = [name for name in NEEDS_WORKSPACE if name not in ws.keys()]
    if missing:
        raise KeyError(
            f"plot_Bz_radial 是会话链的末端（求解器 → 诊断图 → 本脚本），缺少："
            f"{', '.join(missing)}")

    Z = np.asarray(ws["Z"])
    Bz = np.asarray(ws["Bz"])
    R = np.asarray(ws["R"])
    psi = np.asarray(ws["psi"])

    mplot.close_all()  # close all;
    fontsize = mat.scalar(25)
    mplot.figure("unit", "normalized", "DefaultAxesFontSize", fontsize,
                 "DefaultAxesFontWeight", "bold", "DefaultAxesLineWidth", 2,
                 "defaulttextinterpreter", "latex",
                 "position", mat.row(0.1, 0.2, 0.28, 0.3), "color", mat.row(1, 1, 1))
    loaded = mio.load("MRR1d.mat")
    mu0 = mat.scalar(4 * np.pi * 1e-7)

    ido = mat.find(psi == np.min(psi))
    indtmp = psi > 0
    psio = np.where(indtmp, np.nan, psi)
    indtmp = mat.find(psi < 0)
    psii = np.array(psi, copy=True)
    psii[np.asarray(psi < 0)] = np.nan

    Bi = np.zeros(int(np.size(Z, 0)))
    for i in mat.mrange(1, int(np.size(Z, 0))):
        Bi[i - 1] = mat.interp1(mat.take(Z, i), mat.take(Bz, i), -0.72)

    mplot.axes("position", mat.row(0.11, 0.14, 0.85, 0.83))  # fig-1
    mplot.plot(np.asarray(R)[:, 2] * 100, Bi, "k-", "linewidth", 3)   # 这一份是 k-
    mplot.grid_on()
    mplot.hold_on()
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.ylabel("$B_{z} \\ [T]$", "interpreter", "latex", "FontSize", fontsize)
    mplot.xlabel("$r [cm]$", "interpreter", "latex", "FontSize", fontsize)
    mplot.xlim(mat.row(0, 14))
    mplot.set_(mplot.gca(), "xtick", mat.colon(0, 2, 14))
    # 这一份**没有** ylim/ytick 那两行
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
               "GridAlpha", 0.25)
    h = mplot.legend("$\\mathbf{free-boundary}$")
    mplot.set_(h, "interpreter", "latex", "FontSize", 28)
    mplot.legend("boxoff")

    for name, value in (("fontsize", fontsize), ("mu0", mu0), ("ido", ido),
                        ("indtmp", indtmp), ("psio", psio), ("psii", psii),
                        ("Bi", Bi)):
        ws[name] = value
    for name, value in loaded.items():
        ws[name] = value
