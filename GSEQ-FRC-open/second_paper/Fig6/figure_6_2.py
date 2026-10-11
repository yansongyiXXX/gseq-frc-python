"""figure_6_2.m 的移植：Fig6（自由边界）的二维解 vs 线圈真空场 + 一维剖面。

**后处理**脚本（用自由边界求解器 `GSEQ_coil_MRR.m` 留在工作区里的
`psi/R/Z/zw/Z0/rr/Jzeta/P/Rs/Be/CI/psiw_coil/Bz_coil/nR/nZ` 等），图里画了腔体轮廓
与三个线圈矩形。

⚠️ **原脚本按原文也跑不完**（第三份，与 `figure_4_2`/`fix5_2` 同类）：

* 第 75 行用 `Bz(end,:)` 当二维场用，而自由边界求解器工作区里**没有 `Bz`** 这个名字
  （它给的是 `Bz1d`、`Bz_coil`、`BzT`；本机查过加速线的冻结工作区）；
* 第 98/116 行的 `rm` 同样既不在脚本里、也不在求解器里（`figure_4_1` 用的是
  `rm=0.14`、`fix5_1` 用的是 `0.71`，都只在自己脚本里定义）。

移植版照抄这两个依赖（缺变量就在开头报清楚），测试里登记为预期报错。

验收：§7「绘图脚本」那一节的三条。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mio
from gseqfrc import mplot

#: 这个脚本要求调用方工作区里已经有这些（自由边界求解器那次运行留下的）
NEEDS_WORKSPACE = ("psi", "R", "Z", "zw", "Z0", "rr", "Jzeta", "P", "Rs", "Be",
                   "CI", "psiw_coil", "Bz_coil", "nR", "nZ",
                   # 下面两个在求解器工作区里**不存在**（见模块说明）
                   "Bz", "rm")


def run(ws) -> None:
    missing = [name for name in NEEDS_WORKSPACE if name not in ws.keys()]
    if missing:
        raise KeyError(
            f"figure_6_2 需要在自由边界求解器的工作区里跑，缺少：{', '.join(missing)}"
            "（其中 Bz 与 rm 原程序也没定义——照抄原文，需要调用方提供）")

    psi = np.asarray(ws["psi"])
    Bz = np.asarray(ws["Bz"])
    R, Z = np.asarray(ws["R"]), np.asarray(ws["Z"])
    zw = ws["zw"]
    Z0 = int(mat.scalar_value(ws["Z0"]))
    rr = np.asarray(ws["rr"])
    Jzeta, P = np.asarray(ws["Jzeta"]), np.asarray(ws["P"])
    Rs, Be = ws["Rs"], ws["Be"]
    CI = ws["CI"]
    psiw_coil = np.asarray(ws["psiw_coil"])
    Bz_coil = np.asarray(ws["Bz_coil"])
    nR = int(mat.scalar_value(ws["nR"]))
    nZ = int(mat.scalar_value(ws["nZ"]))
    rm = ws["rm"]

    mplot.close_all()  # close all;
    fontsize = mat.scalar(25)
    mplot.figure("unit", "normalized", "DefaultAxesFontSize", fontsize,
                 "DefaultAxesFontWeight", "bold", "DefaultAxesLineWidth", 2,
                 "defaulttextinterpreter", "latex",
                 "position", mat.row(0.5, 0.2, 0.28, 0.6), "color", mat.row(1, 1, 1))
    # % load('MRR1d_s2.mat');
    loaded = mio.load("MRR1d.mat")
    mu0 = mat.scalar(4 * np.pi * 1e-7)
    rr1d = loaded["rr1d"]
    pr1d = loaded["pr1d"]
    j1d = loaded["j1d"]

    mplot.axes("position", mat.row(0.11, 0.75, 0.85, 0.23))  # fig-1
    ido = mat.find(psi == np.min(psi))
    indtmp = psi > 0
    psio = np.where(indtmp, np.nan, psi)
    indtmp = mat.find(psi < 0)
    psii = np.array(psi, copy=True)
    psii[np.asarray(psi < 0)] = np.nan
    mplot.contour(Z, R, psio, 7, "linewidth", 2)
    mplot.hold_on()
    mplot.contour(Z, R, psii, 30, "linewidth", 2)
    mplot.hold_on()   # %colormap('hot');
    c1, h1 = mplot.contour(Z, R, psi, mat.row(0, 0), "r--", "linewidth", 3,
                           return_matrix=True)
    mplot.plot(mat.take(Z, int(mat.el(ido, 1))), mat.take(R, int(mat.el(ido, 1))),
               "mx", "linewidth", 3)   # O-point
    mplot.text(0.01 * mat.scalar_value(zw), mat.take(R, int(mat.el(ido, 1))),
               "O point", "fontsize", 13)

    mplot.ylabel("r [m]")
    zx1 = np.asarray(c1)[0, 1:-1]
    rx1 = np.asarray(c1)[1, 1:-1]
    zx_no = np.size(zx1, 0)
    indx = mat.find(np.abs(zx1) == np.min(np.abs(zx1)))
    mplot.set_(mplot.gca(), "ytick", mat.colon(0, 0.05, 0.15))
    mplot.set_(mplot.gca(), "xtick", mat.colon(-0.6, 0.3, 0.6))
    _ = np.min(psi[:, Z0 - 1])   # 原文裸表达式 min(psi(:,Z0))（只进 ans）
    rs = mat.el(rx1, int(mat.el(indx, 1)))
    mplot.plot(mat.row(-0.75, 0.75), mat.row(0.14, 0.14), "k-",
               mat.row(-0.75, -0.75), mat.row(0, 0.14), "k-",
               mat.row(0.75, 0.75), mat.row(0, 0.14), "k-", "linewidth", 3)
    mplot.hold_on()
    h = mplot.colorbar("east")
    # % h.TickLabels = (0:1:6);
    # % h.Label.String = '\times 10^{-2}';

    w = None
    for pos in (mat.row(-0.75, 0.147, 0.1, 0.006),
                mat.row(-0.60, 0.147, 1.2, 0.006),
                mat.row(0.65, 0.147, 0.1, 0.006)):
        h = mplot.rectangle("Position", pos, "FaceColor", mat.row(0.5, 0.5, 0.5),
                            "linewidth", 1)
        w = h.LineWidth
        h.LineWidth = 3
        mplot.hold_on()

    mplot.ylabel("$r\\ [m]$", "interpreter", "latex", "FontSize", fontsize)
    mplot.xlabel("$z\\ [m]$", "interpreter", "latex", "FontSize", fontsize)
    mplot.text("Interpreter", "latex", "String", "$(a)$",
               "Position", mat.row(-0.7, 0.12), "color", "k", "FontSize", fontsize)

    mplot.axes("position", mat.row(0.12, 0.41, 0.38, 0.26))  # fig-2
    psiw_coil = psiw_coil * CI
    psiw_coil_u = psiw_coil[nR - 1:nR + nZ - 1]     # psiw_coil(nR:nR+nZ-1)
    mplot.plot(np.asarray(Z)[-1, :], psiw_coil_u, "r-", np.asarray(Z)[-1, :],
               np.asarray(psi)[-1, :], "g--", "linewidth", 3)
    mplot.grid_on()
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.xlabel("$z [m]$", "interpreter", "latex", "FontSize", fontsize)
    mplot.ylabel("$\\psi \\ at \\ wall$", "interpreter", "latex", "FontSize", fontsize)
    mplot.xlim(mat.row(-0.75, 0.75))
    mplot.set_(mplot.gca(), "xtick", mat.colon(-0.6, 0.4, 0.6))
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
               "GridAlpha", 0.25)
    mplot.set_(mplot.gcf(), "PaperPositionMode", "auto")
    mplot.text("Interpreter", "latex", "String", "$(b)$",
               "Position", mat.row(-0.7, 0.04), "color", "k", "FontSize", fontsize)
    h = mplot.legend("$\\mathbf{w/o \\ plasma}$", "$\\mathbf{w \\ plasma}$")
    mplot.set_(h, "interpreter", "latex")
    mplot.legend("boxoff")

    mplot.axes("position", mat.row(0.58, 0.41, 0.38, 0.26))  # fig-3
    # % Bz_coil_u=Bz_coil.*CI/2/pi;
    Bz_coil_u = Bz_coil * CI
    Bz_coil_u = np.asarray(Bz_coil_u)[:, 0][nR - 1:nR + nZ - 1]
    mid = (np.size(Bz_coil_u, 0) + 1) // 2
    print(f"B_z at midplane in vacuum= {mat.scalar_value(Bz_coil_u[mid - 1]):5.4f} ")
    print(f"B_z at midplane with plasma= {mat.scalar_value(np.asarray(Bz)[-1, :][mid - 1]):5.4f} ")
    mplot.plot(np.asarray(Z)[-1, :], Bz_coil_u, "r-", np.asarray(Z)[-1, :],
               np.asarray(Bz)[-1, :], "g--", "linewidth", 3)
    mplot.grid_on()
    mplot.hold_on()
    mplot.ylabel("$B_{z} \\ at \\ wall$", "interpreter", "latex", "FontSize", fontsize)
    mplot.xlabel("$z [m]$", "interpreter", "latex", "FontSize", fontsize)
    mplot.xlim(mat.row(-0.75, 0.75))
    mplot.set_(mplot.gca(), "xtick", mat.colon(-0.6, 0.4, 0.6))
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
               "GridAlpha", 0.25)
    mplot.text("Interpreter", "latex", "String", "$(c)$",
               "Position", mat.row(-0.7, 5.6), "color", "k", "FontSize", fontsize)
    h = mplot.legend("$\\mathbf{w/o \\ plasma}$", "$\\mathbf{w \\ plasma}$")
    mplot.set_(h, "interpreter", "latex")
    mplot.legend("boxoff")

    mplot.axes("position", mat.row(0.12, 0.07, 0.38, 0.26))  # fig-4
    Jzetar = -Jzeta[:, Z0 - 1]
    Pr = P[:, Z0 - 1]
    J0 = np.max(np.abs(Jzetar))
    P0 = np.max(np.abs(Pr))
    mplot.plot(rr * 100, Pr, "m-", rr1d * Rs * 100,
               pr1d * (Be ** 2 / (2 * mu0)), "--", "linewidth", 3)
    mplot.grid_on()
    mplot.hold_on()
    mplot.plot(mat.row(100 * float(mat.scalar_value(rs)), 100 * float(mat.scalar_value(rs))),
               mat.row(0, 5e5), "-.k", "LineWidth", 1.5)
    mplot.hold_on()
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.xlabel("$r\\ [cm], z=0$", "interpreter", "latex", "FontSize", fontsize)
    mplot.ylabel("$P$", "interpreter", "latex", "FontSize", fontsize)
    h = mplot.legend("$2D$", "$1D$")
    mplot.set_(h, "interpreter", "latex")
    mplot.legend("boxoff")
    mplot.xlim(mat.row(0, 100 * mat.scalar_value(rm)))
    mplot.set_(mplot.gca(), "xtick", mat.colon(0, 2.5, 100 * mat.scalar_value(rm)))
    mplot.ylim(mat.row(0, 4.3e5))
    mplot.set_(mplot.gca(), "ytick", mat.colon(0, 1e5, 4.5e5))
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
               "GridAlpha", 0.25)
    mplot.text("Interpreter", "latex", "String", "$(d)$",
               "Position", mat.row(1, 3.8e5), "color", "k", "FontSize", fontsize)
    mplot.text("Interpreter", "latex", "String", "$separatrix$", "Position",
               mat.row(100 * mat.scalar_value(rs) - 2, 1e5), "FontSize", 22)

    mplot.axes("position", mat.row(0.58, 0.07, 0.38, 0.26))  # fig-5
    Jzetar = -Jzeta[:, Z0 - 1]
    mplot.plot(rr * 100, Jzetar, "m-", rr1d * Rs * 100,
               j1d * (Be / (2 * mu0 * Rs)), "--", "linewidth", 3)
    mplot.grid_on()
    mplot.hold_on()
    mplot.plot(mat.row(100 * float(mat.scalar_value(rs)), 100 * float(mat.scalar_value(rs))),
               1.1 * mat.row(0, float(J0)), "-.k", "LineWidth", 1.5)
    mplot.hold_on()
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.xlabel("$r\\ [cm], z=0$", "interpreter", "latex", "FontSize", fontsize)
    mplot.ylabel("$J_{\\theta}$", "interpreter", "latex", "FontSize", fontsize)
    h = mplot.legend("$2D$", "$1D$")
    mplot.set_(h, "interpreter", "latex")
    mplot.legend("boxoff")
    mplot.xlim(mat.row(0, 100 * mat.scalar_value(rm)))
    mplot.set_(mplot.gca(), "xtick", mat.colon(0, 2.5, 100 * mat.scalar_value(rm)))
    mplot.ylim(mat.row(0, 5e7))
    mplot.set_(mplot.gca(), "ytick", mat.colon(0, 1e7, 5e7))
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
               "GridAlpha", 0.25)
    mplot.set_(mplot.gcf(), "PaperPositionMode", "auto")
    mplot.text("Interpreter", "latex", "String", "$(e)$",
               "Position", mat.row(1, 4.5e7), "color", "k", "FontSize", fontsize)

    # % figure(2)
    # % plot(rr*100,Bz(:,floor(nZ/2)))

    for name, value in (("fontsize", fontsize), ("mu0", mu0), ("ido", ido),
                        ("indtmp", indtmp), ("psio", psio), ("psii", psii),
                        ("c1", c1), ("h1", h1), ("zx1", zx1), ("rx1", rx1),
                        ("zx_no", mat.scalar(zx_no)), ("indx", indx), ("rs", rs),
                        ("psiw_coil", psiw_coil), ("psiw_coil_u", psiw_coil_u),
                        ("Bz_coil_u", Bz_coil_u), ("Jzetar", Jzetar), ("Pr", Pr),
                        ("J0", mat.scalar(J0)), ("P0", mat.scalar(P0)),
                        ("w", w)):
        ws[name] = value
    for name, value in loaded.items():
        ws[name] = value
