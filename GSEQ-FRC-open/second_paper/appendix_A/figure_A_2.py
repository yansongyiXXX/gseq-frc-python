"""figure_A_2.m 的移植：附录 A 的五联图（分界面 / 壁面 ψ / 壁面 B_z / P(r) / J_θ(r)）。

**这是"后处理"脚本**：它自己只 `load('MRR1d.mat')`（一维 MRR 剖面），二维的东西
全用**调用方工作区里**的变量——MATLAB 里的用法是"跑完 `figure_A_1.m` 之后再 run
这一份"。需要上游留下：`psi/R/Z/zw/nR/nZ/rr/rm`（`figure_A_1` 算的）与
`Jzeta/P/Z0/Bz_coil/psiw_coil/CI/Bz`（前一半是 `psiq1.mat` 带进来的，`Bz` 是
`figure_A_1` 用中心差分算出的二维场）。

所以**单跑这一份会停**（缺上面那些变量），接在 `figure_A_1` 后面就通；移植版把这个
依赖写成开头的显式检查，测试里两条都验（见 `tests/test_plotting_scripts.py`）。

它自己会定义 `rs`（分界面在 z=0 处的半径，从零层等值线的第二个顶点取），供下游的
`figure_A_3.m` 用；`c1` 也一样传给下游。

验收：§7「绘图脚本」那一节的三条（能跑通 / 结构对 / 人眼比对论文图）。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mio
from gseqfrc import mplot

#: 这个脚本要求调用方工作区里已经有这些（来自 `figure_A_1.m` 那次运行）
NEEDS_WORKSPACE = ("psi", "R", "Z", "zw", "nR", "nZ", "rr", "rm",
                   "Jzeta", "P", "Z0", "Bz_coil", "psiw_coil", "CI", "Bz")


def run(ws) -> None:
    missing = [name for name in NEEDS_WORKSPACE if name not in ws.keys()]
    if missing:
        raise KeyError(
            f"figure_A_2 需要在 figure_A_1 之后跑（同一个会话），缺少："
            f"{', '.join(missing)}（原脚本直接用上游算出的二维 Bz/rm 与 psiq1.mat "
            "里的 Jzeta/P/Z0/psiw_coil/CI/Bz_coil）"
        )

    psi = np.asarray(ws["psi"]); R = np.asarray(ws["R"]); Z = np.asarray(ws["Z"])
    zw = ws["zw"]
    nR = int(mat.scalar_value(ws["nR"])); nZ = int(mat.scalar_value(ws["nZ"]))
    rr = np.asarray(ws["rr"]); rm = ws["rm"]
    Jzeta = np.asarray(ws["Jzeta"]); P = np.asarray(ws["P"])
    Z0 = int(mat.scalar_value(ws["Z0"]))
    Bz_coil = np.asarray(ws["Bz_coil"]); psiw_coil = np.asarray(ws["psiw_coil"])
    CI = ws["CI"]
    Bz = np.asarray(ws["Bz"])

    mplot.close_all()  # close all;
    fontsize = mat.scalar(25)
    mplot.figure("unit", "normalized", "DefaultAxesFontSize", fontsize,
                 "DefaultAxesFontWeight", "bold", "DefaultAxesLineWidth", 2,
                 "defaulttextinterpreter", "latex",
                 "position", mat.row(0.5, 0.2, 0.28, 0.6), "color", mat.row(1, 1, 1))
    loaded = mio.load("MRR1d.mat")
    mu0 = mat.scalar(4 * np.pi * 1e-7)
    rr1d = loaded["rr1d"]; pr1d = loaded["pr1d"]; j1d = loaded["j1d"]
    Be = loaded["Be"]; Rs = loaded["Rs"]

    mplot.axes("position", mat.row(0.11, 0.75, 0.85, 0.23))  # fig-1
    ido = mat.find(psi == np.min(psi))
    indtmp = psi > 0
    psio = np.array(psi, copy=True)
    psio[np.asarray(indtmp)] = np.nan
    indtmp = mat.find(psi < 0)
    psii = np.array(psi, copy=True)
    psii[np.asarray(psi < 0)] = np.nan
    mplot.contour(Z, R, psio, 7, "linewidth", 2)
    mplot.hold_on()
    mplot.contour(Z, R, psii, 30, "linewidth", 2)
    mplot.hold_on()
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
    # `min(psi(:,Z0));` 原句是裸表达式（只把结果丢进 ans），这里照抄成无副作用的计算
    _ = np.min(psi[:, Z0 - 1])
    rs = mat.el(rx1, int(mat.el(indx, 1)))
    # 壁面轮廓：只画到 r=0.14 的三段线（底边 + 两条侧边）
    mplot.plot(mat.row(-0.75, 0.75), mat.row(0.14, 0.14), "k-",
               mat.row(-0.75, -0.75), mat.row(0, 0.14), "k-",
               mat.row(0.75, 0.75), mat.row(0, 0.14), "k-", "linewidth", 3)
    mplot.hold_on()
    mplot.colorbar("east")

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
    psiw_coil_u = psiw_coil[nR - 1:nR + nZ - 1]
    mplot.plot(Z[-1, :], psiw_coil_u, "r-", Z[-1, :], psi[-1, :], "g--",
               "linewidth", 3)
    mplot.grid_on()
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.xlabel("$z [m]$", "interpreter", "latex", "FontSize", fontsize)
    mplot.ylabel("$\\psi \\ at \\ wall$", "interpreter", "latex",
                 "FontSize", fontsize)
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
    Bz_coil_u = Bz_coil * CI
    Bz_coil_u = Bz_coil_u[nR - 1:nR + nZ - 1]
    mplot.plot(Z[-1, :], Bz_coil_u, "r-", Z[-1, :], Bz[-1, :], "g--",
               "linewidth", 3)
    mplot.grid_on()
    mplot.hold_on()
    mplot.ylabel("$B_{z} \\ at \\ wall$", "interpreter", "latex",
                 "FontSize", fontsize)
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
    # `A(:,Z0)` 在 MATLAB 里是 n×1 的列，numpy 的 `A[:, k]` 会掉一维，所以按列切片
    Jzetar = -Jzeta[:, Z0 - 1:Z0]
    Pr = P[:, Z0 - 1:Z0]
    J0 = np.max(np.abs(Jzetar))
    P0 = np.max(np.abs(Pr))
    mplot.plot(rr * 100, Pr, "m-", rr1d * Rs * 100,
               pr1d * (mat.scalar_value(Be) ** 2 / (2 * mat.scalar_value(mu0))),
               "--", "linewidth", 3)
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
    mplot.text("Interpreter", "latex", "String", "$separatrix$",
               "Position", mat.row(100 * float(mat.scalar_value(rs)) - 2, 1e5),
               "FontSize", 22)

    mplot.axes("position", mat.row(0.58, 0.07, 0.38, 0.26))  # fig-5
    Jzetar = -Jzeta[:, Z0 - 1:Z0]
    mplot.plot(rr * 100, Jzetar, "m-", rr1d * Rs * 100,
               j1d * (mat.scalar_value(Be) / (2 * mat.scalar_value(mu0) * mat.scalar_value(Rs))),
               "--", "linewidth", 3)
    mplot.grid_on()
    mplot.hold_on()
    mplot.plot(mat.row(100 * float(mat.scalar_value(rs)), 100 * float(mat.scalar_value(rs))),
               mat.row(0, 1.1 * float(J0)), "-.k", "LineWidth", 1.5)
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

    for name, value in (("fontsize", fontsize), ("mu0", mu0), ("ido", ido),
                        ("indtmp", indtmp), ("psio", psio), ("psii", psii),
                        ("c1", c1), ("h1", h1), ("zx1", zx1), ("rx1", rx1),
                        ("zx_no", mat.scalar(zx_no)), ("indx", indx), ("rs", rs),
                        ("w", w), ("psiw_coil", psiw_coil),
                        ("psiw_coil_u", psiw_coil_u), ("Bz_coil_u", Bz_coil_u),
                        ("Jzetar", Jzetar), ("Pr", Pr), ("J0", J0), ("P0", P0),
                        ("h", h)):
        ws[name] = value
    for name, value in loaded.items():
        ws[name] = value
