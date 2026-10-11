"""figure_A_3.m 的移植：附录 A 的"压强平衡"图（一条坐标区、六条曲线）。

**会话链的末端**：`c1`（零层等值线）与 `rs`（分界面半径）来自 `figure_A_2.m`，
`Jzeta/P/Z0/Bz/Br/rr/zz/rm` 来自更上游的 `figure_A_1.m`；它自己只
`load('MRR1d.mat')` 拿一维剖面（`Be` 等）。所以**单跑会停**，按
`figure_A_1 → figure_A_2 → figure_A_3` 的顺序跑就通。

画的是 \(P\)、\(B_z^2/2\mu_0\)、两者之和、\(B_e^2/2\mu_0\)，以及
\(B_e^2/2\mu_0+\int\frac{B_z}{\mu_0}\partial_zB_r\,dr\) 五条曲线，外加一条分界面竖线。

⚠️ 照抄原文的一处**形状不匹配**：第 34 行的 `Be_2+sum_br_curave` 是
1×129 的行加 129×1 的列，MATLAB 的隐式展开给出 129×129（本机 R2023a 实测
`plot(1:5, 5×5)` 画 5 条线），于是这一句实际画的是 **129 条线**（`nR=129`）。
移植版的 `mplot.plot` 按同一条规则处理，所以这里也是 129 条。

验收：§7「绘图脚本」那一节的三条（能跑通 / 结构对 / 人眼比对论文图）。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mio
from gseqfrc import mplot

#: 这个脚本要求调用方工作区里已经有这些（`figure_A_1` + `figure_A_2` 留下的）
NEEDS_WORKSPACE = ("c1", "rs", "Jzeta", "P", "Z0", "Bz", "Br", "zz", "rr", "rm")


def run(ws) -> None:
    missing = [name for name in NEEDS_WORKSPACE if name not in ws.keys()]
    if missing:
        raise KeyError(
            f"figure_A_3 需要在 figure_A_1、figure_A_2 之后跑（同一个会话），"
            f"缺少：{', '.join(missing)}（原脚本直接用上游的 c1/rs 与二维 "
            "Bz/Br，单跑时这些名字在工作区里不存在）"
        )

    c1 = np.asarray(ws["c1"])
    rs = ws["rs"]
    Jzeta = np.asarray(ws["Jzeta"]); P = np.asarray(ws["P"])
    Z0 = int(mat.scalar_value(ws["Z0"]))
    Bz = np.asarray(ws["Bz"]); Br = np.asarray(ws["Br"])
    zz = np.asarray(ws["zz"]); rr = np.asarray(ws["rr"])
    rm = ws["rm"]

    mplot.close_all()  # close all;
    loaded = mio.load("MRR1d.mat")
    mu0 = mat.scalar(4 * np.pi * 1e-7)
    Be = loaded["Be"]

    fontsize = mat.scalar(25)
    mplot.figure("unit", "normalized", "DefaultAxesFontSize", fontsize,
                 "DefaultAxesFontWeight", "bold", "DefaultAxesLineWidth", 2,
                 "defaulttextinterpreter", "latex",
                 "position", mat.row(0.6, 0.2, 0.2, 0.35), "color", mat.row(1, 1, 1))
    # % [c1,h1]=contour(Z,R,psi,[-0,0],'r--','linewidth',3);
    rx1 = c1[1, 1:-1]
    zx1 = c1[0, 1:-1]
    indx = mat.find(np.abs(zx1) == np.min(np.abs(zx1)))

    mplot.axes("position", mat.row(0.105, 0.125, 0.85, 0.81))  # fig-1
    # `A(:,Z0)` 在 MATLAB 里是 **n×1 的列**，numpy 的 `A[:, k]` 会掉一维——这里的
    # 形状直接决定下面 `Be_2+sum_br_curave` 展开成什么，所以按列切片保留二维。
    Jzetar = -Jzeta[:, Z0 - 1:Z0]
    Pr = P[:, Z0 - 1:Z0]
    J0 = np.max(np.abs(Jzetar))
    P0 = np.max(np.abs(Pr))
    Bz_mid = Bz[:, Z0 - 1:Z0]
    Bz_2 = Bz_mid ** 2 / 2 / mat.scalar_value(mu0)
    pr_2 = Bz_2 + Pr
    Be_2 = (mat.scalar_value(Be) ** 2 / 2 / mat.scalar_value(mu0)
            * np.ones((1, np.size(rr, 1))))

    Br_z = mat.diff(Br, 1, 2) / (mat.el(zz, 2) - mat.el(zz, 1))
    Br_zmid = Br_z[:, Z0 - 1:Z0]
    br_curave = Bz_mid * Br_zmid / mat.scalar_value(mu0)
    sum_br_curave = mat.cumtrapz(rr, br_curave)

    mplot.plot(rr * 100, Pr, "m-", "linewidth", 3)
    mplot.grid_on()
    mplot.hold_on()
    mplot.plot(rr * 100, Bz_2, "-", "linewidth", 3)
    mplot.hold_on()
    mplot.plot(rr * 100, pr_2, "r-", "linewidth", 3)
    mplot.hold_on()
    mplot.plot(rr * 100, Be_2, "b--", "linewidth", 3)
    mplot.hold_on()
    # 1×nR 的行 + nR×1 的列 → 隐式展开成 nR×nR（原文如此，见模块说明）
    mplot.plot(rr * 100, Be_2 + sum_br_curave, "--", "linewidth", 3)
    mplot.hold_on()
    mplot.plot(mat.row(100 * float(mat.scalar_value(rs)), 100 * float(mat.scalar_value(rs))),
               mat.row(0, 5e5), "-.k", "LineWidth", 1.5)
    mplot.hold_on()
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.xlabel("$r\\ (cm), z=0$", "interpreter", "latex", "FontSize", fontsize)
    mplot.ylabel("$P$", "interpreter", "latex", "FontSize", fontsize)
    h = mplot.legend("$P$", "$B_z^2/(2\\mu_0)$", "$P+B_z^2/(2\\mu_0)$",
                     "$B_e^2/(2\\mu_0)$",
                     "$B_e^2/(2\\mu_0)+\\int \\frac{B_z}{\\mu_0}"
                     "(\\frac{\\partial{B_r}}{\\partial{z}})dr$")
    mplot.set_(h, "interpreter", "latex")
    mplot.legend("boxoff")
    mplot.xlim(mat.row(0, 100 * mat.scalar_value(rm)))
    mplot.set_(mplot.gca(), "xtick", mat.colon(0, 2.5, 100 * mat.scalar_value(rm)))
    mplot.ylim(mat.row(0, 4.5e5))
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
               "GridAlpha", 0.25)
    mplot.text("Interpreter", "latex", "String", "$separatrix$",
               "Position", mat.row(100 * float(mat.scalar_value(rs)) - 2, 0.4e5),
               "FontSize", 22)

    for name, value in (("fontsize", fontsize), ("mu0", mu0), ("rx1", rx1),
                        ("zx1", zx1), ("indx", indx), ("Jzetar", Jzetar),
                        ("Pr", Pr), ("J0", J0), ("P0", P0), ("Bz_mid", Bz_mid),
                        ("Bz_2", Bz_2), ("pr_2", pr_2), ("Be_2", Be_2),
                        ("Br_z", Br_z), ("Br_zmid", Br_zmid),
                        ("br_curave", br_curave), ("sum_br_curave", sum_br_curave),
                        ("h", h)):
        ws[name] = value
    for name, value in loaded.items():
        ws[name] = value
