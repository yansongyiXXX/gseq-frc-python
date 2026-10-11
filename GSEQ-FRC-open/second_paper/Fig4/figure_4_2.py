"""figure_4_2.m 的移植：二维结果与一维 MRR 剖面的对比图（2×2 布局）。

**这是"后处理"脚本**：它不自己算解，而是用**调用方工作区里**求解器留下的变量
（`psi/R/Z/Bz/zw/rrb/zzb/Z0/rr/Jzeta/P/Be/Rs`）——MATLAB 里就是"跑完
`gseq_frc_fixed.m` 之后再 run 这个脚本"。移植版同样从 `ws` 里读这些名字，
算例/测试喂的是求解器的冻结工作区（`frc_baseline/golden/g_fixed_fig4`）。

⚠️ **原脚本按原文跑不完，有两处缺陷**（都照抄，不修）：

1. **`rm` 没有定义**（第 48、63 行 `xlim([0,100*rm])`）：它既不在本脚本里赋值，
   也不在 `gseq_frc_fixed.m` 里——MATLAB 跑到那里会报"未定义函数或变量"。
   移植版同样要求调用方提供 `rm`（测试里按 `figure_4_1.m` 的写法补 0.14 = Rw）。
2. **第 84 行的两条曲线长度不等**：`Bz` 在 Fig4 的工作区里是 `1×201` 的**壁面
   诊断量**（不是二维场），而 `Z(1,:)` 是 `1×257`。MATLAB 实测这种 `plot` 直接报
   "向量长度必须相同"（`plot(1:257,1:201)` 试过）。所以这个脚本**必然在这一行停下**，
   图只画到第 5 个 panel 为止。

也就是说它是"作者当年手边有别的变量才跑得通"的脚本；移植版保留这个行为，
测试里断言它确实在两处分别停下（见 `tests/test_plotting_scripts.py`）。

验收：§7「绘图脚本」那一节的三条。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mio
from gseqfrc import mplot

#: 这个脚本要求调用方工作区里已经有这些（来自 `gseq_frc_fixed.m` 那次运行）
NEEDS_WORKSPACE = ("psi", "Bz", "R", "Z", "zw", "rrb", "zzb", "Z0", "rr",
                   "Jzeta", "P", "Be", "Rs",
                   # 原脚本用了但**没人定义**的那个（见模块说明）
                   "rm")


def run(ws) -> None:
    missing = [name for name in NEEDS_WORKSPACE if name not in ws.keys()]
    if missing:
        raise KeyError(
            f"figure_4_2 需要在求解器的工作区里跑，缺少：{', '.join(missing)}"
            "（原脚本第 48/63 行用到 rm，而它既不在脚本里也不在求解器里——"
            "照抄原文，需要调用方自己提供）")

    psi = np.asarray(ws["psi"])
    Bz = np.asarray(ws["Bz"])
    R, Z = np.asarray(ws["R"]), np.asarray(ws["Z"])
    zw = ws["zw"]
    rrb, zzb = np.asarray(ws["rrb"]), np.asarray(ws["zzb"])
    Z0 = int(mat.scalar_value(ws["Z0"]))
    rr = np.asarray(ws["rr"])
    Jzeta, P = np.asarray(ws["Jzeta"]), np.asarray(ws["P"])
    Be, Rs = ws["Be"], ws["Rs"]
    rm = ws["rm"]

    mplot.close_all()  # close all;clc;
    fontsize = mat.scalar(25)
    mplot.figure("unit", "normalized", "DefaultAxesFontSize", fontsize,
                 "DefaultAxesFontWeight", "bold", "DefaultAxesLineWidth", 2,
                 "defaulttextinterpreter", "latex",
                 "position", mat.row(0.5, 0.2, 0.28, 0.6), "color", mat.row(1, 1, 1))
    Bz_plasma = Bz[-1, :]          # Bz(end,:)
    psi_plasma = psi[-1, :]        # psi(end,:)

    loaded1 = mio.load("MRR1d_beta=0_6_delta=0_4.mat")
    mu0 = mat.scalar(4 * np.pi * 1e-7)
    rr1d = loaded1["rr1d"]
    pr1d = loaded1["pr1d"]
    j1d = loaded1["j1d"]

    mplot.axes("position", mat.row(0.11, 0.75, 0.85, 0.23))  # fig-1
    ido = mat.find(psi == np.min(psi))
    indtmp = psi > 0
    psio = np.where(indtmp, np.nan, psi)
    indtmp = mat.find(psi < 0)
    psii = np.array(psi, copy=True)
    psii[np.asarray(psi < 0)] = np.nan
    mplot.contour(Z, R, psio, 5, "linewidth", 2)
    mplot.hold_on()
    mplot.contour(Z, R, psii, 30, "linewidth", 2)
    mplot.hold_on()
    mplot.colorbar("east")
    c1, h1 = mplot.contour(Z, R, psi, mat.row(0, 0), "r--", "linewidth", 3,
                           return_matrix=True)
    mplot.plot(mat.take(Z, int(mat.el(ido, 1))), mat.take(R, int(mat.el(ido, 1))),
               "mx", "linewidth", 3)
    mplot.hold_on()  # O-point
    mplot.text(0.01 * mat.scalar_value(zw), mat.take(R, int(mat.el(ido, 1))),
               "O point", "fontsize", 13)
    # 原文 `plot([min(c1(1,:)),max(c1(1,:))],[0,0],'r--','linewidth',4)`：c1 的
    # 第 **1 行** 是"层值 + 各顶点 z 坐标"，所以这条红线横跨整个分界面。
    # （2026-09-17 修：原来写成 `mat.take(c1, 1)`，那是线性索引取第 1 个元素，
    #  线退化成一点，看不见。）
    mplot.plot(mat.row(np.min(np.asarray(c1)[0, :]), np.max(np.asarray(c1)[0, :])),
               mat.row(0, 0), "r--", "linewidth", 4)
    mplot.hold_on()

    zx1 = np.asarray(c1)[0, 1:-1]
    rx1 = np.asarray(c1)[1, 1:-1]
    zx_no = np.size(zx1, 0)
    indx = mat.find(np.abs(zx1) == np.min(np.abs(zx1)))
    strp = "p(\\psi)=\\beta_sexp(\\alpha\\psi)*(\\sigma\\psi+1)^2"
    mplot.xlabel("$z [m]$", "interpreter", "latex", "FontSize", 25)
    mplot.ylabel("$r [m]$", "interpreter", "latex", "FontSize", fontsize)
    mplot.set_(mplot.gca(), "ytick", mat.colon(0, 0.05, 0.15))
    mplot.set_(mplot.gca(), "xtick", mat.colon(-0.75, 0.25, 0.75))
    # `min(psi(:,Z0));` 原句是裸表达式（只把结果丢进 ans），这里照抄成无副作用的计算
    _ = np.min(psi[:, Z0 - 1])
    rs = mat.el(rx1, int(mat.el(indx, 1)))
    mplot.plot(zzb, rrb, "k-", "linewidth", 3)

    w = None
    for pos in (mat.row(-0.75, 0.147, 0.1, 0.006),
                mat.row(-0.60, 0.147, 1.2, 0.006),
                mat.row(0.65, 0.147, 0.1, 0.006)):
        h = mplot.rectangle("Position", pos, "FaceColor", mat.row(0.5, 0.5, 0.5),
                            "linewidth", 1)
        w = h.LineWidth
        h.LineWidth = 3
        mplot.hold_on()
    mplot.text("Interpreter", "latex", "String", "$(a)$",
               "Position", mat.row(-0.7, 0.12), "color", "k", "FontSize", fontsize)

    mplot.axes("position", mat.row(0.12, 0.07, 0.38, 0.26))  # fig-4
    Jzetar = -Jzeta[:, Z0 - 1]
    Pr = P[:, Z0 - 1]
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
    mplot.ylim(mat.row(0, 4.5e5))
    mplot.set_(mplot.gca(), "ytick", mat.colon(0, 1e5, 4e5))
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
               "GridAlpha", 0.25)
    mplot.text("Interpreter", "latex", "String", "$Separatrix$", "Position",
               mat.row(100 * mat.scalar_value(rs) - 2, 1e5), "FontSize", 20)
    mplot.text("Interpreter", "latex", "String", "$(d)$",
               "Position", mat.row(1, 3.9e5), "color", "k", "FontSize", fontsize)

    mplot.axes("position", mat.row(0.58, 0.07, 0.38, 0.26))  # fig-5
    Jzetar = -Jzeta[:, Z0 - 1]
    J0 = np.max(-Jzeta[:, Z0 - 1])
    mplot.plot(rr * 100, -Jzetar, "m", rr1d * Rs * 100,
               j1d * (Be / (2 * mu0 * Rs)), "--", "linewidth", 3)
    mplot.grid_on()
    mplot.hold_on()
    mplot.plot(mat.row(100 * float(mat.scalar_value(rs)), 100 * float(mat.scalar_value(rs))),
               mat.row(0, 5e7), "-.k", "LineWidth", 1.5)
    mplot.hold_on()
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.xlabel("$r\\ [cm], z=0$", "interpreter", "latex", "FontSize", fontsize)
    mplot.ylabel("$J_{\\theta}$", "interpreter", "latex", "FontSize", fontsize)
    h = mplot.legend("$2D$", "$1D$")
    mplot.set_(h, "interpreter", "latex")
    mplot.legend("boxoff")
    mplot.xlim(mat.row(0, 100 * mat.scalar_value(rm)))
    mplot.set_(mplot.gca(), "xtick", mat.colon(0, 2.5, 100 * mat.scalar_value(rm)))
    mplot.ylim(mat.row(0, 5.1e7))
    mplot.set_(mplot.gca(), "ytick", mat.colon(0, 1e7, 4e7))
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
               "GridAlpha", 0.25)
    mplot.set_(mplot.gcf(), "PaperPositionMode", "auto")
    mplot.text("Interpreter", "latex", "String", "$(e)$",
               "Position", mat.row(1, 4.5e7), "color", "k", "FontSize", fontsize)

    loaded2 = mio.load("psib_paper_beta=0_6_delta=0_4.mat")  # read data
    zb = loaded2["zb"]
    psib = loaded2["psib"]
    Bz_wall = loaded2["Bz"]
    mplot.axes("position", mat.row(0.12, 0.41, 0.38, 0.26))  # fig-2
    mplot.plot(zb, psib, "r-", np.asarray(Z)[0, :], psi_plasma, "g--",
               "linewidth", 3)
    mplot.grid_on()
    mplot.hold_on()
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.xlabel("$z [m]$", "interpreter", "latex", "FontSize", fontsize)
    mplot.ylabel("$\\psi \\ at \\ wall$", "interpreter", "latex", "FontSize", fontsize)
    mplot.xlim(mat.row(-0.75, 0.75))
    mplot.set_(mplot.gca(), "xtick", mat.colon(-0.6, 0.4, 0.6))
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
               "GridAlpha", 0.25)
    mplot.set_(mplot.gcf(), "PaperPositionMode", "auto")
    h = mplot.legend("$\\mathbf{w/o \\ plasma}$", "$\\mathbf{w \\ plasma}$")
    mplot.set_(h, "interpreter", "latex")
    mplot.legend("boxoff")
    mplot.text("Interpreter", "latex", "String", "$(b)$",
               "Position", mat.row(-0.7, 0.035), "color", "k", "FontSize", fontsize)
    mplot.ylim(mat.row(0, 0.04))
    mplot.set_(mplot.gca(), "ytick", mat.colon(0, 0.01, 0.04))

    mplot.axes("position", mat.row(0.58, 0.41, 0.38, 0.26))  # fig-3
    mplot.plot(zb, Bz_wall, "r-", np.asarray(Z)[0, :], Bz_plasma, "g--",
               "linewidth", 3)
    mplot.grid_on()
    mplot.hold_on()
    mplot.ylabel("$B_{z} \\ at \\ wall$", "interpreter", "latex", "FontSize", fontsize)
    mplot.xlabel("$z [m]$", "interpreter", "latex", "FontSize", fontsize)
    mplot.xlim(mat.row(-0.75, 0.75))
    mplot.set_(mplot.gca(), "xtick", mat.colon(-0.6, 0.4, 0.6))
    mplot.set_(mplot.gca(), "ytick", mat.colon(0, 2, 10))
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
               "GridAlpha", 0.25)
    h = mplot.legend("$\\mathbf{w/o \\ plasma}$", "$\\mathbf{w \\ plasma}$")
    mplot.set_(h, "interpreter", "latex")
    mplot.legend("boxoff")
    mplot.text("Interpreter", "latex", "String", "$(c)$",
               "Position", mat.row(-0.7, 9), "color", "k", "FontSize", fontsize)

    for name, value in (("fontsize", fontsize), ("Bz_plasma", Bz_plasma),
                        ("psi_plasma", psi_plasma), ("mu0", mu0), ("ido", ido),
                        ("indtmp", indtmp), ("psio", psio), ("psii", psii),
                        ("c1", c1), ("h1", h1), ("zx1", zx1), ("rx1", rx1),
                        ("zx_no", mat.scalar(zx_no)), ("indx", indx),
                        ("strp", strp), ("rs", rs), ("w", w), ("Jzetar", Jzetar),
                        ("Pr", Pr), ("J0", J0), ("zb", zb), ("psib", psib),
                        ("Bz", Bz_wall), ("rm", rm)):
        ws[name] = value
    for name, value in loaded1.items():
        ws[name] = value
    for name, value in loaded2.items():
        ws[name] = value
