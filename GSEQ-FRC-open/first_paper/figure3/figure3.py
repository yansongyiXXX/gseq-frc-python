"""figure3.m 的移植：把三份一维 MRR 结果画成一张四联图。

**这是"绘图脚本"那一批**（`figure*`/`plt_*`/`plot_*`）：原程序没有冻结算例、
也没有基准，所以验收只有两条——能跑通、画出来的东西与原脚本一致（人眼对照论文）。
详见 PORTING.md §7 的「绘图脚本」一节。

数据是随论文提供的成品（`MRR1d_betas=0.7_delta=*.mat`，仓库里**没有**生成它们的
脚本），脚本只做 `load` → 画图。`tight_subplot` 排 2×2，`axes(p(k))` 逐个切到
子图再画，顺序是 (a) 左上、(b) 右上、(c) 左下、(d) 右下（注意源码里落的顺序是
先 p(3) 再 p(1)、p(2)、p(4)，照抄）。
"""

from __future__ import annotations

from gseqfrc import matlab_compat as mat
from gseqfrc import mio
from gseqfrc import mplot


def run(ws) -> None:
    mplot.close_all()  # close all;clear;clc;

    # load('MRR1d_betas=0.7_delta=1.mat'); ...
    d1 = mio.load("MRR1d_betas=0.7_delta=1.mat")
    bz_1 = d1["Bz1d"]; h_1 = d1["h1d"]; j_1 = d1["j1d"]; p_1 = d1["pr1d"]
    psi_1 = d1["psi1d"]; rr_1 = d1["rr1d"]; delta_1 = mat.scalar(1)

    d60 = mio.load("MRR1d_betas=0.7_delta=0.6.mat")
    bz_60 = d60["Bz1d"]; h_60 = d60["h1d"]; j_60 = d60["j1d"]; p_60 = d60["pr1d"]
    psi_60 = d60["psi1d"]; rr_60 = d60["rr1d"]; delta_9 = mat.scalar(0.60)

    d22 = mio.load("MRR1d_betas=0.7_delta=0.22.mat")
    bz_22 = d22["Bz1d"]; h_22 = d22["h1d"]; j_22 = d22["j1d"]; p_22 = d22["pr1d"]
    psi_22 = d22["psi1d"]; rr_22 = d22["rr1d"]; delta_19 = mat.scalar(0.22)

    # MATLAB 的 `load` 会把文件里的变量直接放进工作区；三次 load 之后留下的是
    # 最后一份的值（脚本没有再用它们，但工作区里确实有）
    loaded_last = d22
    rr1d = d22["rr1d"]

    fontsize = mat.scalar(30)
    mplot.figure("unit", "normalized", "DefaultAxesFontSize", fontsize,
                 "DefaultAxesFontWeight", "bold", "DefaultAxesLineWidth", 2,
                 "position", mat.row(0.02, 0.2, 0.5, 0.4))

    # %% INPUT TIGHTPLOT PARAMETERS
    TightPlot = mat.struct(
        ColumeNumber=mat.scalar(2),   # 子图列数
        RowNumber=mat.scalar(2),      # 子图行数
        GapW=mat.scalar(0.06),        # 子图之间的左右间距
        GapH=mat.scalar(0.075),       # 子图之间的上下间距
        MarginsLower=mat.scalar(0.105),
        MarginsUpper=mat.scalar(0.04),
        MarginsLeft=mat.scalar(0.07),
        MarginsRight=mat.scalar(0.03),
    )
    p = mplot.tight_subplot(
        mat.scalar_value(TightPlot["ColumeNumber"][0, 0]),
        mat.scalar_value(TightPlot["RowNumber"][0, 0]),
        mat.row(mat.scalar_value(TightPlot["GapH"][0, 0]),
                mat.scalar_value(TightPlot["GapW"][0, 0])),
        mat.row(mat.scalar_value(TightPlot["MarginsLower"][0, 0]),
                mat.scalar_value(TightPlot["MarginsUpper"][0, 0])),
        mat.row(mat.scalar_value(TightPlot["MarginsLeft"][0, 0]),
                mat.scalar_value(TightPlot["MarginsRight"][0, 0])),
    )

    mplot.axes(p[2])       # axes(p(3))
    mplot.plot(rr1d, psi_1, rr1d, psi_60, "k-", rr1d, psi_22, "-",
               "linewidth", 3, "markersize", 9)
    mplot.hold_on()
    mplot.grid_on()
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.ylabel("$\\psi$", "interpreter", "latex", "Fontsize", 30)
    mplot.xlabel("$r$", "interpreter", "latex", "Fontsize", 35)
    mplot.text("Interpreter", "latex", "FontWeight", "bold", "String", "$(c)$",
               "Position", mat.row(0.05, 0.53), "FontSize", 25)
    mplot.xlim(mat.row(0, 1.5))
    mplot.set_(mplot.gca(), "ytick", mat.colon(-0.2, 0.2, 0.6))
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
               "GridAlpha", 0.25)
    h = mplot.legend("$\\delta_s=1.0\\delta_{s}^{RR}$",
                     "$\\delta_s=0.6\\delta_{s}^{RR}$",
                     "$\\delta_s=0.22\\delta_{s}^{RR}$", "location", "best")
    mplot.set_(h, "interpreter", "latex", "fontsize", 35)
    mplot.legend("boxoff")

    mplot.axes(p[0])       # axes(p(1))
    mplot.plot(rr1d, p_1, rr1d, p_60, "k-", rr1d, p_22, "-",
               "linewidth", 3, "markersize", 9)
    mplot.hold_on()
    mplot.grid_on()
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.ylabel("$P$", "interpreter", "latex", "Fontsize", 30)
    mplot.ylim(mat.row(0, 1.05))
    mplot.set_(mplot.gca(), "ytick", mat.colon(0.0, 0.2, 1.0))
    mplot.set_(mplot.gca(), "xtick", mat.colon(0.0, 0.5, 2))
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
               "GridAlpha", 0.25)
    mplot.text("Interpreter", "latex", "String", "$(a)$",
               "Position", mat.row(0.05, 0.9), "FontSize", 25)

    mplot.axes(p[1])       # axes(p(2))
    mplot.plot(rr1d, j_1, rr1d, j_60, "k-", rr1d, j_22, "-",
               "linewidth", 3, "markersize", 9)
    mplot.hold_on()
    mplot.grid_on()
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.ylabel("$J_\\theta$", "interpreter", "latex", "Fontsize", 30)
    mplot.grid_on()
    mplot.set_(mplot.gca(), "ytick", mat.colon(0, 5, 20))
    mplot.set_(mplot.gca(), "xtick", mat.colon(0.0, 0.5, 2))
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
               "GridAlpha", 0.25)
    mplot.text("Interpreter", "latex", "String", "$(b)$",
               "Position", mat.row(0.05, 14), "FontSize", 25)

    mplot.axes(p[3])       # axes(p(4))
    mplot.plot(rr1d, bz_1, rr1d, bz_60, "k-", rr1d, bz_22, "-",
               "linewidth", 3, "markersize", 9)
    mplot.hold_on()
    mplot.grid_on()
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.ylabel("$B_z$", "interpreter", "latex", "Fontsize", 30)
    mplot.xlabel("$r$", "interpreter", "latex", "Fontsize", 35)
    mplot.ylim(mat.row(-0.6, 1.05))
    mplot.set_(mplot.gca(), "ytick", mat.colon(-0.5, 0.5, 1))
    mplot.set_(mplot.gca(), "xtick", mat.colon(0.0, 0.5, 2))
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
               "GridAlpha", 0.25)
    mplot.text("Interpreter", "latex", "String", "$(d)$",
               "Position", mat.row(0.05, 0.85), "FontSize", 25)

    # 工作区（这一批没有基准，仍然按"脚本里出现过的变量"写出去，便于人工核对）
    for name, value in (("bz_1", bz_1), ("h_1", h_1), ("j_1", j_1), ("p_1", p_1),
                        ("psi_1", psi_1), ("rr_1", rr_1), ("delta_1", delta_1),
                        ("bz_60", bz_60), ("h_60", h_60), ("j_60", j_60),
                        ("p_60", p_60), ("psi_60", psi_60), ("rr_60", rr_60),
                        ("delta_9", delta_9), ("bz_22", bz_22), ("h_22", h_22),
                        ("j_22", j_22), ("p_22", p_22), ("psi_22", psi_22),
                        ("rr_22", rr_22), ("delta_19", delta_19),
                        ("fontsize", fontsize), ("TightPlot", TightPlot),
                        ("p", p), ("rr1d", rr1d)):
        ws[name] = value
    for name, value in loaded_last.items():
        ws[name] = value
