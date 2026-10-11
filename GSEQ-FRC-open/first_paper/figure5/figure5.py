"""figure5.m 的移植：三份一维 MRR 结果（β_s = 0.4/0.6/0.8）画成四联图。

骨架与 `figure3.m` 完全相同（`tight_subplot` 2×2 + `axes(p(k))` 逐个切子图），
差别只有数据文件、变量名与图例文字——按 PORTING.md §1 逐行照抄，不抽公共实现。
验收同 §7 的「绘图脚本」一节：能跑通 + 人眼对照论文图。
"""

from __future__ import annotations

from gseqfrc import matlab_compat as mat
from gseqfrc import mio
from gseqfrc import mplot


def run(ws) -> None:
    mplot.close_all()  # close all;clear;clc;

    d4 = mio.load("MRR1d_betas=0.4_delta=0.148.mat")
    bz_4 = d4["Bz1d"]; h_4 = d4["h1d"]; j_4 = d4["j1d"]; p_4 = d4["pr1d"]
    psi_4 = d4["psi1d"]; rr_4 = d4["rr1d"]; delta_1 = mat.scalar(0.148)

    d6 = mio.load("MRR1d_betas=0.6_delta=0.148.mat")
    bz_6 = d6["Bz1d"]; h_6 = d6["h1d"]; j_6 = d6["j1d"]; p_6 = d6["pr1d"]
    psi_6 = d6["psi1d"]; rr_6 = d6["rr1d"]; delta_3 = mat.scalar(0.148)

    d8 = mio.load("MRR1d_betas=0.8_delta=0.148.mat")
    bz_8 = d8["Bz1d"]; h_8 = d8["h1d"]; j_8 = d8["j1d"]; p_8 = d8["pr1d"]
    psi_8 = d8["psi1d"]; rr_8 = d8["rr1d"]; delta_5 = mat.scalar(0.148)

    loaded_last = d8
    rr1d = d8["rr1d"]

    fontsize = mat.scalar(30)
    mplot.figure("unit", "normalized", "DefaultAxesFontSize", fontsize,
                 "DefaultAxesFontWeight", "bold", "DefaultAxesLineWidth", 2,
                 "position", mat.row(0.02, 0.2, 0.5, 0.4))

    TightPlot = mat.struct(
        ColumeNumber=mat.scalar(2), RowNumber=mat.scalar(2),
        GapW=mat.scalar(0.06), GapH=mat.scalar(0.075),
        MarginsLower=mat.scalar(0.105), MarginsUpper=mat.scalar(0.04),
        MarginsLeft=mat.scalar(0.07), MarginsRight=mat.scalar(0.03),
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
    mplot.plot(rr1d, psi_4, rr1d, psi_6, "k-", rr1d, psi_8, "-",
               "linewidth", 3, "markersize", 9)
    mplot.hold_on()
    mplot.grid_on()
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.ylabel("$\\psi$", "interpreter", "latex", "Fontsize", 30)
    mplot.xlabel("$r$", "interpreter", "latex", "Fontsize", 35)
    mplot.text("Interpreter", "latex", "String", "$(c)$",
               "Position", mat.row(0.05, 0.53), "FontSize", 25)
    mplot.xlim(mat.row(0, 1.5))
    mplot.set_(mplot.gca(), "ytick", mat.colon(-0.2, 0.2, 0.6))
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
               "GridAlpha", 0.25)
    h = mplot.legend("$\\beta_s=0.4$", "$\\beta_s=0.6$", "$\\beta_s=0.8$",
                     "location", "best")
    mplot.set_(h, "interpreter", "latex", "fontsize", 35)
    mplot.legend("boxoff")

    mplot.axes(p[0])       # axes(p(1))
    mplot.plot(rr1d, p_4, rr1d, p_6, "k-", rr1d, p_8, "-",
               "linewidth", 3, "markersize", 9)
    mplot.hold_on()
    mplot.grid_on()
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.ylabel("$P$", "interpreter", "latex", "Fontsize", 30)
    mplot.ylim(mat.row(0, 1.05))
    mplot.set_(mplot.gca(), "ytick", mat.colon(0.0, 0.2, 1))
    mplot.set_(mplot.gca(), "xtick", mat.colon(0.0, 0.5, 2))
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
               "GridAlpha", 0.25)
    mplot.text("Interpreter", "latex", "String", "$(a)$",
               "Position", mat.row(0.05, 0.9), "FontSize", 25)

    mplot.axes(p[1])       # axes(p(2))
    mplot.plot(rr1d, j_4, rr1d, j_6, "k-", rr1d, j_8, "-",
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
    mplot.plot(rr1d, bz_4, rr1d, bz_6, "k-", rr1d, bz_8, "-",
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

    for name, value in (("bz_4", bz_4), ("h_4", h_4), ("j_4", j_4), ("p_4", p_4),
                        ("psi_4", psi_4), ("rr_4", rr_4), ("delta_1", delta_1),
                        ("bz_6", bz_6), ("h_6", h_6), ("j_6", j_6), ("p_6", p_6),
                        ("psi_6", psi_6), ("rr_6", rr_6), ("delta_3", delta_3),
                        ("bz_8", bz_8), ("h_8", h_8), ("j_8", j_8), ("p_8", p_8),
                        ("psi_8", psi_8), ("rr_8", rr_8), ("delta_5", delta_5),
                        ("fontsize", fontsize), ("TightPlot", TightPlot),
                        ("p", p), ("rr1d", rr1d)):
        ws[name] = value
    for name, value in loaded_last.items():
        ws[name] = value
