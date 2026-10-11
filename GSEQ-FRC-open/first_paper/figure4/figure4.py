"""figure4.m 的移植：二维 MRR 结果随 δ_s 变化（画 1.0 / 0.6 / 0.18 三条）。

骨架与 `figure3.m` 相同。注意这一份**加载了 6 份数据**（δ_s = 1.0/0.8/0.6/0.4/0.2/0.18）
但**只画其中三份**（1.0/0.6/0.18）——照抄，不要"顺手"把六条都画上。
另外 (b) 面板里文字的位置是 `[0.05 18]`（不是 figure3 的 `14`）。

验收同 §7 的「绘图脚本」一节：能跑通 + 人眼对照论文图。
"""

from __future__ import annotations

from gseqfrc import matlab_compat as mat
from gseqfrc import mio
from gseqfrc import mplot


def run(ws) -> None:
    mplot.close_all()  # close all;clear;clc;

    d1 = mio.load("MRR2d_betas=0.7_delta=1.mat")
    bz_1 = d1["Bz1d"]; h_1 = d1["h1d"]; j_1 = d1["j1d"]; p_1 = d1["pr1d"]
    psi_1 = d1["psi1d"]; rr_1 = d1["rr1d"]; delta_1 = mat.scalar(1)

    d80 = mio.load("MRR2d_betas=0.7_delta=0.8.mat")
    bz_80 = d80["Bz1d"]; h_80 = d80["h1d"]; j_80 = d80["j1d"]; p_80 = d80["pr1d"]
    psi_80 = d80["psi1d"]; rr_80 = d80["rr1d"]; delta_2 = mat.scalar(0.80)

    d60 = mio.load("MRR2d_betas=0.7_delta=0.6.mat")
    bz_60 = d60["Bz1d"]; h_60 = d60["h1d"]; j_60 = d60["j1d"]; p_60 = d60["pr1d"]
    psi_60 = d60["psi1d"]; rr_60 = d60["rr1d"]; delta_3 = mat.scalar(0.60)

    d40 = mio.load("MRR2d_betas=0.7_delta=0.4.mat")
    bz_40 = d40["Bz1d"]; h_40 = d40["h1d"]; j_40 = d40["j1d"]; p_40 = d40["pr1d"]
    psi_40 = d40["psi1d"]; rr_40 = d40["rr1d"]; delta_4 = mat.scalar(0.40)

    d20 = mio.load("MRR2d_betas=0.7_delta=0.2.mat")
    bz_20 = d20["Bz1d"]; h_20 = d20["h1d"]; j_20 = d20["j1d"]; p_20 = d20["pr1d"]
    psi_20 = d20["psi1d"]; rr_20 = d20["rr1d"]; delta_5 = mat.scalar(0.2)

    d18 = mio.load("MRR2d_betas=0.7_delta=0.18.mat")
    bz_18 = d18["Bz1d"]; h_18 = d18["h1d"]; j_18 = d18["j1d"]; p_18 = d18["pr1d"]
    psi_18 = d18["psi1d"]; rr_18 = d18["rr1d"]; delta_6 = mat.scalar(0.18)

    loaded_last = d18
    rr1d = d18["rr1d"]

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
    mplot.plot(rr1d, psi_1, rr1d, psi_60, "k-", rr1d, psi_18, "-",
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
    h = mplot.legend("$\\delta_s=1.0\\delta_{s}^{RR}$",
                     "$\\delta_s=0.6\\delta_{s}^{RR}$",
                     "$\\delta_s=0.18\\delta_{s}^{RR}$", "location", "best")
    mplot.set_(h, "interpreter", "latex", "fontsize", 35)
    mplot.legend("boxoff")

    mplot.axes(p[0])       # axes(p(1))
    mplot.plot(rr1d, p_1, rr1d, p_60, "k-", rr1d, p_18, "-",
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
    mplot.plot(rr1d, j_1, rr1d, j_60, "k-", rr1d, j_18, "-",
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
               "Position", mat.row(0.05, 18), "FontSize", 25)

    mplot.axes(p[3])       # axes(p(4))
    mplot.plot(rr1d, bz_1, rr1d, bz_60, "k-", rr1d, bz_18, "-",
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

    for name, value in (("bz_1", bz_1), ("h_1", h_1), ("j_1", j_1), ("p_1", p_1),
                        ("psi_1", psi_1), ("rr_1", rr_1), ("delta_1", delta_1),
                        ("bz_80", bz_80), ("h_80", h_80), ("j_80", j_80),
                        ("p_80", p_80), ("psi_80", psi_80), ("rr_80", rr_80),
                        ("delta_2", delta_2), ("bz_60", bz_60), ("h_60", h_60),
                        ("j_60", j_60), ("p_60", p_60), ("psi_60", psi_60),
                        ("rr_60", rr_60), ("delta_3", delta_3), ("bz_40", bz_40),
                        ("h_40", h_40), ("j_40", j_40), ("p_40", p_40),
                        ("psi_40", psi_40), ("rr_40", rr_40), ("delta_4", delta_4),
                        ("bz_20", bz_20), ("h_20", h_20), ("j_20", j_20),
                        ("p_20", p_20), ("psi_20", psi_20), ("rr_20", rr_20),
                        ("delta_5", delta_5), ("bz_18", bz_18), ("h_18", h_18),
                        ("j_18", j_18), ("p_18", p_18), ("psi_18", psi_18),
                        ("rr_18", rr_18), ("delta_6", delta_6),
                        ("fontsize", fontsize), ("TightPlot", TightPlot),
                        ("p", p), ("rr1d", rr1d)):
        ws[name] = value
    for name, value in loaded_last.items():
        ws[name] = value
