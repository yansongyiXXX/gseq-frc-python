"""MRR_figure2.m 的移植：第一篇论文 Fig.2（Tuszewski1984 那一组模型对比）。

与 `MRR_figure1.m` 是同一套版式（上面压强、下面 \(J_\theta\)、左上角叠一个小图），
差别在数据与画的东西：

* 读的是 `*_t84.mat`（Tuszewski84）与 `3PE/3pe_datq.mat`、`dat_t84.mat`；
* 画的是 \(P\) 本身（不是 \(P^2\)）、实验点 `dat_t84(1:3:end,:)`；
* `3PE/3pe_datq.mat` 里的 `datq_3pe` 是**复数**（1501×5）。MATLAB 在这里会给
  "忽略 X/Y 的复数部分"的警告、只画实部；移植版的 `mplot.plot` 同样取实部
  （见 `mplot.plot` 的说明），所以图一样、只是不再打那条警告。

验收：§7「绘图脚本」那一节的三条（能跑通 / 结构对 / 人眼比对论文图）。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mio
from gseqfrc import mplot


def run(ws) -> None:
    mplot.close_all()  # close all;clear;clc;
    loaded = {}
    loaded["rr"] = mio.load("./RR/rr_datq_t84.mat")
    loaded["2pe"] = mio.load("./2PE/2pe_datq_t84.mat")
    loaded["mrr1"] = mio.load("./MRR1/mrr1_datq_t84.mat")
    loaded["mrr2"] = mio.load("./MRR2/mrr2_datq_t84.mat")
    # % load('./MRR2/mrr2_datq.mat');
    # % load('./3PE/3pe_datq_t84.mat');
    loaded["3pe"] = mio.load("./3PE/3pe_datq.mat")
    loaded["dat"] = mio.load("./dat_t84.mat")
    datq_rr = loaded["rr"]["datq_rr"]
    datq_2pe = loaded["2pe"]["datq_2pe"]
    datq_mrr0 = loaded["mrr1"]["datq_mrr0"]
    datq_mrr2 = loaded["mrr2"]["datq_mrr2"]
    datq_3pe = loaded["3pe"]["datq_3pe"]
    dat_t84 = loaded["dat"]["dat_t84"]

    fontsize = mat.scalar(30)
    mplot.figure("unit", "normalized", "DefaultAxesFontSize", fontsize,
                 "DefaultAxesFontWeight", "bold", "DefaultAxesLineWidth", 2,
                 "position", mat.row(0.02, 0.2, 0.5, 0.3))
    # %% INPUT TIGHTPLOT PARAMETERS
    TightPlot = mat.struct(
        ColumeNumber=mat.scalar(1),
        RowNumber=mat.scalar(2),
        GapW=mat.scalar(0.05),
        GapH=mat.scalar(0.07),
        MarginsLower=mat.scalar(0.14),
        MarginsUpper=mat.scalar(0.04),
        MarginsLeft=mat.scalar(0.07),
        MarginsRight=mat.scalar(0.02),
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

    mplot.axes(p[0])
    mplot.plot(datq_rr[:, 0], datq_rr[:, 1], "k-",
               datq_mrr0[:, 0], datq_mrr0[:, 1],
               datq_mrr2[:, 0], datq_mrr2[:, 1], "-",
               datq_2pe[:, 0], datq_2pe[:, 1], "-",
               datq_3pe[:, 0], datq_3pe[:, 1], "-",
               "linewidth", 3, "markersize", 9)
    mplot.hold_on()
    mplot.grid_on()
    mplot.plot(dat_t84[0::3, 0], dat_t84[0::3, 1], "r--", "markersize", 8,
               "linewidth", 2.5)
    mplot.hold_on()
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.ylabel("$P$", "interpreter", "latex", "Fontsize", 30)
    mplot.grid_on()
    mplot.xlim(mat.row(0, 1.5))
    mplot.ylim(mat.row(0, 1.05))
    h = mplot.legend("$RR$", "$MRR-1$", "$MRR-2$", "$2PE$", "$3PE$", "$n/n_m$",
                     "location", "best")
    mplot.set_(h, "interpreter", "latex", "FontSize", 25)
    mplot.legend("boxon")
    mplot.text("Interpreter", "latex", "String", "$(a)$",
               "Position", mat.row(0.05, 0.9), "FontSize", 25)
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k", "GridAlpha", 0.25)
    mplot.set_(mplot.gca(), "ytick", mat.colon(0.0, 0.2, 1))
    mplot.set_(mplot.gca(), "xtick", mat.colon(0.0, 0.5, 1.5))
    mplot.xlabel("$r$", "interpreter", "latex", "Fontsize", 35)

    mplot.axes(p[1])
    mplot.plot(datq_rr[:, 0], datq_rr[:, 4], "k-",
               datq_mrr0[:, 0], datq_mrr0[:, 4],
               datq_mrr2[:, 0], datq_mrr2[:, 4], "-",
               datq_2pe[:, 0], datq_2pe[:, 4], "-",
               datq_3pe[:, 0], datq_3pe[:, 4], "-",
               "linewidth", 3, "markersize", 9)
    mplot.grid_on()
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.ylabel("$J_\\theta$", "interpreter", "latex", "Fontsize", 30)
    mplot.hold_on()
    mplot.grid_on()
    mplot.xlabel("$r$", "interpreter", "latex", "Fontsize", 35)
    mplot.set_(mplot.gca(), "ytick", mat.colon(0.0, 2, 10))
    mplot.set_(mplot.gca(), "xtick", mat.colon(0.0, 0.5, 1.5))
    mplot.text("Interpreter", "latex", "String", "$(b)$",
               "Position", mat.row(0.05, 9), "FontSize", 25)
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k", "GridAlpha", 0.25)

    # %% small figure
    h2 = mplot.axes("Position", mat.row(0.23, 0.6, 0.37, 0.14), "FontSize", 10)
    mplot.axes(h2)
    mplot.plot(datq_rr[:, 0], datq_rr[:, 1], "k-",
               datq_mrr0[:, 0], datq_mrr0[:, 1],
               datq_mrr2[:, 0], datq_mrr2[:, 1], "-",
               datq_2pe[:, 0], datq_2pe[:, 1], "-",
               datq_3pe[:, 0], datq_3pe[:, 1], "-",
               "linewidth", 3, "markersize", 9)
    mplot.hold_on()
    mplot.grid_on()
    mplot.plot(dat_t84[0::3, 0], dat_t84[0::3, 1], "r--", "markersize", 8,
               "linewidth", 2.5)
    mplot.hold_on()
    mplot.set_(mplot.gca(), "ytick", mat.colon(0.0, 0.25, 1))
    mplot.set_(mplot.gca(), "xtick", mat.colon(0.1, 0.2, 1))
    mplot.set_(h2, "xlim", mat.row(0.1, 0.301), "FontSize", 20)

    for name, value in (("datq_rr", datq_rr), ("datq_2pe", datq_2pe),
                        ("datq_mrr0", datq_mrr0), ("datq_mrr2", datq_mrr2),
                        ("datq_3pe", datq_3pe), ("dat_t84", dat_t84),
                        ("fontsize", fontsize), ("TightPlot", TightPlot),
                        ("p", p), ("h", h), ("h2", h2)):
        ws[name] = value
    for group in loaded.values():
        for name, value in group.items():
            ws[name] = value
