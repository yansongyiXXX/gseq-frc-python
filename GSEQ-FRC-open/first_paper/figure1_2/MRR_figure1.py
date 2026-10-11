"""MRR_figure1.m 的移植：第一篇论文 Fig.1（Ikeyama09 那一组模型对比）。

**自包含**：数据是各子目录里的成品 `.mat`（`RR/rr_datq_i09.mat`、`2PE/…`、
`MRR1/…`、`MRR2/…`、`3PE/…` 与 `dat_i09.mat`），每份都是 `datq_*=[rr;Pr;psi;Bz;Jt].'`
的 5 列矩阵。脚本：

* 用 `tight_subplot(1,2,...)` 排上下两格：上面画 \(P^2\)（6 条曲线 + Ikeyama09 实验点），
  下面画 \(J_\theta\)；
* 再在左上角叠一个小图（`axes('Position',[...])`）放大 \(r\in[0.3,0.4]\) 那一段；
* 自己按解析式算了一条 `s-f`（shape factor）曲线 `p_s_f` 作对比。

实验点那一列要按 `dat_i09(:,1)/5.61`、`sqrt(dat_i09(:,2))` 换算（原文如此）。

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
    loaded["rr"] = mio.load("./RR/rr_datq_i09.mat")
    loaded["2pe"] = mio.load("./2PE/2pe_datq_i09.mat")
    loaded["mrr1"] = mio.load("./MRR1/mrr1_datq_i09.mat")
    loaded["mrr2"] = mio.load("./MRR2/mrr2_datq_i09.mat")
    # % load('./MRR2/mrr2_datq.mat');
    loaded["3pe"] = mio.load("./3PE/3pe_datq_i09.mat")
    loaded["dat"] = mio.load("./dat_i09.mat")
    datq_rr = loaded["rr"]["datq_rr"]
    datq_2pe = loaded["2pe"]["datq_2pe"]
    datq_mrr0 = loaded["mrr1"]["datq_mrr0"]
    datq_mrr2 = loaded["mrr2"]["datq_mrr2"]
    datq_3pe = loaded["3pe"]["datq_3pe"]
    diag_data = loaded["dat"]["diag_data"]
    dat_i09 = diag_data
    # `dat_i09(:,1)=dat_i09(:,1)/5.61;` / `dat_i09(:,2)=sqrt(dat_i09(:,2));`
    dat_i09 = np.array(dat_i09, copy=True)
    dat_i09[:, 0] = dat_i09[:, 0] / 5.61
    dat_i09[:, 1] = np.sqrt(dat_i09[:, 1])

    c = 0.58
    delta = 0.0051
    b0 = 1
    a = 0.0561
    rr = mat.colon(0, 0.01, 1) * 2 * a
    u = 2.0 * rr ** 2 / a ** 2 - 1
    bi = 0.5 * c * b0 * (u + u ** 3)
    be = b0 * (1 - (1 - c) * np.exp(-(rr - a) / delta))
    b = bi * (u <= 1) + be * (u > 1)

    p_s_f = 1 - b ** 2 / b0 ** 2
    # % Jr=-(diff(p_s_f))./diff(psi).*(rr(2:end));Jr=[Jr(1);Jr];

    fontsize = mat.scalar(30)
    mplot.figure("unit", "normalized", "DefaultAxesFontSize", fontsize,
                 "DefaultAxesFontWeight", "bold", "DefaultAxesLineWidth", 2,
                 "position", mat.row(0.02, 0.2, 0.5, 0.3))

    # %% INPUT TIGHTPLOT PARAMETERS
    TightPlot = mat.struct(
        ColumeNumber=mat.scalar(1),   # 子图列数
        RowNumber=mat.scalar(2),      # 子图行数
        GapW=mat.scalar(0.05),        # 子图之间的左右间距
        GapH=mat.scalar(0.07),        # 子图之间的上下间距
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
    mplot.plot(datq_rr[:, 0], datq_rr[:, 1] ** 2, "k-",
               datq_mrr0[:, 0], datq_mrr0[:, 1] ** 2,
               datq_mrr2[:, 0], datq_mrr2[:, 1] ** 2, "-",
               datq_2pe[:, 0], datq_2pe[:, 1] ** 2, "-",
               datq_3pe[:, 0], datq_3pe[:, 1] ** 2, "-",
               rr / a, p_s_f ** 2, "-",
               "linewidth", 3, "markersize", 9)
    mplot.hold_on()
    mplot.grid_on()
    mplot.plot(dat_i09[0::1, 0], dat_i09[0::1, 1] ** 2, "ro-", "markersize", 8,
               "linewidth", 2.5)
    mplot.hold_on()
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.ylabel("$P^2$", "interpreter", "latex", "Fontsize", 30)
    mplot.grid_on()
    mplot.xlim(mat.row(0, 1.5))
    mplot.ylim(mat.row(0, 1.05))
    hc = mplot.legend("$\\mathrm{RR}$", "$\\mathrm{MRR-1}$", "$\\mathrm{MRR-2}$",
                      "$\\mathrm{2PE}$", "$\\mathrm{3PE}$", "$\\mathrm{s-f}$",
                      "$\\mathrm{i/i_0}$", "location", "best")
    mplot.set_(hc, "interpreter", "latex", "FontSize", 25)
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
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.ylabel("$J_\\theta$", "interpreter", "latex", "Fontsize", 30)
    mplot.hold_on()
    mplot.grid_on()
    mplot.xlabel("$r$", "interpreter", "latex", "Fontsize", 35)
    # % text('Interpreter','latex','String',0.05,7.0,'(c)','Fontsize',20);
    mplot.text("Interpreter", "latex", "String", "$(b)$",
               "Position", mat.row(0.05, 7.0), "FontSize", 25)
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k", "GridAlpha", 0.25)
    mplot.set_(mplot.gca(), "ytick", mat.colon(0.0, 2, 8))
    mplot.set_(mplot.gca(), "xtick", mat.colon(0.0, 0.5, 1.5))

    # %% small figure
    # % 在原图上叠了一个新的小图窗
    h2 = mplot.axes("Position", mat.row(0.23, 0.6, 0.37, 0.14), "FontSize", 10)
    mplot.axes(h2)
    mplot.plot(datq_rr[:, 0], datq_rr[:, 1] ** 2, "k-",
               datq_mrr0[:, 0], datq_mrr0[:, 1] ** 2,
               datq_mrr2[:, 0], datq_mrr2[:, 1] ** 2, "-",
               datq_2pe[:, 0], datq_2pe[:, 1] ** 2, "-",
               datq_3pe[:, 0], datq_3pe[:, 1] ** 2, "-",
               rr / a, p_s_f ** 2, "-", "linewidth", 3)
    mplot.hold_on()
    mplot.plot(dat_i09[0::1, 0], dat_i09[0::1, 1] ** 2, "ro-", "markersize", 8,
               "linewidth", 2.5)
    mplot.hold_on()
    mplot.set_(mplot.gca(), "ytick", mat.colon(0.0, 0.25, 1))
    mplot.set_(mplot.gca(), "xtick", mat.colon(0.0, 0.1, 1))
    mplot.set_(h2, "xlim", mat.row(0.3, 0.4), "FontSize", 20)

    for name, value in (("datq_rr", datq_rr), ("datq_2pe", datq_2pe),
                        ("datq_mrr0", datq_mrr0), ("datq_mrr2", datq_mrr2),
                        ("datq_3pe", datq_3pe), ("diag_data", diag_data),
                        ("dat_i09", dat_i09), ("c", mat.scalar(c)),
                        ("delta", mat.scalar(delta)), ("b0", mat.scalar(b0)),
                        ("a", mat.scalar(a)), ("rr", rr), ("u", u), ("bi", bi),
                        ("be", be), ("b", b), ("p_s_f", p_s_f),
                        ("fontsize", fontsize), ("TightPlot", TightPlot),
                        ("p", p), ("hc", hc), ("h2", h2)):
        ws[name] = value
    for group in loaded.values():
        for name, value in group.items():
            ws[name] = value
