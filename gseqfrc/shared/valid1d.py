"""valid1d.m 的逐行移植（**两份副本合并后**的唯一实现）。

源文件（两份差 7 处、+28/-11 行，**差的全部是绘图风格**）：

    gseq-main/GSEQ-FRC-open/first_paper/figure1_2/valid1d/valid1d.m   （普通 subplot 风格）
    gseq-main/GSEQ-FRC-open/first_paper/figureB1/valid1d.m            （tight_subplot + LaTeX 风格）

用 Hill 涡的解析 `psi` 校验一维平衡：给定一组合 (E, betas)，直接算 psi/Br/Bz/dBrdz，
再沿 r 累加得到 Pm，最后把 (Pm-Pm0)/Pm0 画出来。**物理部分两份逐字相同**，
差别只有：字号、子图排布（tight_subplot）、标签走 LaTeX、网格与 xtick 单独设置、
线宽 2→3，以及**末尾的打印块在 figureB1 那份里被注释掉了**（所以那个算例不产出
图片、工作区里也没有 `screenposition`）。

2026-09-17 副本合并（PORTING.md §8 的规则 2）：差异提成显式参数 `style`，
两种取值各自原样保留。程序树里两个原位置各留一个薄壳带上自己的取值。

工作区变量也按副本分开写：`figureB1` 多 `fontsize`/`TightPlot`、少 `screenposition`
（这正是两份冻结基准的差别，不是移植自由度）。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mplot

#: `style` 的默认值（figure1_2 那份）
DEFAULT_STYLE = "figure1_2"


def run(ws, style=DEFAULT_STYLE) -> None:
    mplot.close_all()  # close all;clear;clc;

    Be = mat.scalar(1.0)
    a = mat.scalar(1)  # a=rs
    dat = np.array([
        [1, 0.1],
        [1, 0.5],
        [1, 0.9],
        [2, 0.1],
        [2, 0.9],
        [3, 0.5],
        [3, 0.9],
        [4, 0.5],
    ], dtype=float)

    if style == "figureB1":
        fontsize = mat.scalar(30)
        mplot.figure("unit", "normalized", "DefaultAxesFontSize", fontsize,
                     "DefaultAxesFontWeight", "bold", "DefaultAxesLineWidth", 2,
                     "position", mat.row(0.02, 0.2, 0.25, 0.3))

        # %% INPUT TIGHTPLOT PARAMETERS
        TightPlot = mat.struct(
            ColumeNumber=mat.scalar(1),   # 子图列数
            RowNumber=mat.scalar(1),      # 子图行数
            GapW=mat.scalar(0.09),        # 子图之间的左右间距
            GapH=mat.scalar(0.05),        # 子图之间的上下间距
            MarginsLower=mat.scalar(0.15),
            MarginsUpper=mat.scalar(0.04),
            MarginsLeft=mat.scalar(0.15),
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
    else:
        mplot.figure("unit", "normalized", "DefaultAxesFontSize", 13,
                     "DefaultAxesFontWeight", "bold", "DefaultAxesLineWidth", 2,
                     "position", mat.row(0.02, 0.45, 0.4, 0.45))

    nj = int(np.shape(dat)[0])
    strs = []
    for jd in mat.mrange(1, nj):
        # %     E=1; % elongation
        # %     betas=0.3;

        E = mat.scalar(mat.ij(dat, jd, 1))
        betas = mat.scalar(mat.ij(dat, jd, 2))

        b = a * E
        z = mat.scalar(0)

        dr = mat.scalar(0.01)
        rr = mat.colon(0, dr, 1.0)

        B0 = np.sqrt(1 - betas)
        psi = B0 / 2 * rr ** 2 * (1 - rr ** 2 / a ** 2 - z ** 2 / b ** 2)
        Br = B0 * rr * z / b ** 2
        Bz = B0 * (1 - 2 * rr ** 2 / a ** 2 - z ** 2 / b ** 2)
        dBrdz = B0 * rr / b ** 2

        Pm = 1 + mat.cumsum(2 * Bz * dBrdz * dr)

        if style == "figureB1":
            strs.append(f"$E=${mat.num2str(E)}$, \\beta_s=${mat.num2str(betas)}")
        else:
            strs.append(f"E={mat.num2str(E)}, \\beta_s={mat.num2str(betas)}")

        if style == "figureB1":
            mplot.plot(rr, (Pm - mat.el(Pm, 1)) / (mat.el(Pm, 1)), "linewidth", 3)
        else:
            mplot.plot(rr, (Pm - mat.el(Pm, 1)) / (mat.el(Pm, 1)), "linewidth", 2)
        mplot.hold_on()
        if style == "figureB1":
            mplot.grid_on()
        # %     ylim([0,1.3]);

    str_cell = mat.cellstr(strs)
    if style == "figureB1":
        mplot.legend(str_cell, "location", "best", "interpreter", "latex")
        mplot.legend("boxoff")
        mplot.ylabel("$(P_m-P_{m0})/P_{m0}$", "interpreter", "latex", "Fontsize", 25)
        mplot.xlabel("$r$", "interpreter", "latex", "Fontsize", 35)
        mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
                   "GridAlpha", 0.25)
        mplot.set_(mplot.gca(), "xtick", mat.colon(0, 0.2, 1))

        # % set(gcf,'Units','inches');
        # % screenposition = get(gcf,'Position');
        # % set(gcf,'PaperPosition',[0 0 screenposition(3:4)],...
        # %   'PaperSize',[screenposition(3:4)]);
        # % print(gcf,'-dpng',['valid_1d.png']);
    else:
        mplot.legend(str_cell, "location", "best")
        mplot.legend("boxoff")
        mplot.xlabel("r/r_s")
        mplot.ylabel("(P_m-P_{m0})/P_{m0}")

        mplot.set_(mplot.gcf(), "Units", "inches")
        screenposition = mplot.get_(mplot.gcf(), "Position")
        mplot.set_(mplot.gcf(), "PaperPosition",
                   mat.row(0, 0, screenposition[2], screenposition[3]),
                   "PaperSize", mat.row(screenposition[2], screenposition[3]))
        mplot.print_fig(mplot.gcf(), "-dpng", "valid_1d.png")

    ws.Be = Be
    ws.a = a
    ws.dat = dat
    if style == "figureB1":
        ws.fontsize = fontsize
        ws.TightPlot = TightPlot
    ws.nj = mat.scalar(nj)
    ws.jd = mat.scalar(jd)
    ws.E = E
    ws.betas = betas
    ws.b = b
    ws.z = z
    ws.dr = dr
    ws.rr = rr
    ws.B0 = B0
    ws.psi = psi
    ws.Br = Br
    ws.Bz = Bz
    ws.dBrdz = dBrdz
    ws.Pm = Pm
    ws.str = str_cell
    if style != "figureB1":
        ws.screenposition = mat.row(*screenposition)
