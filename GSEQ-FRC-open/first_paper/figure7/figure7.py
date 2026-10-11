"""figure7.m 的移植：第一篇论文 Fig.7 的八联"参数扫描"图。

**自包含**：数据是同目录的 `data2.dat`——210 行 × 13 列（β_s、δ_s 的两半、α、σ、
h、ψ_min、⟨β⟩、I_in、I_total、c_p、ψ_g、x_0），脚本用 `fscanf(fid,'%f',1)`
**一个数一个数**地读进来，然后：

1. 把 13 列散点用 `griddata` 插到 5001×5001 的 (β_s, δ_s) 网格上（网格步长是
   `0.0002×(最大值-最小值)`，两个方向正好各 5001 点）；
2. 按理论边界 \(\delta_s = 1/(\alpha_0\sqrt{1-\beta_s})\) 把边界外的点置 NaN；
3. 画 8 格填色等值线（30 层），每格叠一条理论边界曲线 `semilogy(bb,ll)`。

两处照抄原文的细节：

* `set(h.Parent,'YScale','log')`：`h` 是 `[c,h]=contourf(...)` 的句柄，纵轴取对数；
* `ax.XAxis.Visible='off';`：这一行写在 `ax` 还没赋值之前——MATLAB 对**未定义变量**
  的点赋值会自动建一个 struct（本机实测 `class(ax)=struct`，不报错），到第 99 行
  `ax=gca` 才被真正的坐标区句柄盖掉。移植版用 `mat.struct` 把这一步照抄出来。

`mat.griddata`（scipy 的 Delaunay + 线性插值）与 MATLAB 的对应实现是同一个方法、
同一套三角剖分，所以掩膜之后的有限点数与极值都能对上（见测试里的对照值）。

验收：§7「绘图脚本」那一节的三条（能跑通 / 结构对 / 人眼比对论文图）。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mplot

#: `data2.dat` 的 13 列，顺序与 `fscanf` 的读法一致
COLUMNS = ("betas", "ls_1", "ls_0", "alpha", "sigma", "hx", "psim",
           "ave_beta", "tot_I", "tot_Ix", "p", "psig", "x0")


def run(ws) -> None:
    mplot.close_all()  # close all;clear;clc;
    fname = "data2.dat"
    fid = mat.fopen(fname, "r")

    # 一个数一个数地读（13 列 × 210 行，按空白分词、跨行连着读）
    columns = {name: np.zeros(210) for name in COLUMNS}
    for i in mat.mrange(1, 210):
        for name in COLUMNS:
            columns[name][i - 1] = mat.fscanf(fid, "%f", 1)
    betas = columns["betas"]; ls_1 = columns["ls_1"]; ls_0 = columns["ls_0"]
    alpha = columns["alpha"]; sigma = columns["sigma"]; hx = columns["hx"]
    psim = columns["psim"]; ave_beta = columns["ave_beta"]
    tot_I = columns["tot_I"]; tot_Ix = columns["tot_Ix"]
    p = columns["p"]; psig = columns["psig"]; x0 = columns["x0"]
    ls = ls_1 * ls_0

    xq, yq = mat.meshgrid(
        mat.colon(betas.min(), 0.0002 * (betas.max() - betas.min()), betas.max()),
        mat.colon(ls.min(), 0.0002 * (ls.max() - ls.min()), ls.max()))
    # % [xq,yq] = meshgrid(min(betas):0.01*(max(betas)-min(betas)):max(betas), ...

    psim_new = mat.griddata(betas, ls, psim, xq, yq)
    alpha_new = mat.griddata(betas, ls, alpha, xq, yq)
    sigma_new = mat.griddata(betas, ls, sigma, xq, yq)
    hx_new = mat.griddata(betas, ls, hx, xq, yq)
    ave_beta_new = mat.griddata(betas, ls, ave_beta, xq, yq)
    p_new = mat.griddata(betas, ls, p, xq, yq)

    tot_I_new = mat.griddata(betas, ls, tot_I, xq, yq)
    tot_I_total = mat.griddata(betas, ls, tot_Ix, xq, yq)
    tot_I_1 = mat.griddata(betas, ls, tot_I / tot_Ix, xq, yq)

    alpha0 = np.zeros(np.shape(yq)[0])
    ls0 = np.zeros(np.shape(yq)[0])
    for i in mat.mrange(1, int(np.shape(yq)[0])):
        alpha0[i - 1] = 8 * np.log((np.sqrt(1 - xq[0, i - 1]) + 1) / np.sqrt(xq[0, i - 1]))
        ls0[i - 1] = 1.0 / (alpha0[i - 1] * np.sqrt(1 - xq[0, i - 1]))

    # 原文是两重 for 循环逐点置 NaN（5001×5001 = 2500 万次）：条件就是
    # `yq(j,i) > ls0(i)`，所以这里按同一条件向量化（等价、快得多）
    mask = np.asarray(yq) > ls0.reshape(1, -1)
    for field in (psim_new, alpha_new, sigma_new, hx_new, ave_beta_new,
                  tot_I_new, tot_I_total, tot_I_1, p_new):
        field[np.asarray(mask)] = np.nan

    fontsize = mat.scalar(25)
    mplot.figure("unit", "normalized", "DefaultAxesFontSize", fontsize,
                 "DefaultAxesFontWeight", "bold", "DefaultAxesLineWidth", 2,
                 "defaulttextinterpreter", "latex",
                 "position", mat.row(0.02, 0.2, 0.45, 0.38), "color", mat.row(1, 1, 1))
    mplot.set_(mplot.gcf(), "PaperPositionMode", "auto")  # 使 print 出来的与屏幕显示相同

    bb = mat.colon(0.05, 0.02, 0.95)
    ll = 0.0 * bb
    aa = 0.0 * bb
    pp = 0.0 * bb
    for jb in mat.mrange(1, int(np.size(bb, 1))):
        betas_jb = float(mat.el(bb, jb))
        alpha0_jb = 8 * np.log((np.sqrt(1 - betas_jb) + 1) / np.sqrt(betas_jb))
        ls0_jb = 1 / (alpha0_jb * np.sqrt(1 - betas_jb))
        psim0_jb = np.log(betas_jb) / alpha0_jb
        mat.set_el(aa, jb, alpha0_jb)
        mat.set_el(pp, jb, psim0_jb)
        mat.set_el(ll, jb, ls0_jb)

    No = mat.scalar(30)
    # %%%%%
    up = 0.565; down = 0.13; width1 = 0.225; hight1 = 0.42
    # %%%%%
    panels = (
        (mat.row(0.06, up, width1, hight1), psim_new, "$(a) \\quad \\psi_{min}$",
         None, mat.row(0, mat.mmax(yq)), None),
        (mat.row(0.297, up, width1, hight1), alpha_new, "$(b) \\quad \\alpha$",
         None, mat.row(0, mat.mmax(yq)), None),
        (mat.row(0.535, up, width1, hight1), sigma_new, "$(c) \\quad \\sigma$",
         None, None, mat.row(2, 8, 14)),
        (mat.row(0.77, up, width1, hight1), hx_new, "$(d) \\quad h$",
         "$\\beta_s$", None, mat.row(0.5, 1, 1.5)),
        (mat.row(0.06, down, width1, hight1), ave_beta_new,
         "$(e) \\quad \\left \\langle \\beta \\right \\rangle$", "$\\beta_s$",
         mat.row(0, mat.mmax(yq)), mat.row(0.5, 0.7, 0.9)),
        (mat.row(0.297, down, width1, hight1), tot_I_new, "$(f) \\quad I_{in}$",
         "$\\beta_s$", None, None),
        (mat.row(0.535, down, width1, hight1), tot_I_1, "$(g) \\quad c_{i}$",
         "$\\beta_s$", None, mat.row(0.5, 0.7, 0.9)),
        (mat.row(0.77, down, width1, hight1), p_new, "$(h) \\quad c_{p}$",
         "$\\beta_s$", None, mat.row(0.5, 0.7, 0.9)),
    )
    for index, (position, field, label, xlabel, ylim_limits,
                cbar_ticks) in enumerate(panels):
        mplot.axes("position", position)
        c_matrix, h = mplot.contourf(xq, yq, field, No, return_matrix=True)
        mplot.hold_on()
        mplot.set_(mplot.gca(), "Fontsize", fontsize)
        mplot.set_(h.Parent, "YScale", "log")
        if index == 0:
            # 原文第 88 行：`ax.XAxis.Visible='off';`——这时 `ax` 还没赋值，MATLAB 会
            # 自动建一个 struct（本机实测 class(ax)=struct），不报错也不影响图
            ax = mat.struct(XAxis=mat.struct(Visible="off"))
        if index in (0, 4):
            mplot.ylabel("$\\delta_s$", "fontsize", 28,
                         "position", mat.row(-0.08, 0.35))
        mplot.semilogy(bb, ll, "linewidth", 2)
        if ylim_limits is not None:
            mplot.ylim(ylim_limits)
        # 各格的刻度处理照抄原文（顺序也要对：（e）是先 Xtick 再写刻度标签）
        if index == 0:                              # (a)
            mplot.set_(mplot.gca(), "Ytick", mat.row(0, 0.01, 0.1, 1))
        elif index == 1:                            # (b)
            mplot.set_(mplot.gca(), "Ytick", mat.row(0, 0.1, 1))
            mplot.set_(mplot.gca(), "xtick", [], "ytick", [])
        elif index in (2, 3):                       # (c)(d)
            mplot.set_(mplot.gca(), "xtick", [], "ytick", [])
        elif index == 4:                            # (e)
            mplot.set_(mplot.gca(), "Ytick", mat.row(0, 0.1, 1))
            mplot.set_(mplot.gca(), "Xtick", mat.row(0.2, 0.5, 0.8))
        else:                                       # (f)(g)(h)
            mplot.set_(mplot.gca(), "ytick", [])
            mplot.set_(mplot.gca(), "Xtick", mat.row(0.2, 0.5, 0.8))
        if index in (0, 4):
            mplot.set_(mplot.gca(), "xticklabel", mplot.get_(mplot.gca(), "xtick"),
                       "yticklabel", mplot.get_(mplot.gca(), "ytick"))
        if index in (0, 1, 2, 3):
            mplot.set_(mplot.gca(), "xtick", [])
        if xlabel is not None:
            if index in (4, 5, 6, 7):
                mplot.xlabel(xlabel, "fontsize", 28)
            else:
                mplot.xlabel(xlabel)
        mplot.text("Interpreter", "latex", "String", label,
                   "Position", mat.row(0.07, 1.7),
                   "FontSize", 20 if index == 4 else 25)
        if index == 0:
            mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
                       "GridAlpha", 0.25)

        if cbar_ticks is not None:
            c = mplot.colorbar("Location", "east", "Ticks", cbar_ticks)
        else:
            c = mplot.colorbar("Location", "east")
        ax = mplot.gca()
        axpos = mplot.get_(ax, "Position")
        c.Position[2] = 0.3 * c.Position[2]
        HorizontalAlignment = "right"
        mplot.set_(ax, "Position", axpos)

    mplot.colormap("hot")

    for name, value in (("fname", fname), ("fid", fid), ("columns", columns),
                        ("betas", betas), ("ls_1", ls_1), ("ls_0", ls_0),
                        ("alpha", alpha), ("sigma", sigma), ("hx", hx),
                        ("psim", psim), ("ave_beta", ave_beta), ("tot_I", tot_I),
                        ("tot_Ix", tot_Ix), ("p", p), ("psig", psig), ("x0", x0),
                        ("ls", ls), ("xq", xq), ("yq", yq),
                        ("psim_new", psim_new), ("alpha_new", alpha_new),
                        ("sigma_new", sigma_new), ("hx_new", hx_new),
                        ("ave_beta_new", ave_beta_new), ("p_new", p_new),
                        ("tot_I_new", tot_I_new), ("tot_I_total", tot_I_total),
                        ("tot_I_1", tot_I_1), ("alpha0", alpha0), ("ls0", ls0),
                        ("fontsize", fontsize), ("bb", bb), ("ll", ll),
                        ("aa", aa), ("pp", pp), ("No", No), ("up", mat.scalar(up)),
                        ("down", mat.scalar(down)), ("width1", mat.scalar(width1)),
                        ("hight1", mat.scalar(hight1)), ("ax", ax),
                        ("axpos", np.asarray(axpos, dtype=float)),
                        ("HorizontalAlignment", HorizontalAlignment),
                        ("c", c), ("h", h), ("c_matrix", c_matrix),
                        ("mask", mask)):
        ws[name] = value
