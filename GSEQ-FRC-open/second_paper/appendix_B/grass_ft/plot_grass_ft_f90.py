"""plot_grass_ft_f90.m 的移植：把 GRASS-FT 的 ψ 数据插值到网格上画分界面。

**自包含**：数据来自同目录的 `data/ps_t.dat`（129×385，`nplot=1` 那一支；另外两支
分别是 `ps_pl.dat` 与 `ps_t_ref.dat`，源码用 `if nplot==...` 选），画完把插值结果
存成 `psi_grass.mat`（原程序 `save([filename,'.mat'],'Z1')`）。

⚠️ 与同目录的 `contour_psi.m` **逐字节相同**（MD5 一样，本机核对过；两份都存
`psi_grass.mat`）。按"一个 .m 一个 .py、副本不合并"的约定，这里照样各写一份。

验收：§7「绘图脚本」那一节的三条（能跑通 / 结构对 / 人眼比对论文图）。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mio
from gseqfrc import mplot


def run(ws) -> None:
    mplot.close_all()  # close all;clear;clc;

    mplot.figure("position", mat.row(50, 100, 600, 300))
    mplot.set_(mplot.gca(), "nextplot", "replacechildren")
    mplot.set_(mplot.gcf(), "DefaultAxesFontSize", 15)

    rmin = mat.scalar(0)
    zmin = mat.scalar(-1.5)
    rmax = mat.scalar(1.0)
    zmax = mat.scalar(1.5)

    nplot = mat.scalar(1)
    data = None
    if mat.scalar_value(nplot) == 1:
        data = mat.importdata("data/ps_t.dat")
    elif mat.scalar_value(nplot) == 2:
        data = mat.importdata("data/ps_pl.dat")
    elif mat.scalar_value(nplot) == 3:
        data = mat.importdata("data/ps_t_ref.dat")

    r = mat.linspace(rmin, rmax, int(np.size(data, 0)))
    z = mat.linspace(zmin, zmax, int(np.size(data, 1)))
    X, Y = mat.ndgrid(r, z)

    rr = mat.linspace(rmin, rmax, int(np.size(data, 0)))
    zz = mat.linspace(zmin, zmax, int(np.size(data, 1)))
    X1, Y1 = mat.ndgrid(rr, zz)

    mplot.set_(mplot.gca(), "Fontsize", 15)
    Z1 = mat.griddata(X, Y, data, X1, Y1)
    # % contour(Y1,X1,Z1,200);

    indtmp = Z1 > 0
    psio = np.array(Z1, copy=True)
    psio[np.asarray(indtmp)] = np.nan
    indtmp = mat.find(Z1 < 0)
    psii = np.array(Z1, copy=True)
    psii[np.asarray(Z1 < 0)] = np.nan
    mplot.contour(Y1, X1, psio, 15, "linewidth", 1)
    mplot.hold_on()      # %外部区域
    mplot.contour(Y1, X1, psii, 15, "linewidth", 1)
    mplot.hold_on()      # %内部区域
    mplot.contour(Y1, X1, Z1, mat.row(0, 0, 0), "r--")
    mplot.hold_on()      # %分界面

    mplot.hold_on()
    mplot.box("on")
    mplot.grid_on()
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
               "GridAlpha", 0.5)
    mplot.shading("flat")
    mplot.colorbar()
    mplot.xlim(mat.row(zmin, zmax))
    mplot.ylim(mat.row(0, 1.01))
    mplot.set_(mplot.gca(), "XTick", mat.colon(zmin, 0.5, zmax))
    mplot.xlabel("$Z$", "interpreter", "latex", "FontSize", 17)
    mplot.ylabel("$R$", "interpreter", "latex", "FontSize", 17)

    if mat.scalar_value(nplot) == 1:
        mplot.title("$Grad-Shafranov \\ eq \\ \\psi_{e}$", "interpreter", "latex",
                    "FontSize", 17)
    elif mat.scalar_value(nplot) == 2:
        mplot.title("$Grad-Shafranov \\ eq \\ \\psi_{p}$", "interpreter", "latex",
                    "FontSize", 17)
    elif mat.scalar_value(nplot) == 3:
        mplot.title("$Grad-Shafranov \\ eq \\ \\psi_{tot}$", "interpreter", "latex",
                    "FontSize", 17)

    psiq = data
    filename = "psi_grass"
    mio.save(filename + ".mat", Z1=Z1)
    # 原文下面是一大段注释掉的画法（第 61-77 行），最后一句 `max(max(data))` 是
    # 裸表达式（只把结果丢进 ans）——照抄成无副作用的计算
    _ = np.max(np.max(data))

    for name, value in (("rmin", rmin), ("zmin", zmin), ("rmax", rmax),
                        ("zmax", zmax), ("nplot", nplot), ("data", data),
                        ("r", r), ("z", z), ("X", X), ("Y", Y), ("rr", rr),
                        ("zz", zz), ("X1", X1), ("Y1", Y1), ("Z1", Z1),
                        ("indtmp", indtmp), ("psio", psio), ("psii", psii),
                        ("psiq", psiq), ("filename", filename),
                        ("Z1_max", mat.scalar(np.max(Z1)))):
        ws[name] = value
