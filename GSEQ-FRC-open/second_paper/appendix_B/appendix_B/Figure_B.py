"""Figure_B.m 的移植：附录 B 的四联图（壁面 ψ / 两个二维解 / 两者之差）。

**自包含**：数据都在本目录（`psi.mat` 给 MATLAB 侧的解 `psiq`、`psi_grass1.mat`
给 GRASS-FT 侧的解 `Z1`、`psib.mat` 给壁面剖面 `rb/zb/psib`）。

四格分别是：

* (a) 壁面 ψ（`zb` 对插值到 `zz` 的 `psib`）；
* (b) MATLAB 解的分界面（正/负区域各 15 条等值线）；
* (c) GRASS-FT 解的分界面（外加一条零层红虚线）；
* (d) 两个解之差 `psi_grass-psi_matlab` 的 13 条等值线——两者几乎相同，所以这一格
  的量级是 \(10^{-5}\)，原文才要把色条刻度标签换成 `(-4:2:4)` 再标 `\times 10^{-5}`。

⚠️ 仓库里的三份结果文件其实是**同一份解**（`appendix_B/psi.mat` 的 `psiq`、
`appendix_B/psi_grass1.mat` 与 `grass_ft/psi_grass{,1}.mat` 的 `Z1` 逐字节相同），
所以 (d) 那一格画出来的是浮点残差级别的差；这是随仓库提供的数据如此，照抄不改。

验收：§7「绘图脚本」那一节的三条（能跑通 / 结构对 / 人眼比对论文图）。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mio
from gseqfrc import mplot


def run(ws) -> None:
    mplot.close_all()  # close all;clear;clc;
    fontsize = mat.scalar(25)
    mplot.figure("unit", "normalized", "DefaultAxesFontSize", fontsize,
                 "DefaultAxesFontWeight", "bold", "DefaultAxesLineWidth", 2,
                 "defaulttextinterpreter", "latex",
                 "position", mat.row(0.5, 0.2, 0.28, 0.6), "color", mat.row(1, 1, 1))

    loaded_psi = mio.load("psi.mat")
    psi_matlab = loaded_psi["psiq"]

    loaded_grass = mio.load("psi_grass1.mat")
    psi_grass = loaded_grass["Z1"]

    psi = psi_grass - psi_matlab

    rmin = mat.scalar(0); zmin = mat.scalar(-1.5)
    rmax = mat.scalar(0.17); zmax = mat.scalar(1.5)

    r = mat.linspace(rmin, rmax, int(np.size(psi, 0)))
    z = mat.linspace(zmin, zmax, int(np.size(psi, 1)))
    X, Y = mat.ndgrid(r, z)

    rr = mat.linspace(rmin, rmax, int(np.size(psi, 0)))
    zz = mat.linspace(zmin, zmax, int(np.size(psi, 1)))
    X1, Y1 = mat.ndgrid(rr, zz)

    loaded_b = mio.load("psib.mat")
    zb, rb, psib_in = loaded_b["zb"], loaded_b["rb"], loaded_b["psib"]
    zzb = zz
    rrb = mat.interp1(zb, rb, zzb, "spline")
    psib = mat.interp1(zb, psib_in, zzb, "spline")

    # %% plot
    mplot.axes("position", mat.row(0.11, 0.75, 0.85, 0.218))  # fig-1
    mplot.plot(zb, psib, "m", "linewidth", 2)
    mplot.grid_on()
    mplot.box("on")
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
               "GridAlpha", 0.25)
    mplot.set_(mplot.gca(), "xtick", mat.colon(-2, 1, 2))
    mplot.ylabel("$\\psi_w$", "interpreter", "latex", "FontSize", 25)
    mplot.text("Interpreter", "latex", "String", "$(a)$",
               "Position", mat.row(-1.4, -1.5), "color", "k", "FontSize", 20)

    mplot.axes("position", mat.row(0.11, 0.52, 0.85, 0.218))  # fig-1
    indtmp_1 = psi_matlab > 0
    psio_1 = np.array(psi_matlab, copy=True)
    psio_1[np.asarray(indtmp_1)] = np.nan
    indtmp_1 = mat.find(psi_matlab < 0)
    psii_1 = np.array(psi_matlab, copy=True)
    psii_1[np.asarray(psi_matlab < 0)] = np.nan
    mplot.contour(Y, X, psio_1, 15, "linewidth", 1)
    mplot.hold_on()      # %外部区域
    mplot.contour(Y, X, psii_1, 15, "linewidth", 1)
    mplot.hold_on()      # %内部区域
    # % contour(Y,X,psi_matlab,[0,0,0],'r--'); hold on;
    mplot.set_(mplot.gca(), "xtick", mat.colon(-2, 1, 2))
    mplot.colorbar("east")
    mplot.ylabel("$r[m]$", "interpreter", "latex", "FontSize", 25)
    mplot.text("Interpreter", "latex", "String", "$(b)$",
               "Position", mat.row(-1.4, 0.14), "color", "k", "FontSize", 20)

    mplot.axes("position", mat.row(0.11, 0.293, 0.85, 0.218))  # fig-1
    indtmp = psi_grass > 0
    psio = np.array(psi_grass, copy=True)
    psio[np.asarray(indtmp)] = np.nan
    indtmp = mat.find(psi_grass < 0)
    psii = np.array(psi_grass, copy=True)
    psii[np.asarray(psi_grass < 0)] = np.nan
    mplot.contour(Y, X, psio, 15, "linewidth", 1)
    mplot.hold_on()      # %外部区域
    mplot.contour(Y, X, psii, 15, "linewidth", 1)
    mplot.hold_on()      # %内部区域
    mplot.contour(Y, X, psi_grass, mat.row(0, 0, 0), "r--")
    mplot.hold_on()      # %分界面
    mplot.set_(mplot.gca(), "xtick", mat.colon(-2, 1, 2))
    mplot.colorbar("east")
    mplot.ylabel("$r[m]$", "interpreter", "latex", "FontSize", 25)
    mplot.text("Interpreter", "latex", "String", "$(c)$",
               "Position", mat.row(-1.4, 0.14), "color", "k", "FontSize", 20)
    mplot.xlabel("$z[m]$", "interpreter", "latex", "FontSize", 25)

    psiin = np.array(psi, copy=True)
    psiin[np.asarray(psi_matlab < 0)] = np.nan
    mplot.axes("position", mat.row(0.11, 0.064, 0.85, 0.218))  # fig-1
    mplot.contour(Y, X, psi, 13, "linewidth", 1)
    mplot.hold_on()
    # % pcolor(Y,X,psi); shading interp; hold on;
    mplot.xlabel("$z[m]$", "interpreter", "latex", "FontSize", 25)
    mplot.ylabel("$r[m]$", "interpreter", "latex", "FontSize", 25)
    h = mplot.colorbar("east")
    h.TickLabels = mat.colon(-4, 2, 4)
    h.Label.String = "\\times 10^{-5}"
    mplot.text("Interpreter", "latex", "String", "$(d)$",
               "Position", mat.row(-1.4, 0.14), "color", "k", "FontSize", 20)
    mplot.colormap("jet")

    for name, value in (("fontsize", fontsize), ("psi_matlab", psi_matlab),
                        ("psi_grass", psi_grass), ("psi", psi),
                        ("rmin", rmin), ("zmin", zmin), ("rmax", rmax),
                        ("zmax", zmax), ("r", r), ("z", z), ("X", X), ("Y", Y),
                        ("rr", rr), ("zz", zz), ("X1", X1), ("Y1", Y1),
                        ("zzb", zzb), ("rrb", rrb), ("psib", psib),
                        ("indtmp_1", indtmp_1), ("psio_1", psio_1),
                        ("psii_1", psii_1), ("indtmp", indtmp), ("psio", psio),
                        ("psii", psii), ("psiin", psiin), ("h", h)):
        ws[name] = value
    for name, value in loaded_psi.items():
        ws[name] = value
    for name, value in loaded_grass.items():
        ws[name] = value
    for name, value in loaded_b.items():
        ws[name] = value
