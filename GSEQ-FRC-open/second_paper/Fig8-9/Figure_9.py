"""Figure_9.m 的移植：β_s 扫描（δ_s=0.15 固定）的三联图。

与同目录的 `Figure_8.m` 是同一套版式（分界面 / m 散点 / 中平面压强剖面），差别
只在扫的参数：这里四份数据 `psiq1_fixed_betas=0_{4,5,6,7}_deltas=0_15.mat`，
图 (a) 画 β_s=0.4/0.5/0.6/0.7 四条零层等值线，图 (b) 用写死的
`betas`/`n_factor` 两组数。

**自包含**：数据随论文提供，脚本只做 `load` → 插值 → 画图。

验收：§7「绘图脚本」那一节的三条（能跑通 / 结构对 / 人眼比对论文图）。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mio
from gseqfrc import mplot

#: 四段数据的（变量后缀, 文件名），顺序与源码一致
FILES = (("betas4", "psiq1_fixed_betas=0_4_deltas=0_15.mat"),
         ("betas5", "psiq1_fixed_betas=0_5_deltas=0_15.mat"),
         ("betas6", "psiq1_fixed_betas=0_6_deltas=0_15.mat"),
         ("betas7", "psiq1_fixed_betas=0_7_deltas=0_15.mat"))


def _rgb(*values) -> np.ndarray:
    """MATLAB 的颜色字面量 `[0 0.4470 0.7410]`（matplotlib 要 (3,) 的数组）。"""
    return np.asarray(values, dtype=float)


def run(ws) -> None:
    mplot.close_all()  # close all;clear;clc;
    fontsize = mat.scalar(25)
    mplot.figure("unit", "normalized", "DefaultAxesFontSize", fontsize,
                 "DefaultAxesFontWeight", "bold", "DefaultAxesLineWidth", 2,
                 "position", mat.row(0.45, 0.4, 0.5, 0.4))

    loaded: dict[str, dict] = {}   # 每段 load 覆盖同名变量，最后一份留在工作区里
    psi_fields: dict[str, np.ndarray] = {}
    pr_fields: dict[str, np.ndarray] = {}
    for tag, filename in FILES:
        loaded[tag] = mio.load(filename)
        Rq, Zq, psiq = loaded[tag]["Rq"], loaded[tag]["Zq"], loaded[tag]["psiq"]
        P = loaded[tag]["P"]
        Z0 = int(mat.scalar_value(loaded[tag]["Z0"]))
        if tag == "betas4":
            rw = mat.scalar(np.max(np.max(Rq)))
            zw = mat.scalar(np.max(np.max(Zq)))
            zwa = mat.scalar(np.min(np.min(Zq)))
            nRq, nZq = np.shape(psiq)
            nR, nZ = int(nRq), int(nZq)
            Rlow = mat.scalar(0); Rup = rw
            Zlow = zwa; Zup = zw
            rr = mat.linspace(Rlow, Rup, nR)
            zz = mat.linspace(Zlow, Zup, nZ)
            dR = mat.scalar(mat.el(rr, 2) - mat.el(rr, 1))
            dZ = mat.scalar(mat.el(zz, 2) - mat.el(zz, 1))
            R, Z = mat.ndgrid(rr, zz)
            dR2 = dR * dR
            dZ2 = dZ * dZ
        field = mat.interp2(Zq, Rq, psiq, Z, R, "spline")  # 'cubic'
        for jZ in mat.mrange(1, nZ):
            field[0, jZ - 1] = field[1, jZ - 1]
            field[nR - 1, jZ - 1] = field[nR - 2, jZ - 1]
        psi_fields[tag] = field
        # `P(:,Z0)` 是 n×1 的列，按列切片保留二维（见 `figure_A_3.py` 里的说明）
        pr_fields[tag] = P[:, Z0 - 1:Z0] / np.max(P[:, Z0 - 1:Z0])

    psi_betas4 = psi_fields["betas4"]
    psi_betas5 = psi_fields["betas5"]
    psi_betas6 = psi_fields["betas6"]
    psi_betas7 = psi_fields["betas7"]
    Pr_betas4 = pr_fields["betas4"]
    Pr_betas5 = pr_fields["betas5"]
    Pr_betas6 = pr_fields["betas6"]
    Pr_betas7 = pr_fields["betas7"]
    zl = loaded["betas7"]["zl"]; rl = loaded["betas7"]["rl"]
    hl = loaded["betas7"]["hl"]; wl = loaded["betas7"]["wl"]

    mplot.axes("position", mat.row(0.065, 0.58, 0.91, 0.4))  # fig-1
    ido = mat.find(psi_betas4 == np.min(np.min(psi_betas4)))
    indtmp = psi_betas4 > 0
    psio = np.array(psi_betas4, copy=True)
    psio[np.asarray(indtmp)] = np.nan
    indtmp = mat.find(psi_betas4 < 0)
    psii = np.array(psi_betas4, copy=True)
    psii[np.asarray(psi_betas4 < 0)] = np.nan
    mplot.contour(Z, R, psio, 12, "linewidth", 2)
    mplot.hold_on()
    mplot.contour(Z, R, psii, 15, "linewidth", 2)
    mplot.hold_on()
    c1, h1 = mplot.contour(Z, R, psi_betas4, mat.row(0, 0), "--",
                           "color", _rgb(0, 0.4470, 0.7410), "linewidth", 3,
                           return_matrix=True)
    mplot.hold_on()
    mplot.contour(Z, R, psi_betas5, mat.row(0, 0), "--",
                  "color", _rgb(0.8500, 0.3250, 0.0980), "linewidth", 3)
    mplot.contour(Z, R, psi_betas6, mat.row(0, 0), "--",
                  "color", _rgb(0.9290, 0.6940, 0.1250), "linewidth", 3)
    mplot.contour(Z, R, psi_betas7, mat.row(0, 0), "--",
                  "color", _rgb(0.4940, 0.1840, 0.5560), "linewidth", 3)
    mplot.plot(mat.take(Z, int(mat.el(ido, 1))), mat.take(R, int(mat.el(ido, 1))),
               "mx", "linewidth", 3)   # O-point
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.text(0.01 * mat.scalar_value(zw), mat.take(R, int(mat.el(ido, 1))),
               "O point", "fontsize", 13)
    rm = mat.scalar(0.15)
    zx1 = np.asarray(c1)[0, 1:-1]
    rx1 = np.asarray(c1)[1, 1:-1]
    indx = mat.find(np.abs(zx1) == np.min(np.abs(zx1)))
    rs = mat.el(rx1, int(mat.el(indx, 1)))
    mplot.xlabel("$z [m]$", "interpreter", "latex", "FontSize", fontsize)
    mplot.ylabel("$r [m]$", "interpreter", "latex", "FontSize", fontsize)

    for jl in mat.mrange(1, int(np.size(zl, 1))):   # size(zl,2) 个线圈截面
        mplot.rectangle("position",
                        mat.row(mat.el(zl, jl) - 0.5 * mat.el(wl, jl),
                                mat.el(rl, jl) - 0.5 * mat.el(hl, jl),
                                mat.el(wl, jl), mat.el(hl, jl)),
                        "FaceColor", mat.row(0.5, 0.5, 0.5), "linewidth", 1)
        mplot.hold_on()
    mplot.rectangle("position",
                    mat.row(-mat.scalar_value(zw), 0,
                            2 * mat.scalar_value(zw), mat.scalar_value(rw)),
                    "linewidth", 1)
    mplot.hold_on()
    mplot.set_(mplot.gca(), "xtick", mat.colon(-0.75, 0.25, 0.75))
    mplot.text("interpreter", "latex", "string", "(a)", "position",
               mat.row(-0.7, 0.17), "color", "k", "FontSize", 20)

    # %%
    mplot.axes("position", mat.row(0.55, 0.105, 0.425, 0.37))  # fig-3
    mplot.plot(rr * 100, Pr_betas4, "-", rr * 100, Pr_betas5, rr * 100,
               Pr_betas6, rr * 100, Pr_betas7, "linewidth", 2)
    mplot.grid_on()
    mplot.hold_on()
    mplot.plot(mat.row(100 * mat.scalar_value(rs), 100 * mat.scalar_value(rs)),
               mat.row(0, 1.1), "--", "color", "k", "LineWidth", 1.5)
    mplot.hold_on()
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.xlabel("$r\\ [cm], z=0$", "interpreter", "latex", "FontSize", fontsize)
    mplot.ylabel("$P$", "interpreter", "latex", "FontSize", fontsize)
    h = mplot.legend("$\\beta_s=0.4$", "$\\beta_s=0.5$", "$\\beta_s=0.6$",
                     "$\\beta_s=0.7$")
    mplot.set_(h, "interpreter", "latex")
    mplot.legend("boxoff")
    mplot.xlim(mat.row(0, 100 * mat.scalar_value(rm)))
    mplot.set_(mplot.gca(), "xtick", mat.colon(0, 2, 100 * mat.scalar_value(rm)))
    mplot.set_(mplot.gca(), "ytick", mat.colon(0, 0.2, 1.1))
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
               "GridAlpha", 0.25)
    mplot.text("interpreter", "latex", "string", "$separatrix$", "position",
               mat.row(5, 0.2), "FontSize", 20)
    mplot.text("interpreter", "latex", "string", "(c)", "position",
               mat.row(1, 0.95), "color", "k", "FontSize", 20)

    mplot.axes("position", mat.row(0.065, 0.105, 0.425, 0.37))  # fig-2
    betas = mat.row(0.4, 0.5, 0.6, 0.7)
    n_factor = mat.row(1.17, 2.12, 3.25, 3.52)
    mplot.plot(betas, n_factor, "ko-", "linewidth", 2, "markersize", 8)
    mplot.grid_on()
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
               "GridAlpha", 0.25)
    mplot.xlabel("$\\beta_s$", "interpreter", "latex", "FontSize", fontsize)
    mplot.ylabel("$m$", "interpreter", "latex", "FontSize", fontsize)
    mplot.set_(mplot.gca(), "xtick", mat.colon(0.4, 0.1, 0.7))
    mplot.ylim(mat.row(0, 3.6))
    mplot.text("interpreter", "latex", "string", "(b)", "position",
               mat.row(0.42, 3.05), "color", "k", "FontSize", 20)

    for name, value in (("fontsize", fontsize), ("rw", rw), ("zw", zw),
                        ("zwa", zwa), ("nRq", mat.scalar(nRq)),
                        ("nZq", mat.scalar(nZq)), ("nR", mat.scalar(nR)),
                        ("nZ", mat.scalar(nZ)), ("Rlow", Rlow), ("Rup", Rup),
                        ("Zlow", Zlow), ("Zup", Zup), ("rr", rr), ("zz", zz),
                        ("dR", dR), ("dZ", dZ), ("dR2", dR2), ("dZ2", dZ2),
                        ("R", R), ("Z", Z),
                        ("psi_betas4", psi_betas4), ("psi_betas5", psi_betas5),
                        ("psi_betas6", psi_betas6), ("psi_betas7", psi_betas7),
                        ("Pr_betas4", Pr_betas4), ("Pr_betas5", Pr_betas5),
                        ("Pr_betas6", Pr_betas6), ("Pr_betas7", Pr_betas7),
                        ("ido", ido), ("indtmp", indtmp), ("psio", psio),
                        ("psii", psii), ("c1", c1), ("h1", h1), ("rm", rm),
                        ("zx1", zx1), ("rx1", rx1), ("indx", indx), ("rs", rs),
                        ("betas", betas), ("n_factor", n_factor), ("h", h)):
        ws[name] = value
    # `load` 直接把变量放进工作区；最后一份覆盖同名变量（MATLAB 里也是这样）
    for name, value in loaded["betas7"].items():
        ws[name] = value
