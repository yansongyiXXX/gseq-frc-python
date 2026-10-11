"""Figure_8.m 的移植：δ_s 扫描（β_s=0.7 固定）的三联图。

**自包含**：五份 `psiq1_fixed_betas=0_7_deltas=*.mat` 随论文提供（含
`psiq/Rq/Zq/P/Z0` 与线圈表 `zl/rl/hl/wl`），脚本把 ψ 插值到统一网格后画：

* (a) 分界面——δ_s=0.6/0.8/1.0 三条零层等值线，外加线圈与壁面示意；
* (b) δ_s 与伸长比 m 的散点（数值直接写死在源码里）；
* (c) 中平面归一化压强剖面。

原文是五段几乎重复的代码（只有文件名与变量后缀不同），这里按同样的顺序写成循环，
每段的算式一字未改；**第一段**还要先把统一网格（`rr/zz/R/Z/dR/dZ`）按 `Rq/Zq`
的范围定下来，后面四段复用。

**修改版**：修掉了原脚本第 52 行的笔误。原文那一行的左值是 `psi_delta10`
（少一个 `s`），于是它既没有修正 `psi_deltas10` 的外边界，又凭空建出一个
513×257 的新变量 `psi_delta10`。现在按作者显然的意图写成 `psi_deltas10(nR,jZ)`，
`psi_delta10` 这个笔误变量不再出现（改动记录见 `修改记录.md`）。

验收：§7「绘图脚本」那一节的三条（能跑通 / 结构对 / 人眼比对论文图）。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mio
from gseqfrc import mplot

#: 五段数据的（变量后缀, 文件名），顺序与源码一致
FILES = (("deltas6", "psiq1_fixed_betas=0_7_deltas=0_6.mat"),
         ("deltas7", "psiq1_fixed_betas=0_7_deltas=0_7.mat"),
         ("deltas8", "psiq1_fixed_betas=0_7_deltas=0_8.mat"),
         ("deltas9", "psiq1_fixed_betas=0_7_deltas=0_9.mat"),
         ("deltas10", "psiq1_fixed_betas=0_7_deltas=1.mat"))


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
        if tag == "deltas6":
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
            # 修改版：原文第 52 行这里笔误成 `psi_delta10`，外边界这一行没生效；
            # 现在五段一视同仁地修正外边界。
            field[nR - 1, jZ - 1] = field[nR - 2, jZ - 1]
        psi_fields[tag] = field
        # `P(:,Z0)` 是 n×1 的列，按列切片保留二维（见 `figure_A_3.py` 里的说明）
        pr_fields[tag] = P[:, Z0 - 1:Z0] / np.max(P[:, Z0 - 1:Z0])

    psi_deltas6 = psi_fields["deltas6"]
    psi_deltas7 = psi_fields["deltas7"]
    psi_deltas8 = psi_fields["deltas8"]
    psi_deltas9 = psi_fields["deltas9"]
    psi_deltas10 = psi_fields["deltas10"]
    Pr_deltas6 = pr_fields["deltas6"]
    Pr_deltas7 = pr_fields["deltas7"]
    Pr_deltas8 = pr_fields["deltas8"]
    Pr_deltas9 = pr_fields["deltas9"]
    Pr_deltas10 = pr_fields["deltas10"]
    zl = loaded["deltas10"]["zl"]; rl = loaded["deltas10"]["rl"]
    hl = loaded["deltas10"]["hl"]; wl = loaded["deltas10"]["wl"]

    mplot.axes("position", mat.row(0.065, 0.58, 0.91, 0.4))  # fig-1
    ido = mat.find(psi_deltas6 == np.min(np.min(psi_deltas6)))
    indtmp = psi_deltas10 > 0
    psio = np.array(psi_deltas10, copy=True)
    psio[np.asarray(indtmp)] = np.nan
    indtmp = mat.find(psi_deltas10 < 0)
    psii = np.array(psi_deltas10, copy=True)
    psii[np.asarray(psi_deltas10 < 0)] = np.nan
    mplot.contour(Z, R, psio, 12, "linewidth", 2)
    mplot.hold_on()
    mplot.contour(Z, R, psii, 15, "linewidth", 2)
    mplot.hold_on()
    c1, h1 = mplot.contour(Z, R, psi_deltas6, mat.row(0, 0), "--",
                           "color", _rgb(0, 0.4470, 0.7410), "linewidth", 3,
                           return_matrix=True)
    mplot.hold_on()
    mplot.contour(Z, R, psi_deltas8, mat.row(0, 0), "--",
                  "color", _rgb(0.8500, 0.3250, 0.0980), "linewidth", 3)
    mplot.contour(Z, R, psi_deltas10, mat.row(0, 0), "--",
                  "color", _rgb(0.9290, 0.6940, 0.1250), "linewidth", 3)
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
    mplot.plot(rr * 100, Pr_deltas6, "-", rr * 100, Pr_deltas8, rr * 100,
               Pr_deltas10, "linewidth", 2)
    mplot.grid_on()
    mplot.hold_on()
    mplot.plot(mat.row(100 * mat.scalar_value(rs), 100 * mat.scalar_value(rs)),
               mat.row(0, 1.1), "--", "color", "k", "LineWidth", 1.5)
    mplot.hold_on()
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.xlabel("$r\\ [cm], z=0$", "interpreter", "latex", "FontSize", fontsize)
    mplot.ylabel("$P$", "interpreter", "latex", "FontSize", fontsize)
    h = mplot.legend("$\\delta_s=0.6 \\delta_{s}^{RR}$",
                     "$\\delta_s=0.8\\delta_{s}^{RR}$",
                     "$\\delta_s=1.0\\delta_{s}^{RR}$")
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
    deltas = mat.row(1.0, 0.9, 0.8, 0.7, 0.6)
    n_factor = mat.row(1.65, 2.07, 2.27, 2.72, 3.04)
    mplot.plot(deltas, n_factor, "ko-", "linewidth", 2, "markersize", 8)
    mplot.grid_on()
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
               "GridAlpha", 0.25)
    mplot.xlabel("$\\delta_s/ \\delta_s^{RR}$", "interpreter", "latex",
                 "FontSize", fontsize)
    mplot.ylabel("$m$", "interpreter", "latex", "FontSize", fontsize)
    mplot.set_(mplot.gca(), "xtick", mat.colon(0.6, 0.1, 1))
    mplot.ylim(mat.row(0, 3.1))
    mplot.text("interpreter", "latex", "string", "(b)", "position",
               mat.row(0.63, 2.7), "color", "k", "FontSize", 20)

    for name, value in (("fontsize", fontsize), ("rw", rw), ("zw", zw),
                        ("zwa", zwa), ("nRq", mat.scalar(nRq)),
                        ("nZq", mat.scalar(nZq)), ("nR", mat.scalar(nR)),
                        ("nZ", mat.scalar(nZ)), ("Rlow", Rlow), ("Rup", Rup),
                        ("Zlow", Zlow), ("Zup", Zup), ("rr", rr), ("zz", zz),
                        ("dR", dR), ("dZ", dZ), ("dR2", dR2), ("dZ2", dZ2),
                        ("R", R), ("Z", Z),
                        ("psi_deltas6", psi_deltas6), ("psi_deltas7", psi_deltas7),
                        ("psi_deltas8", psi_deltas8), ("psi_deltas9", psi_deltas9),
                        ("psi_deltas10", psi_deltas10),
                        ("Pr_deltas6", Pr_deltas6), ("Pr_deltas7", Pr_deltas7),
                        ("Pr_deltas8", Pr_deltas8), ("Pr_deltas9", Pr_deltas9),
                        ("Pr_deltas10", Pr_deltas10),
                        ("ido", ido), ("indtmp", indtmp), ("psio", psio),
                        ("psii", psii), ("c1", c1), ("h1", h1), ("rm", rm),
                        ("zx1", zx1), ("rx1", rx1), ("indx", indx), ("rs", rs),
                        ("deltas", deltas), ("n_factor", n_factor), ("h", h)):
        ws[name] = value
    # `load` 直接把变量放进工作区；最后一份 `psiq1_fixed_betas=0_7_deltas=1.mat`
    # 覆盖了同名的 Rq/Zq/psiq/P/Z0/zl/rl/hl/wl/...（MATLAB 里也是这样）
    for name, value in loaded["deltas10"].items():
        ws[name] = value
