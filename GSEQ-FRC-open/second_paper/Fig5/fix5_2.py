"""fix5_2.m 的移植：Fig5（FIX 装置）的二维解 vs 一维剖面，上下对称地画。

**后处理**脚本（用 `GSEQ_FRC_fixed_FIX.m` 留在工作区里的 `psi/R/Z/Jzeta/P/jN0/zl/rl/wl/hl/...`），
图里把装置画成上下镜像（`Z,-R`）、还画了腔体轮廓与 24 个线圈矩形。

**按原文的会话用法是"求解器 → `fix5_1` → 本脚本"**：`Bz_plasma=Bz`（整场）、
`psi_plasma=psi`，二维场都在，本机实测原版能跑完（5 个面板，`w_No`/`rrb` 都是 1×257）。
⚠️ 但如果**只**喂求解器的工作区（没有先跑 `fix5_1`），`Bz` 是 1×201 的壁面诊断量，
第 71-74 行的 `Bz_plasma(w_No(i), i)` 就会"索引超出矩阵维度"——移植版照抄，
测试里断言"单跑必停、接链跑通"。

验收：§7「绘图脚本」那一节的三条。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mio
from gseqfrc import mplot

#: 这个脚本要求调用方工作区里已经有这些（来自 `GSEQ_FRC_fixed_FIX.m`）
NEEDS_WORKSPACE = ("psi", "Bz", "R", "Z", "zw", "Z0", "jN0", "rr", "Jzeta", "P",
                   "Rs", "Be", "zl", "rl", "wl", "hl")


def run(ws) -> None:
    missing = [name for name in NEEDS_WORKSPACE if name not in ws.keys()]
    if missing:
        raise KeyError(
            f"fix5_2 需要在求解器的工作区里跑，缺少：{', '.join(missing)}")

    psi = np.asarray(ws["psi"])
    Bz = np.asarray(ws["Bz"])
    R, Z = np.asarray(ws["R"]), np.asarray(ws["Z"])
    zw = ws["zw"]
    Z0 = int(mat.scalar_value(ws["Z0"]))
    jN0 = int(mat.scalar_value(ws["jN0"]))
    rr = np.asarray(ws["rr"])
    Jzeta, P = np.asarray(ws["Jzeta"]), np.asarray(ws["P"])
    Rs, Be = ws["Rs"], ws["Be"]
    zl, rl = np.asarray(ws["zl"]), np.asarray(ws["rl"])
    wl, hl = np.asarray(ws["wl"]), np.asarray(ws["hl"])

    mplot.close_all()  # close all;
    fontsize = mat.scalar(25)
    mplot.figure("unit", "normalized", "DefaultAxesFontSize", fontsize,
                 "DefaultAxesFontWeight", "bold", "DefaultAxesLineWidth", 2,
                 "defaulttextinterpreter", "latex",
                 "position", mat.row(0.45, 0.4, 0.5, 0.55), "color", mat.row(1, 1, 1))
    Bz_plasma = Bz
    psi_plasma = psi
    # %%
    loaded = mio.load("MRR1d.mat")
    Be0 = loaded["Be"]
    rs0 = loaded["Rs"]
    mu0 = mat.scalar(4 * np.pi * 1e-7)
    rr1d = loaded["rr1d"]
    pr1d = loaded["pr1d"]
    j1d = loaded["j1d"]
    # %%
    # % axes('position',[0.11,0.75,0.85,0.23]);%fig-1
    mplot.axes("position", mat.row(0.065, 0.69, 0.91, 0.3))  # fig-1
    ido = mat.find(psi == np.min(psi))
    indtmp = psi > 0
    psio = np.where(indtmp, np.nan, psi)
    indtmp = mat.find(psi < 0)
    psii = np.array(psi, copy=True)
    psii[np.asarray(psi < 0)] = np.nan
    mplot.contour(Z, R, psio, 12, "linewidth", 2)
    mplot.hold_on()
    mplot.contour(Z, R, psii, 15, "linewidth", 2)
    mplot.hold_on()
    mplot.colorbar("east")  # %colormap(hot);
    mplot.contour(Z, -R, psii, 15, "linewidth", 2)
    mplot.hold_on()
    mplot.contour(Z, -R, psio, 12, "linewidth", 2)
    mplot.hold_on()
    c1, h1 = mplot.contour(Z, R, psi, mat.row(0, 0), "r--", "linewidth", 3,
                           return_matrix=True)
    mplot.hold_on()
    mplot.contour(Z, -R, psi, mat.row(0, 0), "r--", "linewidth", 3)
    mplot.plot(mat.take(Z, int(mat.el(ido, 1))), mat.take(R, int(mat.el(ido, 1))),
               "mx", "linewidth", 3)
    mplot.hold_on()   # O-point
    mplot.plot(mat.take(Z, int(mat.el(ido, 1))), -mat.take(R, int(mat.el(ido, 1))),
               "mx", "linewidth", 3)   # O-point
    mplot.plot(-mat.take(Z, int(mat.el(ido, 1))), mat.take(R, int(mat.el(ido, 1))),
               "mx", "linewidth", 3)
    mplot.hold_on()   # O-point
    mplot.plot(-mat.take(Z, int(mat.el(ido, 1))), -mat.take(R, int(mat.el(ido, 1))),
               "mx", "linewidth", 3)   # O-point
    mplot.set_(mplot.gca(), "Fontsize", fontsize)

    mplot.xlabel("$z [m]$", "interpreter", "latex", "FontSize", 25)
    mplot.ylabel("$r [m]$", "interpreter", "latex", "FontSize", fontsize)
    zx1 = np.asarray(c1)[0, 1:-1]
    rx1 = np.asarray(c1)[1, 1:-1]
    zx_no = np.size(zx1, 0)
    indx = mat.find(np.abs(zx1) == np.min(np.abs(zx1)))
    _ = np.min(psi[:, Z0 - 1])   # 原文是裸表达式 `min(psi(:,Z0))`（只进 ans）
    rs = rs0

    for jl in mat.mrange(1, int(np.size(zl, 1))):
        mplot.rectangle("position",
                        mat.row(mat.el(zl, jl) - 0.5 * mat.el(wl, jl),
                                mat.el(rl, jl) - 0.5 * mat.el(hl, jl),
                                mat.el(wl, jl), mat.el(hl, jl)),
                        "FaceColor", mat.row(0.5, 0.5, 0.5), "linewidth", 1)
        mplot.hold_on()
        mplot.rectangle("position",
                        mat.row(mat.el(zl, jl) - 0.5 * mat.el(wl, jl),
                                -mat.el(rl, jl) - 0.5 * mat.el(hl, jl),
                                mat.el(wl, jl), mat.el(hl, jl)),
                        "FaceColor", mat.row(0.5, 0.5, 0.5), "linewidth", 1)
        mplot.hold_on()

    # %plot_chamber
    rwc = mat.scalar(1)
    zzc = mat.colon(-mat.scalar_value(zw), 0.01 * mat.scalar_value(zw),
                    mat.scalar_value(zw))
    a = mat.scalar(0.15)
    zmc = 0.6 * mat.scalar_value(zw)
    rac = 0.3 * rwc
    rbc = 0.4 * rwc
    ind1 = mat.find(zzc < 0)
    ind2 = mat.find(zzc >= 0)
    # MATLAB 里 `rrc(ind1)=...` 会按最大下标自动长成一维行向量；这里直接按 zzc 的长度建
    rrc = np.zeros(int(np.size(zzc)))
    rrc[np.asarray(ind1).ravel() - 1] = (
        0.5 * (rbc - rac) * (np.tanh(5 * (mat.take(zzc, ind1) + zmc)) + 1.0)
        + rac).ravel()
    rrc[np.asarray(ind2).ravel() - 1] = (
        0.5 * (rbc - rac) * (np.tanh(5 * (zmc - mat.take(zzc, ind2))) + 1.0)
        + rac).ravel()
    mplot.plot(zzc, rrc, "k-", "linewidth", 2)
    mplot.hold_on()
    mplot.plot(zzc, -rrc, "k-", "linewidth", 2)
    mplot.text("Interpreter", "latex", "String", "$(a)$",
               "Position", mat.row(-3.4, 0.5), "color", "k", "FontSize", fontsize)

    mplot.axes("position", mat.row(0.065, 0.065, 0.425, 0.24))  # fig-4
    Jzetar = -Jzeta[:, jN0 - 1]
    Pr = P[:, jN0 - 1]
    J0 = np.max(np.abs(Jzetar))
    P0 = np.max(np.abs(Pr))
    mplot.plot(rr * 100, Pr, "m-", rr1d * Rs * 100,
               pr1d * (Be ** 2 / (2 * mu0)), "--", "linewidth", 3)
    mplot.grid_on()
    mplot.hold_on()
    mplot.plot(mat.row(100 * float(mat.scalar_value(rs)), 100 * float(mat.scalar_value(rs))),
               1.1 * mat.row(0, np.max(Pr)), "-.k", "LineWidth", 1.5)
    mplot.hold_on()
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.xlabel("$r\\ [cm]$", "interpreter", "latex", "FontSize", fontsize)
    mplot.ylabel("$P$", "interpreter", "latex", "FontSize", fontsize)
    h = mplot.legend("$2D$", "$1D$")
    mplot.set_(h, "interpreter", "latex")
    mplot.legend("boxoff")
    mplot.xlim(mat.row(0, 45))
    mplot.set_(mplot.gca(), "xtick", mat.colon(0, 10, 45))
    mplot.set_(mplot.gca(), "ytick", mat.colon(0, 2.5e4, 1e5))
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
               "GridAlpha", 0.25)
    mplot.text("Interpreter", "latex", "String", "$(d)$",
               "Position", mat.row(1.5, 9.5e4), "color", "k", "FontSize", fontsize)
    mplot.text("Interpreter", "latex", "String", "$Separatrix$", "Position",
               mat.row(100 * mat.scalar_value(rs) - 8, 2e4), "FontSize", 20)

    # % axes('position',[0.59,0.07,0.37,0.26]); %fig-5
    mplot.axes("position", mat.row(0.55, 0.065, 0.425, 0.24))  # fig-5
    Jzetar = -Jzeta[:, jN0 - 1]
    mplot.plot(rr * 100, -Jzetar, "m-", rr1d * Rs * 100,
               j1d * (Be / (2 * mu0 * Rs)), "--", "linewidth", 3)
    mplot.grid_on()
    mplot.hold_on()
    mplot.plot(mat.row(100 * float(mat.scalar_value(rs)), 100 * float(mat.scalar_value(rs))),
               1.3 * mat.row(0, float(J0)), "-.k", "LineWidth", 1.5)
    mplot.hold_on()
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.xlabel("$r\\ [cm]$", "interpreter", "latex", "FontSize", fontsize)
    mplot.ylabel("$J_{\\theta}$", "interpreter", "latex", "FontSize", fontsize)
    h = mplot.legend("$2D$", "$1D$")
    mplot.set_(h, "interpreter", "latex")
    mplot.legend("boxoff")
    mplot.xlim(mat.row(0, 45))
    mplot.set_(mplot.gca(), "xtick", mat.colon(0, 10, 45))
    mplot.ylim(mat.row(0, 9e6))
    mplot.set_(mplot.gca(), "ytick", mat.colon(0, 3e6, 9e6))
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
               "GridAlpha", 0.25)
    mplot.text("Interpreter", "latex", "String", "$(e)$",
               "Position", mat.row(1.5, 8e6), "color", "k", "FontSize", fontsize)

    loaded2 = mio.load("psib_FIX.mat")  # read data
    zb = loaded2["zb"]
    rb = loaded2["rb"]
    psib = loaded2["psib"]
    Bz_wall = loaded2["Bz"]
    # 原文 `rrb=interp1(zb,rb,Z(1,:),'spline')`：`Z(1,:)` 是**第 1 行**（1×nZ）。
    # （2026-09-17 修：原来写成 `mat.take(Z, 1)`——线性索引取第 1 个元素，
    #  于是 `w_No` 只有 1 个元素、下面那个循环只跑一次，把真正的行为掩盖了。）
    # `Z(1,:)` 是 1×nZ 的**行**，所以这里用 `[0:1, :]` 保留二维形状（rrb/w_No 都是 1×nZ）
    rrb = mat.interp1(zb, rb, np.asarray(Z)[0:1, :], "spline")
    # MATLAB 里 query 是行向量 → 结果也是行向量，所以后面能写 `size(w_No,2)`
    w_No = np.trunc(np.asarray(rrb).ravel()
                    / (mat.ij(R, 2, 1) - mat.ij(R, 1, 1))).reshape(1, -1)   # fix(...)
    psi_wn = np.zeros(int(np.size(w_No, 1)))
    Bz_wn = np.zeros(int(np.size(w_No, 1)))
    for i in mat.mrange(1, int(np.size(w_No, 1))):
        psi_wn[i - 1] = psi_plasma[int(mat.el(w_No, i)) - 1, i - 1]
        # 原文这里是二维下标 `Bz_plasma(w_No(i),i)`：`Bz_plasma=Bz` 是**整场**，
        # 所以接在 fix5_1 后面跑没问题（本机实测：fix5_2 跑完、axes=5、
        # `w_No`/`rrb` 都是 1×257）。只有在"只喂求解器工作区"时 `Bz` 是 1×201
        # 的壁面诊断量，那时才会越界——测试里就是这么断言"单跑必停"的。
        Bz_wn[i - 1] = Bz_plasma[int(mat.el(w_No, i)) - 1, i - 1]

    # % axes('position',[0.12,0.41,0.37,0.26]); %fig-2 [left bottom width height]
    mplot.axes("position", mat.row(0.065, 0.39, 0.425, 0.24))  # fig-2
    # % plot(zb,psib,'r-',Z(1,:),psi_plasma,'g--','linewidth',3);grid on;hold all;
    mplot.plot(zb, psib, "r-", np.asarray(Z)[0:1, :], psi_wn, "g--", "linewidth", 3)
    mplot.grid_on()
    mplot.hold_on()
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.xlabel("$z [m]$", "interpreter", "latex", "FontSize", fontsize)
    mplot.ylabel("$\\psi \\ at \\ wall$", "interpreter", "latex", "FontSize", fontsize)
    mplot.xlim(mat.row(-3.55, 3.5))
    mplot.set_(mplot.gca(), "xtick", mat.colon(-3, 1, 3))
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k", "GridAlpha", 0.25)
    mplot.set_(mplot.gcf(), "PaperPositionMode", "auto")
    h = mplot.legend("$\\mathbf{w/o \\ plasma}$", "$\\mathbf{w \\ plasma}$")
    mplot.set_(h, "interpreter", "latex")
    mplot.legend("boxoff")
    mplot.text("Interpreter", "latex", "String", "$(b)$",
               "Position", mat.row(-3.3, 0.038), "color", "k", "FontSize", fontsize)

    # % axes('position',[0.59,0.41,0.37,0.26]); %fig-3
    mplot.axes("position", mat.row(0.55, 0.39, 0.425, 0.24))  # fig-3
    mplot.plot(zb, Bz_wall, "r-", np.asarray(Z)[0:1, :], Bz_wn, "g--", "linewidth", 3)
    mplot.grid_on()
    mplot.hold_on()
    mplot.ylabel("$B_{z} \\ at \\ wall$", "interpreter", "latex", "FontSize", fontsize)
    mplot.xlabel("$z [m]$", "interpreter", "latex", "FontSize", fontsize)
    mplot.xlim(mat.row(-3.55, 3.5))
    mplot.set_(mplot.gca(), "xtick", mat.colon(-3, 1, 3))
    mplot.ylim(mat.row(0, 1.5))
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k", "GridAlpha", 0.25)
    h = mplot.legend("$\\mathbf{w/o \\ plasma}$", "$\\mathbf{w \\ plasma}$")
    mplot.set_(h, "interpreter", "latex")
    mplot.legend("boxoff")
    mplot.text("Interpreter", "latex", "String", "$(c)$",
               "Position", mat.row(-3.3, 1.35), "color", "k", "FontSize", fontsize)

    for name, value in (("fontsize", fontsize), ("Bz_plasma", Bz_plasma),
                        ("psi_plasma", psi_plasma), ("Be0", Be0), ("rs0", rs0),
                        ("mu0", mu0), ("ido", ido), ("indtmp", indtmp),
                        ("psio", psio), ("psii", psii), ("c1", c1), ("h1", h1),
                        ("zx1", zx1), ("rx1", rx1), ("zx_no", mat.scalar(zx_no)),
                        ("indx", indx), ("rs", rs), ("rwc", rwc), ("zzc", zzc),
                        ("a", a), ("zmc", zmc), ("rac", rac), ("rbc", rbc),
                        ("ind1", ind1), ("ind2", ind2), ("rrc", rrc),
                        ("Jzetar", Jzetar), ("Pr", Pr), ("J0", mat.scalar(J0)),
                        ("P0", mat.scalar(P0)), ("zb", zb), ("rb", rb),
                        ("psib", psib), ("Bz", Bz_wall), ("rrb", rrb),
                        ("w_No", w_No), ("psi_wn", psi_wn), ("Bz_wn", Bz_wn)):
        ws[name] = value
    for name, value in loaded.items():
        ws[name] = value
    for name, value in loaded2.items():
        ws[name] = value
