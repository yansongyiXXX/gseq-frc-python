"""figure_8_2.m 的移植：第一篇论文 Fig.8 的五联对比图。

**会话链的末端**：它自己只 `load('gs1d.mat')`（一维剖面 `psi_r/r_rr/pmrr/prr/
Bzmrr/Bzrr/Jr_mrr/Jr_rr`）与几个实验数据点文件，二维的 `psi/R/Z/Bz/Jzeta/P/rm/zw/nZ/
zzb/rrb` 全用**调用方工作区里**的变量——MATLAB 里的用法是"跑完 `figure_8_1.m`
之后再 run 这一份"。所以**单跑会停**（缺上游变量），接在 `figure_8_1` 后面就通。

五格分别是：(a) 分界面 + 壁面轮廓；(b) 压强剖面（2D/1D/RR）；(c) 中平面 \(B_z\)；
(d) 中平面 \(J_\theta\)；(e) 归一化 ψ。

⚠️ `max`/`min` 用的是 `mat.mmax/mmin`：MATLAB 的这两个函数**忽略 NaN**，
而这里 `psi` 带着 `jrb` 掩膜的 NaN（例如 `psir=psi(:,Z_mid)`、`abs(min(psir))`），
直接用 numpy 会得到 NaN。这一点在 PORTING.md §5 记了一条。

验收：§7「绘图脚本」那一节的三条（能跑通 / 结构对 / 人眼比对论文图）。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mio
from gseqfrc import mplot

#: 这个脚本要求调用方工作区里已经有这些（来自 `figure_8_1.m` 那次运行）
NEEDS_WORKSPACE = ("psi", "R", "Z", "zw", "nZ", "P", "Bz", "Jzeta", "rm",
                   "zzb", "rrb", "rr")


def run(ws) -> None:
    missing = [name for name in NEEDS_WORKSPACE if name not in ws.keys()]
    if missing:
        raise KeyError(
            f"figure_8_2 需要在 figure_8_1 之后跑（同一个会话），缺少："
            f"{', '.join(missing)}（原脚本直接用上游算出的二维 Bz/P/Jzeta 与 rm）"
        )

    psi = np.asarray(ws["psi"]); R = np.asarray(ws["R"]); Z = np.asarray(ws["Z"])
    zw = ws["zw"]
    nZ = int(mat.scalar_value(ws["nZ"]))
    P = np.asarray(ws["P"]); Bz = np.asarray(ws["Bz"]); Jzeta = np.asarray(ws["Jzeta"])
    rm = ws["rm"]
    zzb = np.asarray(ws["zzb"]); rrb = np.asarray(ws["rrb"])
    rr = np.asarray(ws["rr"])

    mplot.close_all()  # close all;
    fontsize = mat.scalar(25)
    mplot.figure("unit", "normalized", "DefaultAxesFontSize", fontsize,
                 "DefaultAxesFontWeight", "bold", "DefaultAxesLineWidth", 2,
                 "defaulttextinterpreter", "latex",
                 "position", mat.row(0.5, 0.4, 0.5, 0.4), "color", mat.row(1, 1, 1))
    mplot.set_(mplot.gcf(), "PaperPositionMode", "auto")  # 使 print 出来的与屏幕显示相同

    C1 = ["m", "k", "b", "g", "c", "r"]
    i = mat.scalar(1)
    loaded = mio.load("gs1d.mat")

    # subplot(3,2,1:2)
    mplot.axes("position", mat.row(0.08, 0.74, 0.91, 0.25))  # fig-1
    ido = mat.find(psi == mat.mmin(psi))
    indtmp = psi > 0
    psio = np.array(psi, copy=True)
    psio[np.asarray(indtmp)] = np.nan
    indtmp = mat.find(psi < 0)
    psii = np.array(psi, copy=True)
    psii[np.asarray(psi < 0)] = np.nan
    mplot.contour(Z, R, psio, 8, "linewidth", 2)
    mplot.hold_on()
    mplot.contour(Z, R, psii, 15, "linewidth", 2)
    mplot.hold_on()   # % colorbar('northOutside');
    c1, h1 = mplot.contour(Z, R, psi, mat.row(0, 0), "r--", "linewidth", 3,
                           return_matrix=True)
    mplot.plot(mat.take(Z, int(mat.el(ido, 1))), mat.take(R, int(mat.el(ido, 1))),
               "mx", "linewidth", 3)   # O-point
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.text(0.01 * mat.scalar_value(zw), mat.take(R, int(mat.el(ido, 1))),
               "O point", "fontsize", 13)

    mplot.ylabel("$r (m)$", "Fontsize", fontsize)
    zx1 = np.asarray(c1)[0, 1:-1]
    rx1 = np.asarray(c1)[1, 1:-1]
    zx_no = np.size(zx1, 0)
    indx = mat.find(np.abs(zx1) == np.min(np.abs(zx1)))
    mplot.ylim(mat.row(0, 0.18))
    mplot.set_(mplot.gca(), "ytick", mat.colon(0, 0.05, 0.22))
    # % min(psi(:,floor(Zs*nZ)))
    rs = mat.el(rx1, int(mat.el(indx, 1)))
    mplot.plot(zzb, rrb, "k-", "linewidth", 3)
    mplot.xlabel("$Z(m)$", "position", mat.row(-0.2, -0.025), "Fontsize", fontsize)
    mplot.text("Interpreter", "latex", "String", "$(a)$",
               "Position", mat.row(-0.7, 0.163, 1), "FontSize", fontsize)

    Z_mid = int(np.floor(nZ / 2)) + 1
    Pr = P[:, Z_mid - 1:Z_mid]
    Bzr = Bz[:, Z_mid - 1:Z_mid]
    Jzetar = Jzeta[:, Z_mid - 1:Z_mid]
    psir = psi[:, Z_mid - 1:Z_mid]

    # subplot(3,2,3)
    mplot.axes("position", mat.row(0.08, 0.41, 0.425, 0.25))  # fig-2
    mplot.plot(100.0 * loaded["r_rr"], loaded["pmrr"] / mat.mmax(loaded["pmrr"]), "-",
               100.0 * rr, Pr / mat.mmax(Pr), "--",
               100.0 * loaded["r_rr"], loaded["prr"] / mat.mmax(loaded["prr"]),
               "LineWidth", 3)
    mplot.hold_on()
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.grid_on()
    mplot.ylabel("$P/P_{m}$", "FontSize", fontsize)
    h = mplot.legend("$MRR-1D$", "$MRR-2D$", "$RR$", "location", "best")
    mplot.set_(h, "interpreter", "latex")
    mplot.legend("boxoff")
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k", "GridAlpha", 0.25)
    mplot.set_(mplot.gca(), "Ytick", mat.colon(0, 0.25, 1))
    mplot.text("Interpreter", "latex", "String", "$(b)$",
               "Position", mat.row(0.2, 0.8), "FontSize", fontsize)
    mplot.xlim(mat.row(0, 100 * mat.scalar_value(rm)))

    # subplot(3,2,4)
    mplot.axes("position", mat.row(0.565, 0.41, 0.425, 0.25))  # fig-3
    mplot.plot(100.0 * loaded["r_rr"], loaded["Bzmrr"], "-",
               100.0 * rr, Bzr, "--",
               100.0 * loaded["r_rr"], loaded["Bzrr"], "LineWidth", 3)
    mplot.hold_on()
    mplot.set_(mplot.gca(), "Fontsize", 25)
    mplot.grid_on()
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k", "GridAlpha", 0.25)
    mplot.ylabel("$B_z (T)$", "Fontsize", fontsize)
    mplot.text("Interpreter", "latex", "String", "$(c)$",
               "Position", mat.row(0.2, 0.35), "FontSize", fontsize)
    mplot.xlim(mat.row(0, 100 * mat.scalar_value(rm)))
    mplot.set_(mplot.gca(), "Xtick", mat.colon(0, 5, 20))

    # subplot(3,2,5)
    mplot.axes("position", mat.row(0.08, 0.095, 0.425, 0.25))  # fig-4
    mplot.plot(100.0 * loaded["r_rr"],
               loaded["Jr_mrr"] / mat.mmax(loaded["Jr_mrr"]), "-",
               100.0 * rr, -Jzetar / mat.mmax(-Jzetar), "--",
               100.0 * loaded["r_rr"],
               loaded["Jr_rr"] / mat.mmax(loaded["Jr_rr"]), "LineWidth", 3)
    mplot.hold_on()
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.grid_on()
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k", "GridAlpha", 0.25)
    mplot.ylabel("$J_{\\theta}/ J_m$", "Fontsize", fontsize)
    mplot.set_(mplot.gca(), "Ytick", mat.colon(0, 0.25, 1))
    mplot.text("Interpreter", "latex", "String", "$(d)$",
               "Position", mat.row(0.2, 0.8), "FontSize", fontsize)
    mplot.xlim(mat.row(0, 100 * mat.scalar_value(rm)))
    mplot.xlabel("$r [cm] \\ with \\ Z=0$", "interpreter", "latex",
                 "Fontsize", fontsize)

    # subplot(3,2,6)
    mplot.axes("position", mat.row(0.565, 0.095, 0.425, 0.25))  # fig-5
    rr_data = mat.importdata("rr_data.mat", " ")
    rw1 = rr_data[:, 0]; P2w1 = rr_data[:, 1]
    diag_data = mat.importdata("diag_data.mat", " ")
    rw2 = diag_data[:, 0]; P2w2 = diag_data[:, 1]
    f_data = mat.importdata("f.mat", " ")
    rf = f_data[:, 0]; P2f = f_data[:, 1]
    sf_data = mat.importdata("s_f.mat", " ")
    rsf = sf_data[:, 0]; P2sf = sf_data[:, 1]
    p_data = mat.importdata("p.mat", " ")
    rp = p_data[:, 0]; P2p = p_data[:, 1]
    loaded_mrr = mio.load("mrr.mat")
    mrr1d_r = loaded_mrr["rx"]; mrr1d_p2 = loaded_mrr["Px2"]
    mplot.plot(100.0 * loaded["r_rr"],
               loaded["psi_r"] / abs(mat.mmin(loaded["psi_r"])),
               100.0 * rr, psir / abs(mat.mmin(psir)), "--", "LineWidth", 3)
    mplot.hold_on()
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    mplot.grid_on()
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k", "GridAlpha", 0.25)
    mplot.xlim(mat.row(0, 6))
    mplot.set_(mplot.gca(), "Ytick", mat.colon(-1, 0.5, 1))
    mplot.xlabel("$r [cm] \\ with \\ Z=0$", "interpreter", "latex",
                 "Fontsize", fontsize)
    mplot.ylabel("$\\psi / \\psi_{min}$", "interpreter", "latex",
                 "Fontsize", fontsize)
    mplot.text("Interpreter", "latex", "String", "$(e)$",
               "Position", mat.row(0.08, 0.6), "FontSize", fontsize)

    for name, value in (("fontsize", fontsize), ("C1", mat.cellstr(C1)),
                        ("i", i), ("ido", ido), ("indtmp", indtmp), ("psio", psio),
                        ("psii", psii), ("c1", c1), ("h1", h1), ("zx1", zx1),
                        ("rx1", rx1), ("zx_no", mat.scalar(zx_no)), ("indx", indx),
                        ("rs", rs), ("Z_mid", mat.scalar(Z_mid)), ("Pr", Pr),
                        ("Bzr", Bzr), ("Jzetar", Jzetar), ("psir", psir),
                        ("rr_data", rr_data), ("rw1", rw1), ("P2w1", P2w1),
                        ("diag_data", diag_data), ("rw2", rw2), ("P2w2", P2w2),
                        ("f_data", f_data), ("rf", rf), ("P2f", P2f),
                        ("sf_data", sf_data), ("rsf", rsf), ("P2sf", P2sf),
                        ("p_data", p_data), ("rp", rp), ("P2p", P2p),
                        ("mrr1d_r", mrr1d_r), ("mrr1d_p2", mrr1d_p2), ("h", h)):
        ws[name] = value
    for name, value in loaded.items():
        ws[name] = value
    for name, value in loaded_mrr.items():
        ws[name] = value
