"""plt_psi_r.m 的移植：第一篇论文那组剖面图（2×3 布局，其中两格是左右双 y 轴）。

**会话链的末端**：二维量（`psi/R/Z/Bz/Br/Jzeta/P/jrb/rm/zw/nZ/rr/zz/zzb/rrb`）与
`psiq.mat` 带来的 `Izeta/Cq/C1q/C2q` 都来自上游——MATLAB 里的用法是"跑完
`figure_8_1.m` 之后再 run 这一份"，所以**单跑会停**、接在 `figure_8_1` 后面就通。

它比前几份多两样东西：

* `yyaxis left|right`：同一格里画 \(B_z\)（左轴）与 \(B_r\)（右轴），两者量级差很多；
* `trapz` 加权平均：`P_ave`、`Bz_o`、`Bz_w`，以及分界面处压强梯度长度 `lp`。

脚本里有几处 `fprintf` 是**诊断输出**（电流、平均 β、Lp），MATLAB 的 `%5.4d` 遇到
非整数会退回 `%e` 打印，移植版按同样的格式印出来，方便与原始输出对照。

验收：§7「绘图脚本」那一节的三条（能跑通 / 结构对 / 人眼比对论文图）。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mplot

#: 这个脚本要求调用方工作区里已经有这些（来自 `figure_8_1.m` 那次运行）
NEEDS_WORKSPACE = ("psi", "R", "Z", "zw", "nZ", "rm", "rr", "zz", "zzb", "rrb",
                   "Jzeta", "P", "Bz", "Br", "jrb", "Izeta", "Cq", "C1q", "C2q",
                   "dR", "dZ")


def _matlab_int_text(value, precision: int = 4) -> str:
    """MATLAB 的 `fprintf('%5.4d', x)`：x 不是整数时 MATLAB 会退回 `%e` 打印。

    实测（R2023a）：`fprintf('%5.4d', 382704.6)` 印出来是 `3.8270e+05`。
    """
    value = float(value)
    if value == int(value):
        return f"{int(value):d}"
    return f"{value:.{precision}e}"


def run(ws) -> None:
    missing = [name for name in NEEDS_WORKSPACE if name not in ws.keys()]
    if missing:
        raise KeyError(
            f"plt_psi_r 需要在 figure_8_1 之后跑（同一个会话），缺少："
            f"{', '.join(missing)}（原脚本直接用上游算出的二维 Bz/Br/Jzeta 与 rm）"
        )

    psi = np.asarray(ws["psi"]); R = np.asarray(ws["R"]); Z = np.asarray(ws["Z"])
    zw = ws["zw"]
    nZ = int(mat.scalar_value(ws["nZ"]))
    rm = ws["rm"]
    rr = np.asarray(ws["rr"]).ravel(); zz = np.asarray(ws["zz"]).ravel()
    zzb = np.asarray(ws["zzb"]); rrb = np.asarray(ws["rrb"])
    Jzeta = np.asarray(ws["Jzeta"]); P = np.asarray(ws["P"])
    Bz = np.asarray(ws["Bz"]); Br = np.asarray(ws["Br"])
    jrb = np.asarray(ws["jrb"])
    dR = mat.scalar_value(ws["dR"]); dZ = mat.scalar_value(ws["dZ"])
    Izeta = ws["Izeta"]; Cq = ws["Cq"]; C1q = ws["C1q"]; C2q = ws["C2q"]

    mplot.close_all()  # close all;
    hf = mplot.figure("unit", "normalized", "Position", mat.row(0.4, 0.2, 0.6, 0.5),
                      "DefaultAxesFontSize", 14)
    # % global rs;

    Zs = mat.scalar(0.5)
    mplot.subplot(2, 3, mat.colon(1, 1, 2))
    # % Jzetar=-Jzeta(:,floor(Zs*nZ));
    ido = mat.find(psi == mat.mmin(psi))
    indtmp = psi > 0
    psio = np.array(psi, copy=True)
    psio[np.asarray(indtmp)] = np.nan
    indtmp = mat.find(psi < 0)
    psii = np.array(psi, copy=True)
    psii[np.asarray(psi < 0)] = np.nan
    mplot.contour(Z, R, psio, 7, "linewidth", 2)
    mplot.hold_on()
    mplot.contour(Z, R, psii, 15, "linewidth", 2)
    mplot.hold_on()
    mplot.colorbar()
    c1, h1 = mplot.contour(Z, R, psi, mat.row(0, 0), "r--", "linewidth", 3,
                           return_matrix=True)
    mplot.plot(mat.take(Z, int(mat.el(ido, 1))), mat.take(R, int(mat.el(ido, 1))),
               "mx", "linewidth", 3)   # O-point
    mplot.text(0.01 * mat.scalar_value(zw), mat.take(R, int(mat.el(ido, 1))),
               "O point", "fontsize", 13)

    mplot.ylabel("r (m)")
    zx1 = np.asarray(c1)[0, 1:-1]
    rx1 = np.asarray(c1)[1, 1:-1]
    zx_no = np.size(zx1, 0)
    indx = mat.find(np.abs(zx1) == np.min(np.abs(zx1)))
    mplot.title(
        "\\psi, I_z=" + mat.num2str(Izeta / 1e6) + "MA, r_o="
        + mat.num2str(mat.take(R, int(mat.el(ido, 1)))) + ", r_x="
        + mat.num2str(mat.el(rx1, int(mat.el(indx, 1)))) + ", \\psi_{min}="
        + mat.num2str(mat.mmin(psi), 3))
    # % strp='p(\psi)=C*sech[C_2*(\psi-C_3)]';
    # 原文写的是 `-C\cdotexp[...]`（`\cdot` 与 `exp` 之间**没有空格**）。MATLAB 的
    # TeX 解释器按"已知命令"扫描，会渲染成 `\cdot exp`；matplotlib 的 mathtext 会
    # 把 `\cdotexp` 当成一个未知命令直接报 ParseFatalException，所以移植版在这里
    # 补一个空格——渲染出来的字形与原文一致，只是把这条路径留得能跑。
    strp = "p(\\psi)=-C\\cdot exp[-C1\\cdot\\psi] \\times (C2\\cdot\\psi+1)^2"
    mplot.xlabel("z(m), C=" + mat.num2str(Cq) + ", C1=" + mat.num2str(C1q)
                 + ", C2=" + mat.num2str(C2q) + ", " + strp)
    mplot.ylim(mat.row(0, mat.scalar_value(rm) + 0.01))
    # % set(gca,'ytick',[0 0.2 0.4 0.6 0.8]);
    mplot.set_(mplot.gca(), "ytick", mat.colon(0, 0.05, 0.20))

    # 原文这一句是裸表达式（只把结果丢进 ans），照抄成无副作用的计算；
    # MATLAB 的 min 忽略 NaN（这里的 ψ 带 jrb 掩膜）
    _ = mat.mmin(psi[:, int(np.floor(mat.scalar_value(Zs) * nZ)) - 1:][:, :1])

    rs = mat.el(rx1, int(mat.el(indx, 1)))

    mplot.plot(zzb, rrb, "k-", "linewidth", 3)
    mplot.subplot(233)
    R0 = int(np.floor(float(mat.take(R, int(mat.el(ido, 1)))) / mat.scalar_value(
        ws["rw"]) * int(mat.scalar_value(ws["nR"]))))
    mplot.yyaxis("left")
    mplot.plot(zz, Bz[R0 - 1, :], "-h", "LineWidth", 1.5, "MarkerSize", 4)
    mplot.box("on")
    mplot.grid_on()
    mplot.hold_on()
    mplot.ylabel("$B_{z} \\ (T)$", "interpreter", "latex", "FontSize", 17)
    mplot.yyaxis("right")
    mplot.plot(zz, Br[R0 - 1, :], "-h", "LineWidth", 1.5, "MarkerSize", 4)
    mplot.box("on")
    mplot.grid_on()
    mplot.hold_on()
    mplot.set_(mplot.gca(), "Fontsize", 17)
    mplot.xlabel("$z\\ (m),r=r_{o}$", "interpreter", "latex", "FontSize", 17)
    mplot.ylabel("$B_{r} \\ (T)$", "interpreter", "latex", "FontSize", 17)
    mplot.xlim(mat.row(-mat.scalar_value(zw), mat.scalar_value(zw)))
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k", "GridAlpha", 1)
    h = mplot.legend("$B_{z}$", "$B_{r}$")
    mplot.set_(h, "interpreter", "latex")
    mplot.legend("boxoff")

    mplot.subplot(234)
    Jzetar = -Jzeta[:, int(np.floor(mat.scalar_value(Zs) * nZ)) - 1:][:, :1]

    # % tot_I=sum(Jzetar(indx(1)).*dR);
    # 修改版：原文用"等值线点的序号" `indx(1)` 当径向下标的上限。那个序号是
    # 等值线矩阵里的位置，量纲上与网格无关；它在**随论文附的老网格**（nR=257）
    # 上碰巧小于行数，所以没人发现。本脚本现在要接 `gseq.m` 自己产出的解
    # （nR=129），序号 130 直接越界。这里按打印出来的物理含义——"中平面上分界面
    # 以内的电流"——用分界面半径 `rs` 定出径向下标。（`rs` 就是上面那条等值线
    # 在中平面上的 r，语义没变。）
    j_rs = int(round(mat.scalar_value(rs) / dR)) + 1
    j_rs = max(1, min(j_rs, int(np.shape(Jzetar)[0])))
    tot_I1 = 0.0
    for ii in mat.mrange(1, j_rs):
        tot_I1 = tot_I1 + float(Jzetar[ii - 1, 0]) * dR
    # % tot_I=tot_I1*L;

    L = mat.mmax(np.asarray(c1)[0, :]) - mat.mmin(np.asarray(c1)[0, :])
    tot_I = tot_I1 * L
    Jz1 = 0.0
    for jz in mat.mrange(1, nZ):        # 19-06-23 22:00 update
        for jr in mat.mrange(1, int(mat.el(jrb, jz)) - 1):
            if float(psio[jr - 1, jz - 1]) <= 0:      # NaN 比较为假，与 MATLAB 一致
                Jz1 = Jz1 + float(Jzeta[jr - 1, jz - 1]) * dR * dZ
    print(f"I at the midplane inside the separatrix is: {_matlab_int_text(tot_I)} ")
    print(f"I inside the separatrix is: {_matlab_int_text(Jz1)} ")

    Pr = P[:, int(np.floor(mat.scalar_value(Zs) * nZ)) - 1:][:, :1]
    J0 = mat.mmax(np.abs(Jzetar))
    P0 = mat.mmax(np.abs(Pr))
    mplot.plot(rr, Jzetar / J0, "--", rr, Pr / P0, "linewidth", 2)
    mplot.grid_on()
    mplot.hold_on()
    mplot.plot(mat.row(float(np.ravel(rs)[0]), float(np.ravel(rs)[0])),
               mat.row(0, 1), "-.k", "LineWidth", 1.5)
    mplot.hold_on()
    mplot.set_(mplot.gca(), "Fontsize", 17)
    mplot.xlabel("$r\\ (m), z=0$", "interpreter", "latex", "FontSize", 17)
    mplot.ylabel("P_0=" + mat.num2str(P0, 4) + ", J_0=" + mat.num2str(J0, 4))
    h = mplot.legend("$-J/J_0$", "$P/P_0$")
    mplot.set_(h, "interpreter", "latex")
    mplot.legend("boxoff")
    mplot.xlim(mat.row(0, 0.1))
    mplot.set_(mplot.gca(), "xtick", mat.colon(0, 0.02, 0.1))
    mplot.text("Interpreter", "latex", "String", "$Separatrix$",
               "Position", mat.row(float(np.ravel(rs)[0]) - 0.02, 0.15),
               "FontSize", 17)
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k", "GridAlpha", 1)

    P_ave = 0.0
    r_dr = 0.0
    Rend = 0.4
    PP = np.zeros(rr.size)
    rrs = np.zeros(rr.size)
    No = 0
    for ii in mat.mrange(1, int(rr.size)):
        if rr[ii - 1] <= float(np.ravel(rs)[0]):
            PP[ii - 1] = Pr[ii - 1, 0]
            rrs[ii - 1] = rr[ii - 1]
            No = ii
    # %     if rr(ii)<=Rend ... （原文里这几行注释掉了）
    PP = PP[:No]
    rrs = rrs[:No]

    rrzw = np.zeros(rr.size - 1)
    Bwz = np.zeros(rr.size - 1)
    for ii in mat.mrange(1, int(rr.size) - 1):
        if rr[ii] <= Rend:
            rrzw[ii - 1] = rr[ii - 1]
            Bwz[ii - 1] = Bz[ii - 1, int(np.floor(0.99 * nZ)) - 1]

    rrw = np.zeros(rr.size - No)
    Bw = np.zeros(rr.size - No)
    for jj in mat.mrange(No + 1, int(rr.size)):
        rrw[jj - No - 1] = rr[jj - 1]
        Bw[jj - No - 1] = Bz[jj - 1, int(np.floor(0.5 * nZ)) - 1]

    print("********************************************************* ")
    P_ave = mat.trapz(rrs, rrs * PP) / mat.trapz(rrs, rrs)
    mu0 = mat.scalar(4 * np.pi * 1e-7)
    Bx = 0.0341
    ave_beta = 2 * mat.scalar_value(mu0) * P_ave / (Bx * Bx)
    print(f"ave_beta = {ave_beta:5.4f} ")

    # % K=0.530509;
    # % fprintf('ave_beta = %5.4f \n',tanh(K)/K);

    Bz_o = mat.trapz(rrw, rrw * Bw) / mat.trapz(rrw, rrw)
    Bz_w = mat.trapz(rrzw, rrzw * Bwz) / mat.trapz(rrzw, rrzw)

    N_rs = int(np.floor(float(np.ravel(rs)[0]) / mat.scalar_value(ws["rw"])
                        * int(mat.scalar_value(ws["nR"]))))
    ps = (Pr[N_rs - 1, 0] + Pr[N_rs, 0]) / 2
    dpdr = (Pr[N_rs, 0] - Pr[N_rs - 1, 0]) / (rr[N_rs - 1] - rr[N_rs])
    lp = ps / dpdr
    print(f"Lp = {lp:5.4f} ")
    print("********************************************************* ")

    mplot.subplot(235)
    mplot.yyaxis("left")
    mplot.plot(rr, Bz[:, int(np.floor(mat.scalar_value(Zs) * nZ)) - 1], "-h",
               "LineWidth", 1.5, "MarkerSize", 4)
    mplot.box("on")
    mplot.grid_on()
    mplot.hold_on()
    mplot.plot(mat.row(float(np.ravel(rs)[0]), float(np.ravel(rs)[0])),
               mat.row(1.2 * mat.mmin(Bz[:, int(np.floor(mat.scalar_value(Zs) * nZ)) - 1]),
                       mat.mmax(Bz[:, int(np.floor(mat.scalar_value(Zs) * nZ)) - 1])),
               "-.k", "LineWidth", 1.5)
    mplot.hold_on()
    mplot.ylabel("$B_{z} \\ (T)$", "interpreter", "latex", "FontSize", 17)
    mplot.yyaxis("right")
    mplot.plot(rr, Br[:, int(np.floor(mat.scalar_value(Zs) * nZ)) - 1], "-h",
               "LineWidth", 1.5, "MarkerSize", 4)
    mplot.box("on")
    mplot.grid_on()
    mplot.hold_on()
    mplot.set_(mplot.gca(), "Fontsize", 17)
    mplot.xlabel("$r\\ (m),z=0$", "interpreter", "latex", "FontSize", 17)
    mplot.ylabel("$B_{r} \\ (T)$", "interpreter", "latex", "FontSize", 17)
    mplot.xlim(mat.row(0, 0.1))
    mplot.set_(mplot.gca(), "xtick", mat.colon(0, 0.02, 0.1))
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k", "GridAlpha", 1)
    # % legend('B_z','B_r');legend('boxoff');
    h = mplot.legend("$B_{z}$", "$B_{r}$")
    mplot.set_(h, "interpreter", "latex")
    mplot.legend("boxoff")

    mplot.subplot(236)
    mplot.yyaxis("left")
    Z_end = 0.99
    mplot.plot(rr, Bz[:, int(np.floor(Z_end * nZ)) - 1], "-h", "LineWidth", 1.5,
               "MarkerSize", 4)
    mplot.box("on")
    mplot.grid_on()
    mplot.hold_on()
    mplot.ylabel("$B_{z} \\ (T)$", "interpreter", "latex", "FontSize", 17)
    mplot.text("Interpreter", "latex", "String", "$Separatrix$", "Position",
               mat.row(float(np.ravel(rs)[0]) - 0.02,
                       1.1 * mat.mmin(Bz[:, int(np.floor(Z_end * nZ)) - 1])),
               "FontSize", 17)
    mplot.yyaxis("right")
    mplot.plot(rr, Br[:, int(np.floor(Z_end * nZ)) - 1], "-h", "LineWidth", 1.5,
               "MarkerSize", 4)
    mplot.box("on")
    mplot.grid_on()
    mplot.hold_on()
    mplot.set_(mplot.gca(), "Fontsize", 17)
    mplot.xlabel("$r\\ (m),z=0.99 z_{w}$", "interpreter", "latex", "FontSize", 17)
    mplot.ylabel("$B_{r} \\ (T)$", "interpreter", "latex", "FontSize", 17)
    mplot.xlim(mat.row(0, 0.15))
    mplot.set_(mplot.gca(), "xtick", mat.colon(0, 0.05, 0.15))
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k", "GridAlpha", 1)
    h = mplot.legend("$B_{z}$", "$B_{r}$")
    mplot.set_(h, "interpreter", "latex")
    mplot.legend("boxoff")

    mplot.set_(mplot.gcf(), "PaperPositionMode", "auto")
    # % RemovePlotWhiteArea(gca);

    for name, value in (("hf", hf), ("Zs", Zs), ("ido", ido), ("indtmp", indtmp),
                        ("psio", psio), ("psii", psii), ("c1", c1), ("h1", h1),
                        ("zx1", zx1), ("rx1", rx1), ("zx_no", mat.scalar(zx_no)),
                        ("indx", indx), ("strp", strp), ("rs", rs),
                        ("R0", mat.scalar(R0)), ("tot_I1", mat.scalar(tot_I1)),
                        ("L", mat.scalar(L)), ("tot_I", mat.scalar(tot_I)),
                        ("Jz1", mat.scalar(Jz1)), ("Jzetar", Jzetar),
                        ("Pr", Pr), ("J0", mat.scalar(J0)), ("P0", mat.scalar(P0)),
                        ("P_ave", mat.scalar(P_ave)), ("r_dr", mat.scalar(r_dr)),
                        ("Rend", mat.scalar(Rend)), ("PP", PP), ("rrs", rrs),
                        ("No", mat.scalar(No)), ("rrzw", rrzw), ("Bwz", Bwz),
                        ("rrw", rrw), ("Bw", Bw), ("mu0", mu0), ("Bx", mat.scalar(Bx)),
                        ("ave_beta", mat.scalar(ave_beta)), ("Bz_o", mat.scalar(Bz_o)),
                        ("Bz_w", mat.scalar(Bz_w)), ("N_rs", mat.scalar(N_rs)),
                        ("ps", mat.scalar(ps)), ("dpdr", mat.scalar(dpdr)),
                        ("lp", mat.scalar(lp)), ("Z_end", mat.scalar(Z_end)),
                        ("h", h)):
        ws[name] = value
