"""plt_psi_p.m（Bz_fixed_072 副本）的移植：6 面板诊断图 + 留下二维 `Bz` 给下游。

与 `Fig4/figure_4_1.m` 同族（同一套中心差分 → 积分出 p(ψ) → 6 面板 pcolor），差别是：
数据读 `psiq.mat`、多算两个电流积分（`Jztotal`/`Jztotal_inside`）、`rm=0.15`、
**不写** `frceq.dat`（原文注释掉了）。

它是 `plot_Bz_radial.m` 的**正牌上游**：跑完之后工作区里有了二维 `Bz`（以及 `rm`），
下游那个"沿半径画 B_z"的脚本才能跑。测试里就是按这个会话链验的。

验收：§7「绘图脚本」那一节的三条。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mio
from gseqfrc import mplot


def run(ws) -> None:
    mplot.close_all()  # close all; clear; clc;

    # % load('psiq.mat');
    loaded = mio.load("psiq.mat")
    # % load('psiq_I-1.0_ln-040.mat');
    Rq, Zq, psiq, Jzeta = loaded["Rq"], loaded["Zq"], loaded["psiq"], loaded["Jzeta"]
    jrb = loaded["jrb"]

    rw = mat.scalar(np.max(np.max(Rq)))
    zw = mat.scalar(np.max(np.max(Zq)))
    zwa = mat.scalar(np.min(np.min(Zq)))
    nRq, nZq = np.shape(psiq)

    nR, nZ = int(nRq), int(nZq)
    # % nR=2^6+1; nZ=2^8+1;
    Rlow = mat.scalar(0); Rup = rw
    Zlow = zwa; Zup = zw
    rr = mat.linspace(Rlow, Rup, nR)
    zz = mat.linspace(Zlow, Zup, nZ)
    dR = mat.scalar(mat.el(rr, 2) - mat.el(rr, 1))
    dZ = mat.scalar(mat.el(zz, 2) - mat.el(zz, 1))
    dR_v = mat.scalar_value(dR)
    dZ_v = mat.scalar_value(dZ)
    R, Z = mat.ndgrid(rr, zz)
    dR2 = dR * dR
    dZ2 = dZ * dZ
    dR2_v = mat.scalar_value(dR2)
    dZ2_v = mat.scalar_value(dZ2)
    psi = mat.interp2(Zq, Rq, psiq, Z, R, "spline")  # 'cubic'
    # % psi=interp2(Zq,Rq,psiq,Z,R); % not ok for J&dp, due to 'linear' interp
    Jztotal = np.sum(np.sum(np.asarray(Jzeta))) * dR_v * dZ_v

    data_psi = psi < 0
    Jzx = np.array(Jzeta, copy=True)
    Jzx[np.asarray(data_psi)] = 0
    Jztotal_inside = np.sum(np.sum(Jzx)) * dR_v * dZ_v   # 原文是裸表达式（只显示）

    Br = 0.0 * psi
    Bz = 0.0 * psi
    dp = 0.0 * psi
    J = 0.0 * psi
    for jZ in mat.mrange(2, nZ - 1):
        for jR in mat.mrange(2, nR - 1):
            Br[jR - 1, jZ - 1] = -1 / R[jR - 1, jZ - 1] * (
                psi[jR - 1, jZ] - psi[jR - 1, jZ - 2]) / (2 * dZ_v)
            Bz[jR - 1, jZ - 1] = 1 / R[jR - 1, jZ - 1] * (
                psi[jR, jZ - 1] - psi[jR - 2, jZ - 1]) / (2 * dR_v)
            J[jR - 1, jZ - 1] = -1 / R[jR - 1, jZ - 1] * (
                (psi[jR, jZ - 1] - 2 * psi[jR - 1, jZ - 1] + psi[jR - 2, jZ - 1]) / (dR2_v)
                - 1 / R[jR - 1, jZ - 1] * (psi[jR, jZ - 1] - psi[jR - 2, jZ - 1]) / (2 * dR_v)
                + (psi[jR - 1, jZ] - 2 * psi[jR - 1, jZ - 1] + psi[jR - 1, jZ - 2]) / (dZ2_v))
            # %         J(jR,jZ)=((psi(jR,jZ+1)-2*psi(jR,jZ)+psi(jR,jZ-1))); % to check
            dp[jR - 1, jZ - 1] = J[jR - 1, jZ - 1] / R[jR - 1, jZ - 1]

    for jZ in mat.mrange(1, nZ):
        psi[0, jZ - 1] = psi[1, jZ - 1]
        psi[nR - 1, jZ - 1] = psi[nR - 2, jZ - 1]
        Br[0, jZ - 1] = Br[1, jZ - 1]
        Br[nR - 1, jZ - 1] = Br[nR - 2, jZ - 1]
        Bz[0, jZ - 1] = Bz[1, jZ - 1]
        Bz[nR - 1, jZ - 1] = Bz[nR - 2, jZ - 1]
        J[0, jZ - 1] = J[1, jZ - 1]
        J[nR - 1, jZ - 1] = J[nR - 2, jZ - 1]
        dp[0, jZ - 1] = dp[1, jZ - 1]
        dp[nR - 1, jZ - 1] = dp[nR - 2, jZ - 1]

        for jR in mat.mrange(int(mat.el(jrb, jZ)) + 1, nR):
            # %         psi(jR,jZ)=0.0; ... （原文这几行注释掉了，走的是下面的 NaN）
            psi[jR - 1, jZ - 1] = np.nan
            Br[jR - 2, jZ - 1] = np.nan
            Bz[jR - 2, jZ - 1] = np.nan
            J[jR - 2, jZ - 1] = np.nan
            dp[jR - 2, jZ - 1] = np.nan
    # % dp(dp>1e3)=NaN;

    jRo, jZo = mat.find2(psi == np.min(psi))
    jRm, jZm = mat.find2(psi == np.max(psi))
    ro = mat.ij(R, jRo[0], jZo[0])
    zo = mat.ij(Z, jRo[0], jZo[0])
    zm = mat.ij(Z, jRm[0], jZm[0])
    # % rm=R(jRm(1),jZm(1));
    # % rm=0.14;
    rm = mat.scalar(0.15)

    # %%
    mplot.close_all()
    hf = mplot.figure("unit", "normalized", "Position", mat.row(0.6, 0.1, 0.4, 0.7),
                      "DefaultAxesFontSize", 12)
    hs1 = mplot.subplot(611)
    # % set(hs1,'position',[0.1,0.7,0.8,0.1]);
    mplot.pcolor(Z, R, psi)
    mplot.shading("interp")
    mplot.hold_on()
    mplot.colorbar("northOutside")
    mplot.set_(mplot.gca(), "Fontsize", 17)
    # % vv=[0,0.01,-0.001,-0.01,-0.005,0.15,0.4];
    psimin = mat.scalar(np.min(psi))
    vv = mat.row(0, 0.01, 0.15, 0.4, 0.9 * psimin, 0.3 * psimin, 0.03 * psimin)
    C1, h1 = mplot.contour(Z, R, psi, vv, "linewidth", 1, "linestyle", "-",
                           "color", "k", "visible", "off", return_matrix=True)
    mplot.hold_on()
    C2, h2 = mplot.contour(Z, R, psi, mat.row(0, 0), "linewidth", 1, "linestyle",
                           "--", "color", "r", "visible", "off",
                           return_matrix=True)
    mplot.hold_on()
    mplot.ylabel("R (m)")
    mplot.axis("equal")
    mplot.axis("tight")
    mplot.box("on")
    mplot.plot(zo, ro, "rx")
    mplot.ylim(mat.row(0, rm))
    mplot.set_(mplot.gca(), "ytick", mat.colon(0, 0.2, rm))

    for k, field in ((612, Br), (613, Bz), (614, J), (615, dp)):
        mplot.subplot(k)
        mplot.pcolor(Z, R, field)
        mplot.shading("interp")
        mplot.hold_on()
        mplot.colorbar("northOutside")
        mplot.set_(mplot.gca(), "Fontsize", 17)
        mplot.ylabel("R (m)")
        mplot.axis("equal")
        mplot.axis("tight")
        mplot.box("on")
        mplot.set_(mplot.gca(), "ytick", mat.colon(0, 0.2, rm))

    JBz = J * Bz
    JBr = J * Br
    # % subplot(511); hold on;plot(zo,ro,'x',zm,rm,'+'); hold on;

    # % calculate p(r,z), 19-05-30 15:36
    imethod = mat.scalar(2)
    p = 0.0 * R
    if mat.scalar_value(imethod) == 1:  # calculate from dpsi
        for jR in mat.mrange(2, nR):
            p[jR - 1, jZo[0] - 1] = (p[jR - 2, jZo[0] - 1]
                                     + 0.5 * (dp[jR - 2, jZo[0] - 1] + dp[jR - 1, jZo[0] - 1])
                                     * (psi[jR - 1, jZo[0] - 1] - psi[jR - 2, jZo[0] - 1]))
        for jR in mat.mrange(1, nR):
            for jZ in list(range(jZo[0] - 1, 0, -1)):
                p[jR - 1, jZ - 1] = (p[jR - 1, jZ]
                                     + 0.5 * (dp[jR - 1, jZ - 1] + dp[jR - 1, jZ])
                                     * (psi[jR - 1, jZ - 1] - psi[jR - 1, jZ]))
            for jZ in mat.mrange(jZo[0] + 1, nZ):
                p[jR - 1, jZ - 1] = (p[jR - 1, jZ - 2]
                                     + 0.5 * (dp[jR - 1, jZ - 1] + dp[jR - 1, jZ - 2])
                                     * (psi[jR - 1, jZ - 1] - psi[jR - 1, jZ - 2]))
    else:  # calculate from dr and dz
        for jR in mat.mrange(2, nR):
            p[jR - 1, jZo[0] - 1] = (p[jR - 2, jZo[0] - 1]
                                     + 0.5 * (JBz[jR - 2, jZo[0] - 1] + JBz[jR - 1, jZo[0] - 1]) * dR_v)
        for jR in mat.mrange(1, nR):
            for jZ in list(range(jZo[0] - 1, 0, -1)):
                p[jR - 1, jZ - 1] = (p[jR - 1, jZ]
                                     + 0.5 * (JBr[jR - 1, jZ - 1] + JBr[jR - 1, jZ]) * dZ_v)
            for jZ in mat.mrange(jZo[0] + 1, nZ):
                p[jR - 1, jZ - 1] = (p[jR - 1, jZ - 2]
                                     - 0.5 * (JBr[jR - 1, jZ - 1] + JBr[jR - 1, jZ - 2]) * dZ_v)
    p = p - np.min(p) + 0.05 * (np.max(p - np.min(p)))
    mplot.subplot(6, 1, 6)

    mplot.pcolor(Z, R, p)
    mplot.shading("interp")
    mplot.hold_on()
    mplot.colorbar("northOutside")
    mplot.set_(mplot.gca(), "Fontsize", 17)
    mplot.xlabel("Z (m)")
    mplot.ylabel("R (m)")
    mplot.axis("equal")
    mplot.axis("tight")
    mplot.box("on")
    mplot.set_(mplot.gca(), "ytick", mat.colon(0, 0.2, rm))
    # % contour(Z,R,p);axis equal;

    # %%
    C1 = np.array(C1, copy=True)
    C2 = np.array(C2, copy=True)
    C1[C1 > 5] = np.nan
    C2[C2 > 5] = np.nan
    strj = mat.cellstr(["\\psi", "B_r", "B_z", "J_\\zeta", "dp/d\\psi", "p(\\psi)"])
    for j in mat.mrange(1, 6):
        mplot.subplot(6, 1, j)
        # 原文 `plot(C1(1,:),C1(2,:),...)`：第 1、2 **行**才是等值线的 (x, y)。
        # （2026-09-17 修：原来写成 `mat.take(C1,1)`——那是线性索引取第 1 个元素。）
        mplot.plot(np.asarray(C1)[0, :], np.asarray(C1)[1, :], "k", "linewidth", 1)
        mplot.plot(np.asarray(C2)[0, :], np.asarray(C2)[1, :], "r--", "linewidth", 1)
        stj = ["(", mat.char(96 + j), ") "]
        stj = mat.strcat(mat.el(strj, j))
        mplot.text(0.8 * mat.scalar_value(zw), 0.8 * mat.scalar_value(rw), stj,
                   "fontsize", 18, "color", "m")

    # %% （原文下面这段只算不写文件——`frceq.dat` 那几行是注释掉的）
    dpdr = 0.0 * p
    dpdz = 0.0 * p
    for jR in mat.mrange(2, nR - 1):
        for jZ in mat.mrange(2, nZ - 1):
            dpdr[jR - 1, jZ - 1] = (p[jR, jZ - 1] - p[jR - 2, jZ - 1]) / (2 * dR_v)
            dpdz[jR - 1, jZ - 1] = (p[jR - 1, jZ] - p[jR - 1, jZ - 2]) / (2 * dZ_v)
    with np.errstate(divide="ignore", invalid="ignore"):
        JBr_o_dpdz = -JBr / dpdz
        JBz_o_dpdr = JBz / dpdr

    jr, jz = mat.ndgrid(np.arange(1, nR + 1), np.arange(1, nZ + 1))
    eqdat = np.vstack([jr.reshape(1, -1, order="F"), jz.reshape(1, -1, order="F"),
                       R.reshape(1, -1, order="F"), Z.reshape(1, -1, order="F"),
                       psi.reshape(1, -1, order="F"), p.reshape(1, -1, order="F"),
                       J.reshape(1, -1, order="F")])

    for name, value in (("rw", rw), ("zw", zw), ("zwa", zwa),
                        ("nRq", mat.scalar(nRq)), ("nZq", mat.scalar(nZq)),
                        ("nR", mat.scalar(nR)), ("nZ", mat.scalar(nZ)),
                        ("Rlow", Rlow), ("Rup", Rup), ("Zlow", Zlow), ("Zup", Zup),
                        ("rr", rr), ("zz", zz), ("dR", dR), ("dZ", dZ), ("R", R),
                        ("Z", Z), ("dR2", dR2), ("dZ2", dZ2), ("psi", psi),
                        ("Br", Br), ("Bz", Bz), ("dp", dp), ("J", J),
                        ("jRo", jRo), ("jZo", jZo), ("jRm", jRm), ("jZm", jZm),
                        ("ro", ro), ("zo", zo), ("zm", zm), ("rm", rm),
                        ("psimin", psimin), ("vv", vv), ("C1", C1), ("C2", C2),
                        ("JBz", JBz), ("JBr", JBr), ("imethod", imethod), ("p", p),
                        ("strj", strj), ("dpdr", dpdr), ("dpdz", dpdz),
                        ("JBr_o_dpdz", JBr_o_dpdz), ("JBz_o_dpdr", JBz_o_dpdr),
                        ("eqdat", eqdat), ("jr", jr), ("jz", jz), ("Jztotal", mat.scalar(Jztotal)),
                        ("Jztotal_inside", mat.scalar(Jztotal_inside)),
                        ("data_psi", data_psi), ("Jzx", Jzx),
                        ("h1", h1), ("h2", h2), ("hs1", hs1)):
        ws[name] = value
    for name, value in loaded.items():
        ws[name] = value
