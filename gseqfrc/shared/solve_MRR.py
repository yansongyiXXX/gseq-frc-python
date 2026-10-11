"""solve_MRR.m 的逐行移植（**三份副本合并后**的唯一实现）：MRR 参数 + 一维径向平衡。

源文件（三份互不相同）：

    gseq-main/GSEQ-FRC-open/second_paper/Fig4/solve_MRR.m   （基础版）
    gseq-main/GSEQ-FRC-open/second_paper/Fig5/solve_MRR.m   （多一个 `icoil` 开关）
    gseq-main/GSEQ-FRC-open/second_paper/Fig6/solve_MRR.m   （参数块带注释掉的备选、绘图用 k--）

2026-09-17 副本合并（PORTING.md §8 的规则 2：**不是二选一，而是把差异提成显式参数**）：
三份的**求解器骨架逐字相同**（`fsolve(@fmrr0,...)` 解 sigma → 解 psim → 沿 r 逐点
用 `integral` + `fsolve` 解一维平衡 → 派生 Pr/dP/Jt/Bz → 写 `MRR1d.mat`），
差异只有下面这些，全部登记在 `PROFILES` 里：

| 差异 | Fig4 | Fig5 | Fig6 |
|---|---|---|---|
| 参数块 | betas=0.6, Be=1, Rs=0.075, Rw=0.14 | `icoil` 开关（本算例取 1：betas=0.7, Be=0.5, Rs=0.17, Rw=0.4） | betas=0.6, Be=1, Rs=0.075, Rw=0.14 |
| `ls` 系数 | `0.4*ls0` | `0.5*ls0` | `0.40*ls0`（与 0.4 同一个数） |
| `psig` / `x0` | −0.054 / 2.4 | −0.045 / 3.7 | −0.045 / 2.4 |
| 绘图 | 实线、`xlabel('r')` | 同 Fig4 | `k--`、`xlabel('r (cm)')` |
| 控制台输出 | `F1= ...` / `psibc= ...` | 多打印 `******` 与 `Izeta0= ...`，`F1=` 带尾空格 | 两个数值都不带前缀（位置也略不同） |
| 多出来的量 | — | `icoil`（进工作区，也进 `MRR1d.mat`） | — |

打印文本的**位置**在 Fig6 里略有不同，但打印不进工作区、不参与验收（§3），
所以这里统一成一处、文本按 profile 取。其余一字未动。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mglobals
from gseqfrc import mio
from gseqfrc import moptim
from gseqfrc import mpath
from gseqfrc import mplot

#: 三份副本的参数与呈现方式——差异都摆在这里，一条不少
PROFILES: dict[str, dict] = {
    "Fig4": dict(betas=0.6, Be=1.0, Rs=0.075, Rw=0.14, ls_factor=0.4,
                 psig=-0.054, x0=2.4, icoil=None, dashed=False,
                 f1_fmt="F1= {value:17.16f}", psibc_fmt="psibc= {value:17.16f}"),
    "Fig5": dict(betas=0.7, Be=0.5, Rs=0.17, Rw=0.4, ls_factor=0.5,
                 psig=-0.045, x0=3.7, icoil=1, dashed=False,
                 f1_fmt="F1= {value:17.16f} ", psibc_fmt="psibc= {value:17.16f}"),
    "Fig6": dict(betas=0.6, Be=1.0, Rs=0.075, Rw=0.14, ls_factor=0.40,
                 psig=-0.045, x0=2.4, icoil=None, dashed=True,
                 f1_fmt="{value:17.16f}", psibc_fmt="{value:17.16f}"),
}

#: `run()` 的默认 profile
DEFAULT_PROFILE = "Fig4"


def run(ws, profile=DEFAULT_PROFILE) -> None:
    P = PROFILES[profile]
    mplot.close_all()  # close all;clear;clc;
    G = mglobals.G
    mu0 = mat.scalar(4e-7 * np.pi)

    # global betas ls n tol alpha0 ls0 psim0 psig;
    G.declare("betas", "ls", "n", "tol", "alpha0", "ls0", "psim0", "psig")
    G.n = mat.scalar(2)
    G.tol = mat.scalar(1e-10)

    # %%%%%%%% 参数块：Fig5 那份是个 icoil 开关（本算例取 1），另两份是直写 %%%%%%%%
    icoil = None
    if P["icoil"] is not None:
        icoil = mat.scalar(P["icoil"])
        if mat.scalar_value(icoil) == 1:      # five coils
            betas, Be, Rs, Rw = 0.7, 0.5, 0.17, 0.4
            print(f"icoil= {int(mat.scalar_value(icoil)):2d} ")
        elif mat.scalar_value(icoil) == 2:    # three coils
            betas, Be, Rs, Rw = 0.7, 1, 0.075, 0.14
            print(f"icoil= {int(mat.scalar_value(icoil)):2d} ")
        else:
            raise ValueError(f"未覆盖的 icoil={P['icoil']}")
    else:
        betas, Be, Rs, Rw = P["betas"], P["Be"], P["Rs"], P["Rw"]
    G.betas = mat.scalar(betas)
    Be = mat.scalar(Be)
    Rs = mat.scalar(Rs)

    G.alpha0 = 8 * np.log((np.sqrt(1 - G.betas) + 1) / np.sqrt(G.betas))
    G.ls0 = 1 / (G.alpha0 * np.sqrt(1 - G.betas))
    G.psim0 = np.log(G.betas) / G.alpha0
    # % ls=0.148;
    G.ls = P["ls_factor"] * G.ls0

    Rw = mat.scalar(Rw)

    G.psig = mat.scalar(P["psig"])
    options = moptim.optimset(TolFun=G.tol, Tolx=G.tol, Maxfuneval=10000,
                              MaxIter=5000)

    x0 = mat.scalar(P["x0"])
    x = moptim.fsolve(lambda s: mpath.call("fmrr0", s), x0, options)
    F = mpath.call("fmrr0", x)
    sigma = np.real(x)
    alpha = ((1 / G.ls) / np.sqrt(1 - G.betas) + G.n * sigma)

    def F1(psim):
        return mat.exp(-alpha * psim) * (sigma * psim + 1) ** G.n - 1 / G.betas

    options2 = moptim.optimset(TolFun=G.tol, Tolx=G.tol, Display="off")
    psim = moptim.fsolve(F1, G.psig, options2)
    F1 = F1(psim)

    # %%
    mplot.close_all()

    def fpsi(psi_):
        return 1.0 / np.sqrt(1.0 - G.betas * mat.exp(-alpha * psi_)
                             * (sigma * psi_ + 1) ** G.n)

    dr = mat.scalar(0.005)
    rmax = Rw / Rs
    rr = mat.colon(0, dr, rmax)
    # % nr=rmax/dr;
    nr = mat.length(rr)
    psi = 0.0 * rr

    for jr in mat.mrange(2, nr):
        r = mat.el(rr, jr)
        if r <= 1 / np.sqrt(2):
            def FF1(psi_):
                return moptim.integral(fpsi, 0, psi_, 1e-8, 1e-8) + r ** 2 / 2

            # %         psi(jr)=fsolve(FF1,-0.01,options2);
            mat.set_el(psi, jr, moptim.fsolve(FF1, np.real(mat.el(psi, jr - 1)),
                                              options2))
        else:
            def FF2(psi_):
                return (moptim.integral(fpsi, psim, psi_, 1e-8, 1e-8)
                        - (r ** 2 - 1 / 2) / 2)

            # %         psi(jr)=fsolve(FF2,psim0,options2);
            mat.set_el(psi, jr, moptim.fsolve(FF2, 0.99 * np.real(mat.el(psi, jr - 1)),
                                              options2))

    # %%
    mplot.close_all()
    mplot.figure("unit", "normalized", "DefaultAxesFontSize", 16,
                 "DefaultAxesFontWeight", "bold", "DefaultAxesLineWidth", 2,
                 "position", mat.row(0.5, 0.5, 0.5, 0.6))
    fP = lambda psi_: G.betas * mat.exp(-alpha * psi_) * (sigma * psi_ + 1) ** G.n
    Pr = G.betas * mat.exp(-alpha * psi) * (sigma * psi + 1) ** G.n
    fdP = lambda psi_: (G.betas * mat.exp(-alpha * psi_) * (sigma * psi_ + 1) ** (G.n - 1)
                        * (-alpha * (sigma * psi_ + 1) + G.n * sigma))
    dP = (G.betas * mat.exp(-alpha * psi) * (sigma * psi + 1) ** (G.n - 1)
          * (-alpha * (sigma * psi + 1) + G.n * sigma))
    Jt = -rr * dP
    Bz = (-np.sqrt(1 - Pr) * (rr <= 1 / np.sqrt(2))
          + np.sqrt(1 - Pr) * (rr > 1 / np.sqrt(2)))

    ids = mat.el(mat.find(np.abs(rr - 1) == np.min(np.abs(rr - 1))), 1)
    lsp = mat.scalar(np.real(
        -(mat.el(Pr, ids) / ((mat.el(Pr, ids + 1) - mat.el(Pr, ids - 1))
                             / (mat.el(rr, ids + 1) - mat.el(rr, ids - 1))))
    ))

    ind = mat.find(rr <= 1.0)
    h = np.real(fdP(psim) / np.sum(2 * mat.take(dP, ind) * mat.take(rr, ind) * dr))

    # % ave_beta=sum(2*Pr(ind)./max(Pr).*rr(ind)*dr);
    ave_beta = (np.sum(2 * mat.take(Pr, ind) / np.max(Pr) * mat.take(rr, ind) * dr)
                / np.sum(2 * mat.take(rr, ind) * dr))
    tot_I = np.pi * np.sum(2 * mat.take(Jt, ind) * mat.take(rr, ind) * dr)
    tot_I2 = np.pi * np.sum(2 * Jt * rr * dr)

    Izeta0 = tot_I2 * ((Be * Rs) / (2 * mu0))

    if icoil is not None:   # Fig5 那一份多这一段打印（只影响控制台）
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        print("************************** ")
        print(f"Izeta0= {mat.scalar_value(Izeta0):17.16f} ")

    # % load('gs1d.mat');

    # Fig6 那一份把四条线画成 k--、横轴标签写 "r (cm)"，其余相同
    line_kw = dict(color="k", linestyle="--") if P["dashed"] else {}
    xlabel_text = "r (cm)" if P["dashed"] else "r"

    mplot.subplot(221)
    mplot.plot(rr * Rs * 100, psi * (Be * Rs ** 2), "linewidth", 2, **line_kw)
    mplot.hold_on()
    mplot.set_(mplot.gca(), "Fontsize", 15)
    mplot.xlabel(xlabel_text)
    mplot.ylabel("\\psi")
    mplot.xlim(mat.row(0, 100 * Rw))
    mplot.title(f"\\beta_s={mat.num2str(G.betas)}, l_s={mat.num2str(G.ls)}, "
                f"l_{{sp}}={mat.num2str(lsp, 3)}")

    mplot.subplot(222)
    mplot.plot(rr * Rs * 100, Pr * (Be ** 2 / (2 * mu0)), "linewidth", 2, **line_kw)
    mplot.set_(mplot.gca(), "Fontsize", 15)
    mplot.xlabel(xlabel_text)
    mplot.ylabel("P")
    mplot.xlim(mat.row(0, 100 * Rw))
    mplot.title(f"l_{{s0}}={mat.num2str(G.ls0)}, \\alpha={mat.num2str(alpha)}, "
                f"\\psi_g={mat.num2str(G.psig)}")

    mplot.subplot(223)
    mplot.plot(rr * Rs * 100, Jt * (Be / (2 * mu0 * Rs)), "linewidth", 2, **line_kw)
    mplot.set_(mplot.gca(), "Fontsize", 15)
    mplot.xlabel(xlabel_text)
    mplot.ylabel("J_\\theta")
    mplot.xlim(mat.row(0, 100 * Rw))
    mplot.title(f"\\sigma={mat.num2str(sigma)}, h={mat.num2str(h)}, "
                f"n={mat.num2str(G.n)}")

    mplot.subplot(224)
    mplot.plot(rr * Rs * 100, Bz * Be, "linewidth", 2, **line_kw)
    mplot.set_(mplot.gca(), "Fontsize", 15)
    mplot.xlabel(xlabel_text)
    mplot.ylabel("B_z")
    mplot.xlim(mat.row(0, 100 * Rw))
    mplot.title(f"\\psi_m={mat.num2str(psim)}, \\alpha_0={mat.num2str(G.alpha0)}, "
                f"x0={mat.num2str(x0)}")

    # format long;   —— 只影响显示格式
    psibc = np.max(psi * (Be * Rs ** 2))
    print(P["f1_fmt"].format(value=mat.scalar_value(F1)))

    C1s = alpha / (Be * Rs ** 2)
    C2s = sigma / (Be * Rs ** 2)

    print(P["psibc_fmt"].format(value=mat.scalar_value(psibc)))

    psi1d = psi
    pr1d = Pr
    j1d = Jt
    Bz1d = Bz
    rr1d = rr

    if 1 == 1:
        saved = dict(betas=G.betas, ls=G.ls, alpha=alpha, sigma=sigma,
                     psibc=psibc, n=G.n, Rs=Rs, Rw=Rw, psi1d=psi1d, pr1d=pr1d,
                     j1d=j1d, Bz1d=Bz1d, rr1d=rr1d, C1s=C1s, C2s=C2s, Be=Be,
                     Izeta0=Izeta0)
        if icoil is not None:   # Fig5 的 `MRR1d.mat` 里多一个 icoil
            saved["icoil"] = icoil
        mio.save("MRR1d.mat", **saved)

    for name in ("betas", "ls", "n", "tol", "alpha0", "ls0", "psim0", "psig"):
        ws[name] = getattr(G, name)
    ws.mu0 = mu0
    ws.Be = Be
    ws.Rs = Rs
    ws.Rw = Rw
    ws.options = options
    ws.options2 = options2
    ws.x0 = x0
    ws.x = x
    ws.F = mat.scalar(F)
    ws.sigma = sigma
    ws.alpha = alpha
    ws.psim = psim
    ws.F1 = F1
    ws.dr = dr
    ws.rmax = rmax
    ws.rr = rr
    ws.nr = mat.scalar(nr)
    ws.psi = psi
    ws.r = mat.scalar(r)
    ws.jr = mat.scalar(jr)
    ws.Pr = Pr
    ws.dP = dP
    ws.Jt = Jt
    ws.Bz = Bz
    ws.ids = mat.scalar(ids)
    ws.lsp = lsp
    ws.ind = ind
    ws.h = h
    ws.ave_beta = mat.scalar(ave_beta)
    ws.tot_I = mat.scalar(tot_I)
    ws.tot_I2 = mat.scalar(tot_I2)
    ws.Izeta0 = mat.scalar(Izeta0)
    ws.psibc = mat.scalar(psibc)
    ws.C1s = C1s
    ws.C2s = C2s
    ws.psi1d = psi1d
    ws.pr1d = pr1d
    ws.j1d = j1d
    ws.Bz1d = Bz1d
    ws.rr1d = rr1d
    if icoil is not None:
        ws.icoil = icoil
