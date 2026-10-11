"""gseq_frc_fixed.m 的逐行移植：二维定边界 G-S 平衡求解器（Fig4 版本）。

流程：读一维剖面 `MRR1d.mat` → 读壁面磁通 `psib_paper.mat` 并样条插值到网格 →
用一维剖面搭初值 → 反复迭代直到 `dpsim<=tol` 或触到 `nt`。

这一份和 Fig5 的 `GSEQ_FRC_fixed_FIX.m` 是"同一套骨架、不同的配置"，但**不是同一份
代码**（全仓 12 组同名文件里有 8 组内容不同，见 PORTING.md §8），所以按规矩逐行照抄，
不抽公共实现。四处实打实的差别：

* 装置/网格：`zw=0.75`、`Be=1.0`（Fig5 是 3.5 / 0.5），壁面磁通读 `psib_paper.mat`
  （它只随论文提供在 `Fig4/gui/` 下）；
* 初值：`init=0`，走"用一维剖面填 `psi`"那一支（Fig5 是 `init=1`，读上一轮的
  `psiq_fixed_FIX.mat`）。**这一支里原文写的是 `for i=1,nR`——逗号不是冒号**，
  循环只跑一次，而 `psi1d(1)` 恰好是 0，所以初值实际是**整片 0**；
  见下面的注释，这是本算例最容易抄错的一处；
* 中平面下标：`Jztotal`/`Jzetaz0`/`Prz0` 用 `floor(0.5*nZ)=128`，而 `psimid` 与
  绘图段用 `Z0=129`。两个下标**不一样**，照抄；
* 上限 `nt=50000*2=100000`，且实测到顶也没到 `tol`（终值 `dpsim=2.9e-8`）。

其余（`ifd` 三路差分、`ibc` 四路上下边界、`C=C*Izeta0/Izeta` 的支路、`rs` 的
"最小 |psi|"求法、每 100 步一次的绘图段、两份 `psiq*.mat`）与 Fig5 那份写法相同，
但只共用 `fastmath` 里那三个**逐位相同**的 numba 内核，算式仍在这里逐条对应。

`nt` 是脚本里的字面量，harness 通过改这一行得到 `g_fixed_fig4_short`（100000 → 200）；
移植版对应 `run(ws, nt=None)`。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import fastmath
from gseqfrc import mio
from gseqfrc import mpath
from gseqfrc import mplot

#: 脚本里写死的迭代上限（`g_fixed_fig4_short` 把它改写成 200）
DEFAULT_NT = 50000 * 2


def run(ws, nt=None) -> None:
    mplot.close_all()  # close all;clear all;clc;
    if nt is None:
        nt = DEFAULT_NT

    irun = mat.scalar(0)
    icase = mat.scalar(1)
    mu0 = mat.scalar(4 * np.pi * 1e-7)
    init = mat.scalar(0)
    ifd = mat.scalar(2)   # finite difference order, =2 (3-point), =4 (5-point)
    ibc = mat.scalar(2)   # b.c. order
    irs = mat.scalar(mat.scalar_value(init))

    # if icase==1 → load('MRR1d.mat')：这份文件由邻目录的 solve_MRR.m 写出
    # （基准的 prereq 是 s_solveMRR_f4，在同一个沙盒里先生成）。icase==2 的分支
    # 本算例不走。
    loaded = mio.load("MRR1d.mat")
    betas, ls, alpha, sigma, n = (loaded["betas"], loaded["ls"], loaded["alpha"],
                                  loaded["sigma"], loaded["n"])
    Rs, Rw = loaded["Rs"], loaded["Rw"]
    Izeta0 = loaded["Izeta0"]
    psi1d = loaded["psi1d"]      # 初值用的是这份一维剖面

    zw = mat.scalar(0.75)
    nR = 2 ** 9 + 1
    nZ = 2 ** 8 + 1
    Be = mat.scalar(1.0)
    Pm0 = Be ** 2 / (2 * mu0)

    Rlow = 0.0 * Rw; Rup = Rw
    Zlow = -1.0 * zw; Zup = 1.0 * zw
    rr = mat.linspace(Rlow, Rup, nR)
    zz = mat.linspace(Zlow, Zup, nZ)
    dR = mat.scalar(mat.el(rr, 2) - mat.el(rr, 1))
    dZ = mat.scalar(mat.el(zz, 2) - mat.el(zz, 1))
    R, Z = mat.ndgrid(rr, zz)
    dR2 = dR * dR
    dZ2 = dZ * dZ
    Z0 = mat.scalar((np.shape(R)[1] + 1) / 2)

    # load('psib_paper.mat')：随论文提供的壁面磁通（含 Bz 诊断量）。原文用相对
    # 文件名加载，靠搜索路径找到 `Fig4/gui/` 下的那一份（算例的 addpath 里带了它）。
    wall = mio.load("psib_paper.mat")
    zb = wall["zb"]; rb = wall["rb"]; psib_wall = wall["psib"]
    zzb = zz
    rrb = mat.interp1(zb, rb, zzb, "spline")
    psib = mat.interp1(zb, psib_wall, zzb, "spline")
    jrb = np.round(rrb / dR) + 1
    psiw = mat.scalar(np.max(np.abs(psib)))
    drs = mat.scalar(1)

    if mat.scalar_value(init) == 0:
        psi = mat.zeros(nR, nZ)
        # 原文这一行是 `for i=1,nR`——**逗号，不是冒号**。MATLAB 把逗号当成 for
        # 头的结束符，剩下的 `nR` 变成循环体里的一个裸语句（运行时会回显
        # "nR = 513"），于是循环体只以 i=1 执行一次。
        #
        # 而这份数据里 psi1d(1) 正好是 0，所以**初值其实是整片 0**（不是"用一维
        # 剖面铺满整个场"）。顺带说明那个"铺满"的读法为什么跑不通：psi1d 只有
        # 374 个元素，而 nR=513，真写成 `for i=1:nR` 会直接索引越界报错
        # ——原句只有"逗号"这一种读法能跑下去。
        # 证据：MATLAB R2023a 上跑 tools/ml_init_probe.m（真实数据）——
        # 逗号版非零行数 = 0/513、psi1d(1) = 0；冒号版报"索引超出 374"。
        # 移植版照抄"只跑一次"。
        for i in mat.mrange(1, 1):
            psi[i - 1, :] = mat.scalar_value(mat.el(psi1d, i) * Be * Rs ** 2)
        C = mat.scalar(1)
    else:
        prev = mio.load("psiq.mat")
        psi = mat.interp2(prev["Zq"], prev["Rq"], prev["psiq"], Z, R, "nearest")
        C = prev["C"]

    # MATLAB 的 `psi0=psi` 是**按值复制**，Python 里直接赋值是同一个数组的别名——
    # 那样每轮更新 psi 会连带改掉 psi0，dpsim 恒为 0，循环只跑一轮就退出。
    # 顺带把它们做成列优先：numba 内核按原文的 `for j → for i` 走，列优先时内层
    # 才是连续访存（实测快 2~3 倍）。布局只影响速度，不影响数值。
    psi = np.asfortranarray(psi)
    psi0 = np.array(psi, copy=True, order="F")
    psi1 = np.array(psi, copy=True, order="F")
    psi2 = np.array(psi, copy=True, order="F")
    dpsim = mat.scalar(np.max(np.abs(np.asarray(psi) - np.asarray(psi0))) + 0.1)

    it = mat.scalar(1)
    ita = mat.scalar(0)
    runtime0 = mat.scalar(0)
    savepath = "./saveit/"
    savefig = mat.scalar(0)
    if mat.scalar_value(savefig) == 1:
        import pathlib
        pathlib.Path(savepath).mkdir(exist_ok=True)

    runtime = mat.scalar(0)
    tol = mat.scalar(1e-13)

    hf = mplot.figure("unit", "normalized", "Position", mat.row(0.6, 0.35, 0.4, 0.4),
                      "DefaultAxesFontSize", 15)

    # ---- 向量化用：原文的循环范围是 `for j=2:nZ-1, for i=2:jrb(j)-1`。
    #      这里预先把"参与计算的内部点"做成掩码（1 基 → 0 基：i∈[1, jrb(j)-2]，
    #      j∈[1, nZ-2]）。被更新的点集与原文完全一致；壁面外的死区不动。
    jrb_1d = np.asarray(jrb, dtype=int).ravel()
    ii_ = np.arange(nR)[:, None]
    jj_ = np.arange(nZ)[None, :]
    inside = ((ii_ >= 1) & (ii_ <= (jrb_1d[None, :] - 2))
              & (jj_ >= 1) & (jj_ <= nZ - 2))
    # 注意：Jztotal1/Jztotal2 的求和范围与上面不同——原文是 `for jr=1:(jrb(jz)-1)`，
    # **含轴**（r=0 那一列），而且对全部 jz 求和。
    inside_J = (ii_ <= (jrb_1d[None, :] - 2))
    jz_all = np.arange(nZ)

    # `Jztotal`/`Jzetaz0`/`Prz0` 用的是 `floor(0.5*nZ)`（1 基 128），
    # 而 `psimid`、绘图段的 `psi_x`/`Jzeta(:,Z0)` 用的是 `Z0`（129）。两个下标
    # 差一格，原文如此，不能统一。
    jmid = int(np.floor(0.5 * nZ)) - 1        # 0 基：127

    dR_v = mat.scalar_value(dR)
    dZ_v = mat.scalar_value(dZ)

    # ---- [加速] numba 内核（`gseqfrc/fastmath.py`）-----------------------------
    # 与 Fig5 那份用同一组内核：`sources`（exp+dP+P+Jzeta+三个电流积分）、
    # `rhs_relax`（rhs+松弛）、`dpsim_copy`（max+复制）。算式、运算次序、
    # 累加次序与下面的 numpy 版逐条对应，**逐位相同**。numba 不在环境里时
    # 自动走 numpy 版（`_FAST=False`）。
    _FAST = fastmath.NUMBA_AVAILABLE
    if _FAST:
        alpha_v = mat.scalar_value(alpha)
        sigma_v = mat.scalar_value(sigma)
        n_v = mat.scalar_value(n)
        betas_v = mat.scalar_value(betas)
        BeRs2_v = mat.scalar_value(Be) * mat.scalar_value(Rs) ** 2
        Pm0_v = mat.scalar_value(Pm0)
        mu0_v = mat.scalar_value(mu0)
        k1_v = -alpha_v / BeRs2_v                  # 原文 -alpha/(Be*Rs^2).*uu
        k2_v = n_v * sigma_v / BeRs2_v             # 原文 n*sigma/(Be*Rs^2)
        twopi_v = 2 * np.pi
        dP = np.empty((nR, nZ), dtype=float, order="F")
        Jzeta = np.empty((nR, nZ), dtype=float, order="F")
        P = np.empty((nR, nZ), dtype=float, order="F")
        rhs = np.empty((nR, nZ), dtype=float, order="F")
        _R_k = np.asfortranarray(R)
        _box = fastmath.box()
        _box_u = _box.view(np.uint64)
        _rr = np.ascontiguousarray(rr, dtype=float).reshape(-1)
        _jrb_i = np.ascontiguousarray(jrb_1d, dtype=np.int64)

    while (mat.scalar_value(it) <= nt + mat.scalar_value(ita)
           and mat.scalar_value(dpsim) > mat.scalar_value(tol)):
        if _FAST:
            base_v = (mat.scalar_value(C) * Pm0_v) * betas_v
            _jzt, _jzt1, _jzt2 = fastmath.sources(
                psi, _R_k, _jrb_i, jmid, alpha_v, BeRs2_v, sigma_v, n_v - 1.0,
                n_v, k1_v, k2_v, base_v, twopi_v, dR_v, dZ_v, dP, Jzeta, P,
                _box, _box_u)
            Jztotal = mat.scalar(_jzt)
            Jztotal1 = mat.scalar(_jzt1)
            Jztotal2 = mat.scalar(_jzt2)
        else:
            ex = mat.exp(-alpha * psi / (Be * Rs ** 2))
            uu = sigma * psi / (Be * Rs ** 2) + 1
            dP = (C * Pm0 * betas * ex * uu ** (n - 1)
                  * (-alpha / (Be * Rs ** 2) * uu + n * sigma / (Be * Rs ** 2)))

            Jzeta = -R * dP
            # 原文三个累加循环：第一个只走 jz=floor(0.5*nZ) 这一列（写成单值 for），
            # 第二、三个对全部 jz 求和，上界都是 jr=1:(jrb(jz)-1)（含轴）。
            # 三个都是 1×1 标量，比对走"精确相等"，所以必须复现原文的**累加次序**：
            # 外层 jz、内层 jr。numpy 的 .sum() 是成对求和会差 1 ULP，掩码选出来
            # 的是行优先次序（j 最快），要转置过来；cumsum 才是按次序逐个累加。
            ncol = int(jrb_1d[jmid]) - 1
            col = 2 * np.pi * Jzeta[:ncol, jmid] * R[:ncol, jmid] * dR_v
            Jztotal = mat.scalar(float(np.cumsum(col)[-1]) if col.size else 0.0)
            v1 = (Jzeta * dR_v * dZ_v).T[inside_J.T]
            v2 = (2 * np.pi * Jzeta * R * dR_v * dZ_v).T[inside_J.T]
            Jztotal1 = mat.scalar(float(np.cumsum(v1)[-1]) if v1.size else 0.0)
            Jztotal2 = mat.scalar(float(np.cumsum(v2)[-1]) if v2.size else 0.0)

            P = C * Pm0 * betas * ex * uu ** n
        # `Jzeta(:,j)` 在 MATLAB 里是列向量（nR×1）
        Jzetaz0 = np.asarray(Jzeta)[:, jmid].reshape(-1, 1)
        Prz0 = np.asarray(P)[:, jmid].reshape(-1, 1)

        if mat.scalar_value(it) == 1:
            w = mat.scalar(0.1)
            w0 = w
        elif mat.scalar_value(it) >= 100:
            wc = mat.scalar(0.1)
            Izeta = Jztotal
            C = C * Izeta0 / Izeta

        if _FAST:
            np.copyto(rhs, psi0)                     # 原文 rhs=psi0，只改内部点
            fastmath.rhs_relax(psi, psi0, dP, _rr, _jrb_i, mu0_v, dR_v, dZ_v,
                               mat.scalar_value(dR2), mat.scalar_value(dZ2),
                               mat.scalar_value(w), 1 - mat.scalar_value(w), rhs)
        else:
            rhs = _rhs_block(psi0, dP, rr, jrb, nZ, dR, dZ, dR2, dZ2, mu0,
                             inside=inside)
            # 向量化：psi(i,j)=w*psi0(i,j)+rhs(i,j)*(1-w)，只写掩码内的点
            w_v = mat.scalar_value(w)
            psi[inside] = w_v * psi0[inside] + rhs[inside] * (1 - w_v)

        # boundary condition
        psi[0, :] = 0.0  # zero boundary at axis
        # 壁面 Dirichlet：psi(jrb(jz), jz) = psib(jz)（一次向量化赋值）
        psi[jrb_1d - 1, jz_all] = np.asarray(psib).ravel()[:nZ]

        # dpsi/dz=0 at z=min,max
        if mat.scalar_value(ibc) == 1:
            psi[:, 0] = psi[:, 1]
            psi[:, nZ - 1] = psi[:, nZ - 2]
        elif mat.scalar_value(ibc) == 2:
            psi[:, 0] = (4 * psi[:, 1] - psi[:, 2]) / 3
            psi[:, nZ - 1] = (4 * psi[:, nZ - 2] - psi[:, nZ - 3]) / 3
        elif mat.scalar_value(ibc) == 3:
            psi[:, 0] = (18 * psi[:, 1] - 9 * psi[:, 2] + 2 * psi[:, 3]) / 11
            psi[:, nZ - 1] = (18 * psi[:, nZ - 2] - 9 * psi[:, nZ - 3]
                              + 2 * psi[:, nZ - 4]) / 11
        elif mat.scalar_value(ibc) == 4:
            psi[:, 0] = (48 * psi[:, 1] - 36 * psi[:, 2] + 16 * psi[:, 3]
                         - 3 * psi[:, 4]) / 25
            psi[:, nZ - 1] = (48 * psi[:, nZ - 2] - 36 * psi[:, nZ - 3]
                              + 16 * psi[:, nZ - 4] - 3 * psi[:, nZ - 5]) / 25

        psimid = psi[:, int(mat.scalar_value(Z0)) - 1].reshape(-1, 1)

        if mat.scalar_value(irs) == 2:
            rs = mpath.call("solve_rs", mat.take(psimid, mat.mrange(1, nR)), rr)
            drs = np.abs(rs - Rs) / Rs
        elif mat.scalar_value(irs) in (0, 1):
            itmp = int(np.floor(0.15 * nR))
            tail = mat.take(psimid, mat.mrange(itmp, nR))
            idx = mat.el(mat.find(np.abs(tail) == np.min(np.abs(tail))), 1)
            rs = mat.el(rr, int(idx) + itmp - 1)
            rs = mat.scalar(rs)

        it = it + 1
        if _FAST:
            dpsim = mat.scalar(fastmath.dpsim_copy(psi, psi0))
        else:
            dpsim = mat.scalar(np.max(np.abs(np.asarray(psi) - np.asarray(psi0))))
            psi0 = np.array(psi, copy=True, order="F")  # 同上：必须复制

        if int(np.mod(mat.scalar_value(it) - 2, 100)) == 0:  # plot
            # MATLAB 的 %5.4d：零填充到 4 位、宽度 5
            print(f"it = {int(mat.scalar_value(it)):04d}".ljust(7))
            print(f"rs = {int(float(np.asarray(rs).ravel()[0])):04d}".ljust(7))
            ido = mat.find(psi == np.min(psi))
            indtmp = psi > 0
            psio = np.where(indtmp, np.nan, psi)
            indtmp = mat.find(psi < 0)
            psii = np.array(psi, dtype=float, copy=True)
            psii[np.asarray(psi < 0)] = np.nan
            mplot.subplot(2, 3, mat.colon(1, 1, 3))
            mplot.hold_off()
            mplot.contour(Z, R, psio, 10, "linewidth", 2)
            mplot.hold_on()
            mplot.contour(Z, R, psii, 15, "linewidth", 2)
            mplot.hold_on()
            mplot.contour(Z, R, psi, mat.row(0, 0), "r--", "linewidth", 3)
            mplot.hold_on()
            mplot.plot(mat.take(Z, int(mat.el(ido, 1))),
                       mat.take(R, int(mat.el(ido, 1))), "mx", "linewidth", 3)
            mplot.text(0.01 * mat.scalar_value(zw) + mat.take(Z, int(mat.el(ido, 1))),
                       mat.take(R, int(mat.el(ido, 1))), "O point", "fontsize", 13)
            mplot.colorbar()
            mplot.colormap("hot")
            mplot.plot(zb, rb, "-", zzb, rrb, ":", "linewidth", 3)
            mplot.title(f"(a) \\Psi(R,Z), FRC G-S equilibrium, it="
                        f"{int(mat.scalar_value(it))}/{nt}, "
                        f"max(\\Psi^n-\\Psi^{{n-1}})={mat.num2str(dpsim)}")
            mplot.ylabel(f"R, r_w={mat.num2str(Rw)}")
            # 源码里就是带 TeX 转义的字符串，照抄（快照记的是这个文本）
            strp = "p(\\psi)=\\beta_s exp(\\alpha \\psi)*(\\sigma \\psi+1)^2"
            mplot.xlabel(f"Z, z_w={mat.num2str(zw)}; J_zt="
                         f"{mat.num2str(Jztotal / 1e6, 3)}MA; beta_s ="
                         f"{mat.num2str(betas)}; alpha ={mat.num2str(alpha)}; "
                         f"sigma ={mat.num2str(sigma)}; with {strp}")
            mplot.legend("\\psi_{in}", "\\psi_{out}", "separatix")
            mplot.legend("boxoff")
            mplot.set_(mplot.gca(), "xtick", mat.colon(-0.75, 0.25, 0.75))
            mplot.hold_off()

            mplot.subplot(2, 3, 4)
            # **必须复制**：MATLAB 的 `psi_x=psi(:,Z0)` 是快照，而 numpy 里
            # `psi[:, j]` 是视图——列优先数组的一列正好连续，`.reshape(-1,1)`
            # 也还是视图，于是后面每一轮迭代都会把这个"快照"改掉，
            # 快照会变成最后一轮的值（这个坑实测过：差异 1.1e-3）。
            psi_x = np.array(np.asarray(psi)[:, int(mat.scalar_value(Z0)) - 1],
                             copy=True).reshape(-1, 1)
            ind_min = mat.find(psi_x == np.min(psi_x))
            mplot.plot(rr, np.asarray(Jzeta)[:, int(mat.scalar_value(Z0)) - 1]
                       / np.max(np.abs(np.asarray(Jzeta)[:, int(mat.scalar_value(Z0)) - 1])),
                       "m", "linewidth", 3)
            mplot.box("on")
            mplot.grid_on()
            mplot.xlabel("r/R_0")
            mplot.ylabel("J_{\\theta}")

            mplot.subplot(2, 3, 5)
            mplot.contourf(R, Z, dP)
            mplot.title(f"(c) dP, tol={mat.num2str(tol)}")
            mplot.xlabel("R")
            mplot.ylabel("Z")
            mplot.box("on")

            mplot.subplot(2, 3, 6)
            mplot.plot(rr, psi[:, int(mat.scalar_value(Z0)) - 1], "m", "linewidth", 3)
            mplot.box("on")
            mplot.grid_on()
            mplot.title(f"(d) \\Psi, w={mat.num2str(w)}")
            mplot.xlabel("r/R_0")
            mplot.ylabel("\\psi")
            mplot.pause(0.0001)
            # 只写"仅在绘图段出现"的变量；Jzetaz0/Prz0 每轮都赋值，
            # 必须留给末尾那次统一写（否则会停在第一张图那一轮的值上）
            for name in ("ido", "psio", "psii", "indtmp", "psi_x", "ind_min", "strp"):
                ws[name] = locals().get(name)

    runtime = runtime0 + mat.scalar(0) - runtime

    # %%
    Rq = R; Zq = Z; psiq = psi
    filename = "psiq"
    mio.save(filename + ".mat", Rq=Rq, Zq=Zq, psiq=psiq, Jzeta=Jzeta, P=P,
             jrb=jrb, zzb=zzb, rrb=rrb, psib=psib, Z0=Z0, Be=Be, Rs=Rs,
             nR=mat.scalar(nR), nZ=mat.scalar(nZ), C=C)
    mio.save("psiq1.mat", Rq=Rq, Zq=Zq, psiq=psiq, drs=drs, C=C)

    for name in ("mu0", "init", "ifd", "ibc", "irs", "icase", "zw", "Rlow", "Rup",
                 "Zlow", "Zup", "dR", "dZ", "dR2", "dZ2", "Z0", "R", "Z", "rr",
                 "zz", "zb", "rb", "psib", "zzb", "rrb", "jrb", "psiw", "drs",
                 "psi", "psi0", "psi1", "psi2", "C", "Pm0", "dpsim", "it", "ita",
                 "runtime0", "runtime", "nt", "tol", "savepath", "savefig", "w",
                 "w0", "dP", "Jzeta", "Jztotal", "Jztotal1", "Jztotal2", "P",
                 "Jzetaz0", "Prz0", "rhs", "psimid", "rs", "Rq", "Zq", "psiq",
                 "filename", "ido", "psio", "psii", "indtmp", "psi_x", "ind_min",
                 "strp",
                 # 原文把这些也留在工作区里：MATLAB 的 for 循环变量会留值，
                 # 另外 itmp/irun/nR/nZ/wc/Izeta 都是脚本里算出来的标量。
                 "i", "j", "jr", "jz", "itmp", "irun", "nR", "nZ", "wc", "Izeta"):
        value = locals().get(name)
        if value is None or name in ws.keys():
            continue
        ws[name] = mat.scalar(value) if isinstance(value, int) else value
    for name in ("betas", "ls", "alpha", "sigma", "n", "Be", "Rs", "Rw",
                 "psibc", "C1s", "C2s", "Izeta0",
                 # MRR1d.mat 里 load 进来的其余变量（原文 load 之后它们就一直在
                 # 工作区里，快照也记了）
                 "Bz1d", "j1d", "pr1d", "psi1d", "rr1d"):
        if name in loaded:
            ws[name] = loaded[name]
    for name in ("Bz",):
        if name in wall:
            ws[name] = wall[name]
    # 原文的 for 循环变量会留在工作区里（快照也记了）。循环已经向量化，
    # 这里按"最后一轮循环结束时的取值"补上：弛豫/rhs 循环是最后执行的循环，
    # 所以 i=512、j=256；jz/jr 保留自更早那三个电流积分循环。
    i = int(mat.el(jrb, nZ - 1)) - 1      # 松弛循环：for i=2:(jrb(j)-1)
    j = nZ - 1                            # 松弛循环：for j=2:nZ-1
    jr = int(mat.el(jrb, nZ)) - 1         # 累加循环：for jr=1:(jrb(jz)-1)
    jz = nZ                               # 壁面循环：for jz=1:nZ
    for name, value in (("i", i), ("j", j), ("jr", jr), ("jz", jz)):
        # 直接覆盖：初值那一支的 `for i=1,nR` 也会在 locals 里留下 i=1，
        # 但那不是最后一轮循环的取值（实测基准里 i=512）。
        ws[name] = mat.scalar(value)


def _rhs_block(psi0, dP, rr, jrb, nZ, dR, dZ, dR2, dZ2, mu0, inside=None):
    """`rhs` 段（源码 `ifd==2` 三点格式那一支）。

    本算例是 `ifd==2`，走整片向量化：表达式与原文一字不差，只是把
    `for j=2:nZ-1, for i=2:jrb(j)-1` 换成"算全内部片 + 掩码写入"。
    """
    rr_c = np.asarray(rr, dtype=float).reshape(-1, 1)
    rhs = np.array(psi0, dtype=float, copy=True)      # 原文：rhs=psi0，只改内部点
    dR_f, dZ_f = mat.scalar_value(dR), mat.scalar_value(dZ)
    dR2_f, dZ2_f = mat.scalar_value(dR2), mat.scalar_value(dZ2)
    mu0_f = mat.scalar_value(mu0)
    val = 1 / (2 / dR2_f + 2 / dZ2_f) * (
        (psi0[2:, 1:-1] + psi0[:-2, 1:-1]) / dR2_f
        - (psi0[2:, 1:-1] - psi0[:-2, 1:-1]) / (2 * dR_f * rr_c[1:-1])
        + (psi0[1:-1, 2:] + psi0[1:-1, :-2]) / dZ2_f
        + mu0_f * rr_c[1:-1] ** 2 * dP[1:-1, 1:-1])
    sub = inside[1:-1, 1:-1]
    rhs[1:-1, 1:-1][sub] = val[sub]
    return rhs
