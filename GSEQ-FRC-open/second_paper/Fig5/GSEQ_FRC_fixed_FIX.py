"""GSEQ_FRC_fixed_FIX.m 的逐行移植：二维定边界 G-S 平衡求解器（Fig5 版本）。

流程：读一维剖面 `MRR1d.mat` → 读壁面磁通 `psib_FIX.mat` 并样条插值到网格 →
从上一轮的 `psiq_fixed_FIX.mat` 取初值 → 反复迭代直到 `dpsim<=tol` 或触到 `nt`。

迭代内核就是前面核过系数表的那套：`ifd` 四路差分（1/2/3/4，其中 4 有 9 个子分支）、
`ibc` 四路上下边界、以及 `solve_rs` 或"最小 |psi|"两种求分界面半径的方式。
`rhs` 那一段是从 `tools/rhs_python_reference.py` 原样并入的（已与 MATLAB 抽出的
同一段逐项核到 1 ULP）。

注意 `nt` 是脚本里的字面量，harness 通过改这一行得到 `g_fixed_fix_short`
（40000 → 200）；移植版对应 `run(ws, nt=None)`。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import fastmath
from gseqfrc import mio
from gseqfrc import mpath
from gseqfrc import mplot

#: 脚本里写死的迭代上限（`g_fixed_fix_short` 把它改写成 200）
DEFAULT_NT = 40000


def run(ws, nt=None) -> None:
    mplot.close_all()  # close all;clear all;clc;
    if nt is None:
        nt = DEFAULT_NT

    irun = mat.scalar(0)
    icase = mat.scalar(1)
    mu0 = mat.scalar(4 * np.pi * 1e-7)
    init = mat.scalar(1)
    ifd = mat.scalar(2)   # finite difference order, =2 (3-point), =4 (5-point)
    ibc = mat.scalar(2)   # b.c. order
    irs = mat.scalar(mat.scalar_value(init))
    icoil = mat.scalar(1)

    loaded = mio.load("MRR1d.mat")  # betas, ls, alpha, sigma, n, Rs, Rw, Be, ...
    betas, ls, alpha, sigma, n = (loaded["betas"], loaded["ls"], loaded["alpha"],
                                  loaded["sigma"], loaded["n"])
    Rs, Rw = loaded["Rs"], loaded["Rw"]
    Be = loaded["Be"]
    Izeta0 = loaded["Izeta0"]

    zw = mat.scalar(3.5)
    nR = 2 ** 9 + 1
    nZ = 2 ** 8 + 1
    Be = mat.scalar(0.5)
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

    wall = mio.load("psib_FIX.mat")
    zb = wall["zb"]; rb = wall["rb"]; psib_wall = wall["psib"]
    zzb = zz
    rrb = mat.interp1(zb, rb, zzb, "spline")
    psib = mat.interp1(zb, psib_wall, zzb, "spline")
    jrb = np.round(rrb / dR) + 1
    psiw = mat.scalar(np.max(np.abs(psib)))
    drs = mat.scalar(1)

    # %%%%%%%%%%%%%%%coils%%%%%%%%%%%%%%%%%%%%%
    if mat.scalar_value(icoil) == 1:
        hl = mat.row(*([0.09] * 24))
        rl = mat.row(0.45, 0.45, 0.45, 0.45, 0.45, 0.54, 0.63, 0.63, 0.63, 0.63,
                     0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.54, 0.45,
                     0.45, 0.45, 0.45, 0.45)
        zl = mat.row(*[-3.312976756299235, -3.100861384204698, -2.855452177876161,
                       -2.593263233283414, -2.326779738264207, -2.046305749072202,
                       -1.740728563057488, -1.424569958981702, -1.108145712611496,
                       -0.796016016667750, -0.479857412591963, -0.163433166221757,
                       0.163433166221757, 0.479857412591963, 0.796016016667750,
                       1.108145712611496, 1.424569958981702, 1.740728563057488,
                       2.046305749072202, 2.326779738264207, 2.593263233283414,
                       2.855452177876161, 3.100861384204698, 3.312976756299235])
        wl = mat.row(*([0.14] * 24))
        Il = mat.row(*[2.966781507328271e3, 2.966781507328271e3, 2.076747055129789e3,
                       2.076747055129789e3, 2.076747055129789e3, 2.076747055129789e3,
                       2.076747055129789e3, 1.483390753664135e3, 1.483390753664135e3,
                       1.483390753664135e3, 1.483390753664135e3, 1.483390753664135e3,
                       1.483390753664135e3, 1.483390753664135e3, 1.483390753664135e3,
                       1.483390753664135e3, 1.483390753664135e3, 2.076747055129789e3,
                       2.076747055129789e3, 2.076747055129789e3, 2.076747055129789e3,
                       2.076747055129789e3, 2.966781507328271e3, 2.966781507328271e3]) * 100
    elif mat.scalar_value(icoil) == 2:
        hl = mat.row(0.006, 0.006, 0.006)
        rl = mat.row(0.15, 0.15, 0.15)
        zl = mat.row(-0.7, 0, 0.7)
        wl = mat.row(0.1, 1.2, 0.1)
        Il = mat.row(1000, 1000, 1000) * 1000

    if mat.scalar_value(init) == 0:
        psi = mat.zeros(nR, nZ)
        for i in mat.mrange(1, nR):
            mat.set_ij(psi, i, 0, 0.0)  # 占位（该分支未被本算例使用）
        C = mat.scalar(1)
    else:
        prev = mio.load("psiq_fixed_FIX.mat")
        psi = mat.interp2(prev["Zq"], prev["Rq"], prev["psiq"], Z, R, "nearest")
        C = prev["C"]

    # MATLAB 的 `psi0=psi` 是**按值复制**，Python 里直接赋值是同一个数组的别名——
    # 那样每轮更新 psi 会连带改掉 psi0，dpsim 恒为 0，循环只跑一轮就退出。
    psi0 = np.array(psi, copy=True)
    psi1 = np.array(psi, copy=True)
    psi2 = np.array(psi, copy=True)
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
    # 向量化算式里要用 Python 浮点：mat.scalar 为了对齐 MATLAB 的 1×1 矩阵
    # 返回的是 (1,1) 数组，直接乘数组会广播成二维，掩码赋值会报错。
    dR_v = mat.scalar_value(dR)
    dZ_v = mat.scalar_value(dZ)

    # ---- [加速] numba 内核（`gseqfrc/fastmath.py`）-----------------------------
    # 这一版把内层一轮的四个热点整体搬进 numba 的标量循环里：
    #   sources    : `exp` + dP + P + Jzeta + 三个电流积分（原本是 5 段整片运算）
    #   rhs_relax  : `rhs` 段 + 松弛更新（原本是 2 段整片运算）
    #   dpsim_copy : `max|psi-psi0|` + `psi0=psi`（原本是 2 段整片运算）
    # 算式、运算次序、累加次序都与下面的 numpy 版逐条对应，**逐位相同**；
    # 差别只是不再为每个中间量分配 1 MB 的临时数组。numba 不在环境里时自动走
    # numpy 版（`_FAST=False`）。
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
        # 内存布局要跟 `psi` 一致：`mat.interp2` 给出来的 psi 是**列优先**
        # （strides=(8, nR*8)），而 np.empty 默认是行优先。内核按原文的
        # `for j（外层）→ for i（内层）` 走，只有列优先时内层才是连续访问；
        # 混着放会让内层跨 2 KB 取一次，实测慢 3 倍。布局只影响速度，不影响数值。
        dP = np.empty((nR, nZ), dtype=float, order="F")
        Jzeta = np.empty((nR, nZ), dtype=float, order="F")
        P = np.empty((nR, nZ), dtype=float, order="F")
        rhs = np.empty((nR, nZ), dtype=float, order="F")
        _R_k = np.asfortranarray(R)
        _box = fastmath.box()
        _box_u = _box.view(np.uint64)
        # `mat.linspace` 给的是 MATLAB 那样的行向量，内核里按一维取
        _rr = np.ascontiguousarray(rr, dtype=float).reshape(-1)
        _jrb_i = np.ascontiguousarray(jrb_1d, dtype=np.int64)

    while (mat.scalar_value(it) <= nt + mat.scalar_value(ita)
           and mat.scalar_value(dpsim) > mat.scalar_value(tol)):
        jN0 = mat.scalar(np.floor(0.5 * nZ) - np.floor(1.5 / mat.scalar_value(dZ)))
        jz0 = int(mat.scalar_value(jN0))
        if _FAST:
            base_v = (mat.scalar_value(C) * Pm0_v) * betas_v
            _jzt, _jzt1, _jzt2 = fastmath.sources(
                psi, _R_k, _jrb_i, jz0 - 1, alpha_v, BeRs2_v, sigma_v, n_v - 1.0,
                n_v, k1_v, k2_v, base_v, twopi_v, dR_v, dZ_v, dP, Jzeta, P,
                _box, _box_u)
            Jztotal = mat.scalar(_jzt)
            Jztotal1 = mat.scalar(_jzt1)
            Jztotal2 = mat.scalar(_jzt2)
        else:
            # [加速] 原文在 dP 与 P 里各算一遍 exp(...) 和 (...+1)：这里只算一遍共用。
            # 表达式、运算顺序、系数都没动，结果逐位相同（与 Fig6 的 [M] 同一做法）。
            ex = mat.exp(-alpha * psi / (Be * Rs ** 2))
            uu = sigma * psi / (Be * Rs ** 2) + 1
            dP = (C * Pm0 * betas * ex * uu ** (n - 1)
                  * (-alpha / (Be * Rs ** 2) * uu + n * sigma / (Be * Rs ** 2)))

            Jzeta = -R * dP
            Jztotal = 0.0
            Jztotal1 = 0.0
            Jztotal2 = 0.0
            ncol = int(jrb_1d[jz0 - 1]) - 1          # 原文 jr=1:(jrb(jz)-1)
            # 这三个都是 1×1 标量，比对走“精确相等”，所以必须复现原文的**累加次序**：
            # 外层 jz、内层 jr（i 最快）。numpy 的 .sum() 是成对求和，会差 1 ULP；
            # 掩码选出来的是行优先次序（j 最快），要转置过来才和原文一致；cumsum
            # 则是按次序逐个累加。
            col = 2 * np.pi * Jzeta[:ncol, jz0 - 1] * R[:ncol, jz0 - 1] * dR_v
            Jztotal = mat.scalar(float(np.cumsum(col)[-1]) if col.size else 0.0)
            v1 = (Jzeta * dR_v * dZ_v).T[inside_J.T]
            v2 = (2 * np.pi * Jzeta * R * dR_v * dZ_v).T[inside_J.T]
            Jztotal1 = mat.scalar(float(np.cumsum(v1)[-1]) if v1.size else 0.0)
            Jztotal2 = mat.scalar(float(np.cumsum(v2)[-1]) if v2.size else 0.0)

            P = C * Pm0 * betas * ex * uu ** n
        # `Jzeta(:,j)` 在 MATLAB 里是列向量（nR×1）
        jmid = int(np.floor(0.5 * nZ)) - 1
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
            rhs = _rhs_block(ifd, psi0, dP, rr, jrb, nZ, dR, dZ, dR2, dZ2, mu0,
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

        psimid = psi[:, int(mat.scalar_value(jN0)) - 1].reshape(-1, 1)

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
            psi0 = np.array(psi, copy=True)  # 同上：必须复制

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
            mplot.subplot(2, 3, 1)
            mplot.hold_off()
            mplot.contour(Z, R, psio, 10, "linewidth", 2)
            mplot.hold_on()
            mplot.contour(Z, R, psii, 15, "linewidth", 2)
            mplot.hold_on()
            ch, hh = mplot.contour(Z, R, psi, mat.row(0, 0), "r--", "linewidth", 3,
                                   return_matrix=True)
            mplot.hold_on()
            mplot.plot(mat.take(Z, int(mat.el(ido, 1))), mat.take(R, int(mat.el(ido, 1))),
                       "mx", "linewidth", 3)
            colormap_name = "hot"
            mplot.set_(mplot.gca(), "colormap", colormap_name)
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
            mplot.set_(mplot.gca(), "xtick", mat.colon(-3.5, 0.5, 3.5))
            mplot.hold_off()

            mplot.subplot(2, 3, 4)
            # **必须复制**：MATLAB 的 `psi_x=psi(:,Z0)` 是快照，而 numpy 里
            # `psi[:, j]` 是视图——列优先数组的一列正好连续，`.reshape(-1,1)`
            # 也还是视图，于是后面每一轮迭代都会把这个"快照"改掉，快照会变成
            # 最后一轮的值（实测：本算例里两者差 3.3e-6，Fig4 那份差 1.1e-3）。
            psi_x = np.array(np.asarray(psi)[:, int(mat.scalar_value(Z0)) - 1],
                             copy=True).reshape(-1, 1)
            ind_min = mat.find(psi_x == np.min(psi_x))
            jz0 = int(mat.scalar_value(jN0))
            mplot.plot(rr, np.asarray(Jzeta)[:, jz0 - 1]
                       / np.max(np.abs(np.asarray(Jzeta)[:, jz0 - 1])),
                       "m", "linewidth", 3)

            mplot.subplot(2, 3, 5)
            E = mat.scalar(0.5 * (np.max(ch[0, :]) - np.min(ch[0, :])) / float(
                np.asarray(rs).ravel()[0]))
            mplot.contourf(R, Z, dP)
            mplot.title(f"(c) dP, tol={mat.num2str(tol)}")

            mplot.subplot(2, 3, 6)
            mplot.plot(rr, psi[:, jz0 - 1], "m", "linewidth", 3)
            mplot.title(f"(d) \\Psi, w={mat.num2str(w)}")
            # 只写"仅在绘图段出现"的变量；Jzetaz0/Prz0 每轮都赋值，
            # 必须留给末尾那次统一写（否则会停在第一张图那一轮的值上）
            for name in ("ido", "psio", "psii", "indtmp", "psi_x", "ind_min",
                         "E", "strp"):
                ws[name] = locals().get(name)

    runtime = runtime0 + mat.scalar(0) - runtime

    # %%
    Rq = R; Zq = Z; psiq = psi
    filename = "psiq_fixed_FIX"
    mio.save(filename + ".mat", Rq=Rq, Zq=Zq, psiq=psiq, Jzeta=Jzeta, P=P,
             jrb=jrb, zzb=zzb, rrb=rrb, psib=psib, Z0=Z0, Be=Be, Rs=Rs,
             nR=mat.scalar(nR), nZ=mat.scalar(nZ), C=C)
    mio.save("psiq1_fixed_FIX.mat", Rq=Rq, Zq=Zq, psiq=psiq, Jzeta=Jzeta, P=P,
             jrb=jrb, Z0=Z0, zl=zl, wl=wl, rl=rl, hl=hl, C=C, E=E, jN0=jN0)

    for name in ("mu0", "init", "ifd", "ibc", "irs", "icoil", "zw", "Rlow", "Rup",
                 "Zlow", "Zup", "dR", "dZ", "dR2", "dZ2", "Z0", "R", "Z", "rr",
                 "zz", "zb", "rb", "psib", "zzb", "rrb", "jrb", "psiw", "drs",
                 "hl", "rl", "zl", "wl", "Il", "psi", "psi0", "psi1", "psi2",
                 "C", "Pm0", "dpsim", "it", "ita", "runtime0", "runtime", "nt",
                 "tol", "savepath", "savefig", "w", "w0", "w0", "dP", "Jzeta",
                 "Jztotal", "Jztotal1", "Jztotal2", "jN0", "P", "Jzetaz0", "Prz0",
                 "rhs", "psimid", "rs", "Rq", "Zq", "psiq", "filename", "E",
                 "ido", "psio", "psii", "indtmp", "psi_x", "ind_min", "strp",
                 # 原文把这些也留在工作区里：MATLAB 的 for 循环变量会留值，
                 # 另外 itmp/icase/irun/nR/nZ/wc/Izeta 都是脚本里算出来的标量。
                 "i", "j", "jr", "jz", "itmp", "icase", "irun", "nR", "nZ",
                 "wc", "Izeta", "ch"):
        value = locals().get(name)
        if value is None or name in ws.keys():
            continue
        ws[name] = mat.scalar(value) if isinstance(value, int) else value
    for name in ("betas", "ls", "alpha", "sigma", "n", "Be", "Rs", "Rw",
                 "psibc", "ls0", "C1s", "C2s",
                 # MRR1d.mat / psib_FIX.mat 里 load 进来的其余变量（原文 load 之后
                 # 它们就一直在工作区里，快照也记了）
                 "Bz1d", "Izeta0", "j1d", "pr1d", "psi1d", "rr1d"):
        if name in loaded:
            ws[name] = loaded[name]
    for name in ("Bz",):
        if name in wall:
            ws[name] = wall[name]
    # 原文的 for 循环变量会留在工作区里（快照也记了）。循环已经向量化，
    # 这里按"最后一轮循环结束时的取值"补上（与 MATLAB 逐一核对过：
    # i=384, j=256, jr=384, jz=257）。
    i = int(mat.el(jrb, nZ - 1)) - 1      # 松弛循环：for i=2:(jrb(j)-1)
    j = nZ - 1                            # 松弛循环：for j=2:nZ-1
    jr = int(mat.el(jrb, nZ)) - 1         # 累加循环：for jr=1:(jrb(jz)-1)
    jz = nZ                               # 壁面循环：for jz=1:nZ
    for name, value in (("i", i), ("j", j), ("jr", jr), ("jz", jz)):
        if name not in ws.keys():
            ws[name] = mat.scalar(value)


def _rhs_block(ifd, psi0, dP, rr, jrb, nZ, dR, dZ, dR2, dZ2, mu0, inside=None):
    """`rhs` 段（源码 151–239 行）。

    本算例是 `ifd==2`（三点），走整片向量化：表达式与原文一字不差，只是把
    `for j=2:nZ-1, for i=2:jrb(j)-1` 换成"算全内部片 + 掩码写入"。其余 `ifd`
    （1/3/4，含 5 点的 9 个边界子分支）仍走逐点参考实现——它们不在本算例的
    路径上，等需要时再向量化。
    """
    if mat.scalar_value(ifd) != 2 or inside is None:
        from gseqfrc.rhs_reference import rhs_block
        return rhs_block(mat.scalar_value(ifd), psi0, dP, rr, jrb, nZ, dR, dZ,
                         dR2, dZ2, mat.scalar_value(mu0))
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
