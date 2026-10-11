"""gseq_ver_2.m 的逐行移植：二维定边界 G-S 平衡求解器（appendix_B/GSEQ 版本）。

它**不是** Fig4/Fig5 那套骨架的又一份副本，而是同一批代码里另一条分支：

* 装置/网格：`rw=0.17`、`zw=1.5`、`nR=129`、`nZ=385`；壁面磁通读同目录的 `psib.mat`；
* 压力模型用 `mcase` 选：本算例是 `mcase==2`，`dP`/`P` 在**双重循环里逐点**按
  `psi>=0`/`psi<0` 两路解析式算（`psi>=0` 走三次多项式、`psi<0` 走
  `prs*(prw/prs)^(psi/psw)` 那条幂律），不像 Fig4/Fig5 那样是一条 `exp(alpha*psi)` 式子；
* 差分用 `cal_case` 选：本算例是 `cal_case==2`，用的是**另一套系数**——
  `rhs = 1/fact1*(psi0(i+1)+psi0(i-1) - 0.5*h/rr(i)*(psi0(i+1)-psi0(i-1)) + 1/e0^2*(psi0(j+1)+psi0(j-1)) + fact2*rr(i)^2*dP)`，
  而且 `j==1` / `j==nZ` 两端**把同一个邻点用两次**（原文如此，照抄）；
* `cal_case==2` **不施加任何边界条件**（那一整段在 `if cal_case==1` 里）：壁面那
  一行只在循环开始前赋一次 `psi(nR,:)=psib`，之后靠"`rhs` 只在 `i=2:nR-1` 上被写"
  保持不变；
* 松弛是 `psi = psi + w*(rhs-psi)`（本算例 w=0.5），对**所有** i、j 做；
* `C` 的更新是 `C = wc*C + (1-wc)*Ctmp*Izeta/Jztotal`，而 `Izeta` 是写死的 -6e5、
  `Jztotal` 是正的：迭代几次就冲成 Inf，再变成 NaN。**这不是 bug 要修的地方**——
  `mcase==2` 的 dP/P 根本不用 C，所以 C 变 NaN 不影响 psi；基准里的 C 就是 NaN。

`nt` 是脚本里的字面量，harness 通过改这一行得到 `g_ver2_short`（150000 → 200）；
移植版对应 `run(ws, nt=None)`。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mio
from gseqfrc import mplot

#: 脚本里写死的迭代上限（`g_ver2_short` 把它改写成 200）
DEFAULT_NT = 150000


def run(ws, nt=None) -> None:
    mplot.close_all()  # close all;（这一份没有 clear all）
    if nt is None:
        nt = DEFAULT_NT

    irun = mat.scalar(0)   # =0, initial run; =1, from previous run
    # ---- if(irun==0) --------------------------------------------------------
    mu0 = mat.scalar(4 * np.pi * 1e-7)
    init = mat.scalar(0)
    ifd = mat.scalar(2)    # finite difference order, =2 (3-point), =4 (5-point)
    ibc = mat.scalar(2)    # b.c. order
    rw = mat.scalar(0.17)
    zw = mat.scalar(1.5)
    P0 = 1.0 / (2 * mu0 * rw ** 4)

    nR = 129
    nZ = 385

    prs = mat.scalar(5.5)         # 分界线处压强
    prw = mat.scalar(1.0e-5)      # 壁压强
    psw = mat.scalar(-1.0)        # 壁上磁通(外部+内部)
    pm = mat.scalar(-10000.0)     # 常数
    e0 = mat.scalar(5.8824)
    a0 = prs

    # `mat.scalar` 是为了对齐 MATLAB 的 1×1 矩阵（比对时形状也要一致）
    h = mat.scalar(1.0 / (nR - 1))
    fact1 = 2.0 * (1.0 + 1.0 / e0 ** 2)
    fact2 = 0.5 * h ** 2

    # `log(prw/prs)` 在 MATLAB 与 numpy 上**逐位相同**（实测：由基准的 a1/a2
    # 反推出的值与本机 numpy 的值一字不差），所以这里直接用 numpy 的 log。
    # 换机器/换 MATLAB 版本时要重新核一下（PORTING.md §5 那条注意事项）。
    a1 = prs / psw * np.log(prw / prs)
    a2 = 0.5 * prs * (np.log(prw / prs) / psw) ** 2
    a3 = pm

    Rlow = 0.0 * rw; Rup = rw
    Zlow = -1.0 * zw; Zup = 1.0 * zw
    rr = mat.linspace(Rlow, Rup, nR)
    zz = mat.linspace(Zlow, Zup, nZ)
    dR = mat.scalar(mat.el(rr, 2) - mat.el(rr, 1))
    dZ = mat.scalar(mat.el(zz, 2) - mat.el(zz, 1))
    R, Z = mat.ndgrid(rr, zz)
    dR2 = dR * dR
    dZ2 = dZ * dZ

    wall = mio.load("psib.mat")
    zb = wall["zb"]; rb = wall["rb"]; psib_in = wall["psib"]
    zzb = zz
    rrb = mat.interp1(zb, rb, zzb, "spline")
    psib = mat.interp1(zb, psib_in, zzb, "spline")

    jrb = np.round(rrb / dR) + 1

    Izeta = mat.scalar(-6.0e5)
    psiw = mat.scalar(np.max(np.abs(psib)))

    if mat.scalar_value(init) == 0:
        C = mat.scalar(0.0067)
        C3 = mat.scalar(-0.0005)
        # 原文是 `psi=0.0;`（标量），随后那句 `psi(nR,jz)=psib(jz)` 靠 MATLAB 的
        # 自动增长把 psi 撑成 nR×nZ 的零矩阵。移植版直接建同形状的零数组。
        psi = mat.zeros(nR, nZ)
    else:
        prev = mio.load("psiq1.mat")
        C = prev["Cq"]; C3 = prev["C3q"]
        psi = mat.interp2(prev["Zq"], prev["Rq"], prev["psiq"], Z, R, "nearest")

    Ctmp = C
    Ctmp3 = C3

    # % bound condition
    psi[nR - 1, :] = np.asarray(psib).ravel()[:nZ]

    psi0 = np.array(psi, copy=True)
    psi1 = np.array(psi, copy=True)
    psi2 = np.array(psi, copy=True)
    dpsim = mat.scalar(np.max(np.abs(np.asarray(psi) - np.asarray(psi0))) + 0.1)

    it = mat.scalar(1)
    ita = mat.scalar(0)
    runtime0 = mat.scalar(0)
    runtime = mat.scalar(0)
    tol = mat.scalar(1e-10)

    hf = mplot.figure("unit", "normalized", "Position", mat.row(0.5, 0.1, 0.5, 0.4),
                      "DefaultAxesFontSize", 15)

    # ---- 向量化用：Jztotal 的求和范围是 `for jz=1:nZ, for jr=1:(jrb(jz)-1)`
    rr_1d = np.asarray(rr, dtype=float).reshape(-1)
    jrb_1d = np.asarray(jrb, dtype=int).ravel()
    ii_ = np.arange(nR)[:, None]
    inside_J = (ii_ <= (jrb_1d[None, :] - 2))     # jr=1:(jrb(jz)-1)，1 基 → 0 基
    dR_v = mat.scalar_value(dR)
    dZ_v = mat.scalar_value(dZ)
    prs_v = mat.scalar_value(prs)
    prw_v = mat.scalar_value(prw)
    psw_v = mat.scalar_value(psw)
    e0_v = mat.scalar_value(e0)
    fact1_v = mat.scalar_value(fact1)
    fact2_v = mat.scalar_value(fact2)
    h_v = mat.scalar_value(h)
    ratio_v = prw_v / prs_v                       # prw/prs
    log_ratio_v = float(np.log(ratio_v))          # log(prw/prs)
    b1_v = prs_v / psw_v * log_ratio_v            # 负分支里的 prs/psw*log(...)
    rr_col = rr_1d[1:-1].reshape(-1, 1)           # 内点 i=2:nR-1 的 rr(i)

    while (mat.scalar_value(it) <= nt + mat.scalar_value(ita)
           and mat.scalar_value(dpsim) > mat.scalar_value(tol)):
        psi0 = np.array(psi, copy=True)     # 原文循环体第一句就是 psi0=psi
        C2 = mat.scalar(2000)
        mcase = mat.scalar(2)

        if mat.scalar_value(mcase) == 1:
            # 本算例不走这一支（原始公式照抄在此，避免以后要看）
            dP = -C2 * C * np.tanh(C2 * (psi - C3)) / np.cosh(C2 * (psi - C3)) / mu0
            P = C / np.cosh(C2 * (psi - C3)) / mu0
        else:
            # 原来是对每个点判 `psi(i,j)>=0` 再算一路。这里两路都算、再用掩码选：
            # 每个点拿到的是同一批运算的结果，逐位相同；被丢掉的那一路只是白算。
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                positive = psi >= 0
                powr = np.power(ratio_v, psi / psw_v)      # (prw/prs)^(psi/psw)
                dP = np.where(positive,
                              a1 + 2.0 * a2 * psi + 3.0 * a3 * psi * psi,
                              prs_v / psw_v * powr * log_ratio_v)
                P = np.where(positive,
                             a0 + a1 * psi + a2 * psi ** 2 + a3 * psi ** 3,
                             prs_v * powr)

        Jzeta = R * dP
        # 原文 `for jz=1:nZ, for jr=1:(jrb(jz)-1): Jztotal=Jztotal+Jzeta(jr,jz)*dR*dZ`
        # 是逐项累加（1×1 标量、比对走精确相等），所以按列优先后 cumsum 取末项。
        v_J = (Jzeta * dR_v * dZ_v).T[inside_J.T]
        Jztotal = mat.scalar(float(np.cumsum(v_J)[-1]) if v_J.size else 0.0)

        if mat.scalar_value(it) == 1:
            w = mat.scalar(0.5)
            w0 = w
            om = mat.scalar(1.95)
        else:
            wc = w
            # 这一支会让 C 一路冲到 Inf、再变成 NaN（`Izeta` 写死为负、`Jztotal` 是正的）。
            # 原文如此，而且 mcase==2 的 dP/P 不用 C，所以 NaN 不会污染 psi；
            # 基准里的 C/Ctmp 就是 NaN。MATLAB 不会为溢出报警，这里也不报。
            with np.errstate(over="ignore", invalid="ignore"):
                C = wc * C + (1 - wc) * Ctmp * Izeta / Jztotal
            om = mat.scalar(1.95)

        rhs = np.array(psi0, copy=True)
        cal_case = mat.scalar(2)
        if mat.scalar_value(cal_case) == 1:
            raise NotImplementedError("cal_case==1 这一支没有被任何算例覆盖，未移植")
        else:
            # for i=2:nR-1, for j=1:nZ（0 基：i0=1..nR-2）
            upper = psi0[2:, :] + psi0[:-2, :]
            slope = psi0[2:, :] - psi0[:-2, :]
            # 注意 j==1 / j==nZ 两端：原文把同一个邻点写了两遍
            jterm = np.empty((nR - 2, nZ), dtype=float)
            jterm[:, 0] = psi0[1:-1, 1] + psi0[1:-1, 1]
            jterm[:, 1:-1] = psi0[1:-1, 2:] + psi0[1:-1, :-2]
            jterm[:, -1] = psi0[1:-1, nZ - 2] + psi0[1:-1, nZ - 2]
            rhs[1:-1, :] = 1.0 / fact1_v * (
                upper
                - 0.5 * h_v / rr_col * slope
                + 1.0 / e0_v ** 2 * jterm
                + fact2_v * rr_col ** 2 * np.asarray(dP)[1:-1, :])

        # 松弛：psi(i,j) = psi(i,j)+w*(rhs(i,j)-psi(i,j))，对所有 i、j
        w_v = mat.scalar_value(w)
        psi = np.asarray(psi) + w_v * (rhs - np.asarray(psi))

        # %% boundary condition —— cal_case==2 时原文**整段跳过**（不施加任何 b.c.）
        if mat.scalar_value(cal_case) == 1:
            raise NotImplementedError("cal_case==1 的边界条件没有被任何算例覆盖，未移植")

        it = it + 1
        dpsim = mat.scalar(np.max(np.abs(np.asarray(psi) - np.asarray(psi0))))
        psi0 = np.array(psi, copy=True)
        Ctmp = C

        if int(np.mod(mat.scalar_value(it) - 2, 100)) == 0:  # plot
            ido = mat.find(psi == np.min(psi))
            mplot.subplot(2, 3, mat.colon(1, 1, 3))
            mplot.hold_off()
            indtmp = psi > 0
            psio = np.where(indtmp, np.nan, psi)
            indtmp = mat.find(psi < 0)
            psii = np.array(psi, dtype=float, copy=True)
            psii[np.asarray(psi < 0)] = np.nan
            mplot.contour(Z, R, psio, 10, "linewidth", 2)
            mplot.hold_on()
            mplot.contour(Z, R, psii, 20, "linewidth", 2)
            mplot.hold_on()
            mplot.contour(Z, R, psi, mat.row(0, 0), "r--", "linewidth", 3)
            mplot.hold_on()
            mplot.plot(mat.take(Z, int(mat.el(ido, 1))),
                       mat.take(R, int(mat.el(ido, 1))), "mx", "linewidth", 3)
            mplot.text(0.01 * mat.scalar_value(zw) + mat.take(Z, int(mat.el(ido, 1))),
                       mat.take(R, int(mat.el(ido, 1))), "O point", "fontsize", 13)
            mplot.colorbar()
            mplot.title(f"(a) \\Psi(R,Z), FRC G-S equilibrium, it="
                        f"{int(mat.scalar_value(it))}/{nt}, "
                        f"max(\\Psi^n-\\Psi^{{n-1}})={mat.num2str(dpsim)}")
            mplot.ylabel(f"R, r_w={mat.num2str(rw)}")
            strp = "p(\\psi)=C*sech(C2*(psi-C3))"
            mplot.xlabel(f"Z, z_w={mat.num2str(zw)}, I_z={mat.num2str(Izeta / 1e6)}"
                         f"MA, J_zt={mat.num2str(Jztotal / 1e6, 3)}MA, C="
                         f"{mat.num2str(C)}, C2={mat.num2str(C2)}, C3="
                         f"{mat.num2str(C3)} with {strp}")
            mplot.box("on")
            mplot.grid_on()
            mplot.legend("\\psi_{in}", "\\psi_{out}", "separatix")
            mplot.legend("boxoff")
            mplot.hold_off()

            mplot.subplot(2, 3, mat.colon(4, 1, 5))
            mplot.contourf(Z, R, P)
            mplot.title(f"(c) P, tol={mat.num2str(tol)}")
            mplot.xlabel("R")
            mplot.ylabel("Z")
            mplot.box("on")
            mplot.grid_on()
            mplot.colorbar()

            mplot.subplot(2, 3, 6)
            v = mat.row(0, -0.1, 0.1, -0.01, 0.01)
            mplot.contour(Z, R, psi, v)
            mplot.box("on")
            mplot.title(f"(d) \\Psi, w={mat.num2str(w)}")
            mplot.xlabel("R")
            mplot.ylabel("Z")
            mplot.pause(0.0001)
            for name in ("ido", "psio", "psii", "indtmp", "strp", "v"):
                ws[name] = locals().get(name)

    runtime = runtime0 + mat.scalar(0) - runtime

    psiq = psi
    filename = "psi"
    mio.save(filename + ".mat", psiq=psiq)

    for name in ("mu0", "init", "ifd", "ibc", "rw", "zw", "P0", "prs", "prw",
                 "psw", "pm", "e0", "a0", "a1", "a2", "a3", "h", "fact1",
                 "fact2", "Rlow", "Rup", "Zlow", "Zup", "rr", "zz", "dR", "dZ",
                 "R", "Z", "dR2", "dZ2", "zb", "rb", "psib", "zzb", "rrb",
                 "jrb", "Izeta", "psiw", "C", "C3", "Ctmp", "Ctmp3", "psi",
                 "psi0", "psi1", "psi2", "dpsim", "it", "ita", "runtime0",
                 "runtime", "nt", "tol", "dP", "P", "Jzeta", "Jztotal", "rhs",
                 "psiq", "filename", "ido", "psio", "psii", "indtmp", "strp",
                 "v", "om", "mcase", "C2", "cal_case", "w", "w0", "wc",
                 # MATLAB 的 for 循环变量会留值（实测 i=129、j=385、jr=128、jz=385）
                 "i", "j", "jr", "jz", "irun", "nR", "nZ"):
        value = locals().get(name)
        if value is None or name in ws.keys():
            continue
        ws[name] = mat.scalar(value) if isinstance(value, int) else value
    # 循环变量的留存值（逐个核对过基准）
    for name, value in (("i", nR), ("j", nZ),
                        ("jr", int(mat.el(jrb, nZ)) - 1), ("jz", nZ)):
        ws[name] = mat.scalar(value)
