"""gseq.m 的逐行移植：二维定边界 G-S 平衡（NUCTE-III 参数，`icase==3`）。

这一份的特点：

* `init==0` 那一支**没有给 `psi` 赋值**（源码里那句初始化被注释掉了），所以 `psi`
  是被后面那个"填壁面"的循环**自动创建**出来的：MATLAB 见到 `psi(jr,jz)=...` 且
  变量不存在，就建一个刚好够放下最大下标的零矩阵——这里是 nR×nZ 的零矩阵加上
  `jr>=jrb(jz)` 那一片壁面值。移植版等价地先建零矩阵再填。
* 压力模型用 `icase` 选，本算例是 `icase==3`：`dP = C*exp(-C1*psi)*(2*C2*(C2*psi+1)-C1*(C2*psi+1)^2)/mu0`。
* **原文会在第 2 轮中止**：`it~=1` 那一支要调 `solve_lp`（还有 `solve_h`、`solve_rs`），
  而这三个函数在整个仓库里都不存在（`solve_rs` 在，另两个不在）。
  **修改版把这两个函数补上了**（同目录的 `solve_lp.py` / `solve_h.py`，
  定义与来历见那两个文件与 `修改记录.md`），所以脚本现在能按作者的迭代结构跑到底；
  原文只跑得到第 2 轮（`it=2`）。
* `ifd`/`ibc` 是四路分支，harness 靠改这两行覆盖它们：登记表里 `g_gseq_ifd1/3/4`
  与 `g_gseq_ibc1/3/4` 六个变体就是这么来的（都是预期报错，`it=2`），
  移植版对应 `run(ws, ifd=..., ibc=...)`。

**修改版还有一处性能改动**：`rhs` 那一段在 `ifd==2`（本算例的默认分支）下改成整片
向量化，算式与原文逐字相同（与 `Fig5/GSEQ_FRC_fixed_FIX.py` 里已经验收过的写法
一致）。原来的逐点 Python 循环一轮要 0.5 s，跑满 70000 轮是十几小时；向量化之后
一轮不到 1 ms。其余 `ifd` 分支仍走逐点参考实现。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mio
from gseqfrc import mpath
from gseqfrc import mplot

#: 脚本里写死的迭代上限
DEFAULT_NT = 70000

#: 要留在"基本工作区"里的变量。MATLAB 里 gseq.m 是脚本，变量天然留在工作区，
#: 哪怕脚本中途报错（基准正是这种情况：快照记的是第 1 轮跑完 + 第 2 轮算到
#: `solve_lp` 之前的状态）。移植版是个函数，所以每轮都要把这份状态刷进 `ws`。
_WORKSPACE_VARS = (
    "irun", "mu0", "init", "rw", "zw", "eta", "Rlow", "Rup", "Zlow", "Zup",
    "nR", "nZ", "rr", "zz",
    "dR", "dZ", "R", "Z", "dR2", "dZ2", "Z0", "zb", "rb", "psib", "zzb", "rrb",
    "jrb", "Izeta", "psiw", "C", "C1", "C2", "Ctmp", "C1tmp", "C2tmp", "Ln",
    "h", "psi", "psi0", "psi1", "psi2", "dpsim", "it", "ita", "runtime0",
    "runtime", "nt", "tol", "savepath", "savefig", "dP", "P", "Jzeta",
    "Jztotal", "rhs", "icase", "ido", "psio", "psii", "indtmp", "psi_x",
    "ind_min", "strp", "w", "w0", "wc", "P_m", "psi_m", "jr_m",
)


def _flush(ws, env, *, ifd_value: int, ibc_value: int, nR: int, nZ: int, jrb) -> None:
    """把当前状态写进工作区（覆盖写：脚本的变量就是"最新值"）。"""
    for name in _WORKSPACE_VARS:
        value = env.get(name)
        if value is None:
            continue
        ws[name] = mat.scalar(value) if isinstance(value, int) else value
    # harness 的 overrides 是改脚本里那两行，所以工作区里 is 的是被改过的值
    ws["ifd"] = mat.scalar(ifd_value)
    ws["ibc"] = mat.scalar(ibc_value)
    # MATLAB 的 for 循环变量会留值：最后执行的是松弛循环（给出 i、j），
    # jz/jr 保留自更早的壁面循环与电流积分循环（基准实测 113/256/113/257）
    for name, value in (("i", int(mat.el(jrb, nZ - 1)) - 1), ("j", nZ - 1),
                        ("jr", int(mat.el(jrb, nZ)) - 1), ("jz", nZ)):
        ws[name] = mat.scalar(value)


def _rhs_block(ifd, psi0, dP, rr, jrb, nZ, dR, dZ, dR2, dZ2, mu0, inside=None):
    """`rhs` 那一段（源码 130–239 行）。

    **修改版**：`ifd==2`（三点格式，本算例的默认分支）走整片向量化——表达式与原文
    一字不差，只是把 `for j=2:nZ-1, for i=2:jrb(j)-1` 换成"算全内部片 + 掩码写入"。
    写法与 `second_paper/Fig5/GSEQ_FRC_fixed_FIX.py` 里那个已经与 MATLAB 逐变量
    对过的版本一致，所以数值不变，只是把一轮 0.5 s 压到 1 ms 以内。

    其余 `ifd`（1/3/4，含 5 点的那些边界子分支）仍回到逐点参考实现
    （`gseqfrc/rhs_reference.py`）。
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


def run(ws, ifd=None, ibc=None) -> None:
    mplot.close_all()  # close all;clear all;clc;

    irun = mat.scalar(0)
    # ---- if(irun==0) --------------------------------------------------------
    mu0 = mat.scalar(4 * np.pi * 1e-7)
    init = mat.scalar(0)
    ifd_v = mat.scalar(2 if ifd is None else ifd)
    ibc_v = mat.scalar(2 if ibc is None else ibc)
    rw = mat.scalar(0.17)
    zw = mat.scalar(1 * 0.75)          # NUCTE-III
    nR = 2 ** 7 + 1
    nZ = 2 ** 8 + 1
    eta = mat.scalar(0.5)

    Rlow = 0.0 * rw; Rup = rw
    Zlow = -1.0 * zw; Zup = 1.0 * zw
    rr = mat.linspace(Rlow, Rup, nR)
    zz = mat.linspace(Zlow, Zup, nZ)
    dR = mat.scalar(mat.el(rr, 2) - mat.el(rr, 1))
    dZ = mat.scalar(mat.el(zz, 2) - mat.el(zz, 1))
    R, Z = mat.ndgrid(rr, zz)
    dR2 = dR * dR
    dZ2 = dZ * dZ
    Z0 = mat.scalar((np.shape(R)[1] + 1) / 2)

    wall = mio.load("psib.mat")
    zb = wall["zb"]; rb = wall["rb"]; psib_in = wall["psib"]
    zzb = zz
    rrb = mat.interp1(zb, rb, zzb, "spline")
    psib = mat.interp1(zb, psib_in, zzb, "spline")

    jrb = np.round(rrb / dR) + 1

    Izeta = mat.scalar(-25.0e5)
    psiw = mat.scalar(np.max(np.abs(psib)))

    if mat.scalar_value(init) == 0:
        C = mat.scalar(0.0831875)
        C1 = mat.scalar(15623.4955718)
        C2 = mat.scalar(4360.3755823)
        Ln = mat.scalar(0.7)
        h = mat.scalar(0.7)
        # 原文这一支**没有**给 psi 赋值（初始化那句被注释掉了）
    else:
        prev = mio.load("psiq1.mat")
        C = mat.scalar(0.0831875)
        C1 = mat.scalar(15623.4955718)
        C2 = mat.scalar(4360.3755823)
        Ln = prev["Lnq"]; Lntemp = prev["Lntemps"]
        h = prev["hq"]; htemp = prev["htemps"]
        psi = mat.interp2(prev["Zq"], prev["Rq"], prev["psiq"], Z, R, "nearest")

    Ctmp = C
    C1tmp = C1
    C2tmp = C2

    # `psi` 不存在 → MATLAB 的自动建变量（零矩阵 + 壁面那一片）
    psi = mat.zeros(nR, nZ)
    jrb_1d = np.asarray(jrb, dtype=int).ravel()
    psib_1d = np.asarray(psib).ravel()[:nZ]
    for jz in mat.mrange(1, nZ):   # for jz=1:nZ
        for jr in mat.mrange(int(jrb_1d[jz - 1]), nR):   # for jr=jrb(jz):nR
            psi[jr - 1, jz - 1] = psib_1d[jz - 1]

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
    nt = DEFAULT_NT
    tol = mat.scalar(1e-9)

    hf = mplot.figure("unit", "normalized", "Position", mat.row(0.6, 0.35, 0.4, 0.4),
                      "DefaultAxesFontSize", 15)
    mplot.set_(mplot.gcf(), "PaperPositionMode", "auto")

    icase = mat.scalar(3)

    # 松弛/rhs 的循环范围：`for j=2:nZ-1, for i=2:(jrb(j)-1)`
    ii_ = np.arange(nR)[:, None]
    jj_ = np.arange(nZ)[None, :]
    inside = ((ii_ >= 1) & (ii_ <= (jrb_1d[None, :] - 2))
              & (jj_ >= 1) & (jj_ <= nZ - 2))
    inside_J = (ii_ <= (jrb_1d[None, :] - 2))
    dR_v = mat.scalar_value(dR)
    dZ_v = mat.scalar_value(dZ)
    mu0_v = mat.scalar_value(mu0)

    while (mat.scalar_value(it) <= nt + mat.scalar_value(ita)
           and mat.scalar_value(dpsim) > mat.scalar_value(tol)):
        if mat.scalar_value(icase) == 1:
            dP = -2 * C1 * C * np.tanh(C1 * (psi - C2)) / np.cosh(C1 * (psi - C2)) ** 2 / mu0
        elif mat.scalar_value(icase) == 2:
            dP = -C1 * C * mat.exp(-C1 * psi) / mu0
        elif mat.scalar_value(icase) == 3:
            dP = (C * mat.exp(-C1 * psi)
                  * (2 * C2 * (C2 * psi + 1) - C1 * (C2 * psi + 1) ** 2) / mu0)

        Jzeta = R * dP

        # 逐项累加（1×1 标量、精确比对）：jz 外层、jr 内层 → 列优先后 cumsum
        v_J = (Jzeta * dR_v * dZ_v).T[inside_J.T]
        Jztotal = mat.scalar(float(np.cumsum(v_J)[-1]) if v_J.size else 0.0)

        if mat.scalar_value(icase) == 1:
            P = C / np.cosh(C1 * (psi - C2)) / mu0
        elif mat.scalar_value(icase) == 2:
            P = C * mat.exp(-C1 * psi) / mu0
        elif mat.scalar_value(icase) == 3:
            P = C * mat.exp(-C1 * psi) * (C2 * psi + 1) ** 2 / mu0

        if mat.scalar_value(it) == 1:
            w = mat.scalar(0.9)
            w0 = w
            _flush(ws, locals(), ifd_value=int(mat.scalar_value(ifd_v)),
                   ibc_value=int(mat.scalar_value(ibc_v)), nR=nR, nZ=nZ, jrb=jrb)
        else:
            wc = w
            # `P(:,Z0)` 在 MATLAB 里是列向量（nR×1）
            P_m = np.asarray(P)[:, int(mat.scalar_value(Z0)) - 1].reshape(-1, 1)
            psi_m = np.asarray(psi)[:, int(mat.scalar_value(Z0)) - 1].reshape(-1, 1)
            jr_m = np.asarray(Jzeta)[:, int(mat.scalar_value(Z0)) - 1].reshape(-1, 1)
            # 状态刷进工作区之后才调 solve_lp：MATLAB 报错时留下的正是这一份
            # （第 2 轮的 dP/P/Jzeta/Jztotal + 第 1 轮的 psi/绘图变量）
            _flush(ws, locals(), ifd_value=int(mat.scalar_value(ifd_v)),
                   ibc_value=int(mat.scalar_value(ibc_v)), nR=nR, nZ=nZ, jrb=jrb)
            # ↓ 原文这里调 `solve_lp`，它在整个仓库里不存在，脚本就在这一行中止。
            #   修改版在同目录补了 `solve_lp.py` / `solve_h.py`，于是能继续往下跑。
            lp = mpath.call("solve_lp", P_m, psi_m, np.asarray(rr).T)
            Lntemp = -(eta + 1) * lp
            print(f"Ln ={mat.num2str(Lntemp)} ".ljust(7))
            htemp = mpath.call("solve_h", jr_m, rr, psi_m)
            print(f"h ={mat.num2str(htemp)} ".ljust(7))
            rs = mpath.call("solve_rs", psi_m, rr)
            print(f"rs ={mat.num2str(rs)} ".ljust(7))

        rhs = _rhs_block(ifd_v, psi0, dP, rr, jrb, nZ, dR, dZ, dR2, dZ2, mu0,
                         inside=inside)

        w_v = mat.scalar_value(w)
        psi[inside] = w_v * psi0[inside] + rhs[inside] * (1 - w_v)

        # boundary condition
        psi[0, :] = 0.0                       # zero boundary at axis
        psi[jrb_1d - 1, np.arange(nZ)] = psib_1d   # fixed boundary at wall

        # dpsi/dz=0 at z=min,max
        if mat.scalar_value(ibc_v) == 1:
            psi[:, 0] = psi[:, 1]
            psi[:, nZ - 1] = psi[:, nZ - 2]
        elif mat.scalar_value(ibc_v) == 2:
            psi[:, 0] = (4 * psi[:, 1] - psi[:, 2]) / 3
            psi[:, nZ - 1] = (4 * psi[:, nZ - 2] - psi[:, nZ - 3]) / 3
        elif mat.scalar_value(ibc_v) == 3:
            psi[:, 0] = (18 * psi[:, 1] - 9 * psi[:, 2] + 2 * psi[:, 3]) / 11
            psi[:, nZ - 1] = (18 * psi[:, nZ - 2] - 9 * psi[:, nZ - 3]
                              + 2 * psi[:, nZ - 4]) / 11
        elif mat.scalar_value(ibc_v) == 4:
            psi[:, 0] = (48 * psi[:, 1] - 36 * psi[:, 2] + 16 * psi[:, 3]
                         - 3 * psi[:, 4]) / 25
            psi[:, nZ - 1] = (48 * psi[:, nZ - 2] - 36 * psi[:, nZ - 3]
                              + 16 * psi[:, nZ - 4] - 3 * psi[:, nZ - 5]) / 25

        it = it + 1
        print(f"it = {int(mat.scalar_value(it)):04d}".ljust(7))
        dpsim = mat.scalar(np.max(np.abs(np.asarray(psi) - np.asarray(psi0))))
        psi0 = np.array(psi, copy=True)
        Ctmp = C; C1tmp = C1; C2tmp = C2

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
            mplot.contour(Z, R, psii, 15, "linewidth", 2)
            mplot.hold_on()
            mplot.contour(Z, R, psi, mat.row(0, 0), "r--", "linewidth", 3)
            mplot.hold_on()
            mplot.plot(mat.take(Z, int(mat.el(ido, 1))),
                       mat.take(R, int(mat.el(ido, 1))), "mx", "linewidth", 3)
            mplot.text(0.01 * mat.scalar_value(zw) + mat.take(Z, int(mat.el(ido, 1))),
                       mat.take(R, int(mat.el(ido, 1))), "O point", "fontsize", 13)
            mplot.colorbar()
            mplot.plot(zb, rb, "-", zzb, rrb, ":", "linewidth", 3)
            mplot.title(f"(a) \\Psi(R,Z), FRC G-S equilibrium, it="
                        f"{int(mat.scalar_value(it))}/{nt}, "
                        f"max(\\Psi^n-\\Psi^{{n-1}})={mat.num2str(dpsim)}")
            mplot.ylabel(f"R, r_w={mat.num2str(rw)}")
            strp = "p(\\psi)=-C*exp(-C1*psi)*(C2*psi+1)^2"
            mplot.xlabel(f"Z, z_w={mat.num2str(zw)}, I_z={mat.num2str(Izeta / 1e6)}"
                         f"MA, J_zt={mat.num2str(Jztotal / 1e6, 3)}MA, C="
                         f"{mat.num2str(C)}, C1={mat.num2str(C1)}, C2="
                         f"{mat.num2str(C2)} with {strp}")
            mplot.box("on")
            mplot.legend("\\psi_{in}", "\\psi_{out}", "separatix")
            mplot.legend("boxoff")
            mplot.hold_off()

            mplot.subplot(2, 3, 4)
            psi_x = np.array(np.asarray(psi)[:, int(mat.scalar_value(Z0)) - 1],
                             copy=True).reshape(-1, 1)
            ind_min = mat.find(psi_x == np.min(psi_x))
            col = int(mat.scalar_value(Z0)) - 1
            mplot.plot(rr / mat.take(rr, int(mat.el(ind_min, 1))),
                       -np.asarray(Jzeta)[:, col]
                       / np.max(np.abs(np.asarray(Jzeta)[:, col])),
                       "m", "linewidth", 3)
            mplot.box("on")
            mplot.grid_on()
            mplot.xlabel("r")
            mplot.ylabel("J_{\\theta}")

            mplot.subplot(2, 3, 5)
            mplot.contourf(R, Z, dP)
            mplot.title(f"(c) dP, tol={mat.num2str(tol)}")
            mplot.xlabel("R")
            mplot.ylabel("Z")
            mplot.box("on")

            mplot.subplot(2, 3, 6)
            mplot.plot(rr / mat.take(rr, int(mat.el(ind_min, 1))),
                       np.asarray(psi)[:, col], "m", "linewidth", 3)
            mplot.box("on")
            mplot.grid_on()
            mplot.title(f"(d) \\Psi, w={mat.num2str(w)}")
            mplot.xlabel("r")
            mplot.ylabel("\\psi")
            mplot.pause(0.0001)
            for name in ("ido", "psio", "psii", "indtmp", "psi_x", "ind_min", "strp"):
                ws[name] = locals().get(name)

    # ---- 收尾（本算例在第 2 轮就报错了，走不到这里）---------------------------
    runtime = runtime0 + mat.scalar(0) - runtime
    print("**************************************** ")
    print(f"C1 = {float(np.asarray(C).ravel()[0]):7.6f} ")
    print(f"C1 = {float(np.asarray(C1).ravel()[0]):5.3f} ")
    print(f"C2 = {float(np.asarray(C2).ravel()[0]):5.3f} ")

    Rq = R; Zq = Z; psiq = psi; C1q = C1; Cq = C; C2q = C2
    Lnq = Ln; Lntemps = Lntemp; hq = h; htemps = htemp
    if mat.scalar_value(Izeta) == 0:
        filename = "psiq_vac"
    else:
        filename = "psiq"
    mio.save(filename + ".mat", Rq=Rq, Zq=Zq, psiq=psiq, Jzeta=Jzeta, P=P,
             jrb=jrb, zzb=zzb, rrb=rrb, psib=psib, Cq=Cq, C1q=C1q, C2q=C2q,
             nR=mat.scalar(nR), nZ=mat.scalar(nZ), Izeta=Izeta, hq=hq,
             htemps=htemps)
    mio.save(f"{filename}_ifd={int(mat.scalar_value(ifd_v))}"
             f"_ibc={int(mat.scalar_value(ibc_v))}_nR={nR}_nZ={nZ}.mat",
             Rq=Rq, Zq=Zq, psiq=psiq, Jzeta=Jzeta, P=P, jrb=jrb, zzb=zzb,
             rrb=rrb, psib=psib, Cq=Cq, C1q=C1q, C2q=C2q, nR=mat.scalar(nR),
             nZ=mat.scalar(nZ), Izeta=Izeta)
    mio.save("psiq1.mat", Rq=Rq, Zq=Zq, psiq=psiq, Cq=Cq, C1q=C1q, C2q=C2q,
             Lnq=Lnq, Lntemps=Lntemps, hq=hq, htemps=htemps)
    screenposition = mplot.get_(mplot.gcf(), "Position")
    mplot.set_(mplot.gcf(), "PaperPosition",
               mat.row(0, 0, screenposition[2], screenposition[3]), "PaperSize",
               mat.row(screenposition[2], screenposition[3]))
    str_ = (f"gs_frc_rw={mat.num2str(rw)},zw={mat.num2str(zw)},"
            f"Iz={mat.num2str(-Izeta)},nR={mat.num2str(nR)},"
            f"nZ={mat.num2str(nZ)},C={mat.num2str(C)},w={mat.num2str(w)},"
            f"tol={mat.num2str(tol)},it={mat.num2str(it)}")

    _flush(ws, locals(), ifd_value=int(mat.scalar_value(ifd_v)),
           ibc_value=int(mat.scalar_value(ibc_v)), nR=nR, nZ=nZ, jrb=jrb)
