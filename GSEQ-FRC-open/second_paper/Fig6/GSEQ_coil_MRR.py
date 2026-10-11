"""二维自由边界求解器（图 6）——**加速版**，直接顶替原来的 `GSEQ_coil_MRR.m`。

原来那份 `GSEQ_coil_MRR.m`（375 行、逐点循环）在这台机器上要跑 8.3 小时，而且
自己没收敛到底，所以这个位置直接放**加速版**：线圈 Green 函数与 GrzJz 张量向量化、
边界通量走矩阵乘、内层循环用预分配缓冲、最外层 CI 归一化换成带保护的 secant。
同配置下 MATLAB 加速版 665 s、这份 413 s，两者数值差在 \(10^{-13}\) 相对量级
（只来自求和次序），逐点算式一字未改。

**与原文的差异**（详见 `修改记录.md`）：

1. 线圈 Green 函数里的 `Br` 补上了 \(1/(2\pi)\)。原文 `psi`、`Bz` 都除了
   \(2\pi\)，只有 `Br` 没除，于是同一个函数给出的 \(B_r\) 比同一函数据出的
   \(B_z\) 的归一化大 \(2\pi\) 倍（可以和 `fun_Bfield.m` 在同一点的值对照）。
   平衡只用 `Bz`/`psiw`，所以场解一字未动，改的是这个诊断量。
2. 原来的 `GSEQ_coil_MRR.m` 没有逐行保留：它和加速版在同一个位置、做同一件事，
   留着两份只会让人不知道该跑哪个。
3. 内层求解器默认换成 `nk_inner.py` 里的**快速配方**（FreeGSNKE 式：约束内嵌 +
   IMEX + 容差线搜索，卡住回退到原文松弛）——全量从约 415 s 降到约 18 s，
   代价是解在 \(2.8\times10^{-4}\)（相对场幅值）量级上与冻结基准不同。
   要复现冻结基准那份解，用 `solve(..., inner="relax")`。

对应关系（MATLAB 行号 → 本文件）：
  * 参数、网格、边界点枚举      —— `_setup`
  * [A] 线圈 Green 函数（向量化）—— `_coil_green`
  * [B] GrzJz（向量化）          —— `_grz_jz`
  * [C] 边界通量更新（矩阵乘）   —— 主循环里一行
  * [M] 内层逐点松弛（向量化）   —— 主循环里的 `psi[i0:i1, j0:j1] = ...`
  * [F] CI 归一化 secant         —— 主循环末尾

不移植的部分：绘图段（`ch`/`E` 来自 MATLAB 的 `contour`，属绘图派生量）。

用法（与程序树里其它求解脚本一致）：

    mpath.addpath("second_paper/Fig6")
    mpath.load("GSEQ_coil_MRR").run(ws)   # 结果写进 ws，并保存 psiq.mat / psiq1.mat
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

MU0 = 4 * np.pi * 1e-7

#: **修改版**（第 17 条）：互感/线圈↔网格耦合矩阵用的丝状电流网格。
#: 必须用与边界 Green 基同一套 10×100：中间线圈的截面是 6 mm × 1.2 m 的细长条，
#: 粗丝阵（试过 4×10）会把端点附近欠采样，相邻线圈的互感冒估 50%。边界基与这两类
#: 矩阵共用同一离散，才不会出现"同一台机器两套 Green 函数"。
_MUTUAL_NR, _MUTUAL_NZ = 10, 100


def ellipEK_vec(k2):
    """`fun_ellipEK(k2,1)` 的向量化版本（同一组系数、同一运算顺序）。"""
    pk = 1.0 - k2
    ak = (((0.01451196212 * pk + 0.03742563713) * pk + 0.03590092383) * pk
          + 0.09666344259) * pk + 1.38629436112
    bk = (((0.00441787012 * pk + 0.03328355346) * pk + 0.06880248576) * pk
          + 0.12498593597) * pk + 0.5
    K = ak - bk * np.log(pk)
    ae = (((0.01736506451 * pk + 0.04757383546) * pk + 0.0626060122) * pk
          + 0.44325141463) * pk + 1.0
    be = (((0.00526449639 * pk + 0.04069697526) * pk + 0.09200180037) * pk
          + 0.2499836831) * pk
    E = ae - be * np.log(pk)
    bad = (k2 == 1.0)
    if np.any(bad):
        K = np.where(bad, 1.0e300, K)
        E = np.where(bad, 1.0, E)
    return K, E


def _setup(src_dir: Path, icase=2, rw=0.14, zw=0.75, nR=2**7 + 1, nZ=2**8 + 1):
    """参数 + 网格 + 边界点枚举（对应 MATLAB 第一部分）。"""
    m = loadmat(src_dir / "MRR1d.mat", squeeze_me=True)
    betas, ls, alpha, sigma, n = (float(m[k]) for k in
                                  ("betas", "ls", "alpha", "sigma", "n"))
    Rs, Rw, Be = float(m["Rs"]), float(m["Rw"]), float(m["Be"])
    Izeta0 = -float(m["Izeta0"])          # 原文：Izeta0=-Izeta0;
    Pm0 = Be**2 / (2 * MU0)

    if icase == 2:
        hl = np.array([0.006, 0.006, 0.006])
        rl = np.array([0.15, 0.15, 0.15])
        zl = np.array([-0.7, 0.0, 0.7])
        wl = np.array([0.1, 1.2, 0.1])
        Il = np.array([1000.0, 1000.0, 1000.0]) * 1000
    else:
        raise NotImplementedError("只翻译了 icase=2（本算例用的那组线圈）")
    nl = rl.size

    rr = np.linspace(0.0 * Rw, Rw, nR)
    zz = np.linspace(-zw, zw, nZ)
    dR, dZ = rr[1] - rr[0], zz[1] - zz[0]
    R, Z = np.meshgrid(rr, zz, indexing="ij")     # 对应 MATLAB ndgrid(rr,zz)
    dR2, dZ2 = dR * dR, dZ * dZ
    Z0 = (nZ + 1) // 2

    nb = 2 * (nR - 1) + (nZ - 2)
    jrb = np.ones(nb, dtype=int)
    jzb = np.ones(nb, dtype=int)
    for j in range(1, nb + 1):                     # MATLAB 的 1 基编号
        if j < nR:                                 # z = zmin
            jrb[j - 1], jzb[j - 1] = j + 1, 1
        elif j > (nb - nR + 1):                    # z = zmax
            jrb[j - 1], jzb[j - 1] = nb - j + 2, nZ
        else:                                      # r = rmax
            jrb[j - 1], jzb[j - 1] = nR, j - nR + 2
    bnd_lin = (jrb - 1) + (jzb - 1) * nR           # 0 基线性下标
    bnd_ii = bnd_lin % nR                          # 转成二维下标（列优先线性下标）
    bnd_jj = bnd_lin // nR
    Rbnd = R.ravel(order="F")[bnd_lin]
    Zbnd = Z.ravel(order="F")[bnd_lin]
    return dict(betas=betas, ls=ls, alpha=alpha, sigma=sigma, n=n,
                Rs=Rs, Rw=Rw, Be=Be, Izeta0=Izeta0, Pm0=Pm0,
                hl=hl, rl=rl, zl=zl, wl=wl, Il=Il, nl=nl,
                rr=rr, zz=zz, dR=dR, dZ=dZ, R=R, Z=Z, dR2=dR2, dZ2=dZ2, Z0=Z0,
                nR=nR, nZ=nZ, nb=nb, jrb=jrb, jzb=jzb, bnd_lin=bnd_lin,
                bnd_ii=bnd_ii, bnd_jj=bnd_jj,
                Rbnd=Rbnd, Zbnd=Zbnd, rw=rw, zw=zw,
                # `MRR1d.mat` 里随论文一起提供的那几个量：MATLAB 的 `load` 会把
                # 整个文件的变量都放进工作区（快照里就有它们），这里原样带着，
                # 供主树按"物理量契约"逐变量比对（见 PORTABLE.md §7）。
                psibc=float(m["psibc"]), C1s=float(m["C1s"]), C2s=float(m["C2s"]),
                j1d=np.asarray(m["j1d"]), pr1d=np.asarray(m["pr1d"]),
                psi1d=np.asarray(m["psi1d"]), rr1d=np.asarray(m["rr1d"]),
                Bz1d=np.asarray(m["Bz1d"]))


def _coil_filaments(s, jl, nr: int = 10, nz: int = 100):
    """第 `jl` 个线圈的矩形截面丝状电流离散（与 `_coil_green` 原文同一套网格）。

    **修改版**（第 17 条）：把这条离散抽出来，让边界 Green 基、线圈互感、
    线圈↔网格耦合三处共用同一套丝状电流，避免各写一份导致不一致。

    默认 10×100 与原文一致（逐位复现）；互感/耦合矩阵也走同一离散
    （`_MUTUAL_NR × _MUTUAL_NZ`），保证全树只有一套线圈几何。
    """
    rmin = s["rl"][jl] - 0.5 * s["hl"][jl]
    rmax = s["rl"][jl] + 0.5 * s["hl"][jl]
    zmin = s["zl"][jl] - 0.5 * s["wl"][jl]
    zmax = s["zl"][jl] + 0.5 * s["wl"][jl]
    dr = (rmax - rmin) / (nr - 1)
    dz = (zmax - zmin) / (nz - 1)
    rr_c = rmin + dr * np.arange(nr)
    zz_c = zmin + dz * np.arange(nz)
    R1g, Z1g = np.meshgrid(rr_c, zz_c, indexing="ij")   # 与 for jr/for jz 同序
    return (R1g.ravel(order="F"), Z1g.ravel(order="F"), nr, nz)


def _green_G(R_a, Z_a, R_b, Z_b):
    """一般点对点的 `G`（`ellipEK_vec` 那套 Green 函数，形状由广播决定）。

    **修改版**（第 17 条）：`_coil_green` 原来只算"线圈→边界点"，这里抽出一个
    通用版本，供线圈互感、线圈↔网格耦合使用；公式与 `_coil_green`/`_grz_jz` 内
    逐字相同（含 `k2==0` 的裁剪）。返回 `G`，其中 `psi = MU0 * I * G`。
    """
    Ra, Za = np.asarray(R_a), np.asarray(Z_a)
    Rb, Zb = np.asarray(R_b), np.asarray(Z_b)
    k2 = 4 * (Ra * Rb) / ((Ra + Rb) ** 2 + (Za - Zb) ** 2)
    # 自感项里会出现 k2==1（丝重合），`ellipEK_vec` 在 pk=1-k2=0 处会 log(0)；
    # 先夹到 (0, 1-eps) 消掉告警，反正这些点在下面本来就被 mask 掉。
    K, E = ellipEK_vec(np.clip(k2, 0.0, 1.0 - 1e-15))
    G = np.zeros_like(k2)
    # k2==0（在轴上）G 本来就是 0；k2==1（两点重合，只有自感项里会出现）要显式
    # 排掉——那里 K 会取 1e300 的哨兵值，否则会污染求和（见 `_coil_mutual_matrix`）。
    v = (k2 != 0) & (k2 != 1.0)
    G[v] = 1 / (2 * np.pi * np.sqrt(k2[v])) * np.sqrt(Ra * Rb)[v] \
        * ((2 - k2[v]) * K[v] - 2 * E[v])
    return G


def _coil_green_basis(s):
    """[A] 线圈 Green 函数的**每线圈、单位电流**基（`**修改版**` 第 17 条）。

    返回 `psiw_sum`/`Br_sum`/`Bz_sum`，每个都是 `(nl, nb)`：第 `jl` 行是
    "丝状电流求和的原始值"（不含 `MU0 * I1` 系数）。组合方式写回与原文完全相同的
    算式（`MU0 * (Il/(nr*nz)) * sum`），所以默认路径逐位不变；而线圈电流随时间变
    时，只需要把这三张表拿出来做加权和，不必重新跑 Green 的 10×100 求积。
    """
    nb, nl = s["nb"], s["nl"]
    Rbnd, Zbnd = s["Rbnd"], s["Zbnd"]
    psiw_sum = np.zeros((nl, nb))
    Br_sum = np.zeros((nl, nb))
    Bz_sum = np.zeros((nl, nb))
    meta: list[tuple[int, int]] = []
    for jl in range(nl):
        R1f, Z1f, nr, nz = _coil_filaments(s, jl)
        R1 = R1f[None, :]
        Z1 = Z1f[None, :]
        k2 = 4 * (Rbnd[:, None] * R1) / ((Rbnd[:, None] + R1) ** 2
                                         + (Zbnd[:, None] - Z1) ** 2)
        K, E = ellipEK_vec(k2)
        iv, jv = np.nonzero(k2 != 0)
        Rbv, Zbv = Rbnd[iv], Zbnd[iv]
        R1v, Z1v = R1[0, jv], Z1[0, jv]
        k2v, Kv, Ev = k2[iv, jv], K[iv, jv], E[iv, jv]
        den = np.sqrt((Rbv + R1v) ** 2 + (Zbv - Z1v) ** 2)
        G = np.zeros_like(k2)
        G[iv, jv] = 1 / (2 * np.pi * np.sqrt(k2v)) * np.sqrt(Rbv * R1v) \
            * ((2 - k2v) * Kv - 2 * Ev)
        psiw_sum[jl] = G.sum(axis=1)
        BrT = np.zeros_like(k2)
        BrT[iv, jv] = (Zbv - Z1v) / Rbv / den * (
            (Rbv**2 + R1v**2 + (Zbv - Z1v) ** 2)
            / ((Rbv - R1v) ** 2 + (Zbv - Z1v) ** 2) * Ev - Kv)
        Br_sum[jl] = BrT.sum(axis=1)
        BzT = np.zeros_like(k2)
        BzT[iv, jv] = 1 / den * (
            (R1v**2 - Rbv**2 - (Zbv - Z1v) ** 2)
            / ((Rbv - R1v) ** 2 + (Zbv - Z1v) ** 2) * Ev + Kv) / 2 / np.pi
        Bz_sum[jl] = BzT.sum(axis=1)
        meta.append((nr, nz))
    return dict(psiw_sum=psiw_sum, Br_sum=Br_sum, Bz_sum=Bz_sum, meta=meta)


def _coil_green(s, basis=None, Il=None):
    """[A] 线圈 Green 函数，向量化（`psiw_coil`/`Br_coil`/`Bz_coil`/`psib`）。

    **修改版**（第 17 条）：可传入 `basis`（来自 `_coil_green_basis`）与 `Il`
    （线圈电流，默认 `s["Il"]`）。组合算式与原文逐字相同，`basis=None` 时数值
    逐位不变。
    """
    if basis is None:
        basis = _coil_green_basis(s)
    if Il is None:
        Il = s["Il"]
    nb, nl = s["nb"], s["nl"]
    psiw = np.zeros(nb)
    Br = np.zeros(nb)
    Bz = np.zeros(nb)
    for jl in range(nl):
        nr, nz = basis["meta"][jl]
        I1 = Il[jl] / (nr * nz)
        psiw = psiw + MU0 * I1 * basis["psiw_sum"][jl]
        # 修改版：补上 1/(2*pi)。原文这里漏了它，而同一个 Green 函数的 psi、Bz 都除了，
        # 于是 Br_coil 比 fun_Bfield.m 在同一点给出的值大 2*pi 倍。
        Br = Br + MU0 * I1 * basis["Br_sum"][jl] / (2 * np.pi)
        Bz = Bz + MU0 * I1 * basis["Bz_sum"][jl]
    return psiw, Br, Bz, psiw.copy()


def _grz_jz(s):
    """[B] GrzJz（内部格点 → 边界点），向量化，返回按列优先线性下标的二维矩阵。"""
    nR, nZ, nb = s["nR"], s["nZ"], s["nb"]
    Rbnd, Zbnd = s["Rbnd"], s["Zbnd"]
    Rf = s["R"].ravel(order="F")
    Zf = s["Z"].ravel(order="F")
    int_mask = np.zeros(nR * nZ, dtype=bool)
    int_mask[(np.arange(1, nR - 1)[:, None]
              + np.arange(1, nZ - 1)[None, :] * nR).ravel()] = True
    int_lin = np.nonzero(int_mask)[0]
    Rint, Zint = Rf[int_lin], Zf[int_lin]
    G2 = np.zeros((nR * nZ, nb))
    for j in range(nb):
        k2 = 4 * Rbnd[j] * Rint / ((Rbnd[j] + Rint) ** 2 + (Zbnd[j] - Zint) ** 2)
        K, E = ellipEK_vec(k2)
        g = np.zeros_like(k2)
        v = k2 != 0
        g[v] = 1 / (2 * np.pi * np.sqrt(k2[v])) * np.sqrt(Rbnd[j] * Rint[v]) \
            * ((2 - k2[v]) * K[v] - 2 * E[v])
        G2[int_lin, j] = g
    return G2, int_lin


def _coil_mutual_matrix(s):
    """线圈—线圈几何互感矩阵 `M[i, j]`（H），**修改版** 第 17 条。

    用与 `_coil_green` 同一套矩形截面丝状电流离散：第 `j` 个线圈的总电流均分到
    `Nj` 根丝上，第 `i` 个线圈的磁链取 `Ni` 根丝的平均。由 `_green_G` 的约定
    `psi = MU0 * I * G` 以及磁链 `Psi = 2*pi*psi`，有

        M[i,j] = 2*pi * MU0/(Ni*Nj) * sum_{a,b} G(R_a,Z_a; R_b,Z_b)

    注意：**自感项依赖绕组内部结构**（并绕方式、导线截面），这里的几何值只能当
    上界/初值，正式用时应按实机标定覆盖对角线（freegsnke 的 PF 线圈互感/电阻也是
    用真空炮数据标定的）。互项（i≠j）是几何量，可以直接用。
    """
    nl = s["nl"]
    fil = [_coil_filaments(s, jl, _MUTUAL_NR, _MUTUAL_NZ) for jl in range(nl)]
    M = np.zeros((nl, nl))
    for i in range(nl):
        Ra, Za, nri, nzi = fil[i]
        for j in range(nl):
            Rb, Zb, nrj, nzj = fil[j]
            G = _green_G(Ra[:, None], Za[:, None], Rb[None, :], Zb[None, :])
            M[i, j] = 2 * np.pi * MU0 / (nri * nzi * nrj * nzj) * G.sum()
    # 自感项：丝状电流的重合对（k2=1）被排掉了，剩下的"互丝"求和只给出一个
    # 依赖离散（在这里是每线圈 10×100 丝）的几何值，**不是**物理自感——导线内部
    # 磁场贡献缺失。这里标成 NaN，强制使用方按实机绕组/导体截面标定后填入；
    # 互项（i≠j）不含重合对，是干净的几何互感，可以直接用。
    np.fill_diagonal(M, np.nan)
    return M


def _coil_grid_green(s, int_lin, chunk: int = 2048):
    """线圈 ↔ 内部网格点的互感矩阵（`(nl, N_int)`，H），**修改版** 第 17 条。

    给定网格上的等离子体电流分布 `Jz`（A/m^2），每个网格点的电流是
    `I_y = Jz_y * dR * dZ`，穿过第 `i` 个线圈的磁链是
    `Psi_i = sum_y M[i, y] * I_y`。这也是 freegsnke 回路方程 (2.1) 里
    `M_{m,y} dI_y/dt` 那一项需要的矩阵（对称性：`M[i,y]` 同样等于"网格点 y
    的单位电流对线圈 i 的互感"）。

    代价约 97M 次 Green 求值（3 线圈 × 1000 丝 × 32k 网格点），实测 **15 s**
    （本机、有负载），所以默认不建（`MachineModel(plasma_coupling=False)`）；
    需要"等离子体电流反过来耦合线圈"（self-consistent 回路演化）时再一次性建好、
    `model.save(...)` 存盘复用（走编译内核后这一步能降到 ~1 s，属后续优化）。
    """
    Rf = s["R"].ravel(order="F")[int_lin]
    Zf = s["Z"].ravel(order="F")[int_lin]
    nl = s["nl"]
    out = np.zeros((nl, Rf.size))
    for jl in range(nl):
        Ra, Za, nri, nzi = _coil_filaments(s, jl, _MUTUAL_NR, _MUTUAL_NZ)
        acc = np.zeros(Rf.size)
        for k0 in range(0, Rf.size, chunk):
            k1 = min(k0 + chunk, Rf.size)
            G = _green_G(Ra[:, None], Za[:, None], Rf[None, k0:k1], Zf[None, k0:k1])
            acc[k0:k1] = G.sum(axis=0)
        out[jl] = 2 * np.pi * MU0 / (nri * nzi) * acc
    return out


class MachineModel:
    """与线圈电流无关的**机器模型**（**修改版** 第 17 条，步骤 3）。

    时间演化里唯一随时间变的是线圈电流。把不随电流变的东西全部预计算一次：

    * `basis`       ——每线圈、单位电流的边界 Green 基（`_coil_green_basis`）；
    * `G2`/`int_lin`——等离子体电流 → 边界的 GrzJz 张量（`_grz_jz`，约 135 MB）；
    * `Mcoil`       ——线圈—线圈互感矩阵（几何值，自感需标定）；
    * `coil_grid`   ——线圈↔内部网格耦合（可选，回路自洽演化时才需要）。

    每时间步只需要 `coil_flux(Il)` 做一次加权和，拿到 `psib_coil` 交给
    `solve(..., model=model, Il=...)`。
    """

    def __init__(self, s, *, basis=None, G2=None, int_lin=None, mutual=True,
                 plasma_coupling=False, verbose=False):
        self.s = s
        self.nR, self.nZ = s["nR"], s["nZ"]
        self.nb, self.nl = s["nb"], s["nl"]
        t0 = time.perf_counter()
        self.basis = _coil_green_basis(s) if basis is None else basis
        if G2 is None or int_lin is None:
            G2, int_lin = _grz_jz(s)
        self.G2, self.int_lin = G2, int_lin
        self.Mcoil = _coil_mutual_matrix(s) if mutual else None
        self.coil_grid = (_coil_grid_green(s, int_lin) if plasma_coupling
                          else None)
        #: order -> Problem 缓存（算子 A、LU 分解、Gsub 都与线圈电流无关，
        #: 时间演化里逐步复用，省掉每步约 0.2 s 的构造/分解开销）
        self._problems: dict[int, object] = {}
        self.build_s = time.perf_counter() - t0
        if verbose:
            print(f"  [MachineModel] 预计算 {self.build_s:.3f} s："
                  f"Green 基 {self.basis['psiw_sum'].shape}，"
                  f"G2 {self.G2.shape}，互感 {None if self.Mcoil is None else self.Mcoil.shape}")

    # ---- 每步只需要这一行：线圈电流的线性叠加 ----
    def coil_flux(self, Il=None):
        """由线圈电流算 `(psiw, Br, Bz, psib_coil)`，与 `_coil_green` 同式同序。"""
        return _coil_green(self.s, basis=self.basis, Il=Il)

    def problem(self, psib_coil, order: int = 2, params=None):
        """取（或新建）本机器模型上的 `nk_inner.Problem`，并更新线圈边界通量。

        `Problem` 的 `A`、`G2`、`Gsub` 与线圈电流无关，构造一次就够；
        `_luF`/`_luA`（算子分解）也随对象缓存在里面。时间演化每步只是换
        `psib_coil`，再把 `psi_init`（只在冷启动时用作初值）按新的幅值重算——
        暖启动路径给了 `psi0`，根本不走 `psi_init`。

        **修改版**（第 19 条）：多了 `params`，时间演化里每步可以顺带覆盖剖面/
        闭合参数（`Izeta0`、`betas`、`Be`…），同样不必重建 `Problem`。
        """
        nk = _load_nk_inner()
        order = int(order)
        pr = self._problems.get(order)
        if pr is None:
            pr = nk.Problem(self.s, psib_coil, self.G2, order=order)
            self._problems[order] = pr
        else:
            pr.psib_coil = np.asarray(psib_coil, dtype=float).ravel()
            pr.maxpsiw = float(np.abs(pr.psib_coil).max())
            pr.psi_init = pr._initial_psi()
        if params:
            pr.set_params(params)
        return pr

    # ---- 存盘/读盘（跨进程复用；G2 太大，默认不存，重建约 0.4 s）----
    def save(self, path, include_G2: bool = False) -> None:
        path = Path(path)
        payload = dict(
            psiw_sum=self.basis["psiw_sum"], Br_sum=self.basis["Br_sum"],
            Bz_sum=self.basis["Bz_sum"],
            meta=np.asarray(self.basis["meta"], dtype=np.int64),
            int_lin=np.asarray(self.int_lin),
            Mcoil=(np.zeros((0, 0)) if self.Mcoil is None else self.Mcoil))
        np.savez(path, **payload)
        if include_G2:
            np.save(str(path) + ".G2.npy", self.G2)

    @classmethod
    def load(cls, path, s, *, mmap: bool = False, mutual: bool = True,
             plasma_coupling: bool = False):
        path = Path(path)
        npz = path if path.suffix == ".npz" else Path(str(path) + ".npz")
        z = np.load(npz)
        basis = dict(psiw_sum=z["psiw_sum"], Br_sum=z["Br_sum"],
                     Bz_sum=z["Bz_sum"],
                     meta=[tuple(int(v) for v in row) for row in z["meta"]])
        int_lin = z["int_lin"]
        g2_file = Path(str(path) + ".G2.npy")
        G2 = (np.load(g2_file, mmap_mode="r" if mmap else None)
              if g2_file.exists() else None)
        if G2 is None:
            G2, int_lin = _grz_jz(s)
        model = cls(s, basis=basis, G2=G2, int_lin=int_lin, mutual=False,
                    plasma_coupling=plasma_coupling)
        M = z["Mcoil"]
        model.Mcoil = M if (mutual and M.size) else None
        return model


_NK_INNER = None


def _load_nk_inner():
    """按文件路径加载同目录的 `nk_inner.py`（**修改版**新增）。

    程序树的目录名里带连字符，不能当 Python 包导入；`mpath` 加载本模块时也没有
    包上下文。所以这里像 `mpath.load` 那样按**文件路径**加载并缓存。
    """
    global _NK_INNER
    if _NK_INNER is None:
        import importlib.util

        path = Path(__file__).resolve().parent / "nk_inner.py"
        spec = importlib.util.spec_from_file_location("gseqfrc._fig6_nk_inner", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _NK_INNER = module
    return _NK_INNER


def _with_params(s, params=None):
    r"""**修改版**（第 19 条）：在 setup 上叠加本步的剖面参数覆盖，返回**新字典**
    （绝不改动 `model.s`，否则会把机器模型的状态污染掉）。

    只认 `nk_inner.PROFILE_KEYS`（`betas/alpha/sigma/n/Be/Rs/Pm0/Izeta0`）；
    给了 `Be` 而没给 `Pm0` 时按 \\(P_{m0}=B_e^2/2\\mu_0\\) 自动重算——`_setup` 里
    就是这个定义。
    """
    if not params:
        return s
    nk = _load_nk_inner()
    unknown = sorted(set(params) - set(nk.PROFILE_KEYS))
    if unknown:
        raise ValueError(f"未知的剖面参数 {unknown}；可用的是 "
                         f"{list(nk.PROFILE_KEYS)}")
    out = dict(s)
    out.update({k: float(v) for k, v in params.items()})
    if "Be" in params and "Pm0" not in params:
        out["Pm0"] = out["Be"] ** 2 / (2 * MU0)
    return out


def _gs_residual(pr, psi_flat, dP_flat, C):
    """离散 G-S 方程的**幅值相对残差**：\\((\\max F-\\min F)/(\\max\\psi-\\min\\psi)\\)，
    \\(F=A\\psi-\\mathrm{rhs}\\)。与 freegsnke 的 `relative_del_residual` 同口径。"""
    rhs = np.array(pr.bnd_val)
    rhs[pr.int_idx] = pr.dZ2 * MU0 * pr.Rint ** 2 * dP_flat[pr.int_idx] * float(C)
    F = (pr.A @ psi_flat - rhs)[pr.int_idx]
    amp = float(psi_flat.max() - psi_flat.min())
    return float((F.max() - F.min()) / amp) if amp > 0 else float("inf")


def _residual_diagnostics(s, psib_coil, G2, psi, C, psib, order: int = 2, pr=None):
    """**修改版**：终态的残差诊断（不再只看 `dpsim`）。

    两个指标，都是"幅值相对"的（除以 ψ 的幅值 max−min，跟 freegsnke 的
    `relative_del_residual` 同一口径）：

    * `gs_res`  —— 离散 G-S 方程残差：\\(F=A\\psi-\\mathrm{rhs}\\) 的幅值相对残差。
      这条是"方程到底解好没有"的判据（freegsnke 就是拿它停的）。
    * `defect` —— 松弛映射的固定点缺陷：把 ψ 代进**无阻尼**的一轮更新 `G(ψ)`，
      看差多少。`dpsim`（原程序的判据）其实等于 \\((1-w)\\cdot\\) 这个量的逐轮变化，
      所以 `dpsim ≤ tol` 只说明 `defect ≤ tol/(1-w) = 100·tol`——这也是为什么
      冻结基准的 `dpsim` 早就到了 \\(10^{-9}\\)，而方程残差还有 \\(4.7\\times10^{-6}\\)。
    """
    nk = _load_nk_inner()
    if pr is None:
        pr = nk.Problem(s, psib_coil, G2, order=order)
    psi_flat = np.asarray(psi, dtype=float).ravel(order="F")
    psib_flat = np.asarray(psib, dtype=float).ravel()
    amp = float(psi_flat.max() - psi_flat.min())
    if amp <= 0:
        return float("inf"), float("inf")
    pr.set_psib(psib_flat)
    dP = pr.dp_per_unit_C(psi_flat)
    gs_res = _gs_residual(pr, psi_flat, dP, C)
    nxt = nk._relax_rounds(pr, psi_flat, psib_flat, "frozen", "frc",
                           float(C), 1.0, 1, w=0.0)
    defect = float(np.max(np.abs(nxt - psi_flat)) / amp)
    return gs_res, defect


def _solve_fast_inner(s, psib_coil, G2, int_lin, psiw_coil, Br_coil, Bz_coil,
                      setup_s, opts):
    """跑快速内层那一版，并把结果整理成与默认路径**同名的字典**。

    场量（`dP`/`P`/`Jzeta`/`Jztotal*`）按求解器一贯的剖面公式从终态 ψ 重算一遍，
    这样 `run(ws)`、`figure_6_1`/`figure_6_2` 拿到的变量名与默认路径完全一致。
    """
    nk = _load_nk_inner()
    pr = opts.get("problem")
    if pr is None:
        pr = nk.Problem(s, psib_coil, G2, order=opts.get("gs_order", 2))
    else:
        # **修改版**（第 17 条）：复用 MachineModel 里缓存的 Problem；只换本步
        # 的线圈边界通量（A/G2/LU 都与电流无关）。
        pr.psib_coil = np.asarray(psib_coil, dtype=float).ravel()
        pr.maxpsiw = float(np.abs(pr.psib_coil).max())
    fast = nk.full_case(pr, 1.0, psi_base=None, ntout=opts["ntout"],
                        CI_tol=opts["CI_tol"], itCI_max=opts["itCI_max"],
                        verbose=opts["verbose"], inner_tol=opts["inner_tol"],
                        out_tol=opts.get("out_tol", 2e-9),
                        dci_max=opts.get("dci_max"),
                        # **修改版**（第 17 条）：时间演化入口
                        psi0=opts.get("psi0"), CI0=opts.get("CI0", 1.0),
                        secant_state=opts.get("secant_state"),
                        step_tol=opts.get("step_tol"),
                        step_ref=opts.get("step_ref"),
                        step_tol_floor=opts.get("step_tol_floor", 1e-5),
                        mid_floor_rel=opts.get("mid_floor_rel", 3e-5),
                        psib0=opts.get("psib0"),
                        mid_accel=opts.get("mid_accel", "none"),
                        mid_depth=opts.get("mid_depth", 3),
                        mid_reg=opts.get("mid_reg", 1e-10),
                        mid_max_ratio=opts.get("mid_max_ratio", 2.0),
                        ci_fixed=opts.get("ci_fixed"),
                        be_loop=bool(opts.get("be_loop", False)))

    # nk_inner 内部用的是**列优先展平**的一维 ψ（与原文 `x(:)` 一致），
    # 这里按同样的顺序塑回二维，之后所有场量就与默认路径同形状。
    nR, nZ, Z0 = s["nR"], s["nZ"], s["Z0"]
    psi = np.asarray(fast["psi"], dtype=float).reshape(nR, nZ, order="F")
    C, CI, Be1 = fast["C"], fast["CI"], fast["Be1"]
    psib = np.asarray(fast["psib"]).ravel()
    Ppar = pr.P
    R = np.asarray(s["R"])
    rr, dR, dZ = s["rr"], s["dR"], s["dZ"]
    Be, Rs, Pm0, betas = Ppar["Be"], Ppar["Rs"], Ppar["Pm0"], Ppar["betas"]
    alpha, sigma, n = Ppar["alpha"], Ppar["sigma"], Ppar["n"]
    u = sigma * psi / (Be * Rs ** 2) + 1
    ex = np.exp(-alpha * psi / (Be * Rs ** 2))
    dP = C * Pm0 * betas * ex * u ** (n - 1) * (
        -alpha / (Be * Rs ** 2) * u + n * sigma / (Be * Rs ** 2))
    P = C * Pm0 * betas * ex * u ** n
    Jzeta = R * dP
    Jztotal = float(Jzeta.sum() * dR * dZ)
    jr_mid = int(np.asarray(s["jrb"][Z0 - 1])) - 1
    Jztotal_mid = float((2 * np.pi * Jzeta[:jr_mid, Z0 - 1]
                         * R[:jr_mid, Z0 - 1] * dR).sum())

    outer = fast["outer"]
    last_ci = [row for row in outer if row[0] == (outer[-1][0] if outer else 1)]
    itout = len(last_ci) or 1
    it = last_ci[-1][3] if last_ci else 0
    dpsimout = last_ci[-1][2] if last_ci else float("nan")
    dpsim = last_ci[-1][4] if last_ci else float("nan")
    itout_hist = [(row[1], row[2], row[3]) for row in last_ci]

    gs_res, defect = _residual_diagnostics(
        s, psib_coil, G2, psi, C, psib, order=opts.get("gs_order", 2), pr=pr)
    print(f"  残差诊断（**修改版**）：GS 方程幅值相对残差 = {gs_res:.3e}；"
          f"松弛映射固定点缺陷 = {defect:.3e}")

    # **修改版**（第 19 条 / Stage 0）：平衡的物理诊断量（私通量、分离面半径、
    # 反场深度、储能、环向电流…）。只做 O(N) 运算，逐步记录下来供演化用。
    diag = nk.frc_diagnostics(s, psi, C, CI=CI, P=Ppar)

    out = dict(psi=psi, psi0=psi, psi1=psi, psi2=psi, psiq=psi,
               Rq=R, Zq=np.asarray(s["Z"]), C=C, C3=-0.0005, CI=CI, CIq=CI,
               Ctmp=C, CItmp=CI, Be1=Be1, Be=Be, Jzeta=Jzeta, P=P,
               dP=dP, psiw_coil=psiw_coil, Br_coil=Br_coil, Bz_coil=Bz_coil,
               psib=psib, psib_coil=psib_coil,
               maxpsiw=float(np.abs(psib_coil).max()),
               Jztotal=Jztotal, Jztotal_mid=Jztotal_mid,
               jrb=s["jrb"], jzb=s["jzb"],
               R=R, Z=np.asarray(s["Z"]), rr=rr, zz=s["zz"], dR=dR, dZ=dZ,
               Z0=Z0, nR=nR, nZ=nZ, nb=s["nb"], tol=1e-9, nt=20000,
               ntout=opts["ntout"], dpsim=dpsim, dpsimout=dpsimout,
               it=it, itout=itout, itCI=outer[-1][0] if outer else 1,
               setup_s=setup_s, itout_hist=itout_hist,
               # **修改版**：三层各自的"还在降 / 进平台 / 卡死"诊断，见修改记录第 16 条
               layer_diag=fast.get("diag", {}),
               CI_hist=np.array(fast["CI_hist"]), g_hist=np.array(fast["g_hist"]),
               nk_wall=fast["wall"], nk_inner_iters=fast["inner_iters"],
               gs_res=gs_res, defect=defect,
               diag=diag,
               # **修改版**（第 17 条）：时间演化要带走的状态
               dpsi_step=fast.get("dpsi_step"),
               inner_tol_eff=fast.get("inner_tol_eff"),
               out_tol_eff=fast.get("out_tol_eff"),
               step_mode=bool(fast.get("step_mode")),
               secant_state=fast.get("secant_state"))
    for name, value in s.items():
        if name not in ("bnd_ii", "bnd_jj"):
            out.setdefault(name, value)
    return out


def solve(src_dir: Path, *, nt=20000, ntout=30, tol=1e-9, CI_clamp=1.3,
          nR=2**7 + 1, nZ=2**8 + 1, itCI_max=200, CI_tol=5e-4, verbose=False,
          inner="hybrid", inner_tol=1e-5, inner_stop="dpsim", res_tol=1e-6,
          gs_order=2, out_tol=2e-9, dci_max=None, model=None, Il=None,
          psi0=None, CI0=1.0, secant_state=None, step_tol=None, step_ref=None,
          step_tol_floor=1e-5, mid_floor_rel=3e-5, psib0=None,
          mid_accel="none", mid_depth=3, mid_reg=1e-10, mid_max_ratio=2.0,
          params=None, ci_fixed=None, be_loop=False):
    """跑一遍自由边界算例，返回工作区变量（名字与 MATLAB 一致）。

    `inner` 选内层求解器（**修改版**新增的开关）：

    * `"hybrid"`（**默认**）——`nk_inner.py` 里的 FreeGSNKE 式配方（约束内嵌 +
      IMEX + 容差线搜索），卡住就回退到 `"relax"`。全量约 15 s，但停在外层容差带里
      的另一个点，与冻结基准差 2.8e-4（相对场幅值）——见 `修改记录.md` 第 7 条；
    * `"relax"`——原文那条逐点松弛 + 1%/轮 C 控制器，能**逐位复现冻结基准**
      （全量约 415 s）。想复现论文那份解、或与冻结基准逐变量对照时用它。

    `inner_tol` 只对 `"hybrid"` 有意义：内层迭代的收敛判据。默认 1e-5 正是这套配方
    自己的判据（`nk_inner.TOL_RANGE`，freegsnke `forward_solve` 的原始口径）。
    取 1e-6 时 CI2 那几步永远达不到，每次都跑满 20000 轮松弛兜底（全量 ~50 s）；
    取 1e-5 全量约 15 s，解只移动 7e-6（相对场幅值）。

    `out_tol` 是**中间那层**（itout：边界通量不动点）的判据，默认 2e-9（原文口径）。
    它比最外层的 CI 判据（5e-4）严了五个数量级——属于过度收敛：实测放宽到 1e-6，
    线性求解次数从 1318 降到 699（约一半），终态只移动 ~1e-4（C/CI 各 5e-5/8e-5，
    离散残差仍在 8e-6 一档）。默认保持 2e-9 不动（现有数值都建立在它上面）。

    `dci_max`（**修改版**新增，默认 None=关）：把**最外层 CI 的单步变化**限制在
    ±dci_max 以内，即"目标不一次加满、分几步爬过去"（TokaMaker 里 `nR0_ramp=6`
    的同一思路）。**实测是负面结果，见修改记录第 15 条，别开这个开关**：
    CI 的第一步确实是 \(-19.6\%\)、紧跟它的那次 itout 首轮确实要 298 步（占整条
    求解的四分之一），但把这一步摊开之后，每多一轮 CI 就要多付 30 轮 itout，
    而且反而有 8 次内层撞到 300 步上限、触发原文松弛兜底——全量 11.1 s → 30.8 s。
    保留它只为复现那次实验；默认 None 时与原来逐位相同。

    `inner_stop` 只对 `"relax"` 有意义（**修改版**新增）：

    * `"dpsim"`（默认）——原程序的口径：每轮 ψ 的变化量小于 `tol` 就停。它**不等于**
      方程解好了：`dpsim` 是无阻尼固定点缺陷的 \\((1-w)=1\\%\\)，所以 `dpsim ≤ 1e-9`
      只保证缺陷 ≤ 1e-7；实测冻结基准的离散方程残差还有 \\(4.7\\times10^{-6}\\)。
      保留它是为了能**逐位复现冻结基准**。
    * `"residual"`——按离散 G-S 方程残差停（`res_tol`，默认 1e-6），也就是 freegsnke
      的判据。每轮多算一次 \\(A\\psi\\)，所以比 `"dpsim"` 慢一些。

    不管用哪个口径，跑完都会打印/返回两个残差诊断（`gs_res`、`defect`）。

    ---- **修改版**（第 17 条）：时间演化入口 ----

    * `model`：`MachineModel`，与电流无关的量（每线圈 Green 基、G2、互感）
      已经预计算好；给了它就不再做 `_coil_green`/`_grz_jz`（省约 0.7 s/次）。
    * `Il`：本步的线圈电流（默认 `s["Il"]`）。线圈几何固定、只有电流变时，
      `psib_coil` 由 `model.coil_flux(Il)` 一次加权和给出。
    * `psi0` / `CI0` / `secant_state`：从上一步**暖启动**（见 `nk_inner.full_case`）。
    * `step_tol`：打开 freegsnke 式「每步相对」判据；`step_ref` 给上一步实测的
      磁通变化。不开（默认 None）时与原来逐位相同。

    ---- **修改版**（第 19 条）：剖面参数可时变 ----

    * `params`：本步的剖面/闭合参数覆盖，只认 `nk_inner.PROFILE_KEYS`
      （`betas`、`alpha`、`sigma`、`n`、`Be`、`Rs`、`Pm0`、`Izeta0`）。
      给了 `Be` 而没给 `Pm0` 时按 \\(P_{m0}=B_e^2/2\\mu_0\\) 自动重算。
      不传（默认 None）时与原来逐位相同。这就是 Stage 1 的入口：线图电流之外，
      把"电流幅值/剖面形状"也变成随时间可给的量。

    ---- **修改版**（第 23 条）：磁链口径 ----

    * `ci_fixed`：钉住外层归一化因子 \(CI\)（一般给 1.0）。给定时边界通量
      \(\psi_b=\psi_{\rm coil,b}+\psi_{\rm pl,b}\) 就是各电流的**物理**磁链之和，
      于是线圈/导体壳的回路方程与平衡求解器共用一个口径；压强平衡量 \(B_{e1}\)
      退化为诊断量（`out["Be1"]`、`out["diag"]["Be1_rel"]`）。
      默认 None ⇒ 原有路径逐位不变。
    """
    if model is None:
        s = _setup(src_dir, nR=nR, nZ=nZ)
    else:
        s = model.s
    s = _with_params(s, params)
    (betas, ls, alpha, sigma, n, Rs, Be, Izeta0, Pm0) = (
        s["betas"], s["ls"], s["alpha"], s["sigma"], s["n"],
        s["Rs"], s["Be"], s["Izeta0"], s["Pm0"])
    rr, dR, dZ, R, Z = s["rr"], s["dR"], s["dZ"], s["R"], s["Z"]
    nR, nZ, Z0 = s["nR"], s["nZ"], s["Z0"]
    bnd_lin = s["bnd_lin"]

    t0 = time.perf_counter()
    if model is None:
        basis = _coil_green_basis(s)
        psiw_coil, Br_coil, Bz_coil, psib_coil = _coil_green(s, basis=basis, Il=Il)
        G2, int_lin = _grz_jz(s)
    else:
        psiw_coil, Br_coil, Bz_coil, psib_coil = model.coil_flux(Il)
        G2, int_lin = model.G2, model.int_lin
    Gsub = G2[int_lin, :]
    setup_s = time.perf_counter() - t0

    maxpsiw = np.abs(psib_coil).max()
    C, C3 = 1.0, -0.0005

    if inner not in ("hybrid", "relax"):
        raise ValueError(f'inner 只能是 "hybrid" 或 "relax"，收到 {inner!r}')
    if inner == "hybrid":
        # ---- 修改版默认路径：nk_inner.py 的快速内层（FreeGSNKE 式配方）----
        # **修改版**（第 17 条）：给了 MachineModel 就复用缓存的 Problem
        # （算子 A / G2 / LU 分解都与线圈电流无关，只换本步的 psib_coil）。
        # **修改版**（第 19 条）：`params` 在这一步落进缓存的 Problem。这里传的是
        # **完整的一份**（不是只有被覆盖的那几个键）——缓存的 Problem 会被后续调用
        # 复用，只传增量的话，"上一步覆盖过、这一步没覆盖"的键会留在里面（状态泄漏）。
        cached_pr = (model.problem(
            psib_coil, gs_order,
            params=_load_nk_inner().params_from_setup(s))
            if model is not None else None)
        return _solve_fast_inner(s, psib_coil, G2, int_lin, psiw_coil, Br_coil,
                                 Bz_coil, setup_s, dict(
                                     ntout=ntout, CI_tol=CI_tol,
                                     itCI_max=itCI_max, verbose=verbose,
                                     inner_tol=inner_tol, gs_order=gs_order,
                                     out_tol=out_tol, dci_max=dci_max,
                                     psi0=psi0, CI0=CI0,
                                     secant_state=secant_state,
                                     step_tol=step_tol, step_ref=step_ref,
                                     step_tol_floor=step_tol_floor,
                                     mid_floor_rel=mid_floor_rel,
                                     problem=cached_pr, psib0=psib0,
                                     mid_accel=mid_accel, mid_depth=mid_depth,
                                     mid_reg=mid_reg,
                                     mid_max_ratio=mid_max_ratio,
                                     ci_fixed=ci_fixed, be_loop=be_loop))

    if ci_fixed is not None or be_loop:
        raise NotImplementedError(
            'ci_fixed / be_loop 只接在默认的 inner="hybrid" 路径上'
            '（relax 参考路径没有接）')

    if psi0 is None:
        psi = 0.01 * (0.1 - np.exp(-(((R / s["rw"] - 0.2) ** 2 / 0.3 ** 2)
                                     + (Z / s["zw"]) ** 2 / 0.4 ** 2))) * maxpsiw
    else:
        psi = np.asarray(psi0, dtype=float).reshape(nR, nZ, order="F")
    CI = float(CI0)
    CItmp = CI
    Ctmp = C
    psi0 = psi.copy()
    psi1 = psi.copy()
    psi2 = psi.copy()
    dpsim = np.abs(psi - psi0).max() + 0.1
    it, ita, itout = 1, 0, 1
    psiout = psi.copy()
    dpsimout = 100 * tol
    Be1 = 1.1 * Be
    itCI = 1
    wx = 0.95
    CI_hist: list[float] = []
    g_hist: list[float] = []
    w = 0.99
    rc = rr[1:nR - 1][:, None]                     # 列向量，广播到 (nR-2, nZ-2)
    rc2 = rc ** 2
    # ---- Python 侧优化：把循环里每轮都要重算的常量提到外面（都是同一个值，
    #      运算顺序不变，结果逐位相同）----
    coef_r = rc / dR**2                            # rc/dR2
    rc_p = rc + 0.5 * dR                           # rc+0.5*dR
    rc_m = rc - 0.5 * dR                           # rc-0.5*dR
    MU0_rc2 = MU0 * rc2                            # mu0*rc.^2
    half_dZ2 = 0.5 * dZ**2                         # 0.5*dZ2
    one_minus_w = 1 - w
    m_alpha_over = -alpha / (Be * Rs**2)           # 原文：-alpha/(Be*Rs^2)
    n_sig_over = n * sigma / (Be * Rs**2)          # 原文：n*sigma/(Be*Rs^2)
    # 预分配临时量（形状与内部点切片一致），避免每轮 malloc
    buf1 = np.empty((nR - 2, nZ - 2))
    buf2 = np.empty((nR - 2, nZ - 2))
    buf3 = np.empty((nR - 2, nZ - 2))
    buf_full = np.empty((nR, nZ))                  # 算 dpsim 用（全网格）
    buf_delta = np.empty((nR, nZ))                 # **修改版**：有符号的 Δψ（残差要用）
    buf_res = np.empty((nR - 2, nZ - 2))           # 残差场 F（内部点）
    jr_mid = int(s["jrb"][Z0 - 1]) - 1             # 原文 jrb(Z0)-1（1 基计数）
    Jzeta = np.zeros_like(R)
    # **修改版**：把循环里才会赋值的量先占个位——这样 `itCI_max=1` 这类
    # "不进外层循环"的截短配置不会在收尾（残差诊断）处踩未初始化变量。
    P = np.zeros_like(R)
    dP = np.zeros_like(R)
    Jztotal = 0.0
    Jztotal_mid = 0.0
    psib = np.zeros(int(s["nb"]))

    itout_hist: list[tuple[int, float, float]] = []
    res_stop = (inner_stop == "residual")
    if inner_stop not in ("dpsim", "residual"):
        raise ValueError(f'inner_stop 只能是 "dpsim" 或 "residual"，收到 {inner_stop!r}')

    while abs(Be1 / Be - 1) > CI_tol and itCI < itCI_max:
        itout, it = 1, 1
        while itout <= ntout + ita and dpsimout > tol * 2:
            # ---- 由当前 psi 反算 Jzeta（原文是逐点双循环，这里等价向量化）----
            Jzeta[:, :] = 0.0
            Jzeta[1:nR - 1, 1:nZ - 1] = (
                -1.0 / R[1:nR - 1, 1:nZ - 1] * (
                    (psi[2:nR, 1:nZ - 1] - 2 * psi[1:nR - 1, 1:nZ - 1]
                     + psi[:nR - 2, 1:nZ - 1]) / dR**2
                    - 1.0 / R[1:nR - 1, 1:nZ - 1]
                    * (psi[2:nR, 1:nZ - 1] - psi[:nR - 2, 1:nZ - 1]) / (2 * dR)
                    + (psi[1:nR - 1, 2:nZ] - 2 * psi[1:nR - 1, 1:nZ - 1]
                       + psi[1:nR - 1, :nZ - 2]) / dZ**2))
            Jzeta = Jzeta / MU0
            Jztotal = Jzeta.sum() * dR * dZ
            # ---- [C] 边界通量更新：一次矩阵乘 ----
            psib = CI * psib_coil + (MU0 * CI * dR * dZ) * (
                Gsub.T @ Jzeta.ravel(order="F")[int_lin])

            dpsim = 10 * tol
            it = 1
            gs_res_it = float("inf")
            while it <= nt and (gs_res_it > res_tol if res_stop else dpsim > tol):
                # ---- [M] dP / P 共享 exp 与 (...+1) ----
                u = sigma * psi / (Be * Rs**2) + 1
                ex = np.exp(-alpha * psi / (Be * Rs**2))
                # 优化：n=2 时 u**(n-1) 就是 u 本身（pow(x,1) 必须返回 x），
                # 与原文 u.^(n-1) 逐位相同；两个标量系数提前算好，其余顺序不变。
                dP = C * Pm0 * betas * ex * u * (m_alpha_over * u + n_sig_over)
                Jzeta = R * dP
                Jztotal = Jzeta.sum() * dR * dZ
                Jztotal_mid = float((2 * np.pi * Jzeta[:jr_mid, Z0 - 1]
                                     * R[:jr_mid, Z0 - 1] * dR).sum())
                P = C * Pm0 * betas * ex * u ** n
                if itout >= 2:                     # [K] 原文：if(itout>1)
                    C = 0.99 * C + 0.01 * Ctmp * Izeta0 / Jztotal_mid
                # ---- 内层逐点松弛（向量化，与原文逐元素相同）----
                # 优化：用预分配缓冲分步算（同样的运算、同样的顺序，只是不新建数组）
                np.subtract(psi0[2:nR, 1:nZ - 1], psi0[1:nR - 1, 1:nZ - 1], out=buf1)
                np.divide(buf1, rc_p, out=buf1)
                np.subtract(psi0[1:nR - 1, 1:nZ - 1], psi0[:nR - 2, 1:nZ - 1], out=buf2)
                np.divide(buf2, rc_m, out=buf2)
                np.subtract(buf1, buf2, out=buf1)
                np.multiply(coef_r, buf1, out=buf1)
                np.add(psi0[1:nR - 1, 2:nZ], psi0[1:nR - 1, :nZ - 2], out=buf2)
                np.divide(buf2, dZ**2, out=buf2)
                np.add(buf1, buf2, out=buf1)
                np.multiply(MU0_rc2, dP[1:nR - 1, 1:nZ - 1], out=buf2)
                np.add(buf1, buf2, out=buf1)
                np.multiply(half_dZ2, buf1, out=buf1)
                np.multiply(buf1, one_minus_w, out=buf1)
                np.multiply(w, psi0[1:nR - 1, 1:nZ - 1], out=buf2)
                np.add(buf2, buf1, out=buf3)
                psi[1:nR - 1, 1:nZ - 1] = buf3
                psi[0, :] = 0.0
                psi[s["bnd_ii"], s["bnd_jj"]] = psib      # 原文：psi(jrb,jzb)=psib
                it += 1
                if res_stop:
                    # 残差模式：顺手留下**有符号**的 Δψ（默认模式不做这一步，
                    # 免得白多一遍全网格拷贝）
                    np.subtract(psi, psi0, out=buf_delta)
                    np.abs(buf_delta, out=buf_full)
                    dpsim = float(buf_full.max())
                    # **修改版**：按残差停，但**不再每轮做矩阵乘**。
                    #
                    # 恒等式（对这套松弛映射严格成立，实测相对差 5.5e-16）：
                    #     F = Aψ − rhs = 2(ψ0 − ψ̃) = −2·Δψ/(1−w)
                    # 其中 ψ̃ = (ψ_new − w·ψ0)/(1−w) 就是这一轮松弛本来算出来的 Jacobi 值，
                    # Δψ = ψ_new − ψ0 也已经算好了（dpsim 用的就是它）。
                    # 原来这里每轮要 psi/dP 两次 F 序 ravel 拷贝 + 一次稀疏矩阵乘。
                    np.multiply(buf_delta[1:nR - 1, 1:nZ - 1],
                                -2.0 / one_minus_w, out=buf_res)
                    psi_amp = float(psi.max() - psi.min())
                    gs_res_it = ((float(buf_res.max()) - float(buf_res.min()))
                                 / psi_amp) if psi_amp > 0 else float("inf")
                else:
                    np.subtract(psi, psi0, out=buf_full)
                    np.abs(buf_full, out=buf_full)
                    dpsim = float(buf_full.max())
                psi2 = psi1.copy()
                psi1 = psi0.copy()
                psi0 = psi.copy()
                Ctmp = C
            dpsimout = np.abs(psi - psiout).max()
            psiout = psi.copy()
            if verbose:
                print(f"  itout={itout:3d}  dpsimout={dpsimout:.3e}  内层 {it - 1} 轮")
            itout_hist.append((itout, float(dpsimout), it - 1))
            itout += 1
        Be1 = np.sqrt(2 * MU0 * P[:, Z0 - 1].max())
        # ---- [F] CI 归一化：带保护的 secant ----
        g_now = Be1 - Be
        CI_new = CI * Be / Be1
        if CI_hist:
            CI_a, g_a = CI_hist[-1], g_hist[-1]
            if g_now != g_a:
                CI_try = CI - g_now * (CI - CI_a) / (g_now - g_a)
                if np.isfinite(CI_try) and CI_try > 0:
                    CI_new = CI_try
        dCI = CI_new / CI
        if not np.isfinite(dCI) or dCI <= 0:
            dCI = Be / Be1
        dCI = min(max(dCI, 1 / CI_clamp), CI_clamp)
        CI_hist.append(CI)
        g_hist.append(g_now)
        CI = CI * dCI
        itCI += 1
        CItmp = CI
        if verbose:
            print(f"[CI {itCI:3d}] CI={CI:.8f} Be1={Be1:.8f} g={g_now:+.4e} "
                  f"dCI={dCI:.4f}")

    # `Rq`/`Zq` 是 MATLAB 收尾时 `Rq=R;Zq=Z;` 留下的（快照里有）
    gs_res, defect = _residual_diagnostics(s, psib_coil, G2, psi, C, psib)
    print(f"  残差诊断（**修改版**）：GS 方程幅值相对残差 = {gs_res:.3e}；"
          f"松弛映射固定点缺陷 = {defect:.3e}")
    # **修改版**（第 19 条）：原文这里用的是 U+2212 的减号，GBK 控制台（cp936）
    # 打印时会 UnicodeEncodeError 把整条 `inner="relax"` 路径打断，换成 ASCII 的
    # `-`。只动打印文本，数值一字未改。
    print(f"  （原程序只看 dpsim={float(dpsim):.2e}，而 A·ψ-rhs 的残差是 "
          f"{gs_res:.2e}——dpsim 小不等于方程解好了，判收敛要看残差）")

    # **修改版**（第 19 条 / Stage 0）：同 `_solve_fast_inner`，松弛路径也给一份
    # 物理诊断量（两个内层返回的观测量一致，便于对照）。
    diag = _load_nk_inner().frc_diagnostics(
        s, psi, C, CI=CI, P=_load_nk_inner().params_from_setup(s))

    out = dict(psi=psi, psi0=psi0, psi1=psi1, psi2=psi2, psiq=psi,
               Rq=R, Zq=Z, C=C, C3=C3, CI=CI, CIq=CI,
               Ctmp=Ctmp, CItmp=CItmp, Be1=Be1, Be=Be, Jzeta=Jzeta, P=P,
               dP=dP, psiw_coil=psiw_coil, Br_coil=Br_coil, Bz_coil=Bz_coil,
               psib=psib, psib_coil=psib_coil, maxpsiw=maxpsiw,
               Jztotal=Jztotal, Jztotal_mid=Jztotal_mid,
               jrb=s["jrb"], jzb=s["jzb"],
               R=R, Z=Z, rr=rr, zz=s["zz"], dR=dR, dZ=dZ, Z0=Z0, nR=nR, nZ=nZ,
               nb=s["nb"], tol=tol, nt=nt, ntout=ntout, dpsim=dpsim,
               dpsimout=dpsimout, it=it, itout=itout, itCI=itCI,
               setup_s=setup_s, itout_hist=itout_hist,
               CI_hist=np.array(CI_hist), g_hist=np.array(g_hist),
               gs_res=gs_res, defect=defect, diag=diag)
    # 主树并线（2026-09-17）：把"输入/配置/记账"那一类也交出来，让
    # `gseqfrc.golden` 按物理量契约逐变量比对。**Python 侧自己的命名**
    # （`bnd_ii`、`bnd_jj`）不进工作区——MATLAB 那边没有这些名字。
    for name, value in s.items():
        if name not in ("bnd_ii", "bnd_jj"):
            out.setdefault(name, value)
    return out


#: 本文件所在的目录就是算例目录（`MRR1d.mat` 与线圈数据都在这里）
SRC_DIR = Path(__file__).resolve().parent

#: MATLAB 工作区里这些量是**列**向量（`jrb=zeros(nb,1)` 那一路 / 线圈场是 `(nb,1)`）；
#: 加速实现内部为了算法方便用的是一维数组，这里换算回去。
_COLUMN_VARS = ("jrb", "jzb", "Rbnd", "Zbnd", "psib", "psiw_coil",
                "psib_coil", "Br_coil", "Bz_coil")
#: 同理，这些是**行**向量（`linspace` 出来的网格、从 `MRR1d.mat` 读出的剖面）
_ROW_VARS = ("rr", "zz", "hl", "rl", "zl", "wl", "Il", "j1d", "pr1d", "psi1d",
             "rr1d", "Bz1d", "CI_hist", "g_hist")


def _as_matlab(name: str, value):
    """把加速实现的一维数组换算成 MATLAB 那边的形状。"""
    from gseqfrc import matlab_compat as mat

    if name in _COLUMN_VARS:
        return np.asarray(value).reshape(-1, 1)
    if name in _ROW_VARS:
        return np.asarray(value).reshape(1, -1)
    array = np.asarray(value)
    if array.ndim == 0:
        return mat.scalar(array)
    return array if array.ndim >= 2 else value


def run(ws, *, verbose: bool = False, **kwargs) -> None:
    """树里的求解脚本入口：跑完把变量写进工作区，并保存脚本收尾那两份 `.mat`。

    对应原文 `GSEQ_coil_MRR.m` 结尾的
    `save('psiq.mat',...)` / `save('psiq1.mat',...)`（变量表一字不差；
    绘图段不翻译，所以 `psiq1.mat` 里少一个由 MATLAB `contour` 派生的 `E`）。
    """
    from gseqfrc import mio

    out = solve(SRC_DIR, verbose=verbose, **kwargs)
    for name, value in out.items():
        # **修改版**：`layer_diag` 是分层停止判据的诊断字典（不是 MATLAB 工作区
        # 里的量），不进工作区——`_as_matlab` 只处理数值/数组。
        # 第 17 条新增的时间演化量同理：它们不属于原文 `GSEQ_coil_MRR.m` 的工作区。
        if isinstance(value, dict) or name in (
                "layer_diag", "dpsi_step", "inner_tol_eff", "out_tol_eff",
                "step_mode", "secant_state"):
            continue
        ws[name] = _as_matlab(name, value)

    mio.save("psiq.mat", Rq=ws.Rq, Zq=ws.Zq, psiq=ws.psiq, C=ws.C, C3=ws.C3,
             CIq=ws.CIq)
    mio.save("psiq1.mat", Rq=ws.Rq, Zq=ws.Zq, psiq=ws.psiq, Jzeta=ws.Jzeta,
             P=ws.P, jrb=ws.jrb, Z0=ws.Z0, zl=ws.zl, wl=ws.wl, rl=ws.rl,
             hl=ws.hl, C=ws.C, Bz_coil=ws.Bz_coil, psiw_coil=ws.psiw_coil,
             CI=ws.CI)


def _params_at(params_traj, k: int, nt: int):
    """**修改版**（第 19 条 / Stage 1）：取第 `k` 步的剖面参数覆盖。

    两种写法（见 `evolve` 的 `params_traj`）：dict（名 → 序列/标量）或长度 nt 的
    dict 列表。返回 `None` 表示这一步不覆盖（用模型里的当前值）。
    """
    if params_traj is None:
        return None
    if isinstance(params_traj, dict):
        out: dict[str, float] = {}
        for name, value in params_traj.items():
            arr = np.asarray(value, dtype=float)
            if arr.ndim == 0:
                out[name] = float(arr)
            else:
                flat = arr.reshape(-1)
                if flat.size != nt:
                    raise ValueError(
                        f"params_traj['{name}'] 的长度 {flat.size} != 步数 {nt}")
                out[name] = float(flat[k])
        return out or None
    seq = list(params_traj)
    if len(seq) != nt:
        raise ValueError(f"params_traj 列表长度 {len(seq)} != 步数 {nt}")
    item = seq[k]
    if item is None:
        return None
    if isinstance(item, dict):
        return dict(item) or None
    raise TypeError("params_traj 的元素必须是 dict 或 None")


def evolve(Il_traj, *, times=None, model=None, src_dir=SRC_DIR, step_tol=5e-3,
           initial=None, inner_tol=1e-5, ntout=30, CI_tol=5e-4, itCI_max=200,
           verbose=False, progress=True, keep_psi=True, machine_kwargs=None,
           carry_secant=False, params_traj=None, **solve_kwargs):
    """**修改版**（第 17 条，步骤 1–3）：在给定的线圈电流波形下做时间演化。

    这是 freegsnke 同构的「线圈电流驱动」准静态演化：每个时刻由给定的线圈电流
    \(I_l(t)\) 解一个自由边界平衡；`MachineModel` 把与电流无关的量预计算一次
    （每线圈 Green 基、G2、互感），每步只做一次加权和；上一步的
    \(\psi\)、\(CI\)、割线点作为下一步的暖启动；收敛判据用 freegsnke 式
    「残差 < step_tol × 本步磁通变化」（`step_tol=None` 则退回原紧容差）。

    参数
    ----
    Il_traj : (nt, nl) 数组（或长度 nl 的单步）——每一步的线圈电流（A）。
    times   : 可选，(nt,) 时间轴，仅用于记录。
    initial : 可选，第一步的初始状态（`solve()` 的返回或本函数 history 的一项，
              需含 `psiq`/`CI`/`secant_state`/`dpsi_step`）。不给就先冷启动解一次。
    keep_psi: 是否在 history 里保留每一步的 2D ψ（关闭可省内存）。
    carry_secant: 是否把上一时刻的外层割线斜率带进下一步。默认 **False**：实测
              线圈电流一变，上一时刻量到的割线斜率就失效，带着它反而让外层多跑
              1–2 轮（3 轮 → 4–5 轮）；接口保留、可按需打开。
    params_traj: **修改版**（第 19 条，Stage 1）剖面/闭合参数的时间轨迹。两种写法：

               * dict：参数名 → 长度 nt 的序列（或标量，表示整条轨迹都用它）。
                 可用的键见 `nk_inner.PROFILE_KEYS`，最常用的是 `Izeta0`
                 （电流幅值，由求解器的 C 约束嵌入）、`betas`、`Be`、`Rs`…
               * 长度 nt 的 dict 列表：每步一组覆盖（`None`/`{}` 表示该步不覆盖）。

               不给（默认 None）时与第 17 条逐位相同。

    返回 `(history, model)`；`history` 每项含 `psiq/psib/C/CI/gs_res/dpsi_step/…`
    以及 **`diag`**（`nk_inner.frc_diagnostics` 的物理诊断量：私通量、分离面半径、
    反场深度、伸长比、储能、环向电流…）。`model` 可复用于别的波形，也可存盘。
    """
    Il_arr = np.asarray(Il_traj, dtype=float)
    if Il_arr.ndim == 1:
        if model is not None:
            nl = model.nl
        else:
            nl = np.asarray(_setup(src_dir)["Il"]).size
        if Il_arr.size != nl:
            raise ValueError(f"Il_traj 是 1 维且长度 {Il_arr.size} != 线圈数 {nl}；"
                             "多步波形请传 (nt, nl) 的二维数组")
        Il_arr = Il_arr.reshape(1, -1)
    if Il_arr.ndim != 2:
        raise ValueError("Il_traj 必须是 (nt, nl) 的二维数组")

    if model is None:
        s = _setup(src_dir)
        model = MachineModel(s, **(machine_kwargs or {}))
    nt, nl = Il_arr.shape
    if nl != model.nl:
        raise ValueError(f"Il_traj 的列数 {nl} != 机器模型的线圈数 {model.nl}")
    if progress:
        print(f"[evolve] {nt} 步；机器模型预计算 {model.build_s:.3f} s"
              + ("（含线圈互感）" if model.Mcoil is not None else ""))

    history: list[dict] = []
    prev = initial
    for k in range(nt):
        Il = Il_arr[k]
        params = _params_at(params_traj, k, nt)
        kwargs = dict(model=model, Il=Il, inner="hybrid", inner_tol=inner_tol,
                      ntout=ntout, CI_tol=CI_tol, itCI_max=itCI_max,
                      verbose=verbose, params=params)
        kwargs.update(solve_kwargs)
        t0 = time.perf_counter()
        if prev is None:
            # 冷启动（或用户明确要求）：先用原来的紧容差解一个基准初态。
            out = solve(src_dir, **kwargs)
        else:
            out = solve(src_dir, psi0=prev["psiq"], CI0=prev["CI"],
                        psib0=prev.get("psib"),
                        secant_state=(prev.get("secant_state")
                                      if carry_secant else None),
                        step_tol=step_tol, step_ref=prev.get("dpsi_step"),
                        **kwargs)
        wall = time.perf_counter() - t0
        rec = dict(
            i=k, t=(float(times[k]) if times is not None else float(k)), Il=Il.copy(),
            psiq=(np.asarray(out["psiq"]).copy() if keep_psi else None),
            psib=np.asarray(out["psib"], dtype=float).copy(),
            C=float(out["C"]), CI=float(out["CI"]), Be1=float(out["Be1"]),
            gs_res=float(out["gs_res"]), dpsi_step=out.get("dpsi_step"),
            inner_tol_eff=out.get("inner_tol_eff"),
            out_tol_eff=out.get("out_tol_eff"),
            steps=int(np.sum(out["nk_inner_iters"])),
            calls=int(len(out["nk_inner_iters"])),
            nk_wall=float(out["nk_wall"]), wall=wall,
            secant_state=out.get("secant_state"),
            diag=out.get("diag"), params=(dict(params) if params else None))
        history.append(rec)
        prev = rec
        if progress:
            dg = rec.get("diag") or {}
            extra = (f"  Φ={dg.get('Phi_private_Wb', float('nan')):.4e} Wb"
                     f" Rs={dg.get('Rs_mid', float('nan'))*1e2:.3f} cm"
                     f" Bz0={dg.get('Bz_axis_mid', float('nan')):+.4f} T"
                     if dg else "")
            print(f"  [{k+1}/{nt}] Il={np.array2string(Il, precision=3)}  "
                  f"C={rec['C']:.7f} CI={rec['CI']:.7f} res={rec['gs_res']:.2e} "
                  f"步数={rec['steps']:4d} 内层调用={rec['calls']:3d} "
                  f"求解 {rec['nk_wall']:.2f}s 总 {wall:.2f}s{extra}")
    return history, model


if __name__ == "__main__":
    import sys
    import time

    src = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC_DIR
    t = time.perf_counter()
    out = solve(Path(src), verbose="-v" in sys.argv)
    dt = time.perf_counter() - t
    print(f"准备 {out['setup_s']:.1f} s，总计 {dt:.1f} s；"
          f"CI={out['CI']:.6f} Be1={out['Be1']:.6f} "
          f"itCI={out['itCI']} itout={out['itout']} dpsimout={out['dpsimout']:.3e}")
    savemat("GSEQ_coil_MRR_last.mat",
            {k: v for k, v in out.items() if k != "itout_hist"}, do_compression=True)
