"""fun_Bfield.m 的逐行移植（**两份副本合并后**的唯一实现）。

源文件（两份差 2 行）：

    gseq-main/GSEQ-FRC-open/second_paper/Fig4/gui/fun_Bfield.m
    gseq-main/GSEQ-FRC-open/second_paper/Fig5/gui/fun_Bfield.m

**两份的差别就是 `psi` 累加里那个 `/2`**（`Br`、`Bz` 逐字相同）：Fig4 那份多一个
`/2`，与同一个文件算出来的 `Br`、`Bz` 自相矛盾；`GSEQ_coil_MRR.m` 用的是 Fig5 那份
的归一化，所以 Fig5 副本内部自洽、Fig4 不是。冻结基准把两份分别记下来了
（`u_Bfield_fig4` / `u_Bfield_fig5`）。

2026-09-17 副本合并（PORTING.md §8 的规则 2：**不是二选一，而是把差异提成显式
参数**）：两份的差异在这里就是一个显式参数 `psi_div`——Fig5 副本取 1.0（默认），
Fig4 副本取 2.0（原样保留那个 `/2`）。程序树里两个原位置各留一个薄壳带上
自己的取值。除以 1.0 是精确运算，所以两条线的数值与合并前逐位相同。

三个方法（`imethod`）：

1. 单根灯丝的 Green 函数（线圈中心）；
2. 把线圈截面离散成 `nr×nz` 个点求和，注意 `rr=rmin:dr:rmax` 用的是 MATLAB 的
   浮点冒号（见 `matlab_compat.colon`）；
3. Yuan Baoshan 书 p110 的解析级数（Halton 型序列，N=57091 项）。

`psi`/`Br`/`Bz` 在 MATLAB 里始终是 1×1，所以这里从 `mat.scalar(0.0)` 起步，
让整个累加过程保持 1×1。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mpath


def fun_Bfield(r0, z0, rl, zl, hl, wl, Il, Nl, imethod, psi_div=1.0):
    """`psi_div`：`psi` 累加里除的数。Fig5/gui 副本 = 1.0，Fig4/gui 副本 = 2.0。"""
    """[psi,Br,Bz]=fun_Bfield(r0,z0,rl,zl,hl,wl,Il,Nl,imethod)"""
    mu0 = 2.0e-7  # =mu0/(2*pi)
    psi = mat.scalar(0.0)
    Br = mat.scalar(0.0)
    Bz = mat.scalar(0.0)
    nl = mat.length(rl)

    r0 = r0 + 1e-8
    if imethod == 1:  # Green function
        for jl in mat.mrange(1, nl):  # to update
            rl_jl = mat.el(rl, jl)
            zl_jl = mat.el(zl, jl)
            Nl_jl = mat.el(Nl, jl)
            Il_jl = mat.el(Il, jl)
            k2 = 4 * r0 * rl_jl / ((r0 + rl_jl) ** 2 + (z0 - zl_jl) ** 2)
            EK2, EE2 = mpath.call("fun_ellipEK", k2, 1)
            psi = psi + mu0 * Nl_jl * Il_jl / (np.sqrt(k2)) * np.sqrt(
                r0 * rl_jl
            ) * ((2 - k2) * EK2 - 2 * EE2) / psi_div
            Br = Br + mu0 * Nl_jl * Il_jl * (z0 - zl_jl) / r0 / np.sqrt(
                (r0 + rl_jl) ** 2 + (z0 - zl_jl) ** 2
            ) * (
                (r0**2 + rl_jl**2 + (z0 - zl_jl) ** 2)
                / ((r0 - rl_jl) ** 2 + (z0 - zl_jl) ** 2)
                * EE2
                - EK2
            )
            Bz = Bz + mu0 * Nl_jl * Il_jl / np.sqrt(
                (r0 + rl_jl) ** 2 + (z0 - zl_jl) ** 2
            ) * (
                (rl_jl**2 - r0**2 - (z0 - zl_jl) ** 2)
                / ((r0 - rl_jl) ** 2 + (z0 - zl_jl) ** 2)
                * EE2
                + EK2
            )

    elif imethod == 2:

        for jl in mat.mrange(1, nl):  # to update
            rl_jl = mat.el(rl, jl)
            zl_jl = mat.el(zl, jl)
            hl_jl = mat.el(hl, jl)
            wl_jl = mat.el(wl, jl)

            rmin = rl_jl - 0.5 * hl_jl
            rmax = rl_jl + 0.5 * hl_jl
            zmin = zl_jl - 0.5 * wl_jl
            zmax = zl_jl + 0.5 * wl_jl

            nr = 2**4 + 1
            nz = 2**8 + 1
            dr = (rmax - rmin) / (nr - 1)
            dz = (zmax - zmin) / (nz - 1)
            rr = mat.colon(rmin, dr, rmax)
            zz = mat.colon(zmin, dz, zmax)
            I1 = mat.el(Nl, jl) * mat.el(Il, jl) / (nr * nz)
            for jr in mat.mrange(1, nr):
                for jz in mat.mrange(1, nz):
                    r1 = mat.el(rr, jr)
                    z1 = mat.el(zz, jz)

                    k2 = 4 * r0 * r1 / ((r0 + r1) ** 2 + (z0 - z1) ** 2)
                    EK2, EE2 = mpath.call("fun_ellipEK", k2, 1)
                    psi = psi + mu0 * I1 / (np.sqrt(k2)) * np.sqrt(
                        r0 * r1
                    ) * ((2 - k2) * EK2 - 2 * EE2) / psi_div
                    Br = Br + mu0 * I1 * (z0 - z1) / r0 / np.sqrt(
                        (r0 + r1) ** 2 + (z0 - z1) ** 2
                    ) * (
                        (r0**2 + r1**2 + (z0 - z1) ** 2)
                        / ((r0 - r1) ** 2 + (z0 - z1) ** 2)
                        * EE2
                        - EK2
                    )
                    Bz = Bz + mu0 * I1 / np.sqrt(
                        (r0 + r1) ** 2 + (z0 - z1) ** 2
                    ) * (
                        (r1**2 - r0**2 - (z0 - z1) ** 2)
                        / ((r0 - r1) ** 2 + (z0 - z1) ** 2)
                        * EE2
                        + EK2
                    )

    else:  # Yuan Baoshan's book p110
        R1 = r0
        Z1 = z0
        for jl in mat.mrange(1, nl):  # to update

            A1 = mat.el(rl, jl)
            B1 = mat.el(zl, jl)
            C1 = mat.el(hl, jl) * 0.5
            D1 = mat.el(wl, jl) * 0.5
            Nl_jl = mat.el(Nl, jl)
            Il_jl = mat.el(Il, jl)

            A = mat.row(1, 48188, 21101)
            C = mat.row(A1 - C1, B1 - D1, 0.0)
            D = mat.row(A1 + C1, B1 + D1, 0.5 * np.pi)
            N = 57091
            X = 0.0 * C

            for M in mat.mrange(1, N):
                G = 1.0
                for L in mat.mrange(1, 3):
                    Z = M * mat.el(A, L) / N
                    U = Z - np.floor(Z)
                    V = (mat.el(C, L) - mat.el(D, L)) * U
                    W = (U - 1.0) * V
                    mat.set_el(X, L, mat.el(C, L) + (2.0 * W - V) * U)
                    G = 6.0 * W * G
                E1 = 4.0 * R1 * mat.el(X, 1) / (
                    (R1 + mat.el(X, 1)) ** 2 + (Z1 - mat.el(X, 2)) ** 2
                )
                # E2=sqrt(1.0-E1*sin(X(3)^2)); % wrong, 19-06-17 13:18
                E2 = np.sqrt(1.0 - E1 * np.sin(mat.el(X, 3)) ** 2)
                E3 = (Z1 - mat.el(X, 2)) / (R1 + 1e-9) / np.sqrt(
                    (R1 + mat.el(X, 1)) ** 2 + (Z1 - mat.el(X, 2)) ** 2
                )
                E4 = (R1**2 + mat.el(X, 1) ** 2 + (Z1 - mat.el(X, 2)) ** 2) / (
                    (R1 - mat.el(X, 1)) ** 2 + (Z1 - mat.el(X, 2)) ** 2
                )
                E5 = 2e-7 * Nl_jl * Il_jl / (4 * C1 * D1)
                F1 = E5 * E3 * (-1.0 / E2 + E4 * E2)
                Br = Br + F1 * G / N
                E6 = 1.0 / np.sqrt(
                    (R1 + mat.el(X, 1)) ** 2 + (Z1 - mat.el(X, 2)) ** 2
                )
                E7 = (R1**2 - mat.el(X, 1) ** 2 + (Z1 - mat.el(X, 2)) ** 2) / (
                    (R1 - mat.el(X, 1)) ** 2 + (Z1 - mat.el(X, 2)) ** 2
                )
                F2 = E5 * E6 * (1.0 / E2 - E7 * E2)
                Bz = Bz + F2 * G / N

                # psi add by Huasheng Xie, 19-06-17 12:25
                E8 = np.sqrt(
                    (R1 + mat.el(X, 1)) ** 2 + (Z1 - mat.el(X, 2)) ** 2
                )
                E9 = 1.0 - 0.5 * E1
                F3 = E5 * E8 * (E9 / E2 - E2)
                psi = psi + F3 * G / N

    return psi, Br, Bz
