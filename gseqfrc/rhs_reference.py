"""二维求解器里 `rhs` 那一段的**逐点参考实现**。

源码位置：`second_paper/Fig5/GSEQ_FRC_fixed_FIX.m` 第 151–239 行、以及
`first_paper/figure8/gseq.m` 里同一段（分五种差分格式 `ifd=1..4`，其中
`ifd=4` 按边界位置分成 9 个子分支）。

两处求解器在 `ifd==2`（本算例走的那个分支）上用的是向量化写法；其余分支仍然回到
这里——这份实现刻意保持标量、逐点、1 基下标的写法，与 `.m` 里那串分支一字对应，
所以能一行一行对着原文看。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat


def rhs_point(ifd, i, j, psi0, dP, rr, jrb, nZ, dR, dZ, dR2, dZ2, mu0):
    """算 `rhs(i,j)`——与源码里那一串分支一一对应。"""
    p = lambda di, dj: mat.ij(psi0, i + di, j + dj)  # noqa: E731
    r_i = mat.el(rr, i)
    dPij = mat.ij(dP, i, j)
    src = mu0 * r_i ** 2 * dPij

    if ifd == 1:  # previous version
        return 0.5 * dZ2 * (
            r_i / dR2 * (
                (p(1, 0) - p(0, 0)) / (r_i + 0.5 * dR)
                - (p(0, 0) - p(-1, 0)) / (r_i - 0.5 * dR)
            )
            + (p(0, 1) + p(0, -1)) / dZ2
            + src
        )

    if ifd == 2:  # 3-point
        return 1 / (2 / dR2 + 2 / dZ2) * (
            (p(1, 0) + p(-1, 0)) / dR2
            - (p(1, 0) - p(-1, 0)) / (2 * dR * r_i)
            + (p(0, 1) + p(0, -1)) / dZ2
            + src
        )

    if ifd in (3, 4):
        interior = (j > 2 and j < nZ - 1) and (i > 2 and i < mat.el(jrb, j) - 1)
        if ifd == 3:
            if not interior:  # 退化成三点
                return 1 / (2 / dR2 + 2 / dZ2) * (
                    (p(1, 0) + p(-1, 0)) / dR2
                    - (p(1, 0) - p(-1, 0)) / (2 * dR * r_i)
                    + (p(0, 1) + p(0, -1)) / dZ2
                    + src
                )
            return _five_point(p, dR2, dZ2, r_i, dR, src)

        # ifd == 4：9 个子分支
        if interior:
            return _five_point(p, dR2, dZ2, r_i, dR, src)
        if (j > 2 and j < nZ - 1) and i == 2:
            return 1 / (20 / (12 * dR2) - 10 / (12 * dR * r_i) + 30 / (12 * dZ2)) * (
                (-p(3, 0) + 4 * p(2, 0) + 6 * p(1, 0) + 11 * p(-1, 0)) / (12 * dR2)
                - (p(3, 0) - 6 * p(2, 0) + 18 * p(1, 0) - 3 * p(-1, 0)) / (12 * dR * r_i)
                + (-p(0, 2) + 16 * p(0, 1) + 16 * p(0, -1) - p(0, -2)) / (12 * dZ2)
                + src
            )
        if j == 2 and i == 2:
            return 1 / (20 / (12 * dR2) - 10 / (12 * dR * r_i) + 20 / (12 * dZ2)) * (
                (-p(3, 0) + 4 * p(2, 0) + 6 * p(1, 0) + 11 * p(-1, 0)) / (12 * dR2)
                - (p(3, 0) - 6 * p(2, 0) + 18 * p(1, 0) - 3 * p(-1, 0)) / (12 * dR * r_i)
                + (-p(0, 3) + 4 * p(0, 2) + 6 * p(0, 1) + 11 * p(0, -1)) / (12 * dZ2)
                + src
            )
        if j == nZ - 1 and i == 2:
            return 1 / (20 / (12 * dR2) - 10 / (12 * dR * r_i) + 20 / (12 * dZ2)) * (
                (-p(3, 0) + 4 * p(2, 0) + 6 * p(1, 0) + 11 * p(-1, 0)) / (12 * dR2)
                - (p(3, 0) - 6 * p(2, 0) + 18 * p(1, 0) - 3 * p(-1, 0)) / (12 * dR * r_i)
                + (-p(0, -3) + 4 * p(0, -2) + 6 * p(0, -1) + 11 * p(0, 1)) / (12 * dZ2)
                + src
            )
        if (j > 2 and j < nZ - 1) and i == mat.el(jrb, j) - 1:
            return 1 / (20 / (12 * dR2) + 10 / (12 * dR * r_i) + 30 / (12 * dZ2)) * (
                (-p(-3, 0) + 4 * p(-2, 0) + 6 * p(-1, 0) + 11 * p(1, 0)) / (12 * dR2)
                - (-p(-3, 0) + 6 * p(-2, 0) - 18 * p(-1, 0) + 3 * p(1, 0)) / (12 * dR * r_i)
                + (-p(0, 2) + 16 * p(0, 1) + 16 * p(0, -1) - p(0, -2)) / (12 * dZ2)
                + src
            )
        if j == 2 and i == mat.el(jrb, j) - 1:
            return 1 / (20 / (12 * dR2) + 10 / (12 * dR * r_i) + 20 / (12 * dZ2)) * (
                (-p(-3, 0) + 4 * p(-2, 0) + 6 * p(-1, 0) + 11 * p(1, 0)) / (12 * dR2)
                - (-p(-3, 0) + 6 * p(-2, 0) - 18 * p(-1, 0) + 3 * p(1, 0)) / (12 * dR * r_i)
                + (-p(0, 3) + 4 * p(0, 2) + 6 * p(0, 1) + 11 * p(0, -1)) / (12 * dZ2)
                + src
            )
        if j == nZ - 1 and i == mat.el(jrb, j) - 1:
            return 1 / (20 / (12 * dR2) + 10 / (12 * dR * r_i) + 20 / (12 * dZ2)) * (
                (-p(-3, 0) + 4 * p(-2, 0) + 6 * p(-1, 0) + 11 * p(1, 0)) / (12 * dR2)
                - (-p(-3, 0) + 6 * p(-2, 0) - 18 * p(-1, 0) + 3 * p(1, 0)) / (12 * dR * r_i)
                + (-p(0, -3) + 4 * p(0, -2) + 6 * p(0, -1) + 11 * p(0, 1)) / (12 * dZ2)
                + src
            )
        if j == 2 and (i > 2 and i < mat.el(jrb, j) - 1):
            return 1 / (30 / (12 * dR2) + 20 / (12 * dZ2)) * (
                (-p(2, 0) + 16 * p(1, 0) + 16 * p(-1, 0) - p(-2, 0)) / (12 * dR2)
                - (-p(2, 0) + 8 * p(1, 0) - 8 * p(-1, 0) + p(-2, 0)) / (12 * dR * r_i)
                + (-p(0, 3) + 4 * p(0, 2) + 6 * p(0, 1) + 11 * p(0, -1)) / (12 * dZ2)
                + src
            )
        if j == nZ - 1 and (i > 2 and i < mat.el(jrb, j) - 1):
            return 1 / (30 / (12 * dR2) + 20 / (12 * dZ2)) * (
                (-p(2, 0) + 16 * p(1, 0) + 16 * p(-1, 0) - p(-2, 0)) / (12 * dR2)
                - (-p(2, 0) + 8 * p(1, 0) - 8 * p(-1, 0) + p(-2, 0)) / (12 * dR * r_i)
                + (-p(0, -3) + 4 * p(0, -2) + 6 * p(0, -1) + 11 * p(0, 1)) / (12 * dZ2)
                + src
            )
        # 9 个子分支都不匹配时，MATLAB 里 rhs(i,j) 保持 psi0(i,j) 不变（循环也没进来）
        return p(0, 0)

    raise ValueError(f"rhs_point: 未知的 ifd={ifd}")


def _five_point(p, dR2, dZ2, r_i, dR, src):
    """ifd=3/4 的内部五点公式（两路逐字相同）。"""
    return 1 / (30 / (12 * dR2) + 30 / (12 * dZ2)) * (
        (-p(2, 0) + 16 * p(1, 0) + 16 * p(-1, 0) - p(-2, 0)) / (12 * dR2)
        - (-p(2, 0) + 8 * p(1, 0) - 8 * p(-1, 0) + p(-2, 0)) / (12 * dR * r_i)
        + (-p(0, 2) + 16 * p(0, 1) + 16 * p(0, -1) - p(0, -2)) / (12 * dZ2)
        + src
    )


def rhs_block(ifd, psi0, dP, rr, jrb, nZ, dR, dZ, dR2, dZ2, mu0):
    """整个 `rhs` 数组（对应源码里的双重循环）。"""
    rhs = np.array(psi0, dtype=float, copy=True)
    nR = np.shape(psi0)[0]
    for j in mat.mrange(2, nZ - 1):
        for i in mat.mrange(2, mat.el(jrb, j) - 1):
            mat.set_ij(rhs, i, j, rhs_point(
                ifd, i, j, psi0, dP, rr, jrb, nZ, dR, dZ, dR2, dZ2, mu0))
        _ = nR
    return rhs
