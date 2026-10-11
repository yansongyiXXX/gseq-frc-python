"""fun_ellipEK.m 的逐行移植（**三份副本合并后**的唯一实现）。

源文件（三份副本逐字节相同，SHA256 一致）：

    gseq-main/GSEQ-FRC-open/second_paper/Fig4/gui/fun_ellipEK.m
    gseq-main/GSEQ-FRC-open/second_paper/Fig5/gui/fun_ellipEK.m
    gseq-main/GSEQ-FRC-open/second_paper/Fig6/fun_ellipEK.m

2026-09-17 副本合并：实现搬到 `gseqfrc/shared/`，程序树里那三个原位置改成薄壳
（内容只是 `from gseqfrc.shared.fun_ellipEK import fun_ellipEK`）。原程序用
addpath + 函数名解析，薄壳让"原位置仍然有这个名字"这件事保持不变（`mpath` 的
解析行为与合并前一致）。

要点：

* `apx==1` 是真正被调用的分支（`fun_Bfield.m` 与 `GSEQ_coil_MRR.m` 都走它），
  它是 Burkardt special_functions 里的多项式 + 对数近似，**不是** scipy 那种精确
  椭圆积分。换成精确实现会让整个代码库的线圈磁场都变，所以必须照抄这个近似。
* `K = 0.0.*zeros(length(k2),1)` 决定了返回值是**列向量**（n×1），即使输入是行向量。
* `if ( k2 == 1.0 )` 在 `k2` 是向量时是隐含的 `all()` —— 见 `matlab_compat.truth`。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat


def fun_ellipEK(k2, apx=None):
    """[K,E]=fun_ellipEK(k2,apx)：省略 apx 时取 apx=1。"""
    nargin = 1 if apx is None else 2
    if nargin == 1:
        apx = 1

    if apx == 2:
        # 原程序在这里调 MATLAB 的 ellipke(k2,1e-8)（带容差的 AGM）。
        # 没有任何算例覆盖这条分支，scipy 的实现也不保证逐位相同，
        # 所以先按 MATLAB 的 AGM 形式实现，等有算例覆盖时再逐位校验。
        K, E = mat.ellipke(k2, 1e-8)
    elif apx == 0:
        # MATLAB 内建的 ellipticK/ellipticE（不是仓库里的函数），
        # 数值上与 ellipke 的完全椭圆积分一致——`u_ellipEK_alt` 覆盖这条分支。
        K = mat.ellipticK(k2)
        E = mat.ellipticE(k2)
    elif apx == 1:
        # from http://people.sc.fsu.edu/~jburkardt%20/f_src/special_functions/special_functions.f90
        # Accurate and fast
        K = 0.0 * mat.zeros(mat.length(k2), 1)
        E = 0.0 * K
        for j in mat.mrange(1, mat.length(k2)):
            pk = 1.0 - mat.el(k2, j)

            if mat.truth(np.asarray(k2) == 1.0):
                mat.set_el(K, j, 1e300)
                mat.set_el(E, j, 1.0)
            else:
                ak = (
                    (
                        (0.01451196212 * pk + 0.03742563713) * pk + 0.03590092383
                    )
                    * pk
                    + 0.09666344259
                ) * pk + 1.38629436112

                bk = (
                    (
                        (0.00441787012 * pk + 0.03328355346) * pk + 0.06880248576
                    )
                    * pk
                    + 0.12498593597
                ) * pk + 0.5

                mat.set_el(K, j, ak - bk * np.log(pk))

                ae = (
                    (
                        (0.01736506451 * pk + 0.04757383546) * pk + 0.0626060122
                    )
                    * pk
                    + 0.44325141463
                ) * pk + 1.0

                be = (
                    (
                        (0.00526449639 * pk + 0.04069697526) * pk + 0.09200180037
                    )
                    * pk
                    + 0.2499836831
                ) * pk

                mat.set_el(E, j, ae - be * np.log(pk))

    else:  # Taylor expansion for Elliptic function
        K = 0.5 * np.pi * (1 + k2 / 4 + 9 / 64 * k2**2 + (15 / 48) ** 2 * k2**3)
        E = 0.5 * np.pi * (1 - k2 / 4 - 3 / 64 * k2**2 - 45 / 48**2 * k2**3)

    return K, E
