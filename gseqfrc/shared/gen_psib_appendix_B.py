"""gen_psib.m 的逐行移植（**appendix_B 两份副本合并后**的唯一实现）。

源文件：

    gseq-main/GSEQ-FRC-open/second_paper/appendix_B/GSEQ/gen_psib.m
    gseq-main/GSEQ-FRC-open/second_paper/appendix_B/grass_ft/gen_psib.m

2026-09-17 副本合并：这两份**只差一行注释**（`%     zw=1.5;rw=1.0;`），两份冻结
基准（`p_genpsib_gseq_ic*` / `p_genpsib_grass_*`）的工作区变量与数值也**完全相同**，
所以合并成一份实现、两个原位置各留一个薄壳。`first_paper/figure8/gen_psib.m` 是
**另一个程序**（与这两份差 16 处、+73/-76 行：装置尺寸、`zz` 的写法、`icase==7/8`
的参数块、绘图段都不同，`icase==4` 还是负的 `psim`），它继续单独放在
`GSEQ-FRC-open/first_paper/figure8/gen_psib.py`，不并进来。

与 figure8 那份的差别：

* 装置尺寸不同，`zz` 用 `linspace(-zw,zw,385)` 而不是冒号运算符；
* `icase==7` 是 grass_ft 的 psi_bound 构造（`sub_plot=2` 那条分支，按段拼 cos 曲线）；
* `icase==8` 是 `psib` 全 0（zw=1.5, rw=1.0）；
* 绘图段很短，没有 figure 句柄变量、也没有 `save` 之外的文件。

文件尾同样 `save('psib.mat','zb','rb','psib')`。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mio
from gseqfrc import mplot

#: 脚本里写死的默认分支
DEFAULT_ICASE = 7


def run(ws, icase=None) -> None:
    mplot.close_all()  # close all; clear; clc;
    if icase is None:
        icase = DEFAULT_ICASE

    mplot.figure("position", mat.row(50, 100, 1000, 400))
    mplot.set_(mplot.gca(), "nextplot", "replacechildren")
    mplot.set_(mplot.gcf(), "DefaultAxesFontSize", 15)

    if icase == 1:
        zw = mat.scalar(1.0); rw = mat.scalar(0.17)
    elif icase == 2:
        zw = mat.scalar(1.5); rw = mat.scalar(0.17)
    elif icase == 3:  # one frc, for acceleration and translation
        zw = mat.scalar(1.0); rw = mat.scalar(0.17)
    elif icase == 4:
        zw = mat.scalar(1.0); rw = mat.scalar(0.17)
    elif icase == 5:  # two frc, 19-10-04 20:24, for collide
        zw = mat.scalar(1.5); rw = mat.scalar(0.17)
    elif icase == 6:  # one frc, 19-10-07 09:06 for compress
        zw = mat.scalar(1.0); rw = mat.scalar(0.17)
    elif icase == 7:  # one frc, grass_ft psi_bound
        zw = mat.scalar(1.5); rw = mat.scalar(0.17)
        psi_m1 = mat.scalar(-4); psi_wall = mat.scalar(-1); psi_m2 = mat.scalar(-4)
        Z_m1 = mat.scalar(-0.5); Z_m2 = mat.scalar(0.5)
        Z_c1 = mat.scalar(-0.25); Z_c2 = mat.scalar(0.25)
        mo = mat.scalar(385); m = mat.scalar(129)
        rm1 = mat.scalar(4.0); rm2 = mat.scalar(4.0)
        jz1 = mat.scalar(129 + ((m - 1) / 4))
        jz2 = mat.scalar(257 - ((m - 1) / 4))
    elif icase == 8:
        zw = mat.scalar(1.5); rw = mat.scalar(1.0)

    # % zz=-zw:0.005*zw:zw;
    zz = mat.linspace(-zw, zw, 385)
    zb = zz
    rb = rw + 0.0 * zb
    n = mat.length(zz)

    def tanh_pair(ind1, ind2, psim_, psiw_, zm1_, zm2_, psiw2_):
        """icase 1/2/3/4/6 共用的两段 tanh 拼接。"""
        psib = mat.zeros(1, n)
        mat.set_at(psib, ind1,
                   0.5 * (psim_ - psiw_) * (np.tanh(20 * (mat.take(zz, ind1) - zm1_)) + 1.0) + psiw_)
        mat.set_at(psib, ind2,
                   0.5 * (psim_ - psiw2_) * (np.tanh(20 * (zm2_ - mat.take(zz, ind2))) + 1.0) + psiw2_)
        return psib

    if icase == 2:
        psim = mat.scalar(0.00376); psiw = 4.0 * psim
        zc = -0.1 * zw
        ind1 = mat.find(zz < zc); ind2 = mat.find(zz >= zc)
        zm1 = -0.3 * zw + zc; zm2 = 0.3 * zw + zc; psiw2 = 1.0 * psiw
        psib = tanh_pair(ind1, ind2, psim, psiw, zm1, zm2, psiw2)
    elif icase == 1:
        psim = mat.scalar(0.00376); psiw = 2.0 * psim
        zc = -0.4 * zw
        ind1 = mat.find(zz < zc); ind2 = mat.find(zz >= zc)
        zm1 = -0.2 * zw + zc; zm2 = 0.2 * zw + zc; psiw2 = 1.0 * psiw
        psib = tanh_pair(ind1, ind2, psim, psiw, zm1, zm2, psiw2)
    elif icase == 3:
        psim = mat.scalar(0.00376); psiw = 2.0 * psim
        zc = -0.2 * zw
        ind1 = mat.find(zz < zc); ind2 = mat.find(zz >= zc)
        zm1 = -0.2 * zw + zc; zm2 = 0.2 * zw + zc; psiw2 = 1.0 * psiw
        psib = tanh_pair(ind1, ind2, psim, psiw, zm1, zm2, psiw2)
    elif icase == 6:
        psim = mat.scalar(0.00376); psiw = 2.0 * psim
        zc = -0.0 * zw
        ind1 = mat.find(zz < zc); ind2 = mat.find(zz >= zc)
        zm1 = -0.3 * zw + zc; zm2 = 0.3 * zw + zc; psiw2 = 1.0 * psiw
        psib = tanh_pair(ind1, ind2, psim, psiw, zm1, zm2, psiw2)
    elif icase == 4:
        psim = mat.scalar(0.00376); psiw = 2.0 * psim
        zc = -0.0 * zw
        ind1 = mat.find(zz < zc); ind2 = mat.find(zz >= zc)
        zm1 = -0.2 * zw + zc; zm2 = 0.2 * zw + zc; psiw2 = 1.0 * psiw
        psib = tanh_pair(ind1, ind2, psim, psiw, zm1, zm2, psiw2)
    elif icase == 5:  # two frc, 19-10-04 20:24
        psim = mat.scalar(0.00376); psiw = 2.0 * psim
        zc = -0.4 * zw
        zm1 = -0.15 * zw + zc; zm2 = 0.15 * zw + zc
        zm3 = -0.15 * zw - zc; zm4 = 0.15 * zw - zc
        psiw2 = 1.0 * psiw
        psib = 0.0 * zz
        for j in mat.mrange(1, mat.length(zz)):
            z = mat.el(zz, j)
            if z < zc:
                mat.set_el(psib, j, 0.5 * (psim - psiw) * (np.tanh(20 * (z - zm1)) + 1.0) + psiw)
            elif z >= zc and z < 0:
                mat.set_el(psib, j, 0.5 * (psim - psiw2) * (np.tanh(20 * (zm2 - z)) + 1.0) + psiw2)
            elif z >= 0 and z < -zc:
                mat.set_el(psib, j, 0.5 * (psim - psiw2) * (-np.tanh(20 * (zm3 - z)) + 1.0) + psiw2)
            else:
                mat.set_el(psib, j, 0.5 * (psim - psiw) * (-np.tanh(20 * (z - zm4)) + 1.0) + psiw)
    elif icase == 7:  # one frc, grass_ft psi_bound
        sub_plot = mat.scalar(2)
        psib = mat.zeros(1, n)
        if mat.scalar_value(sub_plot) == 1:
            for i in mat.mrange(1, int(np.shape(zz)[1])):
                z = mat.el(zz, i)
                if z <= Z_m1:
                    mat.set_el(psib, i, psi_m1)
                elif z > Z_m1 and z < Z_c1:
                    mat.set_el(psib, i, 0.5 * (psi_m1 + psi_wall)
                               + 0.5 * (psi_m1 - psi_wall)
                               * np.cos(np.pi * (z - Z_m1) / (Z_m1 - Z_c1)))
                elif z < Z_c2 and z >= Z_c1:
                    mat.set_el(psib, i, psi_wall)
                elif z < Z_m2 and z >= Z_c2:
                    mat.set_el(psib, i, 0.5 * (psi_m2 + psi_wall)
                               + 0.5 * (psi_wall - psi_m2)
                               * np.cos(np.pi * (z - Z_c2) / (Z_m2 - Z_c2)))
                elif z >= Z_m2:
                    mat.set_el(psib, i, psi_m2)
        elif mat.scalar_value(sub_plot) == 2:
            for i in mat.mrange(1, int(mat.scalar_value(mo))):
                if i <= mat.scalar_value(jz1) and i > 129:  # two
                    mat.set_el(psib, i, -0.5 * (rm1 + 1.0)
                               - 0.5 * (rm1 - 1.0)
                               * np.cos(np.pi / (mat.el(zz, int(mat.scalar_value(jz1))) + 0.5)
                                        * (mat.el(zz, i) + 0.5)))
                elif i > mat.scalar_value(jz1) and i <= mat.scalar_value(jz2):  # three
                    mat.set_el(psib, i, -1.0)
                elif i > mat.scalar_value(jz2) and i <= 257:  # four
                    mat.set_el(psib, i, -0.5 * (rm2 + 1.0)
                               - 0.5 * (rm2 - 1.0)
                               * np.cos(np.pi / (mat.el(zz, int(mat.scalar_value(jz2))) - 0.5)
                                        * (mat.el(zz, i) - 0.5)))
                elif i <= 129:  # one
                    mat.set_el(psib, i, -0.5 * (rm1 + 1.0)
                               - 0.5 * (rm1 - 1.0)
                               * np.cos(np.pi / (mat.el(zz, int(mat.scalar_value(jz1))) + 0.5)
                                        * (mat.el(zz, 129) + 0.5)))
                elif i >= 257:  # five
                    mat.set_el(psib, i, -0.5 * (rm2 + 1.0)
                               - 0.5 * (rm2 - 1.0)
                               * np.cos(np.pi / (mat.el(zz, int(mat.scalar_value(jz2))) - 0.5)
                                        * (mat.el(zz, 257) - 0.5)))
    elif icase == 8:
        psib = mat.zeros(1, n)
        for i in mat.mrange(1, int(np.shape(zz)[1])):
            mat.set_el(psib, i, 0.0)
    else:
        psib = psim * (1 + np.cos(3 * np.pi * zz / zw) + (1.0 * zz / zw) ** 2) + psiw

    try:
        mplot.plot(zb, psib, "m", "linewidth", 2)
        mplot.xlim(mat.row(-zw, zw))
        mplot.grid_on()
        mplot.set_(mplot.gca(), "box", "on")
        mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k", "GridAlpha", 0.5)
        mplot.xlabel("$Z(m)$", "interpreter", "latex", "FontSize", 17)
        mplot.ylabel("$\\psi$", "interpreter", "latex", "FontSize", 17)
        mio.save("psib.mat", zb=zb, rb=rb, psib=psib)
    finally:
        ws.icase = mat.scalar(icase)
        for name in ("zw", "rw", "zz", "zb", "rb", "psib", "psim", "psiw", "psiw2",
                     "zc", "zm1", "zm2", "zm3", "zm4", "sub_plot", "mo", "m",
                     "rm1", "rm2", "jz1", "jz2", "psi_m1", "psi_m2", "psi_wall",
                     "Z_m1", "Z_m2", "Z_c1", "Z_c2", "ind1", "ind2"):
            value = locals().get(name)
            if value is not None:
                ws[name] = value
        if "j" in locals():
            ws.j = mat.scalar(j)
        if "i" in locals():
            ws.i = mat.scalar(i)
