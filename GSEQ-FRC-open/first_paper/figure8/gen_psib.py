"""gen_psib.m 的逐行移植（figure8 副本）——**这一份不与另两份合并**。

合并情况（2026-09-17，见 PORTING.md §8）：`appendix_B` 下的两份 `gen_psib.m`
只差一行注释，已经并成 `gseqfrc/shared/gen_psib_appendix_B.py`；本份与它们差
16 处、+73/-76 行（装置尺寸、`zz` 的写法、`icase==7/8` 的参数块、绘图段都不同，
`icase==4` 还是负的 `psim` 且会让绘图段的 `ylim` 报错），是**另一个程序**，
所以按"不二选一、但也不硬塞进一个带大段分支的文件"处理：保持独立。

壁面磁通 profile 的生成脚本：按 `icase` 选一组装置尺寸和一条 `psib(z)`，
画图并写出 `psib.mat`（后续的一维/二维求解器都要读它）。

三份副本内容各不相同：本副本的默认 `icase=8`（`zw=0.75, rw=0.17, rend=0.15`），
而且它的 `icase==4` 分支写的是 `psim=-0.00376`（负值），
带到绘图段的 `ylim([0,2*psiw])` 上界为负，MATLAB 会直接报错——
基准把它记成 `expectError`，移植版照抄这个行为。

`icase` 是脚本第 3 行的一个字面量，harness 通过**改这一行**来覆盖分支，
所以移植版的 `run(ws, icase=None)` 用参数表达同一个意思（None 表示用原值）。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mio
from gseqfrc import mpath
from gseqfrc import mplot

#: 脚本里写死的默认分支
DEFAULT_ICASE = 8


def run(ws, icase=None) -> None:
    mplot.close_all()  # close all; clear; clc;
    if icase is None:
        icase = DEFAULT_ICASE

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
    elif icase == 7:  # one frc, 19-10-07 09:06 for compress
        zw = mat.scalar(3.0); rw = mat.scalar(0.7); rend = mat.scalar(0.4)
    elif icase == 8:  # one frc, 19-10-07 09:06 for compress
        zw = mat.scalar(0.75); rw = mat.scalar(0.17); rend = mat.scalar(0.15)

    zz = mat.colon(-zw, 0.005 * zw, zw)
    zb = zz
    rb = rw + 0.0 * zb
    n = mat.length(zz)

    # % psiw=0.00376; psim=0.5*psiw;
    # % psim=0.00376; psiw=2.0*psim;
    if icase == 2:
        psim = mat.scalar(0.00376); psiw = 4.0 * psim
        zc = -0.1 * zw

        ind1 = mat.find(zz < zc)
        ind2 = mat.find(zz >= zc)
        zm1 = -0.3 * zw + zc; zm2 = 0.3 * zw + zc; psiw2 = 1.0 * psiw
        psib = mat.zeros(1, n)
        mat.set_at(psib, ind1,
                   0.5 * (psim - psiw) * (np.tanh(20 * (mat.take(zz, ind1) - zm1)) + 1.0) + psiw)
        mat.set_at(psib, ind2,
                   0.5 * (psim - psiw2) * (np.tanh(20 * (zm2 - mat.take(zz, ind2))) + 1.0) + psiw2)
    elif icase == 1:
        psim = mat.scalar(0.00376); psiw = 2.0 * psim
        zc = -0.4 * zw

        ind1 = mat.find(zz < zc)
        ind2 = mat.find(zz >= zc)
        zm1 = -0.2 * zw + zc; zm2 = 0.2 * zw + zc; psiw2 = 1.0 * psiw
        psib = mat.zeros(1, n)
        mat.set_at(psib, ind1,
                   0.5 * (psim - psiw) * (np.tanh(20 * (mat.take(zz, ind1) - zm1)) + 1.0) + psiw)
        mat.set_at(psib, ind2,
                   0.5 * (psim - psiw2) * (np.tanh(20 * (zm2 - mat.take(zz, ind2))) + 1.0) + psiw2)
    elif icase == 3:
        psim = mat.scalar(0.00376); psiw = 2.0 * psim
        zc = -0.2 * zw

        ind1 = mat.find(zz < zc)
        ind2 = mat.find(zz >= zc)
        zm1 = -0.2 * zw + zc; zm2 = 0.2 * zw + zc; psiw2 = 1.0 * psiw
        psib = mat.zeros(1, n)
        mat.set_at(psib, ind1,
                   0.5 * (psim - psiw) * (np.tanh(20 * (mat.take(zz, ind1) - zm1)) + 1.0) + psiw)
        mat.set_at(psib, ind2,
                   0.5 * (psim - psiw2) * (np.tanh(20 * (zm2 - mat.take(zz, ind2))) + 1.0) + psiw2)
    elif icase == 6:
        psim = mat.scalar(0.00376); psiw = 2.0 * psim
        zc = -0.0 * zw

        ind1 = mat.find(zz < zc)
        ind2 = mat.find(zz >= zc)
        zm1 = -0.8 * zw + zc; zm2 = 0.8 * zw + zc; psiw2 = 1.0 * psiw
        psib = mat.zeros(1, n)
        mat.set_at(psib, ind1,
                   0.5 * (psim - psiw) * (np.tanh(20 * (mat.take(zz, ind1) - zm1)) + 1.0) + psiw)
        mat.set_at(psib, ind2,
                   0.5 * (psim - psiw2) * (np.tanh(20 * (zm2 - mat.take(zz, ind2))) + 1.0) + psiw2)
    elif icase == 7:
        psim = mat.scalar(0.00376); psiw = 2.0 * psim
        zcon1 = mat.scalar(0.85); zcon2 = zcon1 + 0.1
        zc = (zcon1 - 0.0) * zw; zc1 = -(zcon1 - 0) * zw
        zm1 = -zcon2 * zw; zm2 = zcon2 * zw

        ind1 = mat.find(np.abs(zz) <= zc)
        ind2 = mat.find(np.abs(zz) >= zm2)
        ind3 = mat.find((zz >= zc) & (zz < zm2))  # right
        ind4 = mat.find((zz >= zm1) & (zz < zc1))  # right
        psib = mat.zeros(1, n)
        mat.set_at(rb, ind1, 1.0 * rw);   mat.set_at(psib, ind1, psim)
        mat.set_at(rb, ind2, 1.0 * rend); mat.set_at(psib, ind2, psim)
        mat.set_at(rb, ind3,
                   rw + (rend - rw) / (zm2 - zc) * (mat.take(zz, ind3) - zc))
        mat.set_at(psib, ind3, psim)
        mat.set_at(rb, ind4,
                   rend + (rw - rend) / (zc1 - zm1) * (mat.take(zz, ind4) - zm1))
        mat.set_at(psib, ind4, psim)
    elif icase == 8:
        psim = mat.scalar(1.0 * 0.006384168)
        # %     psim=0.0091591;
        # %     zcon1=0.6; zcon2=zcon1+0.05/1.5;
        zcon1 = mat.scalar(0.6); zcon2 = zcon1 + 0.05 / 0.75
        zc = (zcon1 - 0.0) * zw; zc1 = -(zcon1 - 0) * zw
        zm1 = -zcon2 * zw; zm2 = zcon2 * zw

        ind1 = mat.find(np.abs(zz) <= zc)
        ind2 = mat.find(np.abs(zz) >= zm2)
        ind3 = mat.find((zz >= zc) & (zz < zm2))  # right
        ind4 = mat.find((zz >= zm1) & (zz < zc1))  # right
        psib = mat.zeros(1, n)
        mat.set_at(rb, ind1, 1.0 * rw);   mat.set_at(psib, ind1, psim)
        mat.set_at(rb, ind2, 1.0 * rend); mat.set_at(psib, ind2, psim)
        mat.set_at(rb, ind3,
                   rw + (rend - rw) / (zm2 - zc) * (mat.take(zz, ind3) - zc))
        mat.set_at(psib, ind3, psim)
        mat.set_at(rb, ind4,
                   rend + (rw - rend) / (zc1 - zm1) * (mat.take(zz, ind4) - zm1))
        mat.set_at(psib, ind4, psim)
    elif icase == 4:
        psim = mat.scalar(-0.00376); psiw = 2.0 * psim
        zc = -0.0 * zw

        ind1 = mat.find(zz < zc)
        ind2 = mat.find(zz >= zc)
        zm1 = -0.2 * zw + zc; zm2 = 0.2 * zw + zc; psiw2 = 1.0 * psiw
        psib = mat.zeros(1, n)
        mat.set_at(psib, ind1,
                   0.5 * (psim - psiw) * (np.tanh(20 * (mat.take(zz, ind1) - zm1)) + 1.0) + psiw)
        mat.set_at(psib, ind2,
                   0.5 * (psim - psiw2) * (np.tanh(20 * (zm2 - mat.take(zz, ind2))) + 1.0) + psiw2)
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
    else:
        psib = psim * (1 + np.cos(3 * np.pi * zz / zw) + (1.0 * zz / zw) ** 2) + psiw

    # MATLAB 是边跑边往基本工作区写变量，所以脚本中途报错时快照里仍留着已经
    # 算出的那部分。icase==4 正是这种算例（`ylim` 报错、脚本中止、`psib.mat`
    # 根本没写出来），所以工作区要在 finally 里落下、异常照常往外抛。
    try:
        _plot_and_save(icase, zw, rw, zz, zb, rb, psib, locals().get("psiw"))
    finally:
        _write_workspace(ws, icase, locals())


def _plot_and_save(icase, zw, rw, zz, zb, rb, psib, psiw) -> None:
    hf = mplot.figure("unit", "normalized", "Position", mat.row(0.6, 0.3, 0.4, 0.2),
                      "DefaultAxesFontSize", 15)
    if icase == 7 or icase == 8:
        mplot.plot(zz, rb, ".-m")
        mplot.grid_on()
        mplot.set_(mplot.gca(), "Fontsize", 23)
        mplot.xlabel("$z / Z_{max}$", "interpreter", "latex", "FontSize", 17)
        mplot.ylabel("$r / R_{w}$", "interpreter", "latex", "FontSize", 17)
        mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k", "GridAlpha", 1)
        h = mplot.legend("$R_{w}(z)$")
        mplot.legend("boxoff")
        mplot.ylim(mat.row(0, rw))
    else:
        mplot.plot(zb, psib, "linewidth", 2)
        mplot.grid_on()
        mplot.set_(mplot.gca(), "Fontsize", 17)
        mplot.xlim(mat.row(-zw, zw))
        mplot.ylim(mat.row(0, 2 * psiw))
        mplot.xlabel("$Z \\ (m)$", "interpreter", "latex", "FontSize", 17)
        mplot.ylabel("$\\psi \\ (Wb)$", "interpreter", "latex", "FontSize", 17)
        mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k", "GridAlpha", 1)

    # % h=legend('$case \ 1$');
    # % set(h,'interpreter','latex');

    mio.save("psib.mat", zb=zb, rb=rb, psib=psib)


def _write_workspace(ws, icase, values: dict) -> None:
    """把脚本里出现过的变量写进工作区（对应 MATLAB 的基本工作区）。"""
    ws.icase = mat.scalar(icase)
    for name in ("zw", "rw", "zz", "zb", "rb", "psib", "rend", "psim", "psiw",
                 "psiw2", "zc", "zc1", "zm1", "zm2", "zm3", "zm4", "zcon1",
                 "zcon2", "ind1", "ind2", "ind3", "ind4"):
        value = values.get(name)
        if value is not None:
            ws[name] = value
    if "j" in values:
        ws.j = mat.scalar(values["j"])
