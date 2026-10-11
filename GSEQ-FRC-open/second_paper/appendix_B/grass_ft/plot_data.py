"""plot_data.m 的移植：把 PIC 输出的 41 帧 ψ 数据连成一个 GIF（`test.gif`）。

⚠️ **随仓库提供的数据里没有它要的输入**：脚本读 `data/%06d_pse.DAT`
（`i=1:50:2001`，即 41 帧），而 `grass_ft/data/` 里只有 `br/bz/cq/cqp/pr/ps_pl/
ps_t/ps_t_ref/wall` 这些，没有 `*_pse.DAT`。所以按原样跑，它会**停在第 13 行那句
`importdata('data/000000_pse.DAT')` 上**（MATLAB 也是这个行为），后面的循环与 GIF
组装在仓库里是够不着的死代码。移植版照抄全部结构；测试里两条都验：

* 没有原始数据时，停在同一个位置（找不到文件）；
* 另外拿**合成**的 41 帧（临时目录，不进仓库）跑一遍，确认循环与 GIF 那段真的能出图。

验收：§7「绘图脚本」那一节的三条（能跑通 / 结构对 / 人眼比对论文图）；这一份的
"能跑通"只能按上面两条来算。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mplot


def run(ws) -> None:
    mplot.close_all()  # close all;clear;clc;

    mplot.figure("position", mat.row(50, 100, 600, 300))
    mplot.set_(mplot.gca(), "nextplot", "replacechildren")
    mplot.set_(mplot.gcf(), "DefaultAxesFontSize", 15)

    rmin = mat.scalar(0)
    zmin = mat.scalar(-1.5)
    rmax = mat.scalar(1.0)
    zmax = mat.scalar(1.5)

    data = mat.importdata("data/000000_pse.DAT")
    r = mat.linspace(rmin, rmax, int(np.size(data, 0)))
    z = mat.linspace(zmin, zmax, int(np.size(data, 1)))
    X, Y = mat.ndgrid(r, z)

    rr = mat.linspace(rmin, rmax, int(np.size(data, 0)))
    zz = mat.linspace(zmin, zmax, int(np.size(data, 1)))
    X1, Y1 = mat.ndgrid(rr, zz)

    # MATLAB 的 `for i=1:50:2001`：i=1,51,...,2001 共 41 帧
    for i in range(1, 2001 + 1, 50):
        name2 = mat.num2str(i - 1, "%06d")
        xm = "data/" + name2 + "_pse.DAT"
        data = mat.importdata(xm)

        Z1 = mat.griddata(X, Y, data, X1, Y1)
        mplot.set_(mplot.gca(), "Fontsize", 15)
        mplot.contour(Y1, X1, Z1, 30, "linewidth", 1)   # %外部区域
        mplot.title("FRC G-S equilibrium " + " it=" + mat.num2str(i - 1))
        mplot.box("on")
        mplot.grid_on()
        mplot.colorbar()
        mplot.caxis(mat.row(-4, 0))
        mplot.xlim(mat.row(zmin, zmax))
        mplot.ylim(mat.row(0, 1))
        mplot.set_(mplot.gca(), "XTick", mat.colon(zmin, 0.5, zmax))
        mplot.set_(mplot.gca(), "YTick", mat.colon(0, 0.2, 1))
        mplot.xlabel("$Z$", "interpreter", "latex", "FontSize", 17)
        mplot.ylabel("$R$", "interpreter", "latex", "FontSize", 17)
        mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
                   "GridAlpha", 0.5)
        mplot.pause(0.1)

        frame = mplot.getframe(mplot.gcf())
        imind = mplot.frame2im(frame)
        imind, cm = mplot.rgb2ind(imind, 256)
        if i == 1:
            mplot.imwrite(imind, cm, "test.gif", loop_count=np.inf,
                          delay_time=1e-4)
        else:
            mplot.imwrite(imind, cm, "test.gif", write_mode="append",
                          delay_time=1e-4)
    # % imwrite(imind,cm,'test.gif','GIF','Loopcount',inf,'DelayTime',1);

    for name, value in (("rmin", rmin), ("zmin", zmin), ("rmax", rmax),
                        ("zmax", zmax), ("data", data), ("r", r), ("z", z),
                        ("X", X), ("Y", Y), ("rr", rr), ("zz", zz),
                        ("X1", X1), ("Y1", Y1), ("Z1", Z1), ("psiq", data),
                        ("imind", imind), ("cm", cm)):
        ws[name] = value
