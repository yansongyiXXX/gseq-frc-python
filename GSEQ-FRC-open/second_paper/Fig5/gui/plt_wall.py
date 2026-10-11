"""plt_wall.m（Fig5/gui 副本）的移植：画壁面并把壁面上的 ψ 存成 `psib_paper.mat`。

与 Fig4/gui 那份逐行对照只差三处（都照抄）：

* `wall=2` 分支里 `rr` 是 `0.15+0.*zz`（Fig4 是 0.14）；
* 第一次 `save` 只写 `zb/rb/psib`（Fig4 那份多一个 `Bz`）；
* 第二次 `save('../psib_paper.mat',...)` 仍然带 `Bz`。

**是 GUI 的会话脚本**（要用 `hAxes1/hAxes2` 与线圈表），单跑会停。
仓库里 `Fig5/gui/psib_paper.mat` 就是它当年的产物（`rb` 恒为 0.15 那一份），
测试里逐值对照。

验收：§7「绘图/GUI 脚本」那一节的三条 + 与仓库里那份 `.mat` 的逐值对照。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mio
from gseqfrc import mpath
from gseqfrc import mplot

#: 会话里必须已经有的东西（GUI 或者算例工作区留下的）
NEEDS_WORKSPACE = ("hAxes1", "hAxes2", "rl", "zl", "hl", "wl", "Il", "Nl",
                   "imethod")


def run(ws) -> None:
    missing = [name for name in NEEDS_WORKSPACE if name not in ws.keys()]
    if missing:
        raise KeyError(
            f"plt_wall 是 GUI 的会话脚本，缺少：{', '.join(missing)}"
            "（先跑 frc_coil_gui.m，或把它的 frc_coil.mat 与两个坐标区句柄喂进来）")

    hAxes1 = ws["hAxes1"]; hAxes2 = ws["hAxes2"]
    rl = ws["rl"]; zl = ws["zl"]; hl = ws["hl"]; wl = ws["wl"]
    Il = ws["Il"]; Nl = ws["Nl"]
    imethod = int(mat.scalar_value(ws["imethod"]))

    zw = mat.scalar(0.75); rw = mat.scalar(0.15)

    zz = mat.colon(-mat.scalar_value(zw), 0.01 * mat.scalar_value(zw),
                   mat.scalar_value(zw))

    ind1 = mat.find(zz < 0)
    ind2 = mat.find(zz >= 0)
    a = mat.scalar(0.15); zm = 0.75 * zw; ra = 0.45 * rw; rb = 0.8 * rw
    wall = mat.scalar(2)
    if mat.scalar_value(wall) == 1:
        rr = np.zeros(zz.shape)
        mat.set_at(rr, ind1, (
            0.5 * (mat.scalar_value(rb) - mat.scalar_value(ra))
            * (np.tanh(80 * (mat.take(zz, ind1) + mat.scalar_value(zm))) + 1.0)
            + mat.scalar_value(ra)).ravel())
        mat.set_at(rr, ind2, (
            0.5 * (mat.scalar_value(rb) - mat.scalar_value(ra))
            * (np.tanh(80 * (mat.scalar_value(zm) - mat.take(zz, ind2))) + 1.0)
            + mat.scalar_value(ra)).ravel())
    else:
        rr = 0.15 + 0.0 * zz
    plt = mat.scalar(2)
    if mat.scalar_value(plt) == 1:
        mplot.figure()
        mplot.subplot(211)
        mplot.xlabel("z")
        mplot.ylabel("r")
        mplot.hold_on()
    else:
        mplot.set_(mplot.gcf(), "currentaxes", hAxes1)

    mplot.plot(zz, rr, "r--", "linewidth", 2)
    mplot.hold_on()
    mplot.plot(zz, -rr, "r--", "linewidth", 2)

    if mat.scalar_value(plt) == 1:
        mplot.subplot(212)
        mplot.xlabel("z")
        mplot.ylabel("\\psi")
    elif mat.scalar_value(plt) == 2:
        mplot.set_(mplot.gcf(), "currentaxes", hAxes2)
        mplot.cla()
        mplot.xlabel("z")
        mplot.ylabel("\\psi at wall")
        mplot.hold_on()
        mplot.grid_on()
        mplot.xlim(mat.row(-mat.scalar_value(zw), mat.scalar_value(zw)))

    nz = int(np.size(zz, 1))
    psi = 0.0 * zz
    Br = 0.0 * zz
    Bz = 0.0 * zz
    # % imethod=2;
    for jz in mat.mrange(1, nz):
        out = mpath.call("fun_Bfield", float(mat.el(rr, jz)), float(mat.el(zz, jz)),
                         rl, zl, hl, wl, Il, Nl, imethod)
        mat.set_el(psi, jz, out[0])
        mat.set_el(Br, jz, out[1])
        mat.set_el(Bz, jz, out[2])

    if mat.scalar_value(plt) != 3:
        mplot.hold_on()
        mplot.plot(zz, psi, "linewidth", 2)
    # % diff(psi)/(zz(2)-zz(1))

    rb = rr; zb = zz; psib = psi
    # % save('psib_c2u_as.mat','zb','rb','psib');
    mio.save("psib_paper.mat", zb=zb, rb=rb, psib=psib)
    mio.save("psib_paper.mat", zb=zb, rb=rb, psib=psib, Bz=Bz)
    Z0 = int((np.size(psi, 1) + 1) / 2)
    print("psib=%17.16f " % float(mat.el(psi, Z0)))

    for name, value in (("zw", zw), ("rw", rw), ("zz", zz), ("ind1", ind1),
                        ("ind2", ind2), ("a", a), ("zm", zm), ("ra", ra),
                        ("rb", rb), ("wall", wall), ("plt", plt), ("nz", mat.scalar(nz)),
                        ("psi", psi), ("Br", Br), ("Bz", Bz), ("zb", zb),
                        ("psib", psib), ("Z0", mat.scalar(Z0))):
        ws[name] = value
