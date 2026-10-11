"""plt_wall_c2u.m 的移植：C2U 构型的壁面（tanh 形状）与壁面 ψ。

源文件：`gseq-main/GSEQ-FRC-open/second_paper/Fig5/gui/plt_wall_c2u.m`

与同目录 `plt_wall.m` 的差别：`zw=3; rw=1;`、`wall=1`（所以 `rr` 是按
`0.5*(rb-ra)*(tanh(5*(z±zm))+1)+ra` 算出来的 tanh 壁面，`ra=0.4`、`rb=0.7`），
而且只 `save('psib_paper.mat','zb','rb','psib')` 一次。

**是 GUI 的会话脚本**（要用 `hAxes1/hAxes2` 与线圈表），单跑会停。
它的产出（`psib_paper.mat`）后来被 `plt_wall.m` 覆盖掉了、仓库里没有留下，
所以靶子是"能跑通 + 结构对"（外加与 MATLAB 同脚本跑出来的值对照，见测试）。

验收：§7「绘图/GUI 脚本」那一节的三条。
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
            f"plt_wall_c2u 是 GUI 的会话脚本，缺少：{', '.join(missing)}"
            "（先跑 frc_coil_gui.m，或把它的 frc_coil.mat 与两个坐标区句柄喂进来）")

    hAxes1 = ws["hAxes1"]; hAxes2 = ws["hAxes2"]
    rl = ws["rl"]; zl = ws["zl"]; hl = ws["hl"]; wl = ws["wl"]
    Il = ws["Il"]; Nl = ws["Nl"]
    imethod = int(mat.scalar_value(ws["imethod"]))

    zw = mat.scalar(3); rw = mat.scalar(1)

    zz = mat.colon(-mat.scalar_value(zw), 0.01 * mat.scalar_value(zw),
                   mat.scalar_value(zw))

    ind1 = mat.find(zz < 0)
    ind2 = mat.find(zz >= 0)
    # % a=0.15; zm=0.8*zw; ra=0.4*rw; rb=0.8*rw;
    a = mat.scalar(0.15); zm = 0.75 * zw; ra = 0.4 * rw; rb = 0.7 * rw
    wall = mat.scalar(1)
    if mat.scalar_value(wall) == 1:
        rr = np.zeros(zz.shape)
        mat.set_at(rr, ind1, (
            0.5 * (mat.scalar_value(rb) - mat.scalar_value(ra))
            * (np.tanh(5 * (mat.take(zz, ind1) + mat.scalar_value(zm))) + 1.0)
            + mat.scalar_value(ra)).ravel())
        mat.set_at(rr, ind2, (
            0.5 * (mat.scalar_value(rb) - mat.scalar_value(ra))
            * (np.tanh(5 * (mat.scalar_value(zm) - mat.take(zz, ind2))) + 1.0)
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
    Z0 = int((np.size(psi, 1) + 1) / 2)
    print("psib=%17.16f " % float(mat.el(psi, Z0)))

    for name, value in (("zw", zw), ("rw", rw), ("zz", zz), ("ind1", ind1),
                        ("ind2", ind2), ("a", a), ("zm", zm), ("ra", ra),
                        ("rb", rb), ("wall", wall), ("plt", plt), ("nz", mat.scalar(nz)),
                        ("psi", psi), ("Br", Br), ("Bz", Bz), ("zb", zb),
                        ("psib", psib), ("Z0", mat.scalar(Z0))):
        ws[name] = value
