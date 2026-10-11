"""fun_frc_coil.m（Fig5/gui 副本）的移植：算多线圈磁场并把三张图填好。

源文件：`gseq-main/GSEQ-FRC-open/second_paper/Fig5/gui/fun_frc_coil.m`

与 Fig4/gui 那份的差别（照抄）：

* `hAxes1` 用 `pcolor` 铺填色（+色条），只在对称的下半场叠 `contour(Z,-R,...)`；
* 线圈矩形**不填灰色**，外加"电流方向"记号（`Il>0` 画 `r.`/`rx`，否则反过来）；
* **没有** Fig4 那段"重写 `zw/rw/zz/rr` 画壁面线"的代码，所以后面 `hAxes2`/`hAxes3`
  用的 `rr`/`zz` 还是正经的网格向量（Fig4 那份在这里有个自己写出来的坑）。

物理部分同样是 `fun_Bfield`（这一份是**没有** `/2` 的版本，冻结基准 `u_Bfield_fig5`
钉的就是它）——同一套线圈表/采样点下 `fun_frc_coil` 的返回值要与它一致。
速度注意同 Fig4 那份：`imethod=2/3` 单点 0.76 s / 5.7 s，测试只跑 GUI 默认的 method 1。

验收：§7「绘图/GUI 脚本」那一节的三条 + 上面这条物理对照。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mpath
from gseqfrc import mplot

#: 原文里写死的网格数（`nR=2^5+1; nZ=2^6+1;`）
NR = 2 ** 5 + 1
NZ = 2 ** 6 + 1


def fun_frc_coil(r0, z0, rw, zw, rl, zl, hl, wl, Il, Nl, imethod,
                 hAxes1, hAxes2, hAxes3):
    """`[runtime,psi0,Br0,Bz0]=fun_frc_coil(...)`。"""
    mat.disp("Running fun_frc_coil.m ...")
    runtime = mat.cputime()

    psi0, Br0, Bz0 = mpath.call("fun_Bfield", r0, z0, rl, zl, hl, wl, Il, Nl,
                                imethod)

    mplot.set_(mplot.gcf(), "currentaxes", hAxes1)
    mplot.cla()
    # % imethod=3; % =3, YuanBS book p110; =1, Green function; =2, nr*nz Green function

    nl = len(rl)   # length(rl)
    if (nl != len(zl) or nl != len(wl) or nl != len(hl) or nl != len(Il)
            or nl != len(Nl)):
        mplot.warndlg("Length of rl, zl, hl, wl, Il, Nl not consistient!")
        # 原文这里是裸 `return`（4 个输出都没赋值），移植版留 runtime、其余给 None
        return runtime, None, None, None

    nR = NR
    nZ = NZ
    Rlow = mat.scalar(0); Rup = rw; Zlow = -zw; Zup = zw
    rr = mat.linspace(Rlow, Rup, nR)
    zz = mat.linspace(Zlow, Zup, nZ)
    R, Z = mat.ndgrid(rr, zz)
    psi = 0.0 * R
    for jr in mat.mrange(1, nR):
        for jz in mat.mrange(1, nZ):
            psi[jr - 1, jz - 1] = mpath.call(
                "fun_Bfield", float(mat.el(rr, jr)), float(mat.el(zz, jz)),
                rl, zl, hl, wl, Il, Nl, imethod)[0]
    psi[np.abs(psi) > 1e5] = np.nan
    # % contour(Z,R,psiw_coil);
    mplot.pcolor(Z, R, np.real(psi))
    mplot.shading("interp")
    mplot.hold_on()
    mplot.colorbar()

    mplot.contour(Z, -R, np.real(psi), 100)
    mplot.hold_on()

    for jl in mat.mrange(1, nl):
        mplot.rectangle("position",
                        mat.row(mat.el(zl, jl) - 0.5 * mat.el(wl, jl),
                                mat.el(rl, jl) - 0.5 * mat.el(hl, jl),
                                mat.el(wl, jl), mat.el(hl, jl)),
                        "linewidth", 1)
        mplot.hold_on()
        mplot.rectangle("position",
                        mat.row(mat.el(zl, jl) - 0.5 * mat.el(wl, jl),
                                -mat.el(rl, jl) - 0.5 * mat.el(hl, jl),
                                mat.el(wl, jl), mat.el(hl, jl)),
                        "linewidth", 1)
        mplot.hold_on()
        if float(mat.el(Il, jl)) > 0:
            mplot.plot(mat.el(zl, jl), mat.el(rl, jl), "r.",
                       mat.el(zl, jl), -mat.el(rl, jl), "rx", "linewidth", 2)
        else:
            mplot.plot(mat.el(zl, jl), mat.el(rl, jl), "rx",
                       mat.el(zl, jl), -mat.el(rl, jl), "r.", "linewidth", 2)
        mplot.hold_on()

    mplot.xlim(mat.row(Zlow, Zup))
    mplot.ylim(mat.row(-Rup, Rup))
    mplot.xlabel("z")
    mplot.ylabel("r")
    mplot.box("on")

    mplot.set_(mplot.gcf(), "currentaxes", hAxes2)
    mplot.cla()

    Br1 = 0.0 * rr
    Bz1 = 0.0 * rr
    for jr in mat.mrange(1, nR):
        out = mpath.call("fun_Bfield", float(mat.el(rr, jr)), z0, rl, zl, hl, wl,
                         Il, Nl, imethod)
        mat.set_el(Br1, jr, out[1])
        mat.set_el(Bz1, jr, out[2])
    B1 = np.sqrt(Br1 ** 2 + Bz1 ** 2)

    mplot.plot(rr, Br1, ".", rr, Bz1, "--", rr, B1, ":", "linewidth", 2)
    mplot.xlabel("r (m), z=" + mat.num2str(z0) + "m")
    mplot.ylabel("B (T)")
    mplot.legend("B_r", "B_z", "B")
    mplot.legend("boxoff")
    mplot.xlim(mat.row(0, rw))

    mplot.set_(mplot.gcf(), "currentaxes", hAxes3)
    mplot.cla()

    Br2 = 0.0 * zz
    Bz2 = 0.0 * zz
    for jz in mat.mrange(1, nZ):
        out = mpath.call("fun_Bfield", r0, float(mat.el(zz, jz)), rl, zl, hl, wl,
                         Il, Nl, imethod)
        mat.set_el(Br2, jz, out[1])
        mat.set_el(Bz2, jz, out[2])
    B2 = np.sqrt(Br2 ** 2 + Bz2 ** 2)

    mplot.plot(zz, Br2, ".", zz, Bz2, "--", zz, B2, ":", "linewidth", 2)
    mplot.legend("B_r", "B_z", "B")
    mplot.legend("boxoff")
    mplot.xlabel("z (m), r=" + mat.num2str(r0) + "m")
    mplot.ylabel("B (T)")
    mplot.xlim(mat.row(Zlow, Zup))

    runtime = mat.cputime() - runtime

    mplot.set_(mplot.gcf(), "currentaxes", hAxes1)
    mplot.title("method=" + mat.num2str(imethod) + ", runtime=" + mat.num2str(runtime) + "s")

    mat.disp("Finish run fun_frc_coil.m.")

    return runtime, psi0, Br0, Bz0
