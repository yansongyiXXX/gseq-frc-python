"""fun_frc_coil.m（Fig4/gui 副本）的移植：算多线圈磁场并把三张图填好。

源文件：`gseq-main/GSEQ-FRC-open/second_paper/Fig4/gui/fun_frc_coil.m`

它是 GUI 的"运行"回调真正干活的地方：

1. `[psi0,Br0,Bz0]=fun_Bfield(r0,z0,...)` —— **物理部分**就在这里，用的是同目录的
   `fun_Bfield`（`psi` 带那个 `/2` 的 Fig4 副本，冻结基准 `u_Bfield_fig4` 钉的就是它）；
2. `hAxes1`：33×65 网格上逐点算 ψ 画填色/等值线（**逐点调 fun_Bfield**）；
3. `hAxes2` / `hAxes3`：沿 r、沿 z 的 Br/Bz/|B|。

⚠️ 原文有两处"自己写的坑"，照抄不修：

* 第 76-84 行把 `zw/rw/zz/rr` **重新赋值**成壁面画线用的向量，而且这段在
  `for jl=1:nl` 循环**体内**（第 85 行的 `end` 才是循环结束）——所以壁面线会画
  `nl` 遍；更要紧的是后面 `hAxes2` 用的 `rr` 已经变成"全 0.14 的 201 点向量"，
  于是那张图画的是 r=0.14 处的 B（x 轴全是 0.14）；
* `Br1=0.*rr` 长度是 201，而循环只填前 `nR=33` 个，其余留 0。

验收：§7「绘图/GUI 脚本」那一节的三条；物理部分另用 `u_Bfield_fig4` 的冻结值核对
（同一套线圈表/采样点下 `fun_frc_coil` 返回的 `psi0/Br0/Bz0` 必须与 `fun_Bfield`
的冻结结果一致）。**注意速度**：`imethod=1` 单点约 0.4 ms（整网格 ~1 s），
`imethod=2` 单点 0.76 s、`imethod=3` 单点 5.7 s——所以测试只跑 GUI 默认的 method 1。
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
        # 原文这里是裸 `return`（4 个输出都没赋值，取 4 个输出会报"输出参数未赋值"）。
        # 移植版把已经算出来的 runtime 留下、其余给 None，行为上"提示完就退出"。
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
            # `[psi(jr,jz),~,~]=fun_Bfield(...)`：只要第一个输出
            psi[jr - 1, jz - 1] = mpath.call(
                "fun_Bfield", float(mat.el(rr, jr)), float(mat.el(zz, jz)),
                rl, zl, hl, wl, Il, Nl, imethod)[0]
    psi[np.abs(psi) > 1e5] = np.nan
    # % contour(Z,R,psiw_coil);
    # % pcolor(Z,R,real(psi)); shading interp; hold on;
    mplot.contour(Z, R, np.real(psi), 100)
    mplot.hold_on()
    mplot.contour(Z, -R, np.real(psi), 100)
    mplot.hold_on()
    mplot.colorbar("east")

    for jl in mat.mrange(1, nl):
        mplot.rectangle("position",
                        mat.row(mat.el(zl, jl) - 0.5 * mat.el(wl, jl),
                                mat.el(rl, jl) - 0.5 * mat.el(hl, jl),
                                mat.el(wl, jl), mat.el(hl, jl)),
                        "FaceColor", mat.row(0.5, 0.5, 0.5), "linewidth", 1)
        mplot.hold_on()
        mplot.rectangle("position",
                        mat.row(mat.el(zl, jl) - 0.5 * mat.el(wl, jl),
                                -mat.el(rl, jl) - 0.5 * mat.el(hl, jl),
                                mat.el(wl, jl), mat.el(hl, jl)),
                        "FaceColor", mat.row(0.5, 0.5, 0.5), "linewidth", 1)
        mplot.hold_on()
        # % if(Il(jl)>0) plot(zl(jl),rl(jl),'r.',zl(jl),-rl(jl),'rx',...) （原文注释掉了）
        # ↓ 原文这一段在 for 循环**体内**（第 85 行的 end 才关循环），照抄
        zw = 0.75
        rw = 0.15
        zz = mat.colon(-zw, 0.01 * zw, zw)
        rr = 0.14 + 0.0 * zz
        rx = 0.0 * zz
        mplot.plot(zz, rr, "r-", "linewidth", 2)
        mplot.hold_on()
        mplot.plot(zz, -rr, "r-", "linewidth", 2)
        mplot.hold_on()
        mplot.plot(zz, rx, "k--", "linewidth", 2)
        mplot.hold_on()
        mplot.plot(mat.row(-0.4301, 0.4301), mat.row(0, 0), "r--", "linewidth", 4)
        mplot.hold_on()
        mplot.set_(mplot.gca(), "ytick", mat.colon(-0.15, 0.1, 0.15))
        mplot.set_(mplot.gca(), "xtick", mat.colon(-0.75, 0.25, 0.75))

    mplot.xlim(mat.row(Zlow, Zup))
    mplot.ylim(mat.row(-Rup, Rup))
    mplot.xlabel("$z (m)$", "interpreter", "latex", "FontSize", 25)
    mplot.ylabel("$r (m)$", "interpreter", "latex", "FontSize", 25)
    mplot.box("on")

    mplot.set_(mplot.gcf(), "currentaxes", hAxes2)
    mplot.cla()

    Br1 = 0.0 * rr
    Bz1 = 0.0 * rr
    for jr in mat.mrange(1, nR):
        out = mpath.call("fun_Bfield", float(mat.el(rr, jr)), z0, rl, zl, hl, wl,
                         Il, Nl, imethod)
        # `Br1(jr)=...` 是线性索引赋值（Br1 是 1×n 的行向量）
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
