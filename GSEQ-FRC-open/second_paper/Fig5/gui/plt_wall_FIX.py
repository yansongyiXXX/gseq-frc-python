"""plt_wall_FIX.m 的移植：FIX 构型（两个 FRC 拼起来）的壁面 ψ，并把它存下来。

源文件：`gseq-main/GSEQ-FRC-open/second_paper/Fig5/gui/plt_wall_FIX.m`

`zw=3.5; rw=1;`、`wall=1`（tanh 壁面，`ra=0.3`、`rb=0.4`、`zm=0.6*zw`），
存 `psib_FIX.mat`（当前目录 3 个变量、上级目录再加 `Bz` 共 4 个）——
**仓库里 `Fig5/gui/psib_FIX.mat` 就是它当年的产物**（`rb∈[0.3,0.4]`、
`zb∈[-3.5,3.5]` 那一份），测试里逐值对照。

末尾还有三段诊断打印（找两个 FRC 中平面的位置与磁场）：
MATLAB 的 `%5.4d` 遇到非整数会退回 `%e` 打印（本机实测），移植版按同样格式印。

**是 GUI 的会话脚本**（要用 `hAxes1/hAxes2` 与线圈表），单跑会停。

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


def _matlab_int_text(value, precision: int = 4) -> str:
    """MATLAB 的 `fprintf('%5.4d', x)`：x 不是整数时退回 `%e` 打印。"""
    number = float(np.asarray(value).ravel()[0])
    if number == int(number):
        return f"{int(number):d}"
    return f"{number:.{precision}e}"


def run(ws) -> None:
    missing = [name for name in NEEDS_WORKSPACE if name not in ws.keys()]
    if missing:
        raise KeyError(
            f"plt_wall_FIX 是 GUI 的会话脚本，缺少：{', '.join(missing)}"
            "（先跑 frc_coil_gui.m，或把它的 frc_coil.mat 与两个坐标区句柄喂进来）")

    hAxes1 = ws["hAxes1"]; hAxes2 = ws["hAxes2"]
    rl = ws["rl"]; zl = ws["zl"]; hl = ws["hl"]; wl = ws["wl"]
    Il = ws["Il"]; Nl = ws["Nl"]
    imethod = int(mat.scalar_value(ws["imethod"]))

    zw = mat.scalar(3.5); rw = mat.scalar(1)

    zz = mat.colon(-mat.scalar_value(zw), 0.01 * mat.scalar_value(zw),
                   mat.scalar_value(zw))

    ind1 = mat.find(zz < 0)
    ind2 = mat.find(zz >= 0)
    # % a=0.15; zm=0.75*zw; ra=0.4*rw; rb=0.7*rw;
    a = mat.scalar(0.15); zm = 0.6 * zw; ra = 0.3 * rw; rb = 0.4 * rw
    # % rr(ind1)=0.15*(tanh(5*(zz(ind1)+0.8*zw))+1.0)+0.4;
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
    for jz in mat.mrange(1, nz):
        out = mpath.call("fun_Bfield", float(mat.el(rr, jz)), float(mat.el(zz, jz)),
                         rl, zl, hl, wl, Il, Nl, imethod)
        mat.set_el(psi, jz, out[0])
        mat.set_el(Br, jz, out[1])
        mat.set_el(Bz, jz, out[2])

    if mat.scalar_value(plt) != 3:
        mplot.hold_on()
        mplot.plot(zz, psi, "linewidth", 2)

    rb = rr; zb = zz; psib = psi
    # 原文两次 save（当前目录 3 个变量、`../` 再加 `Bz`）；移植版都写进产出目录，
    # 两次落在同一个文件上，最终内容与 `../psib_FIX.mat` 那份一致
    mio.save("psib_FIX.mat", zb=zb, rb=rb, psib=psib)
    mio.save("psib_FIX.mat", zb=zb, rb=rb, psib=psib, Bz=Bz)
    Z0 = int((np.size(psi, 1) + 1) / 2)
    # % fprintf('psib=%17.16f \n',psi(65));

    No = mat.find((zz > -3) & (zz < 0))
    Bz_min_no = mat.find(Bz == mat.mmin(mat.take(Bz, No)))
    print("the position of mid-plane of two FRC: %s "
          % _matlab_int_text(mat.el(Bz_min_no, 1)))
    print("the axial position of mid-plane of two FRC: %s "
          % _matlab_int_text(mat.el(zz, int(mat.el(Bz_min_no, 1)))))
    print("the magnetic field of mid-plane of two FRC: %s "
          % _matlab_int_text(mat.el(Bz, int(mat.el(Bz_min_no, 1)))))

    jN0 = int(np.floor(0.5 * nz) - np.floor(1.5 / (0.01 * mat.scalar_value(zw))))
    print("psib=%17.16f " % float(mat.el(psi, jN0)))

    for name, value in (("zw", zw), ("rw", rw), ("zz", zz), ("ind1", ind1),
                        ("ind2", ind2), ("a", a), ("zm", zm), ("ra", ra),
                        ("rb", rb), ("wall", wall), ("plt", plt), ("nz", mat.scalar(nz)),
                        ("psi", psi), ("Br", Br), ("Bz", Bz), ("zb", zb),
                        ("psib", psib), ("Z0", mat.scalar(Z0)), ("No", No),
                        ("Bz_min_no", Bz_min_no), ("jN0", mat.scalar(jN0))):
        ws[name] = value
