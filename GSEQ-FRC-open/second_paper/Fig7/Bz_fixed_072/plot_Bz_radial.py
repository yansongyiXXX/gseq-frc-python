"""plot_Bz_radial.m（Bz_fixed_072 副本）的移植：把 B_z 沿半径画成一条曲线。

**会话链的末端**：脚本自己只做 `load('MRR1d.mat')` 与一段 `interp1` 插值，然后用
**调用方工作区里**的二维 `Z`、`Bz`（以及 `R`、`psi`、`zw`）——这些是上游
`figure_*_1.m` / `plt_psi_p.m` 那几个诊断图脚本算出来的（它们自己用中心差分造出
二维 `Bz`）。MATLAB 里的用法就是"跑完求解器 → 跑诊断图 → 跑这个"。

⚠️ 单跑这一份会停：工作区里没有二维 `Bz`（求解器留下的是 `Bz1d`/`Bz_coil`，
或干脆没有这个名字）。移植版照抄这个依赖（缺就在开头报清楚），
测试里按会话链喂工作区（见 `tests/test_plotting_scripts.py`）。

原文活跃的代码没几行，其余（分界面、腔体、线圈那些）都是注释——照抄。
验收：§7「绘图脚本」那一节的三条。
"""

from __future__ import annotations

import numpy as np

from gseqfrc import matlab_compat as mat
from gseqfrc import mio
from gseqfrc import mplot

#: 需要在调用方工作区里已经有（上游诊断图脚本留下的二维量）
NEEDS_WORKSPACE = ("Z", "Bz", "R", "psi")


def run(ws) -> None:
    missing = [name for name in NEEDS_WORKSPACE if name not in ws.keys()]
    if missing:
        raise KeyError(
            f"plot_Bz_radial 是会话链的末端（求解器 → 诊断图 → 本脚本），缺少："
            f"{', '.join(missing)}")

    Z = np.asarray(ws["Z"])
    Bz = np.asarray(ws["Bz"])
    R = np.asarray(ws["R"])
    psi = np.asarray(ws["psi"])

    mplot.close_all()  # close all;
    fontsize = mat.scalar(25)
    mplot.figure("unit", "normalized", "DefaultAxesFontSize", fontsize,
                 "DefaultAxesFontWeight", "bold", "DefaultAxesLineWidth", 2,
                 "defaulttextinterpreter", "latex",
                 "position", mat.row(0.1, 0.2, 0.28, 0.3), "color", mat.row(1, 1, 1))
    # % Bz_plasma=Bz(end,:);
    # % psi_plasma=psi(end,:);

    # % load('MRR1d_beta=0_6_delta=0_4.mat');
    loaded = mio.load("MRR1d.mat")
    mu0 = mat.scalar(4 * np.pi * 1e-7)

    # % axes('position',[0.11,0.75,0.85,0.23]);%fig-1
    ido = mat.find(psi == np.min(psi))
    indtmp = psi > 0
    psio = np.where(indtmp, np.nan, psi)
    indtmp = mat.find(psi < 0)
    psii = np.array(psi, copy=True)
    psii[np.asarray(psi < 0)] = np.nan
    # 下面这一整段在原文里全是注释（分界面、腔体、线圈的图都不画）
    # % contour(Z,R,psio,5,...); ... % text(...)

    # % axes('position',[0.58,0.41,0.38,0.26]); %fig-3
    Bi = np.zeros(int(np.size(Z, 0)))
    for i in mat.mrange(1, int(np.size(Z, 0))):     # for i=1:size(Z,1)
        Bi[i - 1] = mat.interp1(mat.take(Z, i), mat.take(Bz, i), -0.72)
    # % plot(zb,Bz,'r-',Z(1,:),Bz_plasma,'g--','linewidth',3);grid on;hold all;
    # % plot(Z(r_x,:),Bz(r_x,:),'c-','linewidth',3);grid on;hold all;
    # % plot(Z(r_x,:),Bz(r_x,:),'-','color',[0.5 0 0],'linewidth',3);grid on;hold all;
    # % plot(R(:,3)*100,Bz(:,3),'k-','linewidth',3);grid on;hold all;
    mplot.axes("position", mat.row(0.11, 0.14, 0.85, 0.83))  # fig-1
    mplot.plot(np.asarray(R)[:, 2] * 100, Bi, "m-", "linewidth", 3)
    mplot.grid_on()
    mplot.hold_on()
    mplot.set_(mplot.gca(), "Fontsize", fontsize)
    # % ylabel('$B_{z} \ at \ wall$','interpreter','latex','FontSize',fontsize);
    mplot.ylabel("$B_{z} \\ [T]$", "interpreter", "latex", "FontSize", fontsize)
    mplot.xlabel("$r [cm]$", "interpreter", "latex", "FontSize", fontsize)
    # % xlim([-0.75,0.75]);set(gca,'xtick',[-0.6:0.4:0.6]);set(gca,'ytick',[0:2:10]);
    mplot.xlim(mat.row(0, 14))
    mplot.set_(mplot.gca(), "xtick", mat.colon(0, 2, 14))
    mplot.ylim(mat.row(2, 6))
    mplot.set_(mplot.gca(), "ytick", mat.colon(2, 1, 6))
    mplot.set_(mplot.gca(), "GridLineStyle", ":", "GridColor", "k",
               "GridAlpha", 0.25)
    # % h=legend('$\mathbf{w/o \ plasma}$','$\mathbf{w \ plasma}$');...
    # % h=legend('$r=0.14$');...
    h = mplot.legend("$\\mathbf{fixed-boundary}$")
    mplot.set_(h, "interpreter", "latex", "FontSize", 28)
    mplot.legend("boxoff")
    # % text('Interpreter','latex','String','$(c)$','Position',[-0.7 9],...)

    for name, value in (("fontsize", fontsize), ("mu0", mu0), ("ido", ido),
                        ("indtmp", indtmp), ("psio", psio), ("psii", psii),
                        ("Bi", Bi)):
        ws[name] = value
    for name, value in loaded.items():
        ws[name] = value
