r"""把设计点 A 的线圈画出来：位置、形状与尺寸、电流大小和方向。

数据来源（都是现成产物，不重跑求解器）：

* `outputs/design_point/forward.npz`：线圈几何 `rl/zl/hl/wl`（R、Z、径向厚度、
  轴向长度）与壁 `Rw`、计算域 `zw`，以及第 0 帧的 \(\psi\) 场（用来叠分界面）；
* `outputs/design_point/design_point.json`：`engineering` 里的线圈安匝数、匝数、
  导体电流、真空壁场。

角度约定：横轴 Z、纵轴 R；\(\hat\theta\) 指向纸面外，所以"电流流出纸面（⊙）"
就是 \(I_\theta>0\)，它产生 \(+Z\) 方向的磁场——与本算例 \(B_e=+0.05\) T 一致
（芯部那个 \(-0.0316\) T 是等离子体自己反向出来的）。

用法：

    python tools/show_coils.py            # 产物 outputs/design_point/coil_layout.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from gseqfrc import mstyle  # noqa: E402

mstyle.install()

OUT_DIR = ROOT / "outputs" / "design_point"
FS = 12.5


def separatrix(Z, R, psi):
    """取 \(\psi=0\) 里"拱起来"的那条（分界面），丢掉贴在 R=0 上的零基准段。"""
    cs = plt.contour(Z, R, psi.T, levels=[0.0])
    out = []
    for seg in cs.allsegs[0]:
        if len(seg) == 0 or np.all(seg[:, 1] < 5e-4):
            continue                      # 整段都在几何轴上 -> 那是 ψ 的零基准
        out.append(seg)
    cs.remove()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", default=str(OUT_DIR / "forward.npz"))
    ap.add_argument("--json", default=str(OUT_DIR / "design_point.json"))
    ap.add_argument("--out", default=str(OUT_DIR / "coil_layout.png"))
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    rl = np.asarray(d["rl"], float)
    zl = np.asarray(d["zl"], float)
    hl = np.asarray(d["hl"], float)          # 径向厚度
    wl = np.asarray(d["wl"], float)          # 轴向长度
    Rw, zw = float(d["Rw"]), float(d["zw"])
    psi = np.asarray(d["psi_all"], float)[args.frame]
    rr, zz = np.asarray(d["rr"], float), np.asarray(d["zz"], float)
    eng = json.loads(Path(args.json).read_text(encoding="utf-8"))["engineering"]
    turns = float(eng["coil_turns"])
    amp_turn = float(eng["I_amp_turn_A"])
    i_cond = float(eng["I_conductor_A"])
    b_wall = float(eng["B_wall_vacuum_T"])
    cond = float(eng["coil_cond_m2"])
    dg = json.loads(Path(args.json).read_text(encoding="utf-8"))["solve"]

    fig = plt.figure(figsize=(13.0, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.22])
    ax = fig.add_subplot(gs[0])
    axn = fig.add_subplot(gs[1]); axn.axis("off")

    # ---- 分界面（第 0 帧）作参照 -----------------------------------------
    Zg, Rg = np.meshgrid(zz, rr, indexing="ij")
    for seg in separatrix(Zg, Rg, psi):
        ax.plot(seg[:, 0], seg[:, 1], color=mstyle.RED, lw=1.8, ls="--",
                zorder=1)
    ax.fill([-zw, zw, zw, -zw], [0, 0, Rw, Rw], color="0.96", zorder=0)

    # ---- 壁与计算域 ------------------------------------------------------
    ax.plot([-zw, zw], [Rw, Rw], color="k", lw=2.0, zorder=3)
    for s in (-1, 1):
        ax.plot([s * zw, s * zw], [0, Rw], color="k", lw=2.0, zorder=3)
    ax.text(0.0, Rw + 0.0035, r"vessel wall  $R_w$ = 14.0 cm",
            ha="center", va="bottom", fontsize=FS * 0.9)

    # ---- 线圈 ------------------------------------------------------------
    for k, (rc, zc, h, w) in enumerate(zip(rl, zl, hl, wl)):
        ax.add_patch(Rectangle((zc - w / 2, rc - h / 2), w, h,
                               facecolor="0.45", edgecolor="k", lw=1.2,
                               zorder=4))
        ax.annotate(rf"coil {k + 1}:  {amp_turn / 1e3:.1f} kA$\cdot$turn  "
                    r"$\odot$",
                    xy=(zc, rc + h / 2), xytext=(zc, rc + 0.0125),
                    ha="center", va="bottom", fontsize=FS * 0.9,
                    arrowprops=dict(arrowstyle="-", lw=1.0, color="0.3"))
    # 尺寸标注
    ax.annotate("", xy=(-wl[1] / 2, rl[1] + 0.006), xytext=(wl[1] / 2, rl[1] + 0.006),
                arrowprops=dict(arrowstyle="<->", lw=1.0, color="0.25"))
    ax.text(0.0, rl[1] + 0.0075, "L = 1.20 m", ha="center", va="bottom",
            fontsize=FS * 0.85)
    for k, s in ((0, -1), (2, 1)):
        ax.annotate("", xy=(zl[k] - wl[k] / 2, 0.135), xytext=(zl[k] + wl[k] / 2, 0.135),
                    arrowprops=dict(arrowstyle="<->", lw=1.0, color="0.25"))
        ax.text(zl[k], 0.138, "0.10 m", ha="center", va="bottom", fontsize=FS * 0.8)
    ax.annotate("", xy=(zw + 0.02, Rw), xytext=(zw + 0.02, rl[0]),
                arrowprops=dict(arrowstyle="<->", lw=1.0, color="0.25"))
    ax.text(zw + 0.025, (Rw + rl[0]) / 2, "1 cm\n(间隙)", ha="left", va="center",
            fontsize=FS * 0.8)

    ax.set_xlim(-zw - 0.06, zw + 0.12)
    ax.set_ylim(0, 0.176)
    ax.set_xlabel(r"$Z$ [m]")
    ax.set_ylabel(r"$R$ [m]")
    ax.set_title("Coil set of design point A   "
                 r"($B_e$ = 0.05 T, wall $R_w$ = 0.14 m; "
                 "dashed red = plasma separatrix at t = 0)")
    mstyle.style_axes(ax, fontsize=FS, fontweight="bold", linewidth=1.6,
                      force=True)

    # ---- 右下角放大：端部线圈的截面 --------------------------------------
    axz = ax.inset_axes([0.60, 0.06, 0.24, 0.34])
    w, h, rc, zc = wl[2], hl[2], rl[2], zl[2]
    axz.add_patch(Rectangle((zc - w / 2, rc - h / 2), w, h,
                            facecolor="0.45", edgecolor="k", lw=1.0))
    axz.set_xlim(zc - 0.09, zc + 0.09)
    axz.set_ylim(rc - 0.010, rc + 0.010)
    axz.set_title("端部线圈截面", fontsize=FS * 0.75)
    axz.annotate("", xy=(zc - w / 2, rc + 0.0035), xytext=(zc + w / 2, rc + 0.0035),
                 arrowprops=dict(arrowstyle="<->", lw=0.9, color="0.25"))
    axz.text(zc, rc + 0.0042, "0.10 m", ha="center", va="bottom",
             fontsize=FS * 0.7)
    axz.annotate("", xy=(zc + w / 2 + 0.006, rc - h / 2),
                 xytext=(zc + w / 2 + 0.006, rc + h / 2),
                 arrowprops=dict(arrowstyle="<->", lw=0.9, color="0.25"))
    axz.text(zc + w / 2 + 0.010, rc, "6 mm", ha="left", va="center",
             fontsize=FS * 0.7)
    mstyle.style_axes(axz, fontsize=FS * 0.7, fontweight="bold",
                      linewidth=1.2, force=True)

    # ---- 表头信息 --------------------------------------------------------
    kind = ("同轴螺线管段" if wl[1] > 0.5 else "环")
    # 含中文的行一律用纯文本（mathtext 里的 Computer Modern 没有汉字字形，
    # 混在一句话里那半句会变方框）
    lines = [
        "三只线圈同轴、同电流方向（⊙ = 电流流出纸面，即 Iθ > 0，"
        "产生 +Z 方向的磁场，也就是 Be）：",
        "  • 线圈 1 / 3（端部）：Z = ∓0.70 m，轴向 0.10 m，径向 6 mm，"
        "R = 0.150 m",
        f"  • 线圈 2（中部）：Z = 0，轴向 1.20 m，径向 6 mm，R = 0.150 m"
        f"（{kind}）",
        f"  • 每只安匝：{amp_turn:,.0f} A·turn = {amp_turn / 1e3:.1f} kA·turn；"
        f"折 {turns:.0f} 匝，导体电流 {i_cond:,.0f} A"
        f"（导体 {np.sqrt(cond) * 1e3:.0f} mm 见方）",
        f"  • 真空壁场 {b_wall:.4f} T；线圈在器壁外 1 cm；"
        f"本时刻等离子体 Rs = {dg['Rs_mid_m'] * 1e2:.2f} cm、"
        f"芯部 Bz0 = {dg['Bz_axis_T']:+.4f} T（反向场）",
        "  • 本算例中三只线圈电流全程恒定：演化由 RMF 驱动承担，"
        "线圈不参与反馈",
    ]
    axn.text(0.0, 1.0, "\n".join(lines), ha="left", va="top",
             fontsize=FS * 0.92, transform=axn.transAxes, linespacing=1.45)

    fig.savefig(args.out, dpi=120)
    print(f"[coils] 产物：{args.out}")
    for k, (rc, zc, h, w) in enumerate(zip(rl, zl, hl, wl), 1):
        print(f"  线圈 {k}: R={rc:.3f} m  Z={zc:+.3f} m  轴向 {w:.3f} m  "
              f"径向 {h * 1e3:.1f} mm  安匝 {amp_turn / 1e3:.2f} kA·turn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
