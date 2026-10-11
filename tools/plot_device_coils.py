r"""画出设计点 A 的线圈布置：位置、形状、尺寸、电流（安匝）。

几何取自 `tools/design_point.py` 用的那套 `icase=2` 装置（PFRC-2 量级的 RMF
维持型 FRC，见 `装置设计点_自定参数.md`）：

* 真空室 / 通量保持器：半径 \(R_w=0.14\) m、半长 \(z_w=0.75\) m；
* 线圈**三组，都在 \(R=0.15\) m**（在壁外 1 cm），径向厚 6 mm：
  中心一组轴向长 1.2 m（覆盖整个中心室），两端各一组 0.1 m；
* 三组**同向、等电流**：56.9 kA·安匝（按 20 匝的绕组算，导体电流 2.84 kA），
  在中平面壁面产生 0.0608 T 的真空场。

画法：横轴 \(z\)、纵轴 \(r\)（与论文图和演化动图同一套坐标）；线圈截面矩形里的
\(\odot\) 表示电流**出纸面**（\(+\theta\) 方向）——按右手定则，它就是产生 \(+z\)
方向磁场的那个方向。红色虚线是初态 FRC 的分离面，用来给尺寸做参照。

用法：`python tools/plot_device_coils.py`
产物：`outputs/design_point/coil_layout.png`
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
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
import design_point as dp     # noqa: E402

mstyle.install()

OUT_DIR = ROOT / "outputs" / "design_point"
FS = 13.0

#: 铜绕组的颜色（画线圈截面用）
COPPER = (0.72, 0.45, 0.20)


def main() -> int:
    s, _model, il_scale, _iz_scale = dp.build_design_model()
    rl = np.asarray(s["rl"], float)          # 线圈中心的 R
    zl = np.asarray(s["zl"], float)          # 线圈中心的 Z
    hl = np.asarray(s["hl"], float)          # 径向厚度
    wl = np.asarray(s["wl"], float)          # 轴向宽度
    amp_turns = np.asarray(s["Il"], float)   # 安匝
    Rw, zw = float(s["Rw"]), float(s["zw"])
    turns = 20.0

    # 初态分离面（从正向演化的第一帧取），给尺寸做个参照
    seam = None
    npz = OUT_DIR / "forward.npz"
    if npz.exists():
        d = np.load(npz, allow_pickle=True)
        psi0 = np.asarray(d["psi_all"], float)[0]
        rr, zz = np.asarray(d["rr"], float), np.asarray(d["zz"], float)
        Zg, Rg = np.meshgrid(zz, rr, indexing="ij")
        seam = (Zg, Rg, psi0.T)

    fig = plt.figure(figsize=(12.5, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.62],
                          width_ratios=[1.0, 1.05])
    ax = fig.add_subplot(gs[0, :])
    axz = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[1, 1])
    axt.axis("off")

    # ---- 真空室 / 通量保持器 -------------------------------------------------
    ax.add_patch(Rectangle((-zw, 0.0), 2 * zw, Rw, fill=False,
                           ec="0.35", lw=2.0, zorder=3))
    ax.text(-0.34, 0.105, "真空室（通量保持器）\n"
                          f"Rw = {Rw * 1e2:.0f} cm，半长 zw = {zw * 1e2:.0f} cm",
            ha="center", va="center", fontsize=FS * 0.85, color="0.15")
    ax.annotate("", xy=(-0.20, 0.132), xytext=(-0.47, 0.112),
                arrowprops=dict(arrowstyle="-", lw=1.0, color="0.4"))

    # ---- 线圈 ---------------------------------------------------------------
    for k, (r, z, h, w, at) in enumerate(zip(rl, zl, hl, wl, amp_turns), 1):
        ax.add_patch(Rectangle((z - w / 2, r - h / 2), w, h,
                               facecolor=COPPER, edgecolor="k", lw=0.7,
                               zorder=4))
        # 注意：标签里别混数学与汉字——matplotlib 的 mathtext 没有汉字字形
        # 端线圈的标注往里挪一点，免得顶到坐标框
        zc = z + (0.03 if z < -0.3 else -0.03 if z > 0.3 else 0.0)
        ax.text(zc, r + 0.008, f"{at / 1e3:.2f} kA·匝",
                ha="center", va="bottom", fontsize=FS * 0.9,
                color="0.1", zorder=6)
        if w > 0.2:      # 中间那组长，符号放两端，免得挤在中间
            for zz_ in (z - 0.25 * w, z + 0.25 * w):
                ax.text(zz_, r, r"$\odot$", ha="center", va="center",
                        fontsize=FS * 1.1, color="w", zorder=6)
        else:
            ax.text(z, r, r"$\odot$", ha="center", va="center",
                    fontsize=FS * 0.9, color="w", zorder=6)

    # ---- 初态分离面 ---------------------------------------------------------
    if seam is not None:
        ax.contour(seam[0], seam[1], seam[2], levels=[0.0], colors=mstyle.RED,
                   linewidths=2.0, linestyles="--", zorder=5)
        ax.text(0.0, 0.030, "初态 FRC 分离面 (t = 0)", ha="center",
                va="center", fontsize=FS * 0.85, color=mstyle.RED, zorder=6)

    ax.set_xlim(-zw - 0.02, zw + 0.02)
    ax.set_ylim(0, 0.192)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$z$ [m]")
    ax.set_ylabel(r"$r$ [m]")
    ax.set_title("设计点 A 的线圈布置（三组同向、等电流）")
    mstyle.style_axes(ax, fontsize=FS, fontweight="bold", linewidth=1.6,
                      force=True)

    # ---- 端线圈放大：把截面尺寸标出来 ---------------------------------------
    k = 2                                   # 右端那组（z = +0.70 m）
    axz.add_patch(Rectangle((zl[k] - wl[k] / 2, rl[k] - hl[k] / 2),
                            wl[k], hl[k], facecolor=COPPER, edgecolor="k",
                            lw=0.9, zorder=4))
    axz.axhline(Rw, color="0.35", lw=2.0, zorder=3)     # 真空室壁
    axz.text(zl[k] + wl[k] / 2 + 0.008, Rw, "壁 Rw", va="center",
             fontsize=FS * 0.8, color="0.35")
    axz.text(zl[k], rl[k], r"$\odot$", ha="center", va="center",
             fontsize=FS * 0.9, color="w", zorder=6)
    axz.annotate("", xy=(zl[k] - wl[k] / 2, rl[k] + 0.0075),
                 xytext=(zl[k] + wl[k] / 2, rl[k] + 0.0075),
                 arrowprops=dict(arrowstyle="<->", lw=1.2, color="k"))
    axz.text(zl[k], rl[k] + 0.0085, f"轴向 {wl[k] * 1e3:.0f} mm",
             ha="center", va="bottom", fontsize=FS * 0.8)
    # 径向尺寸：6 mm 太薄，直接用箭头看不见，先引两条引线再标
    zc = zl[k] + wl[k] / 2
    for off in (-hl[k] / 2, hl[k] / 2):
        axz.plot([zc, zc + 0.028], [rl[k] + off] * 2, color="k", lw=0.9,
                 zorder=5)
    axz.annotate("", xy=(zc + 0.028, rl[k] - hl[k] / 2),
                 xytext=(zc + 0.028, rl[k] + hl[k] / 2),
                 arrowprops=dict(arrowstyle="<->", lw=1.2, color="k"))
    axz.text(zc + 0.034, rl[k], f"径向 {hl[k] * 1e3:.0f} mm",
             ha="left", va="bottom", fontsize=FS * 0.8)
    axz.set_xlim(zl[k] - 0.085, zl[k] + 0.13)
    axz.set_ylim(0.132, 0.172)
    axz.set_aspect("equal", adjustable="box")
    axz.set_xlabel(r"$z$ [m]")
    axz.set_ylabel(r"$r$ [m]")
    axz.set_title("端线圈截面（放大）")
    mstyle.style_axes(axz, fontsize=FS * 0.85, fontweight="bold",
                      linewidth=1.4, force=True)

    # ---- 数字表（每行一项，不用等宽字体也能读）------------------------------
    # 注意：不能用 family="monospace"——DejaVu Sans Mono 没有汉字，中文会变方框
    lines = []
    for k, (r, z, h, w, at) in enumerate(zip(rl, zl, hl, wl, amp_turns), 1):
        lines.append(f"线圈 {k}：中心 (r, z) = ({r:.3f}, {z:+.3f}) m，"
                     f"截面 径向 {h * 1e3:.0f} mm × 轴向 {w * 1e3:.0f} mm，"
                     f"电流 {at / 1e3:.2f} kA·匝")
    lines.append(f"三个线圈同向、同电流、串成同一回路；{turns:.0f} 匝 → 导体电流 "
                 f"{amp_turns[0] / turns / 1e3:.2f} kA；"
                 f"在中平面壁面产生 {dp.B_WALL_PER_MA * il_scale:.4f} T 真空场")
    lines.append("截面里的 ⊙ = 电流出纸面（+θ 方向）；按右手定则，它产生的正是 "
                 "+z 方向的磁场")
    axt.text(0.0, 1.0, "\n".join(lines), ha="left", va="top",
             fontsize=FS * 0.9)

    out = OUT_DIR / "coil_layout.png"
    fig.savefig(out, dpi=120)
    print(f"[coils] 产物：{out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
