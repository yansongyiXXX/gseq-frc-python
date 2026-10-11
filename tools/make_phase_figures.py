r"""形成段与稳定控制段：两张分开的静态汇总图。

仓库里原来是**一张合并的全链动图**（`make_full_chain_animation.py`）。这个工具把
它拆成两张可以直接放进幻灯片的静态图：

* `formation_phase.png` —— **形成段**：θ 箍缩 0D 形成 → 二维交接。
  上排是交接态的二维 \(\psi(R,Z)\)（含装置线圈与壁），下排五格是 0D 形成历史：
  主币线圈电流 / 轴上净场 / 反场因子 / 通量分解 / 轴向长度。
* `sustainment_phase.png` —— **稳定控制段**：平移 → 压缩 → 平顶。
  **左侧竖排**三个相末的二维 \(\psi\) 快照（横排会把每张压得太窄），
  右侧一列是位置与长度跟踪 / 两个控制量 / 私通量。

数据来源（都要先跑过）：

    python tools/frc_formation_0d.py
    python tools/frc_formation_handoff.py
    python tools/frc_discharge_run.py --rmf

用法：

    python tools/make_phase_figures.py              # 两张都出
    python tools/make_phase_figures.py --only formation
    python tools/make_phase_figures.py --only sustainment

产物：`outputs/frc_formation/{formation_phase,sustainment_phase}.png`。

**一处渲染注意**：matplotlib 里中文和 `$...$` 数学混在同一条字符串会把中文画成
方框，所以这里标题一律纯中文、数学只留在轴标签（与 `make_full_chain_animation.py`
同一条规矩）。
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
sys.path.insert(0, str(ROOT / "tools"))
os.chdir(ROOT)
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import SymLogNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from gseqfrc import mstyle  # noqa: E402

mstyle.install()

OUT = ROOT / "outputs" / "frc_formation"
FS = 13.0


def load_products():
    need = ("formation0d.npz", "handoff.npz", "handoff_psi.npz",
            "discharge_psi.npz", "discharge.json")
    missing = [n for n in need if not (OUT / n).is_file()]
    if missing:
        raise SystemExit(
            "[phase] 缺少产物：" + "、".join(missing)
            + "\n        先跑 tools/frc_formation_0d.py、"
              "tools/frc_formation_handoff.py、tools/frc_discharge_run.py --rmf")
    f = np.load(OUT / "formation0d.npz", allow_pickle=True)
    h = np.load(OUT / "handoff.npz", allow_pickle=True)
    hp = np.load(OUT / "handoff_psi.npz")
    d = np.load(OUT / "discharge_psi.npz")
    dj = json.loads((OUT / "discharge.json").read_text(encoding="utf-8"))
    return f, h, hp, d, dj


def make_norm(psi_list, linscale=3.0):
    """跨帧固定的 symlog 归一化 + 等值线层级（与全链动图同一套）。"""
    pmin = min(float(p.min()) for p in psi_list)
    pmax = max(float(p.max()) for p in psi_list)
    linthresh = max(abs(pmin), 1e-30)
    norm = SymLogNorm(linthresh, linscale=linscale, vmin=1.02 * pmin,
                      vmax=1.02 * pmax, base=10)
    core = np.linspace(-linthresh, 0.0, 9)[:-1]
    pos = np.geomspace(linthresh, pmax, 15)
    return norm, core, pos, pmin, pmax


def add_colorbar(fig, norm, rect, pmin, pmax):
    cax = fig.add_axes(rect)
    sm = plt.cm.ScalarMappable(cmap="parula", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$\psi$  [$10^{-5}$ Wb/(2$\pi$)]" "\n(symlog)",
                   fontsize=FS * 0.85, fontweight="bold", labelpad=6)
    units = np.array([-1.5, -1.0, 0.0, 1.0, 10.0, 100.0])
    keep = units[(units * 1e-5 >= 1.02 * pmin) & (units * 1e-5 <= 1.02 * pmax)]
    cbar.set_ticks(keep * 1e-5)
    cbar.set_ticklabels([f"{u:g}" for u in keep])
    cbar.ax.tick_params(labelsize=FS * 0.75)
    for text in cbar.ax.get_yticklabels():
        text.set_fontweight("bold")
    cbar.outline.set_linewidth(1.2)
    return cbar


def coil_drawer(hp):
    """装置线圈 + 壁（几何取交接态产物，两段工况共用同一套线圈）。"""
    rl = np.asarray(hp["rl"], float)
    zl = np.asarray(hp["zl"], float)
    hl = np.asarray(hp["hl"], float)
    wl = np.asarray(hp["wl"], float)
    Rw = float(hp["Rw"])
    zw = float(hp["zw"])

    def draw(ax, labels=True):
        for r, z, hh, w in zip(rl, zl, hl, wl):
            ax.add_patch(Rectangle((z - w / 2, r - hh / 2), w, hh,
                                   fill=False, ec="0.15", lw=1.6))
        ax.plot([-zw, zw], [Rw, Rw], color="k", lw=1.8)
        if labels:
            for lbl, z in zip(("端线圈", "中心线圈", "端线圈"), zl):
                ax.annotate(lbl, (z, 0.158), ha="center", va="bottom",
                            fontsize=FS * 0.62, color="0.35")
        ax.axhline(0.0, color="0.85", lw=0.8, ls="--")
        ax.set_xlim(-zw, zw)
        ax.set_ylim(0, 0.17)
        ax.set_xlabel(r"$z$ [m]")
        ax.set_ylabel(r"$r$ [m]")

    return draw


def draw_map(ax, psi2, zz, rr, norm, core, pos, draw_coils, title,
             coil_labels=True, show_y=True):
    Z, R = np.meshgrid(zz, rr, indexing="ij")
    pT = psi2.T
    ax.pcolormesh(Z, R, pT, cmap="parula", norm=norm, shading="auto")
    ax.contour(Z, R, pT, levels=pos, colors="k", linewidths=0.4, alpha=0.35)
    ax.contour(Z, R, pT, levels=core, colors="k", linewidths=0.6, alpha=0.7)
    ax.contour(Z, R, pT, levels=[0.0], colors=mstyle.RED,
               linewidths=2.0, linestyles="--")
    iR, iZ = np.unravel_index(int(np.argmin(psi2)), psi2.shape)
    ax.plot(zz[iZ], rr[iR], marker="x", color=mstyle.MAGENTA, ms=9, mew=2.2,
            linestyle="none")
    draw_coils(ax, labels=coil_labels)
    if not show_y:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    ax.set_title(title, fontsize=FS)


def style_traces(axes, xlabel):
    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.grid(True, **mstyle.GRID_STYLE)


def figure_formation(f, h, hp, out_png, dpi=120):
    t = np.asarray(f["t"], float)
    Ic = np.asarray(f["Ic"], float)
    B_app = np.asarray(f["B_app"], float)
    F = np.asarray(f["F"], float)
    Phi_open = np.asarray(f["Phi_open"], float)
    Phi_closed = np.asarray(f["Phi_closed"], float)
    l_s = np.asarray(f["l_s"], float)

    t_form = float(h["t_form"])
    Phi_h = float(h["Phi"])
    Rs_h = float(hp["Rs"])
    elong = float(hp["elongation"])
    ls_eq = float(h["ls_equil"])
    F_form = float(h["F_at_form"])

    psi_h = np.asarray(hp["psi"], float)
    zz = np.asarray(hp["zz"], float)
    rr = np.asarray(hp["rr"], float)
    draw_coils = coil_drawer(hp)

    fig = plt.figure(figsize=(14.0, 8.8))
    ax_map = fig.add_axes([0.055, 0.525, 0.775, 0.425])
    norm, core, pos, pmin, pmax = make_norm([psi_h])
    add_colorbar(fig, norm, [0.845, 0.525, 0.013, 0.425], pmin, pmax)
    draw_map(ax_map, psi_h, zz, rr, norm, core, pos, draw_coils,
             f"形成段：θ 箍缩形成 → 二维交接（交接 t_form = {t_form * 1e6:.2f} µs）")

    xs = (0.055, 0.243, 0.431, 0.619, 0.807)
    axs = [fig.add_axes([x, 0.135, 0.168, 0.275]) for x in xs]
    xm = t * 1e6
    tmax = t_form * 1e6 * 1.05

    axs[0].plot(xm, Ic / 1e3, color=mstyle.BLUE, lw=2.2)
    axs[0].axvline(t_form * 1e6, color="0.6", lw=1.0, ls=":")
    axs[0].set_ylabel(r"$I_c$ [kA]")
    axs[0].set_title("形成线圈电流（主币 RLC）", fontsize=FS * 0.85)

    axs[1].plot(xm, B_app, color="C3", lw=2.2)
    axs[1].axhline(0.0, color="0.6", lw=1.0, ls=":")
    axs[1].axvline(t_form * 1e6, color="0.6", lw=1.0, ls=":")
    axs[1].set_ylabel(r"$B_{z0}$ [T]")
    axs[1].set_title("轴上净场（过零即反转）", fontsize=FS * 0.85)

    axs[2].plot(xm, F, color=mstyle.GREEN, lw=2.2)
    axs[2].axhline(1.0, color="0.6", lw=1.0, ls=":")
    axs[2].axvline(t_form * 1e6, color="0.6", lw=1.0, ls=":")
    axs[2].set_ylabel("F")
    axs[2].set_title("反场因子 F（点线 F=1）", fontsize=FS * 0.85)

    axs[3].plot(xm, Phi_open * 1e3, color=mstyle.BLUE, lw=2.2, label="开场线通量")
    axs[3].plot(xm, Phi_closed * 1e3, color=mstyle.RED, lw=2.6, label="闭合私通量")
    axs[3].axvline(t_form * 1e6, color="0.6", lw=1.0, ls=":")
    axs[3].set_ylabel(r"$\Phi$ [mWb]")
    axs[3].legend(fontsize=FS * 0.7, frameon=False)
    axs[3].set_title("通量分解（重联）", fontsize=FS * 0.85)

    axs[4].plot(xm, l_s * 1e2, color=mstyle.BLUE, lw=2.2)
    axs[4].axhline(ls_eq * 1e2, color="0.6", lw=1.0, ls=":")
    axs[4].axvline(t_form * 1e6, color="0.6", lw=1.0, ls=":")
    axs[4].set_ylabel(r"$l_s$ [cm]")
    axs[4].set_title("轴向长度 l_s（点线：平衡）", fontsize=FS * 0.85)

    for ax in axs:
        ax.set_xlim(0, tmax)
    style_traces(axs, r"$t$ [$\mu$s]")

    fig.text(0.5, 0.035,
             f"t_form = {t_form * 1e6:.2f} µs      Φ = {Phi_h:.4e} Wb      "
             f"R_s = {Rs_h * 1e2:.2f} cm      伸长比 = {elong:.2f}      "
             f"l_s(平衡) = {ls_eq * 1e2:.2f} cm      F(t_form) = {F_form:.3f}",
             ha="center", va="center", fontsize=FS * 0.86)

    mstyle.style_figure(fig, fontsize=FS)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


def phase_runs(track):
    """把 track 里连续的相同 phase 合并成若干段：[名字, 起, 止]。"""
    runs = []
    for i, rec in enumerate(track):
        p = str(rec.get("phase", "")).strip()
        if not runs or runs[-1][0] != p:
            runs.append([p, i, i])
        else:
            runs[-1][2] = i
    return runs


def figure_sustainment(h, hp, d, dj, out_png, dpi=120):
    psi_d = np.asarray(d["psi"], float)          # (nt, nR, nZ)
    rr = np.asarray(d["rr"], float)
    zz = np.asarray(d["zz"], float)
    tt = np.asarray(d["t"], float)
    track = dj["track"]
    rmf = dj.get("rmf") or {}
    draw_coils = coil_drawer(hp)

    runs = phase_runs(track)
    if len(runs) >= 4:
        picks = [(runs[-3][2], runs[-3][0]), (runs[-2][2], runs[-2][0]),
                 (runs[-1][2], runs[-1][0])]
    else:
        n = len(tt)
        picks = [(n // 3, ""), (2 * n // 3, ""), (n - 1, "")]

    fig = plt.figure(figsize=(14.0, 12.6))
    rows = (0.730, 0.445, 0.160)          # 三行：平移末 / 压缩末 / 平顶末
    mh = 0.215
    ax_maps = [fig.add_axes([0.062, y, 0.520, mh]) for y in rows]
    norm, core, pos, pmin, pmax = make_norm([psi_d[i] for i, _ in picks])
    add_colorbar(fig, norm, [0.596, rows[-1], 0.011, rows[0] + mh - rows[-1]],
                 pmin, pmax)

    for k, (ax, (i, name)) in enumerate(zip(ax_maps, picks)):
        head = f"{name}末  " if name else ""
        draw_map(ax, psi_d[i], zz, rr, norm, core, pos, draw_coils,
                 f"{head}t = {tt[i] * 1e3:.2f} ms",
                 coil_labels=(k == 0))

    axs = [fig.add_axes([0.735, y, 0.235, mh]) for y in rows]
    xm = tt * 1e3
    zc = np.array([r["zc"] for r in track], float) * 1e2
    zc_s = np.array([r["zc_star"] for r in track], float) * 1e2
    ls = np.array([r["ls"] for r in track], float) * 1e2
    ls_s = np.array([r["ls_star"] for r in track], float) * 1e2
    phi = np.array([r["Phi"] for r in track], float) * 1e3
    amp = np.array([r["a"] for r in track], float) / 1e3
    i2 = np.array([r["i2"] for r in track], float)
    di2 = (i2 - i2[0]) / 1e3

    axs[0].plot(xm, zc, "o-", color=mstyle.BLUE, lw=2.2, ms=5, label="位置 z_c")
    axs[0].plot(xm, zc_s, "--", color="0.55", lw=1.4, label="位置目标")
    axs[0].plot(xm, ls, "s-", color="C2", lw=2.2, ms=5, label="长度 l_s")
    axs[0].plot(xm, ls_s, "--", color="0.55", lw=1.4, label="长度目标")
    axs[0].set_ylabel("[cm]")
    axs[0].legend(fontsize=FS * 0.66, frameon=False, loc="center right")
    axs[0].set_title("位置与长度的三段跟踪", fontsize=FS * 0.85)

    axs[1].plot(xm, amp, "o-", color=mstyle.MAGENTA, lw=2.2, ms=5,
                label="差模 a（管位置）")
    axs[1].plot(xm, di2, "s-", color=mstyle.BLUE, lw=2.2, ms=5,
                label="中心线圈 ΔI2（管长度）")
    axs[1].axhline(0.0, color="0.7", lw=1.0)
    axs[1].set_ylabel("[kA]")
    axs[1].legend(fontsize=FS * 0.66, frameon=False, loc="center right")
    axs[1].set_title("两个控制量（近正交）", fontsize=FS * 0.85)

    axs[2].plot(xm, phi, "o-", color="C2", lw=2.2, ms=5)
    axs[2].set_ylabel(r"$\Phi$ [mWb]")
    axs[2].set_title("私通量（RMF 托住）", fontsize=FS * 0.85)

    for ax in axs:
        for xv in (0.1, 0.2):
            ax.axvline(xv, color="0.8", lw=0.8, ls=":")
    style_traces(axs, r"$t$ [ms]")

    B_w_mT = float(rmf.get("B_omega_T", float("nan"))) * 1e3
    f_hz = float(rmf.get("f_hz", float("nan"))) / 1e6
    fig.text(0.5, 0.040,
             f"Φ末/Φ0 = {float(dj['flux_ratio']):.3f}      "
             f"位置误差 ≤ {float(dj['max_zc_err_cm']):.3f} cm      "
             f"长度误差 ≤ {float(dj['max_ls_err_cm']):.2f} cm      "
             f"RMF：B_ω = {B_w_mT:.1f} mT、f = {f_hz:.2f} MHz",
             ha="center", va="center", fontsize=FS * 0.86)

    mstyle.style_figure(fig, fontsize=FS)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=("both", "formation", "sustainment"),
                    default="both", help="只出其中一张时用")
    ap.add_argument("--dpi", type=int, default=120)
    args = ap.parse_args()

    f, h, hp, d, dj = load_products()
    made = []
    if args.only in ("both", "formation"):
        made.append(figure_formation(f, h, hp, OUT / "formation_phase.png",
                                     dpi=args.dpi))
    if args.only in ("both", "sustainment"):
        made.append(figure_sustainment(h, hp, d, dj,
                                       OUT / "sustainment_phase.png",
                                       dpi=args.dpi))
    for path in made:
        print(f"[phase] 产物：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
