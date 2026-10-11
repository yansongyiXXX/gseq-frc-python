r"""θ 箍缩全过程的磁通演化动图（含装置线圈）。

把仓库里现成的三份产物拼成动图：

* `outputs/frc_formation/formation0d.npz` —— 0D 形成历史（线圈电流、反场因子、
  开场线/闭合通量、轴向长度）；
* `outputs/frc_formation/handoff_psi.npz`（由 `tools/frc_formation_handoff.py`
  生成）—— 交接态的二维 \(\psi(R,Z)\)；
* `outputs/frc_formation/discharge_psi.npz` + `discharge.json`（由
  `tools/frc_discharge_run.py` 生成）—— 平移 → 压缩 → 平顶的逐步 \(\psi\)。

**前后两段分成两张动图**（`--split`，默认就是这个模式）：

* `formation_chain.gif` —— **前段**：θ 箍缩 0D 形成 → 二维交接。
  形成段没有闭合区、也没有二维磁面，所以上排画 0D 仪表盘（开场线/闭合通量、
  反场因子）；交接那几帧换成真正的 \(\psi\) 场。
* `discharge_chain.gif` —— **后段**：平移 → 压缩 → 平顶维持。
  全程是二维 \(\psi\) 场，**装置线圈（端/中心/端）与壁**都按真实几何画出。

（`--part combined` 仍可出原来那张把两段接在一起的 `full_chain.gif`。）

版式（画布 13×7.4 英寸）：上排整幅宽的二维面板；下排三格是随相位切换的
时序曲线 + 游标；最下一条窄带是当前工况数字。色标跨帧固定（symlog，同
`make_flux_animation.py`），并用**全局调色板**写 GIF，避免色条逐帧抖动。

**一处渲染注意**：matplotlib 在同一个字符串里混排中文与 `$...$` 数学时，中文
会取不到字形（画成方框）。所以这里的文字标签要么纯中文、要么纯数学，不混。

用法：

    python tools/frc_formation_0d.py
    python tools/frc_formation_handoff.py
    python tools/frc_discharge_run.py --rmf
    python tools/make_full_chain_animation.py            # 出前后两张（默认）
    python tools/make_full_chain_animation.py --part combined   # 出一张合并版
    python tools/make_full_chain_animation.py --part formation  # 只出前段

产物：`outputs/frc_formation/{formation_chain,discharge_chain}.gif`（+ 关键帧 PNG）。
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
from matplotlib.patches import Rectangle  # noqa: E402

from gseqfrc import mstyle  # noqa: E402
from make_flux_animation import write_gif  # noqa: E402

mstyle.install()

OUT = ROOT / "outputs" / "frc_formation"
FS = 13.0


def fmt(value, spec=".2f", dash="—") -> str:
    value = float(value)
    return dash if not np.isfinite(value) else format(value, spec)


def load_all():
    f = np.load(OUT / "formation0d.npz", allow_pickle=True)
    h = np.load(OUT / "handoff.npz", allow_pickle=True)
    hp = np.load(OUT / "handoff_psi.npz")
    d = np.load(OUT / "discharge_psi.npz")
    dj = json.loads((OUT / "discharge.json").read_text(encoding="utf-8"))
    return f, h, hp, d, dj


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--n-formation", type=int, default=34)
    ap.add_argument("--n-handoff", type=int, default=4)
    ap.add_argument("--part", choices=("formation", "discharge", "combined"),
                    default="combined", help="只出其中一段时用；默认由 --split 决定")
    ap.add_argument("--split", dest="split", action="store_true", default=True,
                    help="出前后两张动图（默认）")
    ap.add_argument("--no-split", dest="split", action="store_false",
                    help="等价于 --part combined：出一张合并版")
    ap.add_argument("--name", default=None, help="覆盖输出名（只出一张时用）")
    ap.add_argument("--linscale", type=float, default=3.0)
    ap.add_argument("--key-frames", default="auto")
    ap.add_argument("--formation2d", default="",
                    help="用二维形成段的产物当前段（给 outputs/frc_formation 里的"
                         "名字，如 formation2d / formation2d_hybrid）；"
                         "留空则用 0D 仪表盘")
    ap.add_argument("--f2-frames", type=int, default=30,
                    help="二维形成段抽多少帧")
    args = ap.parse_args()

    f, h, hp, d, dj = load_all()
    f2 = None
    f2_idx = None
    if args.formation2d:
        p2 = OUT / f"{args.formation2d}.npz"
        if not p2.exists():
            print(f"[full] 找不到 {p2}")
            return 1
        f2 = np.load(p2)
        n_f2 = int(np.asarray(f2["psi"]).shape[0])
        f2_idx = np.unique(np.linspace(0, n_f2 - 1, args.f2_frames).astype(int))
        print(f"[full] 二维形成段：{p2.name}，取 {f2_idx.size}/{n_f2} 帧")
    t0d = np.asarray(f["t"], float)
    Ic = np.asarray(f["Ic"], float)
    F0d = np.asarray(f["F"], float)
    Phi_open = np.asarray(f["Phi_open"], float)
    Phi_closed = np.asarray(f["Phi_closed"], float)
    ls0d = np.asarray(f["l_s"], float)
    t_form = float(h["t_form"])
    Phi_target = float(h["Phi"])
    Il_handoff = float(h["Il"])
    ls_eq_0d = float(h["ls_equil"])

    i_form = max(2, min(int(np.searchsorted(t0d, t_form, side="right")),
                        len(t0d) - 1))
    form_idx = np.unique(np.linspace(0, i_form, args.n_formation).astype(int))
    Phi_max_mWb = float(np.nanmax(np.maximum(Phi_open[:i_form + 1],
                                             Phi_closed[:i_form + 1]))) * 1e3

    # ---------------- 2D 数据 ----------------
    psi_h = np.asarray(hp["psi"], float)               # (nR, nZ)
    rr_h, zz_h = np.asarray(hp["rr"], float), np.asarray(hp["zz"], float)
    rl, zl = np.asarray(hp["rl"], float), np.asarray(hp["zl"], float)
    hl, wl = np.asarray(hp["hl"], float), np.asarray(hp["wl"], float)
    Rw, zw = float(hp["Rw"]), float(hp["zw"])
    Phi_h, Rs_h = float(hp["Phi"]), float(hp["Rs"])
    E_h = float(hp["elongation"])
    ls_h = 2.0 * E_h * Rs_h

    psi_d = np.asarray(d["psi"], float)                # (nt, nR, nZ)
    rr_d, zz_d = np.asarray(d["rr"], float), np.asarray(d["zz"], float)
    tt_d = np.asarray(d["t"], float)
    track = dj["track"]
    rmf = dj.get("rmf") or {}

    pmin = min(float(psi_h.min()), float(psi_d.min()))
    pmax = max(float(psi_h.max()), float(psi_d.max()))
    if f2 is not None:
        psi_f2_all = np.asarray(f2["psi"], float)
        pmin = min(pmin, float(psi_f2_all.min()))
        pmax = max(pmax, float(psi_f2_all.max()))
    else:
        psi_f2_all = None
    linthresh = abs(pmin)
    norm = matplotlib.colors.SymLogNorm(linthresh, linscale=args.linscale,
                                        vmin=1.02 * pmin, vmax=1.02 * pmax,
                                        base=10)
    core_levels = np.linspace(-linthresh, 0.0, 9)[:-1]
    pos_levels = np.geomspace(linthresh, pmax, 15)

    # ---------------- 帧序列（每帧：(kind, index, 下排模式)）----------------
    F_FORM = [("formation", int(i), "formation") for i in form_idx]
    F_2D = ([("f2d", int(k), "f2d") for k in f2_idx]
            if psi_f2_all is not None else [])
    F_HAND_F = [("handoff", 0, "formation")] * args.n_handoff   # 前段里的交接
    F_HAND_D = [("handoff", 0, "discharge")] * args.n_handoff   # 合并版里的交接
    F_DISCH = [("discharge", int(j), "discharge")
               for j in range(psi_d.shape[0])]
    head = F_2D if F_2D else F_FORM
    sets = {"formation": head + F_HAND_F,
            "discharge": list(F_DISCH),
            "combined": head + F_HAND_D + list(F_DISCH)}

    if args.split:
        jobs = [("formation", "formation_chain"), ("discharge", "discharge_chain")]
    else:
        part = args.part if args.part != "combined" else "combined"
        default = {"formation": "formation_chain", "discharge": "discharge_chain",
                   "combined": "full_chain"}[part]
        jobs = [(part, args.name or default)]

    # ---------------- 画布（显式定位，避开布局引擎）----------------
    fig = plt.figure(figsize=(13.0, 7.4))
    ax_map = fig.add_axes([0.055, 0.455, 0.775, 0.465])
    cax = fig.add_axes([0.842, 0.455, 0.014, 0.465])
    ax_a = fig.add_axes([0.075, 0.150, 0.215, 0.245])
    ax_b = fig.add_axes([0.385, 0.150, 0.215, 0.245])
    ax_c = fig.add_axes([0.695, 0.150, 0.215, 0.245])

    sm = plt.cm.ScalarMappable(cmap="parula", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$\psi$  [$10^{-5}$ Wb/(2$\pi$)]" "\n(symlog)",
                   fontsize=FS * 0.9, fontweight="bold", labelpad=6)
    units = np.array([-1.5, -1.0, 0.0, 1.0, 10.0, 100.0])
    keep_u = units[(units * 1e-5 >= 1.02 * pmin) & (units * 1e-5 <= 1.02 * pmax)]
    cbar.set_ticks(keep_u * 1e-5)
    cbar.set_ticklabels([f"{u:g}" for u in keep_u])
    cbar.ax.tick_params(labelsize=FS * 0.8)
    for _t in cbar.ax.get_yticklabels():
        _t.set_fontweight("bold")
    cbar.outline.set_linewidth(1.2)

    info = fig.text(0.5, 0.048, "", ha="center", va="center", fontsize=FS * 0.88)

    Zg, Rg = np.meshgrid(zz_h, rr_h, indexing="ij")
    Zd, Rd = np.meshgrid(zz_d, rr_d, indexing="ij")
    if psi_f2_all is not None:
        rr_f = np.asarray(f2["rr"], float)
        zz_f = np.asarray(f2["zz"], float)
        Zf, Rf = np.meshgrid(zz_f, rr_f, indexing="ij")
        T_f = np.asarray(f2["t"], float) * 1e6
        IP_f = np.asarray(f2["Iphi"], float) / 1e3
        BZ_f = np.asarray(f2["Bz0"], float)
        PH_f = np.asarray(f2["Phi"], float) * 1e3
        RS_f = np.asarray(f2["Rs"], float)

    def draw_coils(ax):
        for r, z, hh, w in zip(rl, zl, hl, wl):
            ax.add_patch(Rectangle((z - w / 2, r - hh / 2), w, hh,
                                   fill=False, ec="0.15", lw=1.8))
        ax.plot([-zw, zw], [Rw, Rw], color="k", lw=2.0)
        for lbl, z in zip(("端线圈", "中心线圈", "端线圈"), zl):
            ax.annotate(lbl, (z, 0.158), ha="center", va="bottom",
                        fontsize=FS * 0.68, color="0.35")
        ax.axhline(0.0, color="0.8", lw=0.8, ls="--")
        ax.set_xlim(-zw, zw)
        ax.set_ylim(0, 0.17)
        ax.set_xlabel(r"$z$ [m]")
        ax.set_ylabel(r"$r$ [m]")

    def draw_map_psi(ax, psi2, Z, R, zz, rr, title):
        pT = psi2.T
        ax.pcolormesh(Z, R, pT, cmap="parula", norm=norm, shading="auto")
        ax.contour(Z, R, pT, levels=pos_levels, colors="k",
                   linewidths=0.4, alpha=0.35)
        ax.contour(Z, R, pT, levels=core_levels, colors="k",
                   linewidths=0.6, alpha=0.7)
        ax.contour(Z, R, pT, levels=[0.0], colors=mstyle.RED,
                   linewidths=2.0, linestyles="--")
        iR, iZ = np.unravel_index(int(np.argmin(psi2)), psi2.shape)
        ax.plot(zz[iZ], rr[iR], marker="x", color=mstyle.MAGENTA,
                ms=10, mew=2.2, linestyle="none")
        draw_coils(ax)
        ax.set_title(title, fontsize=FS)

    def draw_map_formation(ax, i, title):
        tt = t0d[:i + 1] * 1e6
        ax.plot(tt, Phi_open[:i + 1] * 1e3, color=mstyle.BLUE, lw=2.2,
                label="开场线通量")
        ax.plot(tt, Phi_closed[:i + 1] * 1e3, color=mstyle.RED, lw=2.6,
                label="闭合私通量")
        ax.plot(t0d[i] * 1e6, Phi_closed[i] * 1e3, "o", color=mstyle.MAGENTA,
                ms=9, linestyle="none")
        ax.set_xlim(0, t_form * 1e6 * 1.05)
        ax.set_ylim(0, max(Phi_max_mWb * 1.1, 1e-3))
        ax.set_xlabel(r"$t$ [$\mu$s]")
        ax.set_ylabel(r"$\Phi$ [mWb]")
        ax.legend(loc="upper left", fontsize=FS * 0.85, frameon=False)
        ax.grid(True, **mstyle.GRID_STYLE)
        ax.set_title(title, fontsize=FS)
        ax.text(0.985, 0.30,
                f"当前 F = {fmt(F0d[i], '.3f')}    "
                f"形成线圈 I_c = {fmt(Ic[i] / 1e3)} kA    "
                f"l_s = {fmt(ls0d[i] * 1e2)} cm",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=FS * 0.8, color="0.25")
        ax.axvline(t_form * 1e6, color="0.6", lw=1.0, ls=":")

    def bottom_formation(i):
        for ax in (ax_a, ax_b, ax_c):
            ax.clear()
        xm = t0d[:i + 1] * 1e6
        ax_a.plot(xm, Ic[:i + 1] / 1e3, color=mstyle.BLUE, lw=2.2)
        ax_a.plot(t0d[i] * 1e6, Ic[i] / 1e3, "o", color=mstyle.RED, ms=7)
        ax_a.set_ylabel(r"$I_c$ [kA]")
        ax_a.set_title("形成线圈电流（主币 RLC）", fontsize=FS * 0.85)
        ax_b.plot(xm, F0d[:i + 1], color=mstyle.GREEN, lw=2.2)
        ax_b.axhline(1.0, color="0.5", lw=1.2, ls=":")
        ax_b.plot(t0d[i] * 1e6, F0d[i], "o", color=mstyle.RED, ms=7)
        ax_b.set_ylabel("F")
        ax_b.set_title("反场因子 F（点线 F=1）", fontsize=FS * 0.85)
        ax_c.plot(xm, ls0d[:i + 1] * 1e2, color=mstyle.BLUE, lw=2.2)
        ax_c.axhline(ls_eq_0d * 1e2, color="0.5", lw=1.2, ls=":")
        ax_c.plot(t0d[i] * 1e6, ls0d[i] * 1e2, "o", color=mstyle.RED, ms=7)
        ax_c.set_ylabel(r"$l_s$ [cm]")
        ax_c.set_title("轴向长度 l_s（点线：平衡）", fontsize=FS * 0.85)
        for ax in (ax_a, ax_b, ax_c):
            ax.set_xlim(0, t_form * 1e6 * 1.05)
            ax.set_xlabel(r"$t$ [$\mu$s]")
            ax.grid(True, **mstyle.GRID_STYLE)

    def bottom_discharge(j, cursor=True):
        for ax in (ax_a, ax_b, ax_c):
            ax.clear()
        xm = tt_d * 1e3
        ph = np.array([r["Phi"] for r in track]) * 1e3
        zc = np.array([r["zc"] for r in track]) * 1e2
        zs = np.array([r["zc_star"] for r in track]) * 1e2
        ls = np.array([r["ls"] for r in track]) * 1e2
        lss = np.array([r["ls_star"] for r in track]) * 1e2
        ax_a.plot(xm, ph, color="0.72" if not cursor else "0.6", lw=1.5)
        if cursor:
            ax_a.plot(xm[:j + 1], ph[:j + 1], color=mstyle.BLUE, lw=2.4)
            ax_a.plot(xm[j], ph[j], "o", color=mstyle.RED, ms=7)
        ax_a.set_ylabel("Phi [mWb]")
        ax_a.set_title("私通量（RMF 托住）", fontsize=FS * 0.85)
        ax_b.plot(xm, zs, color="0.55", lw=1.2, ls="--")
        ax_b.plot(xm, zc, color="0.72" if not cursor else "0.6", lw=1.5)
        if cursor:
            ax_b.plot(xm[:j + 1], zc[:j + 1], color=mstyle.BLUE, lw=2.4)
            ax_b.plot(xm[j], zc[j], "o", color=mstyle.RED, ms=7)
        ax_b.set_ylabel("z_c [cm]")
        ax_b.set_title("轴向位置 z_c（虚线：目标）", fontsize=FS * 0.85)
        ax_c.plot(xm, lss, color="0.55", lw=1.2, ls="--")
        ax_c.plot(xm, ls, color="0.72" if not cursor else "0.6", lw=1.5)
        if cursor:
            ax_c.plot(xm[:j + 1], ls[:j + 1], color=mstyle.BLUE, lw=2.4)
            ax_c.plot(xm[j], ls[j], "o", color=mstyle.RED, ms=7)
        ax_c.set_ylabel("l_s [cm]")
        ax_c.set_title("轴向长度 l_s（虚线：目标）", fontsize=FS * 0.85)
        for ax in (ax_a, ax_b, ax_c):
            ax.set_xlabel(r"$t$ [ms]")
            ax.grid(True, **mstyle.GRID_STYLE)
            for xv in (0.1, 0.2):
                ax.axvline(xv, color="0.8", lw=0.8, ls=":")

    def bottom_f2d(i):
        """二维形成段的下排：环向电流 / 轴上净场 / 私通量（时间轴 µs）。"""
        for ax in (ax_a, ax_b, ax_c):
            ax.clear()
        ax_a.plot(T_f[:i + 1], IP_f[:i + 1], color=mstyle.RED, lw=2.2)
        ax_a.plot(T_f[i], IP_f[i], "o", color=mstyle.MAGENTA, ms=7)
        ax_a.set_ylabel("I_phi [kA]")
        ax_a.set_title("等离子体环向电流（通量守恒）", fontsize=FS * 0.85)
        ax_b.plot(T_f[:i + 1], BZ_f[:i + 1], color=mstyle.BLACK, lw=2.2)
        ax_b.axhline(0.0, color="0.75", lw=0.9)
        ax_b.plot(T_f[i], BZ_f[i], "o", color=mstyle.MAGENTA, ms=7)
        ax_b.set_ylabel("B_z0 [T]")
        ax_b.set_title("轴上净场（过零即反转）", fontsize=FS * 0.85)
        ax_c.plot(T_f[:i + 1], PH_f[:i + 1], color=mstyle.BLUE, lw=2.2)
        ax_c.plot(T_f[i], PH_f[i], "o", color=mstyle.MAGENTA, ms=7)
        ax_c.set_ylabel("Phi [mWb]")
        ax_c.set_title("私通量", fontsize=FS * 0.85)
        for ax in (ax_a, ax_b, ax_c):
            ax.set_xlabel("t [us]")
            ax.grid(True, **mstyle.GRID_STYLE)

    def draw_frame(k, frames):
        kind, i, bottom = frames[k]
        for ax in (ax_map, ax_a, ax_b, ax_c):
            ax.clear()

        if kind == "f2d":
            cax.set_visible(True)
            rs_txt = "-" if not np.isfinite(RS_f[i]) else f"{RS_f[i] * 1e2:.2f}"
            draw_map_psi(ax_map, psi_f2_all[i], Zf, Rf, zz_f, rr_f,
                         f"θ箍缩形成（二维场演化，不做平衡求解）    "
                         f"t = {T_f[i]:.2f} µs    "
                         f"（R_s = {rs_txt} cm，Φ = {PH_f[i]:.4f} mWb）")
            bottom_f2d(i)
            info.set_text(
                f"形成段：线圈 I_c = {np.asarray(f2['Idrive'])[i] / 1e3:.2f} kA        "
                f"等离子体电流 I_phi = {IP_f[i]:+.2f} kA        "
                f"B_z0 = {BZ_f[i]:+.4f} T        "
                f"F_drive = {np.asarray(f2['Fdrive'])[i]:.2f}        "
                f"Φ = {PH_f[i]:.4f} mWb")
        elif kind == "formation":
            cax.set_visible(False)
            draw_map_formation(
                ax_map, i,
                f"θ箍缩形成（0D 集总模型）    t = {t0d[i] * 1e6:.2f} µs"
                f"    （尚无二维磁面）")
            bottom_formation(i)
            info.set_text(
                f"形成段（0D）：反场因子 F = {fmt(F0d[i], '.3f')}        "
                f"Φ_open = {fmt(Phi_open[i] * 1e3, '.4f')} mWb        "
                f"Φ_closed = {fmt(Phi_closed[i] * 1e3, '.4f')} mWb        "
                f"l_s = {fmt(ls0d[i] * 1e2)} cm")
        elif kind == "handoff":
            cax.set_visible(True)
            draw_map_psi(ax_map, psi_h, Zg, Rg, zz_h, rr_h,
                         "二维交接    t_form 处的自由边界平衡    "
                         f"（I_l = {Il_handoff / 1e3:.2f} kA，"
                         f"R_s = {Rs_h * 1e2:.2f} cm，"
                         f"l_s = 2·E·R_s = {ls_h * 1e2:.1f} cm）")
            if bottom == "formation":
                bottom_formation(i_form)
                note = (f"交接态：三线圈同值 I_l = {Il_handoff / 1e3:.2f} kA        "
                        f"Φ = {Phi_h:.4e} Wb（0D 目标 {Phi_target:.3e}）        "
                        f"R_s = {Rs_h * 1e2:.2f} cm        伸长比 = {E_h:.2f}\n"
                        f"二维交接态 l_s = {ls_h * 1e2:.1f} cm；"
                        f"0D 在 t_form 时列长仍为 {ls0d[i_form] * 1e2:.0f} cm"
                        "（轴向收缩在交接之后才走完）")
            else:
                bottom_discharge(0, cursor=False)
                note = (f"交接态：三线圈同值 I_l = {Il_handoff / 1e3:.2f} kA        "
                        f"Φ = {Phi_h:.4e} Wb（0D 目标 {Phi_target:.3e}）        "
                        f"R_s = {Rs_h * 1e2:.2f} cm        伸长比 = {E_h:.2f}\n"
                        "（下排灰线是后面放电段的时间线：它从设计点 A 名义平衡起，"
                        f"l_s = {track[0]['ls'] * 1e2:.1f} → "
                        f"{track[-1]['ls'] * 1e2:.1f} cm）")
            info.set_text(note)
        else:
            cax.set_visible(True)
            row = track[i]
            draw_map_psi(ax_map, psi_d[i], Zd, Rd, zz_d, rr_d,
                         f"{row['phase']}    t = {row['t'] * 1e3:.3f} ms    "
                         f"（z_c = {row['zc'] * 1e2:+.2f} cm，"
                         f"l_s = {row['ls'] * 1e2:.2f} cm，"
                         f"R_s = {row['Rs'] * 1e2:.2f} cm）")
            bottom_discharge(i)
            il = row["Il"]
            rmf_txt = (f"RMF：B_w = {rmf['B_omega_T'] * 1e3:.1f} mT，"
                       f"f = {rmf['f_hz'] / 1e6:.2f} MHz" if rmf else "无 RMF")
            info.set_text(
                f"线圈 I_1 / I_2 / I_3 = {il[0] / 1e3:.2f} / "
                f"{il[1] / 1e3:.2f} / {il[2] / 1e3:.2f} kA        "
                f"差模 a = {row['a'] / 1e3:+.2f} kA        "
                f"Phi = {row['Phi']:.4e} Wb"
                f"（{row['Phi'] / track[0]['Phi']:.3f}×）        " + rmf_txt)

        for ax in (ax_map, ax_a, ax_b, ax_c):
            mstyle.style_axes(ax, fontsize=FS * 0.95, fontweight="bold",
                              linewidth=1.4, force=True)
        return fig

    # ---------------- 渲染每一段 ----------------
    def keys_of(frames):
        """按帧序列自动找关键帧（对合并版/前段/后段都成立）。"""
        kinds = [fr[0] for fr in frames]
        out = []
        for nm, kind in (("f2d_end", "f2d"), ("form_end", "formation"),
                         ("handoff", "handoff")):
            at = [k for k, kk in enumerate(kinds) if kk == kind]
            if at:
                out.append((nm, at[-1]))
        for nm, ph in (("trans_end", "平移"), ("comp_end", "压缩"),
                       ("flat_end", "平顶")):
            at = [k for k, fr in enumerate(frames)
                  if fr[0] == "discharge" and track[fr[1]]["phase"] == ph]
            if at:
                out.append((nm, at[-1]))
        return out

    auto_keys = {name: keys_of(frames) for name, frames in sets.items()}

    for part, name in jobs:
        frames = sets[part]
        print(f"[full] {name}：{len(frames)} 帧")
        rgb = []
        for k in range(len(frames)):
            draw_frame(k, frames)
            fig.canvas.draw()
            rgb.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        gif = write_gif(rgb, OUT / f"{name}.gif", args.fps)
        print(f"[full] 动图：{gif}  ({gif.stat().st_size / 1024:.0f} KB)")

        if args.key_frames == "auto":
            keys = auto_keys[part]
        elif args.key_frames.strip():
            keys = [(w.strip(), int(w)) for w in args.key_frames.split(",")
                    if w.strip().isdigit()]
        else:
            keys = []
        for label, k in keys:
            draw_frame(int(k), frames)
            png = OUT / f"{name}_frame_{label}.png"
            fig.savefig(png, dpi=120)
            print(f"[full] 关键帧：{png}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
