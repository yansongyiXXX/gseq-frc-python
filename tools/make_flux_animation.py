r"""把设计点 A 的正向演化做成磁通演化动图（GIF）。

输入：`outputs/design_point/forward.npz`（由 `tools/frc_evolve_design.py` 生成，
里面存了每一步的 \(\psi(R,Z)\) 快照）。

版式照论文 Fig.4 的样子：**上排是一整幅宽的 \(\psi(z,r)\) 场**（分离面、开放场线、
线圈与壁、色条），下面并排两张轨迹图——左边私通量 \(\Phi(t)\)（带游标），右边耗散率
\(S_0\) 与驱动 \(S_{\rm drive}\)；参数说明单独占最下面一条窄带，不压在任何曲线上。

**色标**（`--scale`）：默认 `symlog`——芯部 \(|\psi|\lesssim|\psi_{\min}|\approx
1.7\times10^{-5}\) 与壁面 \(2.1\times10^{-3}\) 差两个数量级，线性全量程会把芯部
压成一块均匀深蓝、只看芯部又得把壁面裁掉（留白）。对称对数色标在零点附近
\(\pm|\psi_{\min}|\) 内保持线性（芯部层次照看得见）、之外按 \(\log_{10}\) 铺开，
**整个求解域都填色、没有留白**；线性区那一段的宽度由 `--linscale` 调（默认 3）。
`--scale linear` 回到原来那版（线性 + 超出 \(4.2|\psi_{\min}|\) 留白）。

**动图是用全局调色板写的**（`_write_gif`）：`PillowWriter` 每帧各自量化一次，
同一条色条上同一个像素在不同帧会落到不同的调色板格子，看起来就是色条在抖。

用法：

    python tools/frc_evolve_design.py --steps 24 --dt 1e-5   # 先生成更多帧
    python tools/make_flux_animation.py                      # 再出动图

产物：`outputs/design_point/flux_evolution.gif`（以及同名 .png 关键帧）。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from gseqfrc import mstyle  # noqa: E402

mstyle.install()          # 全套 MATLAB 默认值（字体、parula、刻度朝里…）

OUT_DIR = ROOT / "outputs" / "design_point"

#: 这张图里的字号。画布 13 英寸宽，13 pt 的字与 MATLAB 最大化图窗观感相当。
FS = 13.0


def fmt(value, spec: str = ".2f", dash: str = "—") -> str:
    """数值格式化：最后几步平衡已经不存在（\(R_s\)/\(\beta\) 是 NaN）时写个破折号，
    别在图上印 `nan`。"""
    value = float(value)
    return dash if not np.isfinite(value) else format(value, spec)


def _quantizer():
    """Pillow 9.1 起枚举搬到了 `Image.Quantize` / `Image.Dither`。"""
    from PIL import Image

    method = getattr(getattr(Image, "Quantize", Image), "MEDIANCUT", 0)
    dither = getattr(getattr(Image, "Dither", Image), "NONE", 0)
    return method, dither


def write_gif(frames, path, fps) -> Path:
    """把逐帧 RGB 数组写成 GIF（全局调色板，不做逐帧抖动）。

    为什么要自己写：`PillowWriter` 让每一帧各自自适应量化，同一条色条上同一个
    像素在不同帧会落到不同的调色板格子里——动起来就是色条在"呼吸"。这里先用
    所有帧的像素统计出一张全局调色板，再让每帧映射到**同一张**调色板上，
    颜色条就完全钉死了（代价是渐变可能出现轻微色带，但帧间绝对一致）。
    """
    from PIL import Image

    method, dither = _quantizer()
    # 抽稀后把所有帧拼在一起统计调色板：整幅图用同一套颜色
    sample = np.concatenate([frame[::3, ::3] for frame in frames], axis=0)
    palette = Image.fromarray(sample).quantize(colors=255, method=method)
    images = [Image.fromarray(frame).quantize(palette=palette, dither=dither)
              for frame in frames]
    images[0].save(str(path), save_all=True, append_images=images[1:],
                   duration=int(round(1000.0 / fps)), loop=0, disposal=2,
                   optimize=False)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", default=str(OUT_DIR / "forward.npz"))
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--nlevels", type=int, default=21)
    ap.add_argument("--scale", choices=("symlog", "linear"), default="symlog",
                    help="上排填色的色标：symlog（对称对数，默认，铺满整个求解域）"
                         "或 linear（线性、超出 4.2|ψmin| 的部分留白）")
    ap.add_argument("--linscale", type=float, default=3.0,
                    help="symlog 色标里线性区（|ψ|<|ψmin|，也就是芯部那段）"
                         "占几个十倍频程的宽度，越大芯部的颜色越铺得开")
    ap.add_argument("--name", default="flux_evolution")
    ap.add_argument("--key-frame", type=int, default=None,
                    help="额外导出的静止关键帧序号（默认中间那帧）")
    ap.add_argument("--overlay-ref", default=None,
                    help="叠加参考算例（MATLAB .fig）的分离面，便于对照形状")
    args = ap.parse_args()

    d = np.load(args.npz)
    psi_all = np.asarray(d["psi_all"], dtype=float)      # (nt, nR, nZ)
    rr, zz = np.asarray(d["rr"], float), np.asarray(d["zz"], float)
    t = np.asarray(d["t"], float)
    phi = np.asarray(d["Phi"], float)
    S0 = np.asarray(d["S0"], float)
    drive = np.asarray(d["drive"], float)
    Rs, Bz0 = np.asarray(d["Rs"], float), np.asarray(d["Bz0"], float)
    beta, Iz = np.asarray(d["beta"], float), np.asarray(d["Izeta0"], float)
    Rw, zw, Be = float(d["Rw"]), float(d["zw"]), float(d["Be"])
    rl, zl = np.asarray(d["rl"], float), np.asarray(d["zl"], float)
    hl, wl = np.asarray(d["hl"], float), np.asarray(d["wl"], float)
    nt = psi_all.shape[0]
    print(f"[gif] {nt} 帧；t ∈ [0, {t[-1] * 1e3:.3f}] ms；"
          f"Φ ∈ [{phi.min():.3e}, {phi.max():.3e}] Wb")

    # ---- 色标：跨帧固定（固定标尺才看得出通量在衰减）------------------------
    # 芯部和壁面差两个数量级（|ψ_min|≈1.7e-5 对 ψ_wall≈2.1e-3）：线性全量程会把
    # 芯部压成一块均匀深蓝，只看芯部就得把壁面那片裁掉（留白）。`--scale symlog`
    # 用对称对数：零点附近 ±linthresh 之内是线性的（芯部层次照看得见），之外按
    # log10 铺开（开放场区不再被压成一条），而且**不用裁掉任何区域**。
    psi_scale = float(abs(psi_all[0].min()))
    psi_min_all = float(np.min(psi_all))
    psi_max_all = float(np.max(psi_all))
    #: symlog 色条的刻度（单位 10⁻⁵ Wb/(2π)）。刻度不能给太密——对数轴上
    #: 2/5/10/30/100 这几档在十来厘米高的色条上会叠在一起，只留整量级那几档。
    log_tick_units = np.array([-1.5, -1.0, 0.0, 1.0, 10.0, 100.0])
    if args.scale == "symlog":
        linthresh = abs(psi_min_all)
        vmin, vmax = 1.02 * psi_min_all, 1.02 * psi_max_all
        norm = matplotlib.colors.SymLogNorm(linthresh, linscale=args.linscale,
                                            vmin=vmin, vmax=vmax, base=10)
        core_levels = np.linspace(-linthresh, 0.0, 9)[:-1]
        pos_levels = np.geomspace(linthresh, psi_max_all, 15)
        cbar_label = (r"$\psi$   $[10^{-5}\ \mathrm{Wb}/(2\pi)]$"
                      "\n" r"(symlog)")
    else:
        levels = np.linspace(-1.05 * psi_scale, 0.0, args.nlevels)
        vmin, vmax = levels[0], -levels[0] * 4.0
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        clip_pos = vmax
        core_levels = levels[1:]
        pos_levels = np.linspace(0.0, np.nanmax(np.abs(psi_all[0])), 31)[1:]
        cbar_label = r"$\psi$ [Wb/(2$\pi$)]"

    # 可选的参考算例（从 MATLAB .fig 里取 ψ 场，画它的 ψ=0 分离面）
    ref = None
    if args.overlay_ref:
        from scipy.io import loadmat
        hg = loadmat(args.overlay_ref, squeeze_me=True,
                     struct_as_record=False)["hgS_070000"]
        pr0 = np.atleast_1d(hg.children[0].children)[0].properties
        ref = (np.asarray(pr0.XData, float), np.asarray(pr0.YData, float),
               np.asarray(pr0.ZData, float))
        print(f"[gif] 参考算例：ψ_min={np.nanmin(ref[2]):.4e}，"
              f"分离面用灰色点线叠加")

    # 画法与原文的 figure_4_1 / figure_6_1 一致：横轴 z、纵轴 r，单位 m。
    Zg, Rg = np.meshgrid(zz, rr, indexing="ij")   # 形状 (nZ, nR)
    # 版式照论文 Fig.4：上面一整幅宽的 ψ 场，下面两张轨迹图，参数说明另占一条窄带
    fig = plt.figure(figsize=(13.0, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1.45, 0.14])
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    axinfo = fig.add_subplot(gs[2, :])
    axinfo.axis("off")
    # 色条只建一次，不能放进逐帧的绘制里，否则每帧都会往图上叠一根。
    sm = plt.cm.ScalarMappable(cmap="parula", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax0, pad=0.0120, fraction=0.030)
    cbar.set_label(cbar_label, fontsize=FS * 0.95, fontweight="bold",
                   labelpad=8)
    if args.scale == "symlog":
        # 刻度按"每 10⁻⁵ 一个单位"给，好读；刻度值本身就是数据的绝对值
        units = log_tick_units[(log_tick_units * 1e-5 >= vmin)
                               & (log_tick_units * 1e-5 <= vmax)]
        cbar.set_ticks(units * 1e-5)
        cbar.set_ticklabels([f"{u:g}" for u in units])
    else:
        cbar.locator = MaxNLocator(nbins=6)
        cbar.update_ticks()
        cbar.formatter.set_useMathText(True)   # 量级标成 ×10⁻⁵，MATLAB 的写法
    cbar.ax.tick_params(labelsize=FS * 0.85)
    cbar.ax.yaxis.get_offset_text().set_fontsize(FS * 0.85)
    for _txt in cbar.ax.get_yticklabels():
        _txt.set_fontweight("bold")
    cbar.outline.set_linewidth(1.2)

    # 参数说明：单独放最下面那条窄带，永远压不到曲线上；每帧只改文字
    info = axinfo.text(0.5, 0.5, "", transform=axinfo.transAxes,
                       ha="center", va="center", fontsize=FS * 0.95)

    def draw_frame(k: int):
        for ax in (ax0, ax1, ax2):
            ax.clear()
        psi = psi_all[k]

        # ---- 上：ψ 全场 ---------------------------------------------------
        pT, ZT, RT = psi.T, Zg, Rg          # (nZ, nR)
        if args.scale == "symlog":
            p_show = pT                      # 对称对数色标：整个求解域都铺满，无留白
        else:
            # 线性色标只画到分离面外侧这一层，其余留白（壁面通量比芯部大两个量级，
            # 全画出来芯部会压成一片）
            p_show = np.ma.masked_where(pT > clip_pos, pT)
        ax0.pcolormesh(ZT, RT, p_show, cmap="parula", norm=norm, shading="auto")
        # 开放场线（正场区）
        ax0.contour(ZT, RT, pT, levels=pos_levels, colors="k",
                    linewidths=0.4, alpha=0.35)
        # 反场区内部等值线（原文那几条负水平）
        ax0.contour(ZT, RT, pT, levels=core_levels, colors="k",
                    linewidths=0.6, alpha=0.7)
        # 分离面 ψ=0（原文用红色虚线）
        ax0.contour(ZT, RT, pT, levels=[0.0], colors=mstyle.RED,
                    linewidths=2.0, linestyles="--")
        if ref is not None:      # 参考算例的分离面（灰色点线）
            ax0.contour(ref[0], ref[1], ref[2], levels=[0.0], colors="0.35",
                        linewidths=1.4, linestyles=":")

        # ---- O 点：np.argmin 是 C 序索引，必须用 unravel_index 解 --------
        iR, iZ = np.unravel_index(int(np.argmin(psi)), psi.shape)
        ax0.plot(zz[iZ], rr[iR], marker="x", color=mstyle.MAGENTA,
                 ms=10, mew=2.2, linestyle="none")
        ax0.annotate("O point", (zz[iZ], rr[iR]),
                     textcoords="offset points", xytext=(9, 9),
                     fontsize=FS * 0.8, color=mstyle.BLACK)

        # ---- 线圈与壁（横轴 z、纵轴 r）------------------------------------
        for r, z, h, w in zip(rl, zl, hl, wl):
            ax0.add_patch(Rectangle((z - w / 2, r - h / 2), w, h,
                                    fill=False, ec="0.5", lw=1.6))
        ax0.plot([-zw, zw], [Rw, Rw], color="k", lw=2.0)
        ax0.axhline(0.0, color="0.75", lw=0.8, ls="--")
        ax0.set_xlim(-zw, zw)
        ax0.set_ylim(0, 0.165)
        ax0.set_xlabel(r"$z$ [m]")
        ax0.set_ylabel(r"$r$ [m]")
        ax0.set_title(rf"$\psi(z,r)$ at $t={t[k] * 1e3:.3f}$ ms"
                      rf"   ($R_s$={fmt(Rs[k] * 1e2)} cm)")

        # ---- 左下：Φ(t) ---------------------------------------------------
        ax1.plot(t * 1e3, phi, color="0.45", lw=1.2)              # 全程（浅）
        ax1.plot(t[:k + 1] * 1e3, phi[:k + 1], color=mstyle.BLUE, lw=2.4)
        ax1.plot(t[k] * 1e3, phi[k], marker="o", color=mstyle.RED,
                 ms=7, linestyle="none")
        ax1.set_xlabel(r"$t$ [ms]")
        ax1.set_ylabel(r"$\Phi$ [Wb]")
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
        # MATLAB 遇到这种小量级会在轴顶标一个 ×10⁻⁵，而不是把 0.0000xx 写全
        ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0),
                             useMathText=True)
        ax1.grid(True, **mstyle.GRID_STYLE)
        ax1.set_title(rf"$\Phi={phi[k]:.3e}$ Wb,  "
                      rf"$R_s={fmt(Rs[k] * 1e2)}$ cm")

        # ---- 右下：S0 与 drive ---------------------------------------------
        ax2.plot(t * 1e3, S0, color=mstyle.RED, lw=2.0,
                 label=r"$S_0$ (loss)")
        ax2.plot(t * 1e3, drive, color=mstyle.GREEN, lw=2.0, ls="--",
                 label=r"$S_{drive}$ (RMF)")
        ax2.plot(t[k] * 1e3, S0[k], marker="o", color=mstyle.RED,
                 ms=6, linestyle="none")
        ax2.plot(t[k] * 1e3, drive[k], marker="s", color=mstyle.GREEN,
                 ms=6, linestyle="none")
        ax2.set_xlabel(r"$t$ [ms]")
        ax2.set_ylabel("[Wb/s]")
        ax2.grid(True, **mstyle.GRID_STYLE)
        ax2.legend(loc="upper right", frameon=False)
        ax2.set_ylim(0, 1.08 * float(max(S0.max(), drive.max())))

        ratio = f"{drive[k] / S0[k]:.3f}" if S0[k] > 0 else "—"
        info.set_text(rf"$B_e$ = {Be:.3f} T      "
                      rf"$B_{{z0}}$ = {fmt(Bz0[k], '+.4f')} T      "
                      rf"$\langle\beta\rangle$ = {fmt(beta[k], '.3f')}      "
                      rf"$I_{{\zeta0}}$ = {fmt(Iz[k] / 1e3)} kA      "
                      rf"$S_{{drive}}/S_0$ = {ratio}"
                      "\n"
                      "色标 symlog：|ψ| < ψmin 内线性、之外对数"
                      "（--scale linear 恢复线性+留白）")

        # 每帧都 clear() 过，样式要重来一遍（force 跳掉幂等短路）
        for ax in (ax0, ax1, ax2):
            mstyle.style_axes(ax, fontsize=FS, fontweight="bold",
                              linewidth=1.6, force=True)
        return fig

    # ---- 逐帧渲染，最后用全局调色板写 GIF ---------------------------------
    frames = []
    for k in range(nt):
        draw_frame(k)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    gif = write_gif(frames, OUT_DIR / f"{args.name}.gif", args.fps)
    print(f"[gif] 产物：{gif}")

    # ---- 关键帧（静止图，方便贴到报告里）----------------------------------
    kk = args.key_frame if args.key_frame is not None else nt // 2
    draw_frame(int(kk))
    png = OUT_DIR / f"{args.name}_frame{kk:02d}.png"
    fig.savefig(png, dpi=120)
    print(f"[gif] 关键帧：{png}（第 {kk} 帧）")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
