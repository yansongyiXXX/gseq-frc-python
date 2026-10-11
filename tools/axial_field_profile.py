r"""On-axis vacuum field profile of the design-point A coil set.

Romero et al. (Nucl. Fusion 66 (2026) 016042) show that the axial
(in)stability of an elongated FRC is set by the shape of the external
axial field on axis:

* **U 形** —— |B| 在中平面取局部极小，等离子体被两端镜场压住，轴向稳定；
* **W 形** —— |B| 在中平面取局部极大，等离子体被推向两端，轴向不稳定，
  必须靠 trim 线圈做位置反馈。

本脚本用设计点 A 的线圈 Green 函数扫描线圈电流组合，给出轴上 |B_z(z)| 与
U/W 形判定，作为要不要加位置反馈、以及中间线圈能推到哪一步的依据。

用法：python tools/axial_field_profile.py
产物：outputs/design_point/axial_profile.json / .png
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.chdir(ROOT)

import design_point as dp  # noqa: E402

OUT = ROOT / "outputs" / "design_point"


MU0 = 4 * np.pi * 1e-7


def on_axis_bz(s, il, z_grid, nr=6, nz=24):
    r"""轴上真空场 \(B_z(0,z)=\sum \mu_0\,\Delta I\,R^2/2\,(R^2+(z-z_0)^2)^{-3/2}\)。

    每个线圈按矩形截面分成 `nr × nz` 根丝，安匝数均分——与求解器里
    `_coil_filaments` 的做法一致。
    """
    il = np.asarray(il, dtype=float)
    bz = np.zeros_like(z_grid)
    for i in range(il.size):
        r0 = float(s["rl"][i])
        z0 = float(s["zl"][i])
        h = float(s["hl"][i])
        w = float(s["wl"][i])
        rs = r0 + h * (np.arange(nr) + 0.5) / nr
        zs = z0 - 0.5 * w + w * (np.arange(nz) + 0.5) / nz
        dI = il[i] / (nr * nz)
        RR, ZZ = np.meshgrid(rs, zs, indexing="ij")
        for Rf, Zf in zip(RR.ravel(), ZZ.ravel()):
            bz += MU0 * dI * Rf ** 2 / (2.0 * (Rf ** 2 + (z_grid - Zf) ** 2) ** 1.5)
    return bz


def profile(model, s, il):
    """轴上 B_z(z) [T]，z 用计算网格的 zz。"""
    zz = np.asarray(s["zz"], dtype=float)
    return zz, on_axis_bz(s, il, zz)


def describe(zz, bz, zw):
    """U/W 形判定：中平面与 ±zw/2 处的场强比较。"""
    iz = lambda z: int(np.argmin(abs(zz - z)))
    b_mid = float(bz[iz(0.0)])
    b_side = float(np.mean([bz[iz(-0.30)], bz[iz(0.30)]]))
    well = b_mid - b_side                     # >0 = W 形（中平面凸起）
    return dict(B_mid=b_mid, B_side=b_side, well=well,
                shape="W" if well > 0 else "U",
                well_rel=well / max(abs(b_side), 1e-12))


def main() -> None:
    s, model, _, _ = dp.build_design_model()
    s0 = dp.gcm._setup(dp.SRC)
    il0 = np.asarray(s["Il"], dtype=float)
    zw = float(s0["zw"])

    cases = {
        "nominal (1:1:1)": il0.copy(),
        "centre +20%": il0 * np.array([1.0, 1.20, 1.0]),
        "centre +50%": il0 * np.array([1.0, 1.50, 1.0]),
        "ends +20%": il0 * np.array([1.20, 1.0, 1.20]),
        "ends -20%": il0 * np.array([0.80, 1.0, 0.80]),
        "ends -50%": il0 * np.array([0.50, 1.0, 0.50]),
        "ends -80%": il0 * np.array([0.20, 1.0, 0.20]),
        "ends off": il0 * np.array([0.00, 1.0, 0.00]),
    }
    out = {}
    for name, il in cases.items():
        zz, bz = profile(model, s0, il)
        d = describe(zz, bz, zw)
        out[name] = dict(Il=il.tolist(), **d)
        print(f"[axial] {name:>16s}  Il={np.array2string(il / 1e3, precision=1)} kA·匝  "
              f"B(r=0,z=0)={d['B_mid']:+.5f} T  侧向={d['B_side']:+.5f} T  "
              f"井深={d['well'] * 1e3:+7.3f} mT  → {'W 形（轴向不稳定）' if d['shape'] == 'W' else 'U 形（轴向稳定）'}")

    (OUT / "axial_profile.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    plt = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        try:
            from gseqfrc import mstyle
            mstyle.install()
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        plt = None

    if plt is not None:
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        for name, il in cases.items():
            zz, bz = profile(model, s0, il)
            ax.plot(zz, bz, lw=1.6, label=name)
        ax.axhline(0, color="k", lw=0.6, alpha=0.3)
        ax.set_xlabel(r"$z$ [m]")
        ax.set_ylabel(r"$B_z(r=0,z)$ [T]")
        ax.set_title("设计点 A：轴上真空场分布（U/W 形判定）")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "axial_profile.png", dpi=150)
        print(f"[axial] wrote {OUT / 'axial_profile.png'}")


if __name__ == "__main__":
    main()
