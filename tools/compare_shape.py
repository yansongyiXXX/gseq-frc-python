r"""对照两个 FRC 平衡的形状：参考算例（Fig4 固定边界）vs 设计点 A（Fig6 自由边界）。

用途：解释"为什么两张图的等离子体形状不一样"。三个面板：

1. 中平面 \(\psi(R)\) 剖面（各自按 \(|\psi_{\min}|\) 归一化）——看径向形状；
2. 反场区的径向包络 \(R_{\max}(z)\)——看轴向长度与细长比；
3. 壁面通量 \(\psi_b(z)\)——**这是形状差异的根源**：固定边界算例直接给这条剖面，
   自由边界算例的这条剖面由线圈电流产生。

用法：

    python tools/compare_shape.py --ref "<参考 .fig 或 Fig4 的 psiq mat>"
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scipy.io import loadmat  # noqa: E402


def load_ref(path: str):
    """参考算例：既支持 MATLAB .fig（从 contourgroup 取 ψ 面），也支持 .mat。"""
    if path.lower().endswith(".fig"):
        hg = loadmat(path, squeeze_me=True, struct_as_record=False)["hgS_070000"]
        pr = np.atleast_1d(hg.children[0].children)[0].properties
        return (np.asarray(pr.XData, float), np.asarray(pr.YData, float),
                np.asarray(pr.ZData, float))
    m = loadmat(path, squeeze_me=True)
    psi = np.asarray(m["psiq"], float)
    Rq = np.asarray(m["Rq"], float) if "Rq" in m else None
    Zq = np.asarray(m["Zq"], float) if "Zq" in m else None
    if Rq is None or Zq is None or Rq.shape != psi.shape:
        nR, nZ = psi.shape
        rr = np.linspace(0, 0.14, nR)
        zz = np.linspace(-0.75, 0.75, nZ)
        Zq, Rq = np.meshgrid(zz, rr, indexing="ij")
        psi = psi.T
    return Zq, Rq, psi


def envelope(Zq, Rq, psi):
    """把 (Z,R,ψ) 摆成 psi[iR, iZ] 的规范形式，返回 (rs, zs, psi_reg)。

    不同来源的数据轴向不一样：Fig4 那份是 (nR=513, nZ=129)，Fig6 这份是
    (nR=129, nZ=257)——所以要按坐标矩阵**哪个轴在变**来判断，不能假定。
    """
    # psi 的两个轴与坐标矩阵的两个轴一一对应：轴 0 是 z 时 psi 是 (nZ, nR)，
    # 轴 0 是 r 时 psi 是 (nR, nZ)。统一摆成 (nR, nZ)。
    z_axis0 = np.ptp(Zq[:, 0]) > 0
    if z_axis0:                       # 轴 0 是 z
        zs, rs, psi_reg = Zq[:, 0], Rq[0, :], psi.T
    else:                             # 轴 0 是 r
        zs, rs, psi_reg = Zq[0, :], Rq[:, 0], psi
    return rs, zs, psi_reg


def frc_envelope(rs, zs, psi_reg):
    """反场区（ψ<0）在每个 z 截面上的最大 R。"""
    neg = psi_reg < 0
    out = np.full(zs.size, np.nan)
    for j in range(zs.size):
        col = np.nonzero(neg[:, j])[0]
        if col.size:
            out[j] = rs[col].max()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", default=r"C:\Users\dell\Desktop\untitled.fig")
    ap.add_argument("--forward", default="outputs/design_point/forward.npz")
    ap.add_argument("--out", default="outputs/design_point/shape_compare.png")
    args = ap.parse_args()

    Zq, Rq, psi_ref = load_ref(args.ref)
    rs_r, zs_r, psi_r = envelope(Zq, Rq, psi_ref)        # 规范成 (nR, nZ)
    d = np.load(args.forward)
    psi_dp = np.asarray(d["psi_all"][0], float)
    rr = np.asarray(d["rr"], float)
    zz = np.asarray(d["zz"], float)
    psi_d = psi_dp                                        # 已经是 (nR, nZ)

    neg_r = psi_r < 0
    neg_d = psi_d < 0
    print("参考算例 : psi_min=%+.4e  反场区 R<=%.2f cm, Z=+-%.2f cm  细长比=%.2f"
          % (np.nanmin(psi_r), rs_r[neg_r.any(axis=1)].max() * 1e2,
             np.abs(zs_r[neg_r.any(axis=0)]).max() * 1e2,
             np.abs(zs_r[neg_r.any(axis=0)]).max() / rs_r[neg_r.any(axis=1)].max()))
    print("设计点 A : psi_min=%+.4e  反场区 R<=%.2f cm, Z=+-%.2f cm  细长比=%.2f"
          % (psi_d.min(), rr[neg_d.any(axis=1)].max() * 1e2,
             np.abs(zz[neg_d.any(axis=0)]).max() * 1e2,
             np.abs(zz[neg_d.any(axis=0)]).max() / rr[neg_d.any(axis=1)].max()))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

    Z0r = int(np.argmin(np.abs(zs_r)))
    Z0d = int(np.argmin(np.abs(zz)))
    ax[0].plot(rs_r * 1e2, psi_r[:, Z0r] / abs(np.nanmin(psi_r)),
               "k-", label="reference (Fig4)")
    ax[0].plot(rr * 1e2, psi_d[:, Z0d] / abs(psi_d.min()), "r--",
               label="design point A (Fig6)")
    ax[0].axhline(0, color="0.7", lw=0.6)
    ax[0].set_xlabel("R [cm]")
    ax[0].set_ylabel(r"$\psi/\psi_{\min}$")
    ax[0].set_title("midplane profile")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].plot(zs_r * 1e2, frc_envelope(rs_r, zs_r, psi_r) * 1e2, "k-",
               label="reference (Fig4)")
    ax[1].plot(zz * 1e2, frc_envelope(rr, zz, psi_d) * 1e2, "r--",
               label="design point A")
    ax[1].set_xlabel("z [cm]")
    ax[1].set_ylabel(r"$R_{\max}$ of $\psi<0$ [cm]")
    ax[1].set_title("FRC envelope")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    ax[2].plot(zs_r * 1e2, psi_r[-1, :], "k-", label="reference wall flux")
    ax[2].plot(zz * 1e2, psi_d[-1, :], "r--", label="design point wall flux")
    ax[2].set_xlabel("z [cm]")
    ax[2].set_ylabel(r"$\psi_{\rm wall}$ [Wb/(2$\pi$)]")
    ax[2].set_title("wall flux (what sets the shape)")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    fig.savefig(args.out, dpi=120)
    print(f"产物：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
