"""时间演化演示：不对称线圈电流波形下，逐步演化磁通（**修改版** 第 17 条）。

用 `MachineModel` 把与电流无关的量预计算一次，然后对一段**上下不对称**的线圈
电流波形逐步推进：每步以上一步的 (psi, psib, CI) 暖启动，判据用 freegsnke 式
"残差 < step_tol × 本步磁通变化"。

用法（在本仓库根目录）：

    python tools/evolve_demo.py                # 默认：从保存的 Fig6 平衡起步
    python tools/evolve_demo.py --cold         # 没有保存的解时，先冷启动解一次

产物：`outputs/evolve_demo/history.npz`（每步的 psi、C、CI、psib、耗时）。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from gseqfrc import mpath  # noqa: E402


def _initial_from_saved(gcm, model, mat_path: Path):
    """从 `outputs/p2-Fig6/psiq.mat` 造一个暖启动状态（补算边界通量 psib）。"""
    m = loadmat(mat_path, squeeze_me=True)
    psi0 = np.asarray(m["psiq"], dtype=float)
    CI0 = float(m["CIq"])
    psib_coil0 = model.coil_flux(model.s["Il"])[3]
    pr = model.problem(psib_coil0, 2)
    psif = psi0.ravel(order="F")
    C0 = pr.P["Izeta0"] / pr.jzmid_per_unit_C(psif)
    Jz = (pr.R.ravel(order="F") * pr.dp_per_unit_C(psif) * C0)[pr.int_idx]
    psib0 = CI0 * psib_coil0 + gcm.MU0 * CI0 * pr.dR * pr.dZ * (pr.Gsub.T @ Jz)
    return dict(psiq=psi0, psib=psib0, CI=CI0, C=C0, secant_state=None,
                dpsi_step=None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold", action="store_true",
                    help="忽略保存的平衡，从冷启动解一个初始状态")
    ap.add_argument("--steps", type=int, default=6, help="演化步数（默认 6）")
    ap.add_argument("--step-tol", type=float, default=5e-3)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--anderson", action="store_true",
                    help="中层用 Anderson 加速（默认关；仅在 step_tol 路径生效）")
    args = ap.parse_args()

    mpath.addpath("second_paper/Fig6")
    gcm = mpath.load("GSEQ_coil_MRR")
    s = gcm._setup(gcm.SRC_DIR)
    model = gcm.MachineModel(s)

    mat_path = ROOT / "outputs" / "p2-Fig6" / "psiq.mat"
    initial = None
    if not args.cold and mat_path.exists():
        initial = _initial_from_saved(gcm, model, mat_path)
        print(f"初始状态：{mat_path}（C={initial['C']:.6f}，CI={initial['CI']:.6f}）")
    else:
        print("初始状态：冷启动（约 10–30 s）")

    base = np.asarray(s["Il"], dtype=float)          # (nl,)
    nl = base.size
    # 特意做成**上下不对称**：上线圈 +0.5%/步、中线圈 +0.2%/步、下线圈 −0.5%/步。
    # 对称性因此被打破（半域那类加速在这里不能用），但可分离快解不受影响。
    slope = np.array([+0.005, +0.002, -0.005])[:nl]
    Il_traj = base[None, :] * (1.0 + slope[None, :] * np.arange(args.steps)[:, None])
    times = np.arange(args.steps) * 0.1                       # ms，占位时间轴

    hist, model = gcm.evolve(Il_traj, times=times, model=model, initial=initial,
                             step_tol=args.step_tol, progress=args.progress,
                             mid_accel=("anderson" if args.anderson else "none"))

    print("\n步  Il(kA)                     C          CI          res       "
          "步数  调用  求解/s")
    for h in hist:
        print(f"{h['i']:2d}  {np.array2string(h['Il']/1e3, precision=2):24s} "
              f"{h['C']:.7f}  {h['CI']:.7f}  {h['gs_res']:.2e}  "
              f"{h['steps']:4d}  {h['calls']:4d}  {h['nk_wall']:6.2f}")

    out_dir = ROOT / "outputs" / "evolve_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "history.npz",
        times=np.asarray(times),
        Il=np.asarray([h["Il"] for h in hist]),
        C=np.asarray([h["C"] for h in hist]),
        CI=np.asarray([h["CI"] for h in hist]),
        gs_res=np.asarray([h["gs_res"] for h in hist]),
        wall=np.asarray([h["wall"] for h in hist]),
        psiq=np.asarray([h["psiq"] for h in hist if h["psiq"] is not None]))
    print(f"\n产物：{out_dir / 'history.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
