r"""展示物 3：三种**磁链口径**的对照表（初态半径/通量、壁面通量漂移）。

同一个机器模型（线圈 + 理想导体壳 2 环）跑三遍 Stage 4 演化，只换磁链口径：

* ``ci``       —— 论文口径：边界通量带 \(CI\) 归一化，压强平衡是硬约束（A 路线）
* ``physical`` —— 物理口径：钉住 \(CI\equiv1\)，压强平衡退化为**预测**（B 路线）
* ``be-loop``  —— \(B_e\) 环：\(CI\equiv1\) 且把剖面参考场 \(B_e\) 当未知量反解，
                  物理磁链与压强平衡**同时**成立（C 路线）

输出 ``outputs/showcase/fluxnorm.md``、``fluxnorm.csv`` 与三个 ``fluxnorm_<口径>.npz``。

用法（仓库根目录）::

    python tools/showcase_fluxnorm.py                      # 3 步 × 30 μs，理想壳 2 环
    python tools/showcase_fluxnorm.py --steps 2 --dt 3e-5
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

OUT_DIR = ROOT / "outputs" / "showcase"


def load_stage04():
    """按路径加载 `tools/frc_evolve_stage04.py`（它导入时会自己 addpath）。"""
    path = ROOT / "tools" / "frc_evolve_stage04.py"
    spec = importlib.util.spec_from_file_location("frc_evolve_stage04", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["frc_evolve_stage04"] = mod
    spec.loader.exec_module(mod)
    return mod


def stage04_args(steps: int, dt: float, flux_norm: str) -> Namespace:
    """与 `frc_evolve_stage04.main()` 的默认值一致，只改要改的那几项。"""
    return Namespace(
        steps=steps, dt=dt, te=50.0, eta=None, shell="ideal", flux_norm=flux_norm,
        be_from_coils=False, n_shell=2, shell_t=5e-3, shell_rho=1.7e-8,
        coil_L=None, coil_R=None, coil_area=None, voltage="ri", drive_rate=0.0,
        no_plasma_coupling=False, rtol=2e-3, max_eval=4, step_tol=5e-3,
        verify=False)


def summarise(hist: list[dict], n_active: int) -> dict:
    """从一条演化历史里抽出这张表要的量。"""
    first, last = hist[0], hist[-1]
    d_psi_wall = (last["psi_wall"] / first["psi_wall"] - 1.0)
    det = np.asarray([h["I"] for h in hist], dtype=float)
    return {
        "Rs0_cm": first["Rs"] * 1e2,
        "Rs_end_cm": last["Rs"] * 1e2,
        "Phi0_Wb": first["Phi"],
        "Phi_end_Wb": last["Phi"],
        "Phi_rel": last["Phi"] / first["Phi"],
        "psi_wall0": first["psi_wall"],
        "psi_wall_end": last["psi_wall"],
        "dpsi_wall_rel": d_psi_wall,
        "CI0": first["CI"],
        "CI_end": last["CI"],
        "C_end": None,
        "S0_0": first["S0"],
        "S0_end": last["S0"],
        "I_shell_end_kA": (last["I_shell"] / 1e3 if n_active < det.shape[1] else 0.0),
        "I_coil_end_kA": det[-1, :n_active] / 1e3,
        "t_end_ms": last["t"] * 1e3,
        "wall_s": float(sum(h["wall"] for h in hist)),
        "n_eval": int(sum(h["n_eval"] for h in hist)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--dt", type=float, default=3e-5)
    ap.add_argument("--te", type=float, default=50.0)
    ap.add_argument("--n-shell", type=int, default=2)
    ap.add_argument("--shell", choices=("none", "resistive", "ideal"), default="ideal")
    args = ap.parse_args()

    mod = load_stage04()
    eta = mod.spitzer_resistivity(args.te)
    s0 = mod.gcm._setup(mod.gcm.SRC_DIR)
    print(f"[fluxnorm] T_e={args.te:g} eV → eta={eta:.3e} Ohm*m；"
          f"壳={args.shell}，{args.n_shell} 环；{args.steps} 步 × {args.dt * 1e6:.0f} μs")
    t0 = time.perf_counter()
    machine = mod.build_machine(s0, shell=args.shell, n_shell=args.n_shell,
                                plasma_coupling=True)
    print(f"[fluxnorm] 机器模型（含线圈↔等离子体耦合）构建 {time.perf_counter() - t0:.1f} s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for norm in ("ci", "physical", "be-loop"):
        a = stage04_args(args.steps, args.dt, norm)
        print(f"\n[fluxnorm] === 口径 {norm} ===", flush=True)
        hist = mod.run(args=a, machine=machine, eta=eta, drive=0.0, voltage="ri")
        results[norm] = summarise(hist, machine["n_active"])
        np.savez_compressed(
            OUT_DIR / f"fluxnorm_{norm.replace('-', '_')}.npz",
            times=np.array([h["t"] for h in hist]),
            Phi=np.array([h["Phi"] for h in hist]),
            Rs=np.array([h["Rs"] for h in hist]),
            psi_wall=np.array([h["psi_wall"] for h in hist]),
            CI=np.array([h["CI"] for h in hist]),
            S0=np.array([h["S0"] for h in hist]),
            Izeta0=np.array([h["Izeta0"] for h in hist]),
            I=np.array([h["I"] for h in hist]),
            gs_res=np.array([h["gs_res"] for h in hist]),
            psiq=np.array([h["psiq"] for h in hist]))
        r = results[norm]
        print(f"[fluxnorm] {norm}: Rs0={r['Rs0_cm']:.3f} cm  Φ0={r['Phi0_Wb']:.4e} Wb  "
              f"Φ末/Φ0={r['Phi_rel']:.4f}  Δψ_wall={r['dpsi_wall_rel'] * 100:+.2f}%  "
              f"外壳电流={r['I_shell_end_kA']:+.2f} kA  用时 {r['wall_s']:.1f} s",
              flush=True)

    names = {"ci": "A · 论文口径（CI 缩放）",
             "physical": "B · 物理口径（CI=1，Be=1）",
             "be-loop": "C · Be 环（CI=1，反解 Be）"}
    lines = [
        f"# 三种磁链口径对照（{args.steps} 步 × {args.dt * 1e6:.0f} μs，"
        f"{args.shell} 壳 {args.n_shell} 环，T_e={args.te:g} eV）",
        "",
        "同一套线圈电流、同一张网格；只换边界通量的磁链口径。",
        "",
        "| 口径 | 初态 Rs [cm] | 初态 Φ [Wb] | Φ末/Φ0 | ψ_wall 变化 | 外壳电流 [kA] | 终态 CI | 平衡求解次数 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for norm in ("ci", "physical", "be-loop"):
        r = results[norm]
        lines.append(
            f"| {names[norm]} | {r['Rs0_cm']:.3f} | {r['Phi0_Wb']:.4e} "
            f"| {r['Phi_rel']:.4f} | {r['dpsi_wall_rel'] * 100:+.2f}% "
            f"| {r['I_shell_end_kA']:+.2f} | {r['CI_end']:.6f} | {r['n_eval']} |")
    lines += [
        "",
        "读数：口径越自洽（A → B → C），壁面通量的漂移越小——壳的约束越咬得住。"
        "代价是平衡族不再与论文那一支相同：复现论文图用 A，算回路/壳/驱动用 C。",
    ]
    (OUT_DIR / "fluxnorm.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import csv
    keys = ["Rs0_cm", "Rs_end_cm", "Phi0_Wb", "Phi_end_Wb", "Phi_rel",
            "psi_wall0", "psi_wall_end", "dpsi_wall_rel", "CI0", "CI_end",
            "S0_0", "S0_end", "I_shell_end_kA", "t_end_ms", "wall_s", "n_eval"]
    with (OUT_DIR / "fluxnorm.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["flux_norm", *keys])
        for norm in ("ci", "physical", "be-loop"):
            w.writerow([norm, *[results[norm][k] for k in keys]])

    print("\n" + "\n".join(lines))
    print(f"\n[fluxnorm] 产物：{OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
