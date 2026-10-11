r"""给定目标 \(l_s\)，反解线圈电流分布（\(l_s\) 控制的反问题）。

正问题（`tools/frc_ls_control.py`）：线圈电流 \(\to\) 伸长比 \(E\) \(\to\)
\(l_s=2ER_s\)。这里做它的反解：

* 用响应矩阵（`outputs/design_point/shaping.json` 的 \(\partial l_s/\partial I_j\)）
  给牛顿初值；
* 每步解一次二维自由边界平衡（暖启动），用**割线**更新雅可比；
* 收敛判据 \(|l_s-l_s^*|/l_s^*\le\) `--tol`。

默认只动**中心线圈**（`tools/frc_ls_control.py` 量出它是主控旋钮，+1% → +0.78%）；
`--knob mirror` 改用"端升中降"的镜像模（等价于反向调中心线圈）。

用法：

    python tools/frc_ls_target.py --ls-cm 55        # 目标 55 cm
    python tools/frc_ls_target.py --ls-cm 45 --knob mirror

产物：`outputs/frc_formation/ls_target.json`（解出的 Il、E、Rs、Φ、ΔIl、爬升时间）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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

import design_point as dp  # noqa: E402

OUT = ROOT / "outputs" / "frc_formation"
SHAPING = ROOT / "outputs" / "design_point" / "shaping.json"
#: 设计点回路（用于把 ΔIl 折成爬升时间；与 shaping.json 一致）
COIL_L_H = 314.1624953042869e-6
COIL_TURNS = 20.0
CTRL_VOLT = 450.0


def measure(psi, diag, s) -> dict:
    """与 `tools/equilibrium_shaping.py` 同口径的整形观测量。"""
    nR, nZ = s["nR"], s["nZ"]
    zz = np.asarray(s["zz"], float)
    p2 = np.asarray(psi, float).reshape(nR, nZ, order="F")
    out = dict(Phi=float(diag["Phi_private_Wb"]), Rs=float(diag["Rs_mid"]),
               Be1_rel=float(diag.get("Be1_rel", np.nan)))
    cols = np.nonzero((p2 < 0.0).any(axis=0))[0]
    if cols.size:
        zlo, zhi = float(zz[cols].min()), float(zz[cols].max())
        out["zc"] = 0.5 * (zlo + zhi)
        out["zh"] = 0.5 * (zhi - zlo)
        out["E"] = out["zh"] / out["Rs"]
        out["ls"] = 2.0 * out["zh"]
    else:
        out["zc"] = out["zh"] = out["E"] = out["ls"] = float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="给定目标 l_s，反解线圈电流")
    ap.add_argument("--ls-cm", type=float, required=True, help="目标 l_s [cm]")
    ap.add_argument("--knob", choices=("center", "mirror"), default="center",
                    help="控制方向：只动中心线圈 / 镜像模")
    ap.add_argument("--tol", type=float, default=5e-3, help="相对容差")
    ap.add_argument("--max-iter", type=int, default=6, help="最大 Newton 步数")
    args = ap.parse_args()
    ls_target = args.ls_cm * 1e-2

    s, model, il_scale, _ = dp.build_design_model()
    s0 = dp.gcm._setup(dp.gcm.SRC_DIR)
    il_base = np.asarray(s["Il"], dtype=float)
    psi, ci = dp.warm_start_guess(il_scale)

    hist: list[dict] = []

    def run(tag, il):
        nonlocal psi, ci
        il = np.asarray(il, float)
        iz = float(s["Izeta0"]) * float(np.mean(il / il_base))  # Iζ0 随电流尺度同步
        t0 = time.perf_counter()
        out = dp.gcm.solve(dp.gcm.SRC_DIR, model=model, Il=il,
                           params=dict(Izeta0=iz), psi0=psi, CI0=ci)
        wall = time.perf_counter() - t0
        psi, ci = np.asarray(out["psiq"], float), float(out["CI"])
        m = measure(psi, out["diag"], s0)
        m["Il"] = np.asarray(il, float).tolist()
        m["C"] = float(out["C"])
        m["wall_s"] = wall
        hist.append(dict(tag=tag, **{k: m[k] for k in
                                     ("Il", "ls", "E", "Rs", "Phi", "zc", "wall_s")}))
        print(f"[ls-target] {tag:>8s} I_l={np.asarray(il) / 1e3} kA  "
              f"ls={m['ls'] * 1e2:6.2f} cm  E={m['E']:5.3f}  "
              f"Rs={m['Rs'] * 1e2:6.3f} cm  Phi={m['Phi']:.4e}  ({wall:.1f}s)")
        return m

    # ---- 基线 & 初始雅可比 ----
    m = run("base", il_base)
    if SHAPING.exists():
        sh = json.loads(SHAPING.read_text(encoding="utf-8"))
        dzh = np.asarray(sh["jacobian"]["zh"], float)
        J = 2.0 * dzh[1] if args.knob == "center" else -2.0 * dzh[1]
    else:
        J = 2.0 * 3.43e-6 if args.knob == "center" else -2.0 * 3.43e-6
    print(f"[ls-target] 目标 l_s={args.ls_cm:.2f} cm，旋钮={args.knob}，"
          f"初始 dls/dIl={J:.3e} m/A")

    il = il_base.copy()
    prev_dI = None
    for it in range(1, args.max_iter + 1):
        err = ls_target - m["ls"]
        rel = abs(err) / ls_target
        print(f"[ls-target] it{it}: l_s={m['ls'] * 1e2:.2f} cm  "
              f"err={err * 1e2:+.3f} cm（{rel:.2e}）")
        if rel <= args.tol:
            break
        if not np.isfinite(J) or J == 0:
            print("[ls-target] 雅可比失效，停")
            break
        dI = err / J
        # 限步：单步电流变化不超过 8%（大步长会把平衡推没）
        lim = 0.08 * il_base[1]
        dI = float(np.clip(dI, -lim, lim))
        if args.knob == "center":
            il_new = il.copy(); il_new[1] = il[1] + dI
        else:
            mv = np.array([0.5, -1.0, 0.5])      # 无量纲方向；步长 dI 是幅度
            il_new = il + dI * mv
        m_prev, il_prev = m, il.copy()
        il = il_new
        m = run(f"it{it}", il)
        if np.isfinite(m["ls"]) and np.isfinite(m_prev["ls"]):
            dls = m["ls"] - m_prev["ls"]
            dIl = (il[1] - il_prev[1]) if args.knob == "center" else dI
            if dIl != 0 and (dls / dIl) != 0:
                J = dls / dIl
        prev_dI = dI

    conv = abs(m["ls"] - ls_target) / ls_target
    dIl = np.asarray(il, float) - il_base
    dI_cond = dIl / COIL_TURNS
    t_ramp = np.abs(dI_cond) * COIL_L_H / CTRL_VOLT
    res = dict(
        target_ls_cm=args.ls_cm, knob=args.knob, converged=bool(conv <= args.tol),
        rel_err=float(conv),
        solution=dict(Il_J=[float(v) for v in il], Il_kA=[float(v / 1e3) for v in il],
                      dIl_J=[float(v) for v in dIl], ls_cm=float(m["ls"] * 1e2),
                      E=float(m["E"]), Rs_cm=float(m["Rs"] * 1e2),
                      Phi_Wb=float(m["Phi"]), C=float(m["C"]),
                      t_ramp_s=[float(v) for v in t_ramp]),
        history=hist,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ls_target.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n[ls-target] 结果：{'命中' if res['converged'] else '未完全命中'}"
          f"（相对差 {conv:.2e}）")
    print(f"  目标 l_s={args.ls_cm:.2f} cm → 实得 {m['ls'] * 1e2:.2f} cm")
    print(f"  线圈电流 I_l = {np.array2string(np.asarray(il) / 1e3, precision=2)} kA 安匝")
    print(f"  相对设计点 ΔI_l = {np.array2string(dIl / 1e3, precision=2)} kA 安匝")
    print(f"  ΔI_conductor = {np.array2string(dI_cond / 1e3, precision=3)} kA，"
          f"450 V 下爬升 {np.array2string(t_ramp * 1e3, precision=3)} ms")
    print(f"  E={m['E']:.3f}  Rs={m['Rs'] * 1e2:.3f} cm  Phi={m['Phi']:.4e} Wb")
    print(f"[ls-target] 产物：{OUT / 'ls_target.json'}")
    return 0 if res["converged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
