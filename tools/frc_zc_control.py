r"""FRC 轴向平移控制：差模线圈 → 位置 z_c。

总目标第 3 条里的"平移与俘获"：在**同一套线圈**内把 FRC 沿轴移动。控制旋钮是
**差模**（两端线圈反向）——tools/equilibrium_shaping.py 已量出它只平移、几乎不改变
长度和通量。

本工具做两件事：

1. **正向**：从 outputs/design_point/shaping.json 读灵敏度，给出"差模 +1 kA →
   平移多少"；
2. **反解**：给目标 z_c*，用割线牛顿解出需要的差模电流（每步解一次二维平衡）。

用法：

    python tools/frc_zc_control.py                 # 只打印正向灵敏度
    python tools/frc_zc_control.py --zc-cm 3       # 反解到 +3 cm

产物：outputs/frc_formation/zc_control.json
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
COIL_L_H = 314.1624953042869e-6
COIL_TURNS = 20.0
CTRL_VOLT = 450.0
DIFF = np.array([1.0, 0.0, -1.0])          # 差模方向：两端反向
ZC_QUANT = 0.75 / 128.0 / 2.0              # z_c 网格量化步长 [m]


def measure(psi, diag, s) -> dict:
    nR, nZ = s["nR"], s["nZ"]
    zz = np.asarray(s["zz"], float)
    p2 = np.asarray(psi, float).reshape(nR, nZ, order="F")
    out = dict(Phi=float(diag["Phi_private_Wb"]), Rs=float(diag["Rs_mid"]))
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
    ap = argparse.ArgumentParser(description="FRC 轴向平移控制（差模）")
    ap.add_argument("--zc-cm", type=float, default=None,
                    help="目标轴向位置 [cm]；不给就只打印正向灵敏度")
    ap.add_argument("--tol", type=float, default=1.0, help="网格量化步数容差")
    ap.add_argument("--max-iter", type=int, default=6)
    args = ap.parse_args()

    if not SHAPING.exists():
        print(f"[zc] 找不到 {SHAPING}")
        return 1
    sh = json.loads(SHAPING.read_text(encoding="utf-8"))
    nom = sh["nominal"]
    Jc = np.asarray(sh["jacobian"]["zc"], float)
    Il_base = np.asarray(nom["Il"], float)
    J = float(Jc @ DIFF)
    print(f"[zc] 设计点：z_c={nom['zc'] * 1e2:+.3f} cm  l_s={2 * nom['zh'] * 1e2:.2f} cm")
    print("[zc] 正向灵敏度（差模 = +I1 / -I3）：")
    for j in range(3):
        print(f"     单路线圈 I{j + 1}: dzc/dIj = {Jc[j]:+.3e} m/A"
              f"（+1 kA -> {Jc[j] * 1e3 * 1e2:+.3f} cm）")
    print(f"     差模合成: dzc/d(a) = {J:+.3e} m/A"
          f"（差模 +1 kA -> {J * 1e3 * 1e2:+.3f} cm）")
    print(f"     量化: z_c 的网格步长 = {ZC_QUANT * 1e3:.3f} mm")

    res = dict(nominal=dict(zc_m=float(nom["zc"]), ls_m=2 * float(nom["zh"]),
                            Il_J=[float(v) for v in Il_base]),
               jacobian_zc=[float(v) for v in Jc],
               diff_gain_m_per_A=J, zc_quant_m=ZC_QUANT)
    if args.zc_cm is None:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "zc_control.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[zc] 产物：{OUT / 'zc_control.json'}")
        return 0

    zc_target = args.zc_cm * 1e-2
    s, model, il_scale, _ = dp.build_design_model()
    s0 = dp.gcm._setup(dp.gcm.SRC_DIR)
    psi, ci = dp.warm_start_guess(il_scale)
    hist = []

    def run(tag, il):
        nonlocal psi, ci
        t0 = time.perf_counter()
        out = dp.gcm.solve(dp.gcm.SRC_DIR, model=model, Il=np.asarray(il, float),
                           params=dict(Izeta0=float(s["Izeta0"])), psi0=psi, CI0=ci)
        psi, ci = np.asarray(out["psiq"], float), float(out["CI"])
        m = measure(psi, out["diag"], s0)
        m["wall_s"] = time.perf_counter() - t0
        hist.append(dict(tag=tag, a=float(0.5 * (il[0] - il[2])),
                         zc=float(m["zc"]), ls=float(m["ls"]), Rs=float(m["Rs"])))
        print(f"[zc] {tag:>7s} 差模 a={0.5 * (il[0] - il[2]) / 1e3:+7.2f} kA  "
              f"z_c={m['zc'] * 1e2:+7.3f} cm  l_s={m['ls'] * 1e2:6.2f} cm  "
              f"Rs={m['Rs'] * 1e2:6.3f} cm（{m['wall_s']:.1f}s）")
        return m

    m = run("base", Il_base)
    a = 0.0
    for it in range(1, args.max_iter + 1):
        err = zc_target - m["zc"]
        print(f"[zc] it{it}: z_c={m['zc'] * 1e2:+.3f} cm  "
              f"err={err * 1e2:+.3f} cm（{abs(err) / ZC_QUANT:.1f} 格）")
        if abs(err) <= args.tol * ZC_QUANT:
            break
        if not np.isfinite(J) or J == 0:
            print("[zc] 雅可比失效，停")
            break
        da = float(np.clip(err / J, -0.2 * Il_base[0], 0.2 * Il_base[0]))
        a += da
        il = Il_base + a * DIFF
        m_prev = m
        m = run(f"it{it}", il)
        if np.isfinite(m["zc"]) and np.isfinite(m_prev["zc"]) and da != 0:
            Jn = (m["zc"] - m_prev["zc"]) / da
            if Jn != 0 and np.isfinite(Jn):
                J = Jn

    conv = abs(m["zc"] - zc_target)
    il = Il_base + a * DIFF
    d_il = a * DIFF
    t_ramp = np.abs(d_il / COIL_TURNS) * COIL_L_H / CTRL_VOLT
    res.update(target_zc_cm=args.zc_cm, converged=bool(conv <= args.tol * ZC_QUANT),
               final=dict(a_A=float(a), Il_J=[float(v) for v in il],
                          zc_m=float(m["zc"]), ls_m=float(m["ls"]),
                          Rs_m=float(m["Rs"]), dIl_J=[float(v) for v in d_il],
                          t_ramp_s=[float(v) for v in t_ramp]),
               history=hist)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "zc_control.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[zc] {'命中' if res['converged'] else '未完全命中'}："
          f"目标 {args.zc_cm:+.3f} cm -> 实得 {m['zc'] * 1e2:+.3f} cm")
    print(f"  差模幅度 a={a / 1e3:+.2f} kA -> I_l 两端 {il[0] / 1e3:.2f} / "
          f"{il[2] / 1e3:.2f} kA")
    print(f"  dIl={np.array2string(d_il / 1e3, precision=2)} kA，"
          f"450 V 下爬升 {np.array2string(t_ramp * 1e3, precision=2)} ms")
    print(f"  l_s={m['ls'] * 1e2:.2f} cm（应基本不变）")
    print(f"[zc] 产物：{OUT / 'zc_control.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
