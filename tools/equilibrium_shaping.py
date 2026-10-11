r"""Slow equilibrium shaping: 3 coil currents -> axial position / elongation.

Solves the design-point A equilibrium for the nominal coil set and for three
single-coil perturbations, builds the response matrix
\(\partial (z_c, z_h, R_s, \Phi)/\partial I_j\), then reports the sensitivities
of the three physical coil modes:

* 共模  (all three up)                -> size / private flux
* 差模  (end coils opposite)          -> axial position
* 镜像模 (ends up, centre down)       -> elongation / mirror ratio

Finally it tries a 3x3 Newton solve for a target
\((z_c, E, \Phi)=(\text{shift}, E_{\rm new}, \Phi_0)\).

用法：
    python tools/equilibrium_shaping.py
产物：outputs/design_point/shaping.json
"""

from __future__ import annotations

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

gcm = dp.gcm
SRC = dp.SRC
OUT = ROOT / "outputs" / "design_point" / "shaping.json"
TARGET_ZC = 0.02        # +2 cm axial shift
TARGET_ELONG = 3.6      # from 3.16


def measure(psi, diag, s):
    """Shape observables of one equilibrium."""
    nR, nZ = s["nR"], s["nZ"]
    rr = np.asarray(s["rr"], float)
    zz = np.asarray(s["zz"], float)
    p2 = np.asarray(psi, float).reshape(nR, nZ, order="F")
    core = p2 < 0.0
    out = dict(Phi=float(diag["Phi_private_Wb"]), Rs=float(diag["Rs_mid"]),
               Be1_rel=float(diag.get("Be1_rel", np.nan)))
    cols = np.nonzero(core.any(axis=0))[0]
    if cols.size:
        zlo, zhi = float(zz[cols].min()), float(zz[cols].max())
        out["zc"] = 0.5 * (zlo + zhi)
        out["zh"] = 0.5 * (zhi - zlo)
        out["E"] = out["zh"] / out["Rs"]
    else:
        out["zc"] = out["zh"] = out["E"] = float("nan")
    return out


def main() -> None:
    s, model, il_scale, _ = dp.build_design_model()
    s0 = gcm._setup(SRC)
    params = dict(Izeta0=float(s["Izeta0"]))
    il0 = np.asarray(s["Il"], dtype=float)

    psi0, ci0 = dp.warm_start_guess(il_scale)
    psi, ci = psi0, ci0
    cases = {}

    def run(tag, il):
        nonlocal psi, ci
        t0 = time.perf_counter()
        out = gcm.solve(SRC, model=model, Il=il, params=params, psi0=psi, CI0=ci)
        wall = time.perf_counter() - t0
        psi, ci = np.asarray(out["psiq"], float), float(out["CI"])
        m = measure(psi, out["diag"], s0)
        m["Il"] = np.asarray(il, float).tolist()
        m["wall_s"] = wall
        cases[tag] = m
        print(f"[shape] {tag:>10s} Il={np.array2string(np.asarray(il)/1e3, precision=2)} kA  "
              f"zc={m['zc']*1e2:+6.3f} cm  zh={m['zh']*1e2:6.3f} cm  Rs={m['Rs']*1e2:6.3f} cm  "
              f"E={m['E']:5.3f}  Phi={m['Phi']:.4e} Wb  Be1/Be-1={m['Be1_rel']:+.2e}  ({wall:.1f}s)")
        return m

    nom = run("nominal", il0)

    eps = 0.03
    keys = ("zc", "zh", "Rs", "Phi", "E")
    J = np.zeros((len(keys), 3))
    for j in range(3):
        il = il0.copy()
        il[j] *= 1.0 + eps
        m = run(f"I{j+1}+3%", il)
        for i, k in enumerate(keys):
            J[i, j] = (m[k] - nom[k]) / (il0[j] * eps)

    pct = 0.01 * il0                          # +1% 的安匝数
    print("\n[shape] 每 +1% 单路线圈电流引起的变化：")
    for i, k in enumerate(keys):
        print(f"   d{k:>3s} = " + "  ".join(f"I{j+1}:{J[i, j]*pct[j]:+.4g}" for j in range(3))
              + f"   (基准 {nom[k]:.4g})")

    # physical modes: common / differential / mirror
    common = np.ones(3)
    diff = np.array([1.0, 0.0, -1.0])
    mirror = np.array([0.5, -1.0, 0.5])
    modes = {}
    for name, v in (("common", common), ("differential", diff), ("mirror", mirror)):
        modes[name] = {k: float(J[i] @ v) for i, k in enumerate(keys)}
    print("\n[shape] 物理模式的灵敏度（每 +1 kA·匝 的该模式）：")
    for name, d in modes.items():
        print(f"   {name:>12s}: " + "  ".join(f"{k}={d[k]:+.4g}" for k in keys))

    # ---- 目标：位置用差模、伸长用中间线圈（通量由剖面闭合决定）------------
    il = il0.copy()
    y = np.array([nom["zc"], nom["E"]])
    dzc_dI1 = J[0, 0]
    dE_dI2 = J[4, 1]
    hist = []
    for it in range(4):
        ez, eE = TARGET_ZC - y[0], TARGET_ELONG - y[1]
        if abs(ez) < 2e-4 and abs(eE) < 2e-3:
            break
        d = ez / (2.0 * dzc_dI1)            # 两端反向等量 = 纯差模
        il = il + np.array([d, eE / dE_dI2, -d])
        lim = 0.5 * il0
        il = np.clip(il, il0 - lim, il0 + lim)
        m = run(f"target{it+1}", il)
        y_new = np.array([m["zc"], m["E"]])
        if it == 0 and abs(il[1] - il0[1]) > 1e-9:
            dE_dI2 = (y_new[1] - nom["E"]) / (il[1] - il0[1])
        y = y_new
        hist.append(dict(iter=it + 1, Il=il.tolist(), y=y.tolist()))

    conv = np.array([abs(y[0] - TARGET_ZC) / TARGET_ZC, abs(y[1] - TARGET_ELONG) / TARGET_ELONG])

    # ---- engineering: how long to ramp at 450 V ---------------------------
    L = 314.1624953042869e-6
    dI_cond = (il - il0) / 20.0            # amp-turns -> conductor amps
    t_ramp = np.abs(dI_cond) * L / 450.0

    payload = dict(
        nominal=nom, cases=cases, jacobian={k: J[i].tolist() for i, k in enumerate(keys)},
        modes=modes, target=dict(zc=TARGET_ZC, E=TARGET_ELONG, Phi=nom["Phi"]),
        newton_history=hist, final=dict(Il=il.tolist(), y=y.tolist(), conv=conv.tolist(),
                                        ramp_s=t_ramp.tolist()),
        coil_L_H=L,
    )
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    fin = cases[f"target{len(hist)}"] if hist else nom
    print(f"\n[shape] 目标 zc={TARGET_ZC*1e2:+.2f} cm、E={TARGET_ELONG}")
    print(f"[shape] 最终 zc={y[0]*1e2:+.3f} cm、E={y[1]:.3f}、Phi={fin['Phi']:.4e} Wb"
          f"（基准 {nom['Phi']:.4e}）、Be1/Be-1={fin['Be1_rel']:+.2e}、"
          f"残差 {conv[0]*100:.2f}% / {conv[1]*100:.2f}%")
    print(f"[shape] 线圈电流 {np.array2string(il/1e3, precision=2)} kA·匝"
          f"（相对基准 {np.array2string((il/il0-1)*100, precision=2)} %）")
    print(f"[shape] 450 V 下最慢一路爬升需 {t_ramp.max()*1e3:.2f} ms")
    print(f"[shape] wrote {OUT}")


if __name__ == "__main__":
    main()
