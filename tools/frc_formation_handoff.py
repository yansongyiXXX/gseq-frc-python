r"""把 0D 形成末态交给二维求解器（形成段 → Stage 3/4 的交接验证）。

读 `outputs/frc_formation/handoff.npz`（由 `tools/frc_formation_0d.py` 生成），
用其中的 \((I_l,\Phi)\) 在设计点 A 的几何上解一次二维自由边界平衡：
用割线调 \(I_{\zeta0}\) 命中 \(\Phi\)，再看解出来的 \(R_s,B_e\), 伸长比是否
与 0D/族关系一致。

这是"形成段闭环"的最后一步：0D 给出时间线与交接态，二维确认它是一个**合法的
平衡初值**。

用法：`python tools/frc_formation_handoff.py`
产物：`outputs/frc_formation/handoff_check.json`、`handoff_psi.npz`（交接态的二维 ψ 场，
供全链动图使用）
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

OUT = ROOT / "outputs" / "frc_formation"
#: 族自相似常数 c = Φ/(B_e R_s²)（Stage 1/2 实测，设计点 A 的 c=0.337）
C_FAMILY = 0.336


def solve_at(Il, izeta0, psi0, ci0, model):
    """解一次二维平衡（沿用设计点那条暖启动路径）。"""
    return dp.gcm.solve(dp.gcm.SRC_DIR, model=model, Il=Il,
                        params=dict(Izeta0=float(izeta0)),
                        psi0=psi0, CI0=ci0)


def main() -> int:
    hp = OUT / "handoff.npz"
    if not hp.exists():
        print(f"[handoff] 找不到 {hp}，先跑 tools/frc_formation_0d.py")
        return 1
    d = np.load(hp)
    Il_h = float(d["Il"])
    Phi_t = float(d["Phi"])
    t_form = float(d["t_form"])
    ls_equil_0d = float(d["ls_equil"]) if "ls_equil" in d else float("nan")
    print(f"[handoff] 0D 交接态：t_form={t_form * 1e6:.2f} µs  "
          f"Il={Il_h / 1e3:.2f} kA  Φ_target={Phi_t:.3e} Wb  "
          f"l_s^eq={ls_equil_0d:.3f} m")

    s, model, il_scale, iz_scale = dp.build_design_model()
    Il = np.full(3, Il_h, dtype=float)
    psi0, ci0 = dp.warm_start_guess(il_scale)
    s0 = dp.gcm._setup(dp.gcm.SRC_DIR)
    iz0 = float(s0["Izeta0"]) * iz_scale * (Il_h / float(np.asarray(s["Il"])[0]))

    xs, fs, outs = [], [], []
    t0 = time.perf_counter()
    x, warm = iz0, None
    for n in range(1, 4):
        out = solve_at(Il, x, psi0 if warm is None else warm["psiq"], ci0, model)
        phi = float(out["diag"]["Phi_private_Wb"])
        xs.append(x); fs.append(phi - Phi_t); outs.append(out)
        warm = out
        err = abs(phi - Phi_t) / max(Phi_t, 1e-30)
        print(f"  [{n}] Iz0={x / 1e3:8.2f} kA  Φ={phi:.4e} Wb  err={err:.2e}")
        if err <= 2e-3:
            break
        if n == 1:
            x = xs[0] * np.sqrt(Phi_t / max(phi, 1e-30))
        else:
            df, dx = fs[-1] - fs[-2], xs[-1] - xs[-2]
            if df == 0 or not np.isfinite(df):
                break
            x = xs[-1] - fs[-1] * dx / df
    wall = time.perf_counter() - t0

    k = int(np.argmin([abs(f) for f in fs]))
    out = outs[k]
    dg = out["diag"]
    Phi_2d = float(dg["Phi_private_Wb"])
    Rs_2d = float(dg["Rs_mid"])
    Be_2d = float(np.sqrt(max(2 * dp.MU0 * float(dg.get("Pmax", 0.0)), 0.0))) \
        if "Pmax" in dg else float("nan")
    Rs_family = float(np.sqrt(Phi_2d / (C_FAMILY * s["Be"]))) if Phi_2d > 0 else float("nan")
    E_2d = float(dg.get("elongation", np.nan))
    ls_2d = 2.0 * E_2d * Rs_2d if np.isfinite(E_2d) and np.isfinite(Rs_2d) else float("nan")

    res = dict(
        t_form_s=t_form, Il_handoff_A=Il_h, Phi_target_Wb=Phi_t,
        solve=dict(converged=bool(out.get("C") is not None), wall_s=wall,
                   C=float(out["C"]), CI=float(out["CI"]),
                   Izeta0_A=float(xs[k]), Phi_2d_Wb=Phi_2d,
                   Rs_mid_m=Rs_2d, Be1_rel=float(dg.get("Be1_rel", np.nan)),
                   elongation=float(dg.get("elongation", np.nan)),
                   beta_avg_sep=float(dg.get("beta_avg_sep", np.nan)),
                   gs_res=float(out.get("gs_res", np.nan))),
        consistency=dict(
            phi_rel_err=abs(Phi_2d - Phi_t) / max(Phi_t, 1e-30),
            Rs_2d_m=Rs_2d, Rs_family_m=Rs_family,
            Rs_rel_err=abs(Rs_2d - Rs_family) / Rs_family if Rs_family else float("nan"),
            ls_equil_0d_m=ls_equil_0d, ls_2d_m=ls_2d,
            ls_rel_err=(abs(ls_2d - ls_equil_0d) / ls_equil_0d
                        if ls_equil_0d and np.isfinite(ls_2d) else float("nan"))),
    )
    (OUT / "handoff_check.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    # 交接态的二维 ψ 场（新增可选产物，供"形成 → 交接 → 放电"全链动图使用；
    # 只多写一个文件，不改任何数值路径）
    psi_2d = np.asarray(out["psiq"], float).reshape(s0["nR"], s0["nZ"], order="F")
    np.savez_compressed(
        OUT / "handoff_psi.npz",
        psi=np.ascontiguousarray(psi_2d),
        rr=np.asarray(s0["rr"], float), zz=np.asarray(s0["zz"], float),
        rl=np.asarray(s0["rl"], float), zl=np.asarray(s0["zl"], float),
        hl=np.asarray(s0["hl"], float), wl=np.asarray(s0["wl"], float),
        Rw=float(s0["Rw"]), zw=float(s0["zw"]),
        Il=np.asarray(Il, float), Izeta0=float(xs[k]), Phi=float(Phi_2d),
        Rs=float(Rs_2d), elongation=float(dg.get("elongation", np.nan)),
        beta_avg_sep=float(dg.get("beta_avg_sep", np.nan)),
        t_form=float(t_form), C=float(out["C"]), CI=float(out["CI"]))
    print(f"[handoff] 交接态 ψ 场：{OUT / 'handoff_psi.npz'}")

    print("[handoff] 二维结果：")
    print(f"  Iz0={xs[k] / 1e3:.2f} kA  C={out['C']:.4f}  CI={out['CI']:.4f}  "
          f"gs_res={out.get('gs_res', float('nan')):.2e}  ({wall:.1f} s)")
    print(f"  Φ_2d={Phi_2d:.4e} Wb（目标 {Phi_t:.4e}，相对差 "
          f"{res['consistency']['phi_rel_err']:.2e}）")
    print(f"  Rs_2d={Rs_2d * 1e2:.3f} cm  族预测 Rs={Rs_family * 1e2:.3f} cm  "
          f"（相对差 {res['consistency']['Rs_rel_err']:.2e}）")
    print(f"  伸长比={dg.get('elongation', float('nan')):.3f}  "
          f"<β>={dg.get('beta_avg_sep', float('nan')):.3f}")
    print(f"  l_s：0D 平衡 {ls_equil_0d * 1e2:.2f} cm  二维(2·E·Rs) {ls_2d * 1e2:.2f} cm  "
          f"（相对差 {res['consistency']['ls_rel_err']:.2e}）")
    print(f"[handoff] 产物：{OUT / 'handoff_check.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
