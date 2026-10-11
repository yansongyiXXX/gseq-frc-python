r"""FRC 轴向平移的轨迹跟踪：z_c*(t) → 差模线圈电流波形。

和 `tools/frc_ls_track.py` 同构，只是把目标量从长度换成位置：

* 控制旋钮：**差模** a（两端反向 I1+a / I3−a），来自 8.19 的灵敏度；
* 可选 `--hold-ls`：每步再用**中心线圈**把长度压回参考值，把 8.19 里那 2.3% 的
  位置↔长度二阶耦合解掉；
* 每步解一次二维平衡（暖启动），割线牛顿命中 z_c*，并检查所需电压。

用法：

    python tools/frc_zc_track.py --final-cm 3 --t-ms 0.2 --n 6
    python tools/frc_zc_track.py --final-cm 3 --t-ms 0.2 --n 6 --hold-ls

产物：`outputs/frc_formation/zc_track.{npz,json,png}`
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
DIFF = np.array([1.0, 0.0, -1.0])
ZC_QUANT = 0.75 / 128.0 / 2.0


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
    ap = argparse.ArgumentParser(description="平移轨迹跟踪 → 差模电流波形")
    ap.add_argument("--final-cm", type=float, default=3.0, help="末位置 [cm]")
    ap.add_argument("--t-ms", type=float, default=0.20, help="平移时长 [ms]")
    ap.add_argument("--n", type=int, default=6, help="轨迹采样点数")
    ap.add_argument("--tol", type=float, default=1.0, help="网格量化步数容差")
    ap.add_argument("--hold-ls", action="store_true",
                    help="用中心线圈把长度压回参考值（解掉位置↔长度耦合）")
    args = ap.parse_args()

    sh = json.loads(SHAPING.read_text(encoding="utf-8"))
    Il_base = np.asarray(sh["nominal"]["Il"], float)
    J = float(np.asarray(sh["jacobian"]["zc"], float) @ DIFF)   # dzc/da
    J_ls = 2.0 * float(np.asarray(sh["jacobian"]["zh"], float)[1])  # dls/dI2
    J_ref, J_ls_ref = J, J_ls

    s, model, il_scale, _ = dp.build_design_model()
    s0 = dp.gcm._setup(dp.gcm.SRC_DIR)
    psi, ci = dp.warm_start_guess(il_scale)

    def solve(il):
        nonlocal psi, ci
        out = dp.gcm.solve(dp.gcm.SRC_DIR, model=model, Il=np.asarray(il, float),
                           params=dict(Izeta0=float(s["Izeta0"])), psi0=psi, CI0=ci)
        psi, ci = np.asarray(out["psiq"], float), float(out["CI"])
        m = measure(psi, out["diag"], s0)
        m["Il"] = np.asarray(il, float)
        return m

    m = solve(Il_base)
    zc0, ls_ref = m["zc"], m["ls"]
    print(f"[zc-track] 起点 z_c={zc0 * 1e2:+.3f} cm  l_s={ls_ref * 1e2:.2f} cm")

    ts = np.linspace(0.0, args.t_ms * 1e-3, args.n + 1)
    zc_star = np.linspace(zc0, args.final_cm * 1e-2, args.n + 1)
    a = 0.0
    il = Il_base.copy()
    rows = [dict(t=0.0, zc_star=zc0, zc=zc0, a=0.0, Il=il.tolist(),
                 ls=ls_ref, Rs=m["Rs"])]

    t0 = time.perf_counter()
    for k in range(1, args.n + 1):
        zt = float(zc_star[k])
        # ---- 差模牛顿：命中位置 ----
        for _it in range(3):
            err = zt - m["zc"]
            if abs(err) <= args.tol * ZC_QUANT:
                break
            if not np.isfinite(J) or J == 0:
                J = J_ref
            da = float(np.clip(err / J, -0.2 * Il_base[0], 0.2 * Il_base[0]))
            il = Il_base + (a + da) * DIFF
            m_prev = m
            m = solve(il)
            if np.isfinite(m["zc"]) and np.isfinite(m_prev["zc"]) and da != 0:
                Jn = (m["zc"] - m_prev["zc"]) / da
                if Jn != 0 and np.isfinite(Jn):
                    J = Jn
            a += da
        # ---- 可选：中心线圈把长度压回参考 ----
        if args.hold_ls:
            for _it in range(2):
                lerr = ls_ref - m["ls"]
                if abs(lerr) <= 0.5 * 1.17e-2:      # 半格
                    break
                if not np.isfinite(J_ls) or J_ls == 0:
                    J_ls = J_ls_ref
                dI2 = float(np.clip(lerr / J_ls, -0.1 * Il_base[1], 0.1 * Il_base[1]))
                il = Il_base + a * DIFF
                il = il.copy(); il[1] += dI2
                m_prev = m
                m = solve(il)
                if np.isfinite(m["ls"]) and np.isfinite(m_prev["ls"]) and dI2 != 0:
                    Jn = (m["ls"] - m_prev["ls"]) / dI2
                    if Jn != 0 and np.isfinite(Jn):
                        J_ls = Jn
        il = m["Il"]
        rows.append(dict(t=float(ts[k]), zc_star=zt, zc=float(m["zc"]),
                         a=float(0.5 * (il[0] - il[2])), Il=il.tolist(),
                         ls=float(m["ls"]), Rs=float(m["Rs"])))
        print(f"[zc-track] t={ts[k] * 1e3:.3f} ms  目标 {zt * 1e2:+6.3f} cm  "
              f"实得 {m['zc'] * 1e2:+6.3f} cm  a={a / 1e3:+6.2f} kA  "
              f"I2={il[1] / 1e3:6.2f} kA  l_s={m['ls'] * 1e2:5.2f} cm  "
              f"err={abs(m['zc'] - zt) / ZC_QUANT:.1f} 格")
    wall = time.perf_counter() - t0

    tt = np.array([r["t"] for r in rows])
    zc = np.array([r["zc"] for r in rows])
    zs = np.array([r["zc_star"] for r in rows])
    a_arr = np.array([r["a"] for r in rows])
    ls_arr = np.array([r["ls"] for r in rows])
    dt = np.diff(tt)
    dIdt = np.diff(a_arr / COIL_TURNS) / np.where(dt > 0, dt, 1.0)
    Vreq = COIL_L_H * np.abs(dIdt)
    err = np.abs(zc - zs)
    res = dict(final_cm=args.final_cm, t_ms=args.t_ms, n=args.n,
               hold_ls=bool(args.hold_ls),
               max_err_cm=float(np.max(err) * 1e2),
               max_err_cells=float(np.max(err) / ZC_QUANT),
               ls_drift_cm=float((ls_arr[-1] - ls_arr[0]) * 1e2),
               a_final_kA=float(a_arr[-1] / 1e3),
               max_voltage_V=float(np.max(Vreq)) if Vreq.size else 0.0,
               track=[dict(t=r["t"], zc_star=r["zc_star"], zc=r["zc"], a=r["a"],
                           I2=r["Il"][1], ls=r["ls"], Rs=r["Rs"]) for r in rows],
               wall_s=wall)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "zc_track.npz", t=tt, zc_star=zs, zc=zc,
                        a=a_arr, ls=ls_arr, Vreq=Vreq,
                        Il=np.array([r["Il"] for r in rows]))
    (OUT / "zc_track.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[zc-track] 跟踪误差 max={res['max_err_cm']:.3f} cm"
          f"（{res['max_err_cells']:.1f} 格）  差模末值 {res['a_final_kA']:+.2f} kA")
    print(f"[zc-track] l_s 漂移 {res['ls_drift_cm']:+.2f} cm"
          f"{'（--hold-ls 已压）' if args.hold_ls else ''}   "
          f"所需电压 max={res['max_voltage_V']:.0f} V   总耗时 {wall:.0f} s")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from gseqfrc import mstyle
        mstyle.install()
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
        ax[0].plot(tt * 1e3, zs * 1e2, "k--", lw=1.6, label="目标 z_c*")
        ax[0].plot(tt * 1e3, zc * 1e2, "o-", color="C0", label="实得 z_c")
        ax[0].set_xlabel("t (ms)"); ax[0].set_ylabel("z_c (cm)")
        ax[0].set_title("位置跟踪"); ax[0].legend()
        ax[1].plot(tt * 1e3, a_arr / 1e3, "o-", color="C3")
        ax[1].set_xlabel("t (ms)"); ax[1].set_ylabel("差模 a (kA)")
        ax[1].set_title("差模线圈波形")
        ax[2].plot(tt * 1e3, ls_arr * 1e2, "s-", color="C2")
        ax[2].set_xlabel("t (ms)"); ax[2].set_ylabel("l_s (cm)")
        ax[2].set_title(f"长度漂移 {res['ls_drift_cm']:+.2f} cm")
        fig.tight_layout()
        fig.savefig(OUT / "zc_track.png", dpi=140)
        plt.close(fig)
        print(f"[zc-track] 产物：{OUT / 'zc_track.json'} / {OUT / 'zc_track.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[zc-track] 出图跳过：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
