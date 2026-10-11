r"""\(l_s\) 闭环跟踪：给目标长度轨迹 \(l_s^*(t)\)，生成线圈电流波形。

把反解（`tools/frc_ls_target.py`）接到一条**时间轨迹**上：压缩段希望长度从
\(l_s^0\) 变到 \(l_s^{\rm final}\)，就在若干时刻上逐点反解线圈电流，拼成
\(I_l(t)\)，并检查

* 跟踪误差 \(|l_s(t)-l_s^*(t)|/l_s^*\)；
* 所需电压 \(V=L\,\mathrm dI_{\rm cond}/\mathrm dt\)（\(L\) 取 20 匝回路）；
* 准静态假设是否成立（压缩时间 vs Alfvén 时间）。

注意：反解给的是**每个目标长度对应的平衡电流**，是准静态（慢压缩）口径。真做
快压缩要把它当"参考轨迹 + 反馈"，动力学部分仍由时间推进负责。

用法：

    python tools/frc_ls_track.py --final-cm 45 --t-ms 0.2 --n 6

产物：`outputs/frc_formation/ls_track.{npz,json,png}`
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
COIL_L_H = 314.1624953042869e-6      # 20 匝维持回路自感
COIL_TURNS = 20.0
MU0 = 4 * np.pi * 1e-7
QE = 1.602176634e-19
MP = 1.67262192369e-27


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
    ap = argparse.ArgumentParser(description="l_s 闭环跟踪 → 线圈电流波形")
    ap.add_argument("--final-cm", type=float, default=45.0, help="压缩末长度 [cm]")
    ap.add_argument("--t-ms", type=float, default=0.20, help="压缩时长 [ms]")
    ap.add_argument("--n", type=int, default=6, help="轨迹采样点数")
    ap.add_argument("--tol", type=float, default=1.5e-2,
                    help="单点相对容差（l_s 由网格量出，量化约 1.2 cm）")
    ap.add_argument("--te", type=float, default=30.0, help="电子温度 [eV]（准静态判据）")
    ap.add_argument("--ti", type=float, default=30.0, help="离子温度 [eV]")
    ap.add_argument("--zc-cm", type=float, default=0.0,
                    help="压缩前先把 FRC 平移到这个位置 [cm]（差模固定住）")
    args = ap.parse_args()

    s, model, il_scale, _ = dp.build_design_model()
    s0 = dp.gcm._setup(dp.gcm.SRC_DIR)
    il_base = np.asarray(s["Il"], dtype=float)
    iz_base = float(s["Izeta0"])
    psi, ci = dp.warm_start_guess(il_scale)
    J = 6.866e-6                       # dls/dIl（中心线圈），响应矩阵量级
    if SHAPING.exists():
        J = 2.0 * float(np.asarray(json.loads(
            SHAPING.read_text(encoding="utf-8"))["jacobian"]["zh"], float)[1])
    J_ref = J                          # 备用：割线失效（量化导致 dls=0）时回到它

    def solve(il):
        nonlocal psi, ci
        il = np.asarray(il, float)
        iz = iz_base * float(np.mean(il / il_base))
        out = dp.gcm.solve(dp.gcm.SRC_DIR, model=model, Il=il,
                           params=dict(Izeta0=iz), psi0=psi, CI0=ci)
        psi, ci = np.asarray(out["psiq"], float), float(out["CI"])
        m = measure(psi, out["diag"], s0)
        m["Il"] = il
        return m

    m = solve(il_base)
    ls0 = m["ls"]
    print(f"[track] 起点 l_s={ls0 * 1e2:.2f} cm  z_c={m['zc'] * 1e2:+.3f} cm（设计点）")

    # ---- 可选：先把 FRC 平移到 --zc-cm，差模固定住（位置保持）----
    a_diff = 0.0
    if abs(args.zc_cm) > 1e-9:
        Jz = 6.866e-6
        for _it in range(3):
            err = args.zc_cm * 1e-2 - m["zc"]
            if abs(err) <= 2.93e-3:
                break
            da = float(np.clip(err / Jz, -0.2 * il_base[0], 0.2 * il_base[0]))
            il = il_base + (a_diff + da) * np.array([1.0, 0.0, -1.0])
            m_prev = m
            m = solve(il)
            if da != 0:
                Jn = (m["zc"] - m_prev["zc"]) / da
                if Jn != 0 and np.isfinite(Jn):
                    Jz = Jn
            a_diff += da
        ls0 = m["ls"]
        print(f"[track] 平移到位：z_c={m['zc'] * 1e2:+.3f} cm（差模 {a_diff / 1e3:+.2f} kA）"
              f"  l_s={ls0 * 1e2:.2f} cm（平移已把它改了）")

    ts = np.linspace(0.0, args.t_ms * 1e-3, args.n + 1)
    ls_star = np.linspace(ls0, args.final_cm * 1e-2, args.n + 1)
    il = il_base + a_diff * np.array([1.0, 0.0, -1.0])   # 差模固定，长度用中心线圈调
    rows = [dict(t=0.0, ls_star=ls0, ls=ls0, Il=il.tolist(), E=m["E"],
                 Rs=m["Rs"], Phi=m["Phi"])]

    t0 = time.perf_counter()
    for k in range(1, args.n + 1):
        tgt = float(ls_star[k])
        for _ in range(3):
            err = tgt - m["ls"]
            if abs(err) / tgt <= args.tol:
                break
            if not np.isfinite(J) or J == 0:
                J = J_ref
            dI = float(np.clip(err / J, -0.08 * il_base[1], 0.08 * il_base[1]))
            il = il.copy(); il[1] += dI
            m_prev = m
            m = solve(il)
            if np.isfinite(m["ls"]) and np.isfinite(m_prev["ls"]) and dI != 0:
                Jnew = (m["ls"] - m_prev["ls"]) / dI
                if Jnew != 0 and np.isfinite(Jnew):
                    J = Jnew
        rows.append(dict(t=float(ts[k]), ls_star=tgt, ls=float(m["ls"]),
                         Il=il.tolist(), E=float(m["E"]), Rs=float(m["Rs"]),
                         Phi=float(m["Phi"])))
        print(f"[track] t={ts[k] * 1e3:.3f} ms  目标 {tgt * 1e2:6.2f} cm  "
              f"实得 {m['ls'] * 1e2:6.2f} cm  I2={il[1] / 1e3:6.2f} kA  "
              f"err={abs(m['ls'] - tgt) / tgt:.1e}")
    wall = time.perf_counter() - t0

    # ---- 波形与可行性 ----
    tt = np.array([r["t"] for r in rows])
    ls_arr = np.array([r["ls"] for r in rows])
    I2 = np.array([r["Il"][1] for r in rows])
    Icond = I2 / COIL_TURNS
    dt = np.diff(tt)
    dIdt = np.diff(Icond) / np.where(dt > 0, dt, 1.0)
    Vreq = COIL_L_H * np.abs(dIdt)
    ls_err = np.abs(ls_arr - ls_star) / ls_star

    # 准静态判据：压缩时间 vs 轴向 Alfvén 时间
    cs = float(np.sqrt((args.te + args.ti) * QE / MP))
    tau_ax = float(ls0 / cs)
    quasi = "成立" if args.t_ms * 1e-3 > 10 * tau_ax else \
        f"偏快（压缩 {args.t_ms * 1e-3 * 1e6:.0f} µs vs 10τ_ax={10 * tau_ax * 1e6:.0f} µs）"

    res = dict(
        final_ls_cm=args.final_cm, duration_ms=args.t_ms, n=args.n,
        max_rel_err=float(np.max(ls_err)),
        max_voltage_V=float(np.max(Vreq)) if Vreq.size else 0.0,
        max_dIdt_A_per_s=float(np.max(np.abs(dIdt))) if dIdt.size else 0.0,
        c_s_km_s=cs / 1e3, tau_ax_us=tau_ax * 1e6, quasi_static=quasi,
        track=[dict(t=r["t"], ls_star=r["ls_star"], ls=r["ls"],
                    I2_A=r["Il"][1], E=r["E"], Rs=r["Rs"], Phi=r["Phi"])
               for r in rows],
        wall_s=wall,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "ls_track.npz", t=tt, ls_star=ls_star, ls=ls_arr,
                        I2=I2, Icond=Icond, Vreq=Vreq, ls_err=ls_err)
    (OUT / "ls_track.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n[track] 跟踪误差 max={res['max_rel_err']:.2e}  "
          f"所需电压 max={res['max_voltage_V']:.0f} V  "
          f"max|dI_c/dt|={res['max_dIdt_A_per_s']:.3e} A/s")
    print(f"[track] 准静态：{quasi}（c_s={cs / 1e3:.1f} km/s，τ_ax={tau_ax * 1e6:.1f} µs）")
    print(f"[track] 总耗时 {wall:.0f} s")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from gseqfrc import mstyle
        mstyle.install()
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
        ax[0].plot(tt * 1e3, ls_star * 1e2, "k--", lw=1.6, label="目标 l_s*")
        ax[0].plot(tt * 1e3, ls_arr * 1e2, "o-", color="C0", label="实得 l_s")
        ax[0].set_xlabel("t (ms)"); ax[0].set_ylabel("l_s (cm)")
        ax[0].set_title("长度跟踪"); ax[0].legend()
        ax[1].plot(tt * 1e3, I2 / 1e3, "o-", color="C3")
        ax[1].set_xlabel("t (ms)"); ax[1].set_ylabel("I_2 (kA 安匝)")
        ax[1].set_title("中心线圈电流波形")
        ax[2].plot(tt[1:] * 1e3, Vreq, "s-", color="C2")
        ax[2].set_xlabel("t (ms)"); ax[2].set_ylabel("|V| (V)")
        ax[2].set_title("所需电压 |L dI_c/dt|")
        fig.tight_layout()
        fig.savefig(OUT / "ls_track.png", dpi=140)
        plt.close(fig)
        print(f"[track] 产物：{OUT / 'ls_track.json'} / {OUT / 'ls_track.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[track] 出图跳过：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
