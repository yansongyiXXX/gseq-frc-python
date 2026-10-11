r"""端到端放电时间线：平移 → 压缩 → 平顶维持（同一条回路、同一套线圈）。

三段串起来跑，控制量是二维的：

* **差模** a（两端反向 I1+a / I3−a）→ 轴向位置 z_c；
* **中心线圈** I2 → 长度 l_s；
* 通量按 \(\Phi^{n+1}=\Phi^n-(S_0-S_{\rm RMF})\Delta t\) 推进，`--rmf` 时由 RMF 托住。

每步做一次二维牛顿（割线更新增益），把 (z_c, l_s) 同时拉到各自的目标。
响应矩阵来自 `outputs/design_point/shaping.json`，在这套几何下它对
(z_c, l_s) 是**严格对角**的（差模只动位置、中心线圈只动长度）。

用法：

    python tools/frc_discharge_run.py --rmf                 # 默认三段
    python tools/frc_discharge_run.py --rmf --zc-cm 3 --ls-cm 45

产物：`outputs/frc_formation/discharge.{npz,json,png}`
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

import design_point as dp                      # noqa: E402
import frc_evolve_stage04 as s4                # noqa: E402
from frc_compress_run import rmf_drive         # noqa: E402

OUT = ROOT / "outputs" / "frc_formation"
SHAPING = ROOT / "outputs" / "design_point" / "shaping.json"
COIL_L_H = 314.1624953042869e-6
COIL_TURNS = 20.0
DIFF = np.array([1.0, 0.0, -1.0])
ZC_QUANT = 0.75 / 128.0 / 2.0                  # 2.93 mm
LS_QUANT = 2.0 * 0.75 / 128.0                  # 5.86 mm


def measure(psi, diag, s) -> dict:
    nR, nZ = s["nR"], s["nZ"]
    zz = np.asarray(s["zz"], float)
    p2 = np.asarray(psi, float).reshape(nR, nZ, order="F")
    out = dict(Phi=float(diag["Phi_private_Wb"]), Rs=float(diag["Rs_mid"]),
               Bz0=float(diag.get("Bz_axis_mid", np.nan)),
               beta=float(diag.get("beta_avg_sep", np.nan)))
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


def draw(path, *, tt, zc, zs, ls, lss, ph, a_arr, i2off_arr, args, res) -> None:
    """三段放电的三联图。中间一幅标清两个控制量各自的含义。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gseqfrc import mstyle
    mstyle.install()
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    # ---- (a) 位置与长度 ----
    ax[0].plot(tt * 1e3, zs * 1e2, "k--", lw=1.4, label="目标 z_c*（位置）")
    ax[0].plot(tt * 1e3, zc * 1e2, "o-", color="C0", label="实得 z_c")
    ax[0].plot(tt * 1e3, lss * 1e2, ":", color="0.5", lw=1.4, label="目标 l_s*（长度）")
    ax[0].plot(tt * 1e3, ls * 1e2, "s-", color="C2", label="实得 l_s")
    ax[0].set_xlabel("t (ms)"); ax[0].set_ylabel("cm")
    ax[0].set_title("位置与长度的三段跟踪"); ax[0].legend(fontsize=8)
    # ---- (b) 两个控制量（这一幅要标清各自含义）----
    ax[1].plot(tt * 1e3, a_arr / 1e3, "o-", color="C3",
               label="差模 a=(I1-I3)/2：两端反向 → 控位置 z_c")
    ax[1].plot(tt * 1e3, i2off_arr / 1e3, "s-", color="C0",
               label="中心线圈 ΔI2（相对 56.89 kA）→ 控长度 l_s")
    ax[1].set_xlabel("t (ms)"); ax[1].set_ylabel("电流 (kA)")
    ax[1].set_title("两个控制量"); ax[1].legend(fontsize=8, loc="center right")
    # ---- (c) 通量 ----
    ax[2].plot(tt * 1e3, ph * 1e3, "s-", color="C4")
    ax[2].set_xlabel("t (ms)"); ax[2].set_ylabel("Φ (mWb)")
    ax[2].set_title(f"私通量（Φ_end/Φ_0={res['flux_ratio']:.2f}）")
    for x_ in (args.t_trans_ms, args.t_trans_ms + args.t_comp_ms):
        for a_ in ax:
            a_.axvline(x_, color="gray", ls=":", lw=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="端到端放电：平移 → 压缩 → 平顶")
    ap.add_argument("--zc-cm", type=float, default=3.0, help="平移目标位置 [cm]")
    ap.add_argument("--ls-cm", type=float, default=45.0, help="压缩目标长度 [cm]")
    ap.add_argument("--t-trans-ms", type=float, default=0.10)
    ap.add_argument("--n-trans", type=int, default=4)
    ap.add_argument("--t-comp-ms", type=float, default=0.10)
    ap.add_argument("--n-comp", type=int, default=4)
    ap.add_argument("--flat-ms", type=float, default=0.40)
    ap.add_argument("--flat-dt", type=float, default=5e-5)
    ap.add_argument("--rmf", action="store_true", help="RMF 托住通量")
    ap.add_argument("--ne", type=float, default=1.4e20)
    ap.add_argument("--te", type=float, default=30.0)
    ap.add_argument("--eta-cd", type=float, default=None)
    ap.add_argument("--beta-w", type=float, default=None)
    ap.add_argument("--omega-tau", type=float, default=1.0)
    ap.add_argument("--r-o", type=float, default=0.70 * 0.0569)
    ap.add_argument("--plot-only", action="store_true",
                    help="不跑仿真，直接用 outputs/frc_formation/discharge.npz 重画")
    args = ap.parse_args()

    sh = json.loads(SHAPING.read_text(encoding="utf-8"))
    Il_base = np.asarray(sh["nominal"]["Il"], float)
    if args.plot_only:
        d = np.load(OUT / "discharge.npz")
        rr = json.loads((OUT / "discharge.json").read_text(encoding="utf-8"))
        draw(OUT / "discharge.png", tt=d["t"], zc=d["zc"], zs=d["zc_star"],
             ls=d["ls"], lss=d["ls_star"], ph=d["Phi"], a_arr=d["a"],
             i2off_arr=d["I2"] - Il_base[1], args=args, res=rr)
        print(f"[disch] 重画完成：{OUT / 'discharge.png'}")
        return 0
    Jz = float(np.asarray(sh["jacobian"]["zc"], float) @ DIFF)      # dzc/da
    Jl = 2.0 * float(np.asarray(sh["jacobian"]["zh"], float)[1])    # dls/dI2
    Jz_ref, Jl_ref = Jz, Jl

    s, model, il_scale, _ = dp.build_design_model()
    s0 = dp.gcm._setup(dp.gcm.SRC_DIR)
    machine = s4.build_machine(s, shell="none", verbose=False)
    eta = s4.spitzer_resistivity(args.te)
    psi, ci = dp.warm_start_guess(il_scale)

    # ---- 初态（保留**原始求解输出**给 match_flux 做暖启动）----
    out0 = dp.gcm.solve(dp.gcm.SRC_DIR, model=model, Il=Il_base,
                        params=dict(Izeta0=float(s["Izeta0"])), psi0=psi, CI0=ci)
    psi = np.asarray(out0["psiq"], float)
    m = measure(psi, out0["diag"], s0)
    m["Il"] = Il_base
    zc0, ls0 = m["zc"], m["ls"]
    iz_cur = float(s["Izeta0"])
    prev = out0
    pr = model.problem(model.coil_flux(Il_base)[3], 2)
    S0_0, _sp = s4.flux_loss_rate(psi, pr, eta)
    rmf = None
    if args.rmf:
        rmf = rmf_drive(te_ev=args.te, ne=args.ne, s0_wb_s=S0_0,
                        be=float(s["Be"]), r_o=args.r_o, eta_cd=args.eta_cd,
                        beta_w=args.beta_w, omega_tau=args.omega_tau)
        print(f"[disch] 初态 z_c={zc0 * 1e2:+.3f} cm  l_s={ls0 * 1e2:.2f} cm  "
              f"Φ={m['Phi']:.4e} Wb")
        print(f"[disch] RMF：η_cd={rmf['eta_cd']:.3g}  β_ω={rmf['beta_w']:.3f}"
              f"  B_ω={rmf['B_omega_T'] * 1e3:.2f} mT  f={rmf['f_hz'] / 1e6:.2f} MHz"
              f"  S_RMF/S0={rmf['drive_frac']:.3f}")
    else:
        print(f"[disch] 初态 z_c={zc0 * 1e2:+.3f} cm  l_s={ls0 * 1e2:.2f} cm  "
              f"Φ={m['Phi']:.4e} Wb（无 RMF）")

    # ---- 时间表：平移 / 压缩 / 平顶 ----
    z_t, l_t = args.zc_cm * 1e-2, args.ls_cm * 1e-2
    sched = []
    t = 0.0
    for k in range(1, args.n_trans + 1):          # 平移：z_c 走，l_s 保持
        dt = args.t_trans_ms * 1e-3 / args.n_trans
        t += dt
        frac = k / args.n_trans
        sched.append((t, "平移", zc0 + (z_t - zc0) * frac, ls0, dt))
    for k in range(1, args.n_comp + 1):           # 压缩：l_s 走，z_c 保持
        dt = args.t_comp_ms * 1e-3 / args.n_comp
        t += dt
        frac = k / args.n_comp
        sched.append((t, "压缩", z_t, ls0 + (l_t - ls0) * frac, dt))
    if args.flat_ms > 0:                          # 平顶：两个都保持
        n_flat = max(1, int(round(args.flat_ms * 1e-3 / args.flat_dt)))
        for _j in range(n_flat):
            t += args.flat_dt
            sched.append((t, "平顶", z_t, l_t, args.flat_dt))

    a, i2off = 0.0, 0.0
    rows = [dict(phase="初态", t=0.0, zc_star=zc0, ls_star=ls0, zc=zc0, ls=ls0,
                 a=0.0, i2=float(Il_base[1]), Il=Il_base.tolist(),
                 Phi=float(m["Phi"]), Rs=float(m["Rs"]))]
    # 存真场：每步的 ψ 降采样一帧（供可视化画真实磁面）
    psi_frames = []

    def stash(psiq):
        p2 = np.asarray(psiq, float).reshape(s0["nR"], s0["nZ"], order="F")
        psi_frames.append(np.ascontiguousarray(p2[::4, ::4]))

    stash(prev["psiq"])
    t0 = time.perf_counter()
    for (t, phase, zstar, lstar, dt) in sched:
        S0, _sp = s4.flux_loss_rate(prev["psiq"], pr, eta)
        phi_now = float(prev["diag"]["Phi_private_Wb"])
        dr = (rmf["drive_frac"] * S0) if rmf else 0.0
        phi_target = max(0.0, phi_now - (S0 - dr) * dt)   # 通量按 ODE 走
        guess = iz_cur
        out = prev
        for _it in range(3):
            ez = zstar - m["zc"]
            el = lstar - m["ls"]
            if abs(ez) <= 1.0 * ZC_QUANT and abs(el) <= 1.0 * LS_QUANT:
                break
            if not np.isfinite(Jz) or Jz == 0:
                Jz = Jz_ref
            if not np.isfinite(Jl) or Jl == 0:
                Jl = Jl_ref
            da = float(np.clip(ez / Jz, -0.15 * Il_base[0], 0.15 * Il_base[0]))
            dI2 = float(np.clip(el / Jl, -0.15 * Il_base[1], 0.15 * Il_base[1]))
            # 半步重试：求解器掉真空解就把步长减半再来
            ok = False
            for _try in range(3):
                il = Il_base + (a + da) * DIFF
                il = il.copy(); il[1] += i2off + dI2
                g = guess * float(np.mean(il / Il_base))
                out, iz, _phi, _ne, _er = s4.match_flux(
                    phi_target, g, model, il, prev, None, 2e-3, 4)
                if out is not None:
                    m_try = measure(out["psiq"], out["diag"], s0)
                    if np.isfinite(m_try["ls"]) and m_try["Phi"] > 0.5 * phi_now:
                        ok = True
                        break
                da *= 0.5
                dI2 *= 0.5
            if not ok:
                print(f"[disch] {phase} t={t * 1e3:.3f} ms：平衡消失，停")
                break
            iz = float(iz)
            m_prev = m
            m = m_try
            m["Il"] = il
            guess = iz
            if np.isfinite(m["zc"]) and np.isfinite(m_prev["zc"]) and da != 0:
                Jn = (m["zc"] - m_prev["zc"]) / da
                if Jn != 0 and np.isfinite(Jn):
                    Jz = Jn
            if np.isfinite(m["ls"]) and np.isfinite(m_prev["ls"]) and dI2 != 0:
                Jn = (m["ls"] - m_prev["ls"]) / dI2
                if Jn != 0 and np.isfinite(Jn):
                    Jl = Jn
            a += da
            i2off += dI2
        prev = out
        iz_cur = guess
        stash(out["psiq"])
        rows.append(dict(phase=phase, t=t, zc_star=zstar, ls_star=lstar,
                         zc=float(m["zc"]), ls=float(m["ls"]), a=float(a),
                         i2=float(m["Il"][1]), Il=m["Il"].tolist(),
                         Phi=float(m["Phi"]), Rs=float(m["Rs"])))
        print(f"[disch] {phase} t={t * 1e3:6.3f} ms  z_c={m['zc'] * 1e2:+6.3f}"
              f"（目标 {zstar * 1e2:+6.3f}）  l_s={m['ls'] * 1e2:6.2f}"
              f"（目标 {lstar * 1e2:6.2f}）  a={a / 1e3:+6.2f} kA  "
              f"I2={m['Il'][1] / 1e3:6.2f} kA  Φ={m['Phi']:.3e}"
              f"（{m['Phi'] / rows[0]['Phi']:.3f}×）")
    wall = time.perf_counter() - t0

    tt = np.array([r["t"] for r in rows])
    zc = np.array([r["zc"] for r in rows])
    zs = np.array([r["zc_star"] for r in rows])
    ls = np.array([r["ls"] for r in rows])
    lss = np.array([r["ls_star"] for r in rows])
    ph = np.array([r["Phi"] for r in rows])
    res = dict(zc_cm=args.zc_cm, ls_cm=args.ls_cm, rmf=rmf,
               total_ms=float(tt[-1] * 1e3),
               max_zc_err_cm=float(np.max(np.abs(zc - zs)) * 1e2),
               max_ls_err_cm=float(np.max(np.abs(ls - lss)) * 1e2),
               flux_ratio=float(ph[-1] / ph[0]),
               a_final_kA=float(rows[-1]["a"] / 1e3),
               i2_final_kA=float(rows[-1]["i2"] / 1e3),
               track=[{k: (v if not isinstance(v, np.floating) else float(v))
                       for k, v in r.items()} for r in rows],
               wall_s=wall)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "discharge.npz", t=tt, zc=zc, zc_star=zs,
                        ls=ls, ls_star=lss, Phi=ph,
                        a=np.array([r["a"] for r in rows]),
                        I2=np.array([r["i2"] for r in rows]))
    np.savez_compressed(OUT / "discharge_psi.npz",
                        t=tt, psi=np.array(psi_frames),
                        rr=np.asarray(s0["rr"], float)[::4],
                        zz=np.asarray(s0["zz"], float)[::4])
    (OUT / "discharge.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[disch] 总时长 {res['total_ms']:.3f} ms   "
          f"z_c 误差 max={res['max_zc_err_cm']:.3f} cm   "
          f"l_s 误差 max={res['max_ls_err_cm']:.3f} cm")
    print(f"[disch] 通量 Φ_end/Φ_0={res['flux_ratio']:.3f}   "
          f"差模末值 {res['a_final_kA']:+.2f} kA   中心线圈末值 "
          f"{res['i2_final_kA']:.2f} kA   耗时 {wall:.0f} s")

    try:
        draw(OUT / "discharge.png", tt=tt, zc=zc, zs=zs, ls=ls, lss=lss, ph=ph,
             a_arr=np.array([r["a"] for r in rows]),
             i2off_arr=np.array([r["i2"] for r in rows]) - Il_base[1],
             args=args, res=res)
        print(f"[disch] 产物：{OUT / 'discharge.json'} / {OUT / 'discharge.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[disch] 出图跳过：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
