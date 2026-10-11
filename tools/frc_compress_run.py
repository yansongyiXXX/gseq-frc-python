r"""压缩段动力学：把 \(l_s\) 闭环给出的线圈波形接回时间推进。

`tools/frc_ls_track.py` 给的是一条**准静态参考波形**（每个目标长度对应的平衡电流）。
这里把它当**预设线圈电流**喂进 stage04 的"回路 → 平衡 → 通量"演化里，让压缩段真正
动起来，并回答一个关键问题：

> 准静态参考波形，在**真实通量衰减**下还追得上目标长度吗？

做法（复用 `frc_evolve_stage04` 的机器/回路/求解接口，跑在与轨迹**同几何**的设计点）：

1. 每一步把线圈电流设成波形值 \(I_l(t)\)（中心线圈，两端不变）；
2. 通量按 \(\Phi^{n+1}=\Phi^n-S_0\Delta t\) 推进，用 `match_flux` 调 \(I_{\zeta0}\) 命中；
3. 记录 \(l_s=2E R_s\)、\(\Phi\)、\(R_s\)，与目标轨迹对比。

用法：

    python tools/frc_compress_run.py                       # 用 ls_track.npz 的波形
    python tools/frc_compress_run.py --te 30 --dt-factor 1

产物：`outputs/frc_formation/compress_run.{npz,json,png}`
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

import design_point as dp          # noqa: E402
import frc_evolve_stage04 as s4    # noqa: E402

OUT = ROOT / "outputs" / "frc_formation"
M_E = 9.1093837015e-31
E_CH = 1.602176634e-19


def rmf_drive(*, te_ev, ne, s0_wb_s, be, r_o, eta_cd=None, beta_w=None,
              omega_tau=1.0) -> dict:
    r"""RMF 驱动率（第 25–30 条那套标度模型）。

    维持条件 \(\eta_{\rm cd}\beta_\omega^2F(x)=1\)；驱动率
    \(S_{\rm RMF}=\eta_{\rm cd}\beta_\omega^2F(x)\,S_0\)。\(\eta_{\rm cd}\) 不给就
    按**刚性转子上限**标定（\(S^{RR}=2\pi\eta n_ee\omega r_O^2\)，\(\beta_{\rm ref}=1\)）：

    $$\eta_{\rm cd}=\frac{S^{RR}(\omega^*)}{F(x^*)S_0},\qquad \omega^*=x^*/\tau_e .$$

    \(\beta_\omega\) 不给就取**刚好维持**（\(S_{\rm RMF}=S_0\)）所需的值。
    """
    eta = s4.spitzer_resistivity(te_ev)
    tau_e = M_E / (ne * E_CH ** 2 * eta)
    omega = omega_tau / tau_e
    F = omega_tau / (1.0 + omega_tau ** 2)
    if eta_cd is None:
        S_rr = 2 * np.pi * eta * ne * E_CH * omega * r_o ** 2
        eta_cd = S_rr / (F * s0_wb_s)
    if beta_w is None:
        beta_w = 1.0 / float(np.sqrt(max(eta_cd * F, 1e-300)))
    drive_frac = float(eta_cd * beta_w ** 2 * F)
    return dict(eta=eta, tau_e=tau_e, omega=omega, F=F, eta_cd=float(eta_cd),
                beta_w=float(beta_w), B_omega_T=float(beta_w * be),
                drive_frac=drive_frac, f_hz=float(omega / (2 * np.pi)))


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
        out["zh"] = 0.5 * (zhi - zlo)
        out["E"] = out["zh"] / out["Rs"]
        out["ls"] = 2.0 * out["zh"]
    else:
        out["zh"] = out["E"] = out["ls"] = float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="压缩段动力学（预设波形 → 时间推进）")
    ap.add_argument("--waveform", default=str(OUT / "ls_track.npz"),
                    help="ls_track.npz（含 t 与 I2）")
    ap.add_argument("--te", type=float, default=30.0, help="电子温度 [eV]（电阻率）")
    ap.add_argument("--dt-factor", type=float, default=1.0,
                    help="时间步因子（>1 表示把波形时间轴拉长）")
    ap.add_argument("--step-tol", type=float, default=5e-3)
    ap.add_argument("--closed-loop", action="store_true",
                    help="闭环：每步按**实际通量**重解线圈电流去追目标 l_s*"
                         "（而不是照搬准静态波形）")
    ap.add_argument("--loop-tol", type=float, default=1e-2,
                    help="闭环单步相对容差")
    ap.add_argument("--rmf", action="store_true",
                    help="接 RMF 驱动：把 η_cd·β_ω²F 那一份 S0 补回来（托住通量）")
    ap.add_argument("--ne", type=float, default=1.4e20, help="电子密度 [m^-3]")
    ap.add_argument("--eta-cd", type=float, default=None,
                    help="RMF 效率；不给按刚性转子上限标定")
    ap.add_argument("--beta-w", type=float, default=None,
                    help="B_ω/B_e；不给取刚好维持所需")
    ap.add_argument("--omega-tau", type=float, default=1.0, help="ω·τ_e（滑移参数）")
    ap.add_argument("--r-o", type=float, default=0.70 * 0.0569,
                    help="O 点半径 [m]（标定 η_cd 用）")
    ap.add_argument("--flat-top-ms", type=float, default=0.0,
                    help="压缩结束后再跑多久的平顶维持 [ms]（0=不跑）")
    ap.add_argument("--flat-dt", type=float, default=5e-5,
                    help="平顶段的时间步 [s]（慢段可以用大步）")
    args = ap.parse_args()

    wf = np.load(args.waveform)
    t_wf = np.asarray(wf["t"], float)
    I2_wf = np.asarray(wf["I2"], float)
    ls_star = np.asarray(wf["ls_star"], float)
    n_steps = t_wf.size - 1
    dt = float(t_wf[1] - t_wf[0]) * args.dt_factor
    print(f"[compress] 波形 {n_steps} 步，dt={dt * 1e6:.2f} µs，"
          f"总时长 {dt * n_steps * 1e3:.3f} ms")

    s, model, il_scale, _ = dp.build_design_model()
    s0 = dp.gcm._setup(dp.gcm.SRC_DIR)
    machine = s4.build_machine(s, shell="none", verbose=False)
    eta = s4.spitzer_resistivity(args.te)
    Il_base = np.asarray(s["Il"], dtype=float)
    psi0, ci0 = dp.warm_start_guess(il_scale)

    # 起点：波形首点的电流（设计点必须暖启动，冷启动不收敛）
    Il = Il_base.copy(); Il[1] = float(I2_wf[0])
    out = dp.gcm.solve(dp.gcm.SRC_DIR, model=model, Il=Il,
                       params=dict(Izeta0=float(s["Izeta0"])),
                       psi0=psi0, CI0=ci0)
    pr = model.problem(model.coil_flux(Il)[3], 2)
    S0, _sp = s4.flux_loss_rate(out["psiq"], pr, eta)
    phi = float(out["diag"]["Phi_private_Wb"])
    m = measure(out["psiq"], out["diag"], s0)
    rows = [dict(step=0, t=0.0, Il=Il.tolist(), ls_star=float(ls_star[0]),
                 ls=float(m["ls"]), E=float(m["E"]), Rs=float(m["Rs"]),
                 Phi=float(m["Phi"]), Izeta0=float(s["Izeta0"]))]
    print(f"[compress] 初态 l_s={m['ls'] * 1e2:.2f} cm  Φ={m['Phi']:.4e} Wb  "
          f"S0={S0:.3f} Wb/s  τ_Φ={m['Phi'] / S0 * 1e3:.3f} ms")

    rmf = None
    if args.rmf:
        rmf = rmf_drive(te_ev=args.te, ne=args.ne, s0_wb_s=S0, be=float(s["Be"]),
                        r_o=args.r_o, eta_cd=args.eta_cd, beta_w=args.beta_w,
                        omega_tau=args.omega_tau)
        print(f"[compress] RMF：η_cd={rmf['eta_cd']:.3g}  β_ω={rmf['beta_w']:.3f}"
              f"  B_ω={rmf['B_omega_T'] * 1e3:.2f} mT  f={rmf['f_hz'] / 1e6:.2f} MHz"
              f"  τ_e={rmf['tau_e'] * 1e9:.2f} ns")
        print(f"          驱动率 S_RMF/S0 = {rmf['drive_frac']:.3f}"
              f"{'（刚好维持：通量不再衰减）' if abs(rmf['drive_frac'] - 1) < 0.05 else ''}")

    prev = out
    t = 0.0
    J = 6.866e-6                    # dls/dIl（中心线圈），闭环用；随迭代更新
    J_ref = J
    # 闭环的初值：从准静态波形出发（照搬版不变）
    Il_cl = Il_base.copy(); Il_cl[1] = float(I2_wf[1]) if n_steps >= 1 else Il_base[1]
    # 时间表：压缩段（波形的 n 步）+ 可选平顶段（保持末长度，大时间步）
    schedule = [(float(t_wf[k]) * args.dt_factor, float(ls_star[k]), dt)
                for k in range(1, n_steps + 1)]
    if args.flat_top_ms > 0:
        t_end = schedule[-1][0] if schedule else 0.0
        n_flat = max(1, int(round(args.flat_top_ms * 1e-3 / args.flat_dt)))
        for _j in range(1, n_flat + 1):
            schedule.append((t_end + _j * args.flat_dt, float(ls_star[-1]),
                             args.flat_dt))
    t0 = time.perf_counter()
    for k, (t, ls_k, dt_k) in enumerate(schedule, start=1):
        if args.closed_loop:
            Il = Il_cl.copy()                 # 用上一步闭环解出的电流做初值
        else:
            Il = Il_base.copy()
            Il[1] = float(I2_wf[k]) if k <= n_steps else float(I2_wf[-1])
        S0, _sp = s4.flux_loss_rate(prev["psiq"], pr, eta)
        phi_now = float(prev["diag"]["Phi_private_Wb"])
        drive_rmf = (rmf["drive_frac"] * S0) if rmf else 0.0
        phi_target = max(0.0, phi_now - (S0 - drive_rmf) * dt_k)
        guess = float(prev["Izeta0"]) * np.sqrt(phi_target / max(phi_now, 1e-30))
        out = None
        prev_ls, dI_prev = None, 0.0
        for _it in range(3 if args.closed_loop else 1):
            out, iz, phi, n_eval, err = s4.match_flux(
                phi_target, guess, model, Il, prev, args.step_tol, 2e-3, 4)
            if out is None:
                break
            m = measure(out["psiq"], out["diag"], s0)
            if not args.closed_loop:
                break
            lerr = float(ls_k) - m["ls"]
            if abs(lerr) / float(ls_k) <= args.loop_tol:
                break
            if prev_ls is not None and dI_prev != 0:
                Jnew = (m["ls"] - prev_ls) / dI_prev
                if Jnew != 0 and np.isfinite(Jnew):
                    J = Jnew
            if not np.isfinite(J) or J == 0:
                J = J_ref
            dI = float(np.clip(lerr / J, -0.08 * Il_base[1], 0.08 * Il_base[1]))
            prev_ls, dI_prev = m["ls"], dI
            Il = Il.copy(); Il[1] += dI
            guess = iz
        if out is None:
            print(f"[compress] t={t * 1e3:.3f} ms：平衡消失，停在上一态")
            break
        Il_cl = Il.copy()
        rows.append(dict(step=k, t=t, Il=Il.tolist(), ls_star=float(ls_k),
                         ls=float(m["ls"]), E=float(m["E"]), Rs=float(m["Rs"]),
                         Phi=float(m["Phi"]), Izeta0=float(iz)))
        print(f"[compress] t={t * 1e3:6.3f} ms  I2={Il[1] / 1e3:6.2f} kA  "
              f"目标 {ls_k * 1e2:6.2f} cm  实得 {m['ls'] * 1e2:6.2f} cm  "
              f"Φ={m['Phi']:.3e}（{m['Phi'] / rows[0]['Phi']:.3f}×）  err="
              f"{abs(m['ls'] - ls_k) / ls_k:.1e}")
        prev = out
    wall = time.perf_counter() - t0

    tt = np.array([r["t"] for r in rows])
    ls = np.array([r["ls"] for r in rows])
    lss = np.array([r["ls_star"] for r in rows])
    ph = np.array([r["Phi"] for r in rows])
    err = np.abs(ls - lss) / lss
    res = dict(n_steps=len(rows) - 1, te_ev=args.te, dt_s=dt,
               total_ms=float(tt[-1] * 1e3),
               flux_ratio=float(ph[-1] / ph[0]),
               max_rel_err=float(np.max(err)),
               mean_rel_err=float(np.mean(err)),
               rmf=rmf, closed_loop=bool(args.closed_loop),
               tau_phi_ms=float(ph[0] / s4.flux_loss_rate(prev["psiq"], pr, eta)[0] * 1e3),
               track=rows, wall_s=wall)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "compress_run.npz", t=tt, ls=ls, ls_star=lss,
                        Phi=ph, err=err,
                        Il=np.array([r["Il"] for r in rows]))
    (OUT / "compress_run.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[compress] 通量比 Φ_end/Φ_0 = {res['flux_ratio']:.3f}"
          f"（τ_Φ={res['tau_phi_ms']:.3f} ms，窗口 {res['total_ms']:.3f} ms）")
    print(f"[compress] 跟踪误差：max={res['max_rel_err']:.2e} "
          f"mean={res['mean_rel_err']:.2e}   总耗时 {wall:.0f} s")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from gseqfrc import mstyle
        mstyle.install()
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
        ax[0].plot(tt * 1e3, lss * 1e2, "k--", lw=1.6, label="目标 l_s*")
        ax[0].plot(tt * 1e3, ls * 1e2, "o-", color="C0", label="动力学实得")
        ax[0].set_xlabel("t (ms)"); ax[0].set_ylabel("l_s (cm)")
        ax[0].set_title("压缩段长度跟踪（动力学）"); ax[0].legend()
        ax[1].plot(tt * 1e3, np.array([r["Il"][1] for r in rows]) / 1e3, "o-", color="C3")
        ax[1].set_xlabel("t (ms)"); ax[1].set_ylabel("I_2 (kA)")
        ax[1].set_title("预设线圈波形")
        ax[2].plot(tt * 1e3, ph * 1e3, "s-", color="C4")
        ax[2].set_xlabel("t (ms)"); ax[2].set_ylabel("Φ (mWb)")
        ax[2].set_title(f"私通量演化（Φ_end/Φ_0={res['flux_ratio']:.2f}）")
        fig.tight_layout()
        fig.savefig(OUT / "compress_run.png", dpi=140)
        plt.close(fig)
        print(f"[compress] 产物：{OUT / 'compress_run.json'} / {OUT / 'compress_run.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[compress] 出图跳过：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
