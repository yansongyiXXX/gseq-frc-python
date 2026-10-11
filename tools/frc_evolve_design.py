r"""设计点 A 的正向演化：通量 ODE + RMF 驱动（接在 `tools/design_point.py` 上）。

模型与第 21 条（`tools/frc_evolve_stage03.py`）完全一致，只换了装置参数：

1. 状态量是私通量 \(\Phi\)，每步推进
   \(\Phi^{n+1}=\max(0,\Phi^n-(S_0-S_{\rm drive})\,dt)\)；
2. \(S_0=2\pi(\eta/\mu_0)\Delta^*\psi|_{\text{O 点}}\) 由**当前平衡**算出；
3. 驱动项按第 25/26 条的 RMF 标度给：

   $$S_{\rm drive}=\eta_{cd}\,\beta_\omega^{2}F(\omega\tau_e)\,S_0,\qquad
   F(x)=\frac{x}{1+x^{2}}$$

   所以驱动是**随耗散率一起变**的（\(\kappa=\eta_{cd}\beta_\omega^2F\) 是常数），
   这比"给一个固定 Wb/s"更接近真实 RMF 行为。
4. 每步再用割线法找 \(I_{\zeta0}\)，使二维平衡的 \(\Phi\) 等于目标值。

用法：

    python tools/frc_evolve_design.py                    # 12 步 × 20 μs
    python tools/frc_evolve_design.py --beta-w 0.447     # 临界维持（η_cd=10 时）
    python tools/frc_evolve_design.py --beta-w 0.6 --steps 20

产物：`outputs/design_point/forward.npz` 与 `forward.png`。
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
import frc_evolve_stage03 as s3    # noqa: E402
import rmf_efficiency as rme       # noqa: E402
from gseqfrc import mstyle         # noqa: E402

mstyle.install()   # 出图统一按 MATLAB 那一套（见 gseqfrc/mstyle.py）

gcm = s3.gcm
OUT_DIR = ROOT / "outputs" / "design_point"


def slip(x: float) -> float:
    r"""标准滑移因子 \(F(x)=x/(1+x^2)\)。"""
    return x / (1.0 + x ** 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--dt", type=float, default=2e-5, help="时间步 [s]")
    ap.add_argument("--te", type=float, default=30.0, help="电子温度 [eV]")
    ap.add_argument("--eta-cd", type=float, default=10.0, help="RMF 效率（假设值）")
    ap.add_argument("--beta-w", type=float, default=0.2, help="B_w/B_e")
    ap.add_argument("--f", type=float, default=2.0e6, help="RMF 频率 [Hz]")
    ap.add_argument("--ne", type=float, default=1.41e20, help="电子密度 [m^-3]")
    ap.add_argument("--theory", action="store_true",
                    help="驱动项里的 η_cd 改用理论版：每步按当前平衡剖面重算 "
                         "η_cd^th（Hugrass 1984 内部效率，见 tools/rmf_efficiency.py）")
    ap.add_argument("--x-ref", type=float, default=1.0,
                    help="标定 η_cd^th 用的参考滑移 ωτ_e（默认 1，与维持边界一致）")
    ap.add_argument("--nu-ratio", type=float, default=1.0,
                    help="有效/经典碰撞频率之比 nu/nu0（论文取 ~1，默认 1）")
    ap.add_argument("--hold", action="store_true",
                    help="闭环：每步反解 β_ω 把 Φ 钉在 Φ_ref 上（给出幅度/功率预算）")
    ap.add_argument("--hold-frac", type=float, default=1.0,
                    help="目标 Φ_ref 线性降到 frac × Φ_0（默认 1.0 = 完全维持、不漂）")
    ap.add_argument("--ctrl-gain", type=float, default=1.0,
                    help="Φ 误差的比例增益（1 = 一步到位；<1 = 更软的控制器）")
    ap.add_argument("--bw-max", type=float, default=None,
                    help="B_ω/B_e 上限，用于检查幅度余量")
    ap.add_argument("--p-max", type=float, default=None,
                    help="RMF 功率上限 [kW]，用于检查功率余量")
    ap.add_argument("--rtol", type=float, default=2e-3, help="割线法 Φ 相对容差")
    ap.add_argument("--max-eval", type=int, default=4)
    ap.add_argument("--step-tol", type=float, default=5e-3)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    s, model, il_scale, iz_scale = dp.build_design_model()
    eta = rme.spitzer_resistivity(args.te)
    # 电子—离子碰撞时间 τ_e = m_e/(n_e e² η)
    tau_e = rme.electron_collision_time(args.ne, eta)
    x_slip = 2 * np.pi * args.f * tau_e
    kappa = args.eta_cd * args.beta_w ** 2 * slip(x_slip)
    Il = np.asarray(s["Il"], dtype=float)
    Rg = np.asarray(s["rr"], dtype=float)
    Zg = np.asarray(s["zz"], dtype=float)

    use_prof = bool(args.theory or args.hold)
    dR = float(Rg[1] - Rg[0])
    dZ = float(Zg[1] - Zg[0])
    R_ring = 2 * np.pi * Rg[:, None]          # 轴对称体积元里的 2πR

    def eta_cd_at(out_step):
        """当前平衡剖面对应的驱动效率：`--theory` 时按 Hugrass 1984 每步重算。

        `--hold`（闭环）也要剖面——功率预算要用内部效率 \(\mathcal{E}_i=1/\langle\gamma\rangle\)；
        但那种情况下 η_cd 本身仍由 `--theory` 决定用不用理论值。
        """
        if not use_prof:
            return float(args.eta_cd), {"gamma": float("nan"), "E_i": float("nan")}
        prof = rme.midplane_integrals(Rg, Zg, out_step["psiq"], out_step["Jzeta"])
        th = rme.theory_eta_cd(args.ne, eta, prof, x_ref=args.x_ref,
                               nu_ratio=args.nu_ratio)
        return (float(th["eta_cd"]) if args.theory else float(args.eta_cd)), th

    def ohm_power(out_step):
        r"""等离子体欧姆耗散功率 \(P_\Omega=\int\eta J_\zeta^{2}\,\mathrm{d}V\) [W]。"""
        J = np.asarray(out_step["Jzeta"], dtype=float)
        return float(eta * np.sum(J ** 2 * R_ring) * dR * dZ)

    def tag(rec, eta_cd_v, th_v, out_step, beta_w_applied):
        """把效率 / 内部效率 / 维持需求 / 功率预算记进一条 history。"""
        rec["eta_cd"] = eta_cd_v
        rec["gamma"] = th_v["gamma"]
        rec["E_i"] = th_v["E_i"]
        rec["beta_req"] = rme.required_beta(eta_cd_v, x_slip)
        rec["beta_w"] = float(beta_w_applied)
        p_ohm = ohm_power(out_step)
        rec["P_ohm"] = p_ohm
        ei = th_v["E_i"]
        rec["P_rmf"] = (p_ohm / ei) if (ei and np.isfinite(ei) and ei > 0) else float("nan")

    print(f"[fwd] 设计点：B_e={s['Be']} T、R_s={s['Rs']} m、β_s={s['betas']}、"
          f"线圈安匝={Il[0] / 1e3:.1f} kA")
    print(f"[fwd] 输运：η={eta:.3e} Ω·m、τ_e={tau_e:.3e} s、"
          f"ωτ_e={x_slip:.3f}、F={slip(x_slip):.4f}")
    print(f"[fwd] 驱动：η_cd={args.eta_cd:g}、B_ω/B_e={args.beta_w:g} "
          f"→ κ=S_drive/S_0={kappa:.4f}"
          + ("（维持）" if abs(kappa - 1) < 0.05 else
             "（净衰减）" if kappa < 1 else "（净增长）")
          + ("；--theory：κ 每步按剖面重算" if args.theory else ""))

    t0 = time.perf_counter()
    psi0, ci0 = dp.warm_start_guess(il_scale)
    out = gcm.solve(gcm.SRC_DIR, model=model, Il=Il,
                    params={"Izeta0": float(s["Izeta0"])},
                    psi0=psi0, CI0=ci0)
    pr = model.problem(model.coil_flux(Il)[3], 2)
    phi = float(out["diag"]["Phi_private_Wb"])
    S0, spread = s3.flux_loss_rate(out["psiq"], pr, eta)
    iz = float(s["Izeta0"])
    eta_cd_now, th = eta_cd_at(out)
    kappa_now = eta_cd_now * args.beta_w ** 2 * slip(x_slip)
    phi0 = phi
    hist = [s3._record(0, 0.0, S0, spread, phi, out, iz, phi, 0, 0.0,
                       time.perf_counter() - t0)]
    hist[0]["drive"] = kappa_now * S0
    hist[0]["Phi_ref"] = phi0
    tag(hist[0], eta_cd_now, th, out, args.beta_w)
    print(f"[fwd] 初态：Φ={phi:.4e} Wb、S0={S0:.4f} Wb/s、"
          f"τ_Φ={phi / S0 * 1e3:.3f} ms（冷+暖启动 {time.perf_counter() - t0:.1f} s）")
    if np.isfinite(hist[0]["P_rmf"]):
        print(f"[fwd] 功率：初态 P_Ω={hist[0]['P_ohm'] / 1e3:.3f} kW，"
              f"RMF 需输送 P_Ω/ℰ_i={hist[0]['P_rmf'] / 1e3:.3f} kW")
    if args.hold:
        print(f"[fwd] 闭环：Φ_ref 从 Φ_0={phi0:.4e} Wb 线性降到 "
              f"{args.hold_frac:g}×Φ_0={args.hold_frac * phi0:.4e} Wb、"
              f"增益={args.ctrl_gain:g}"
              + (f"、B_ω/B_e 上限={args.bw_max:g}" if args.bw_max is not None else ""))
    if args.theory:
        print(f"[fwd] 理论版 η_cd（初态）：{eta_cd_now:.3f}"
              f"（⟨γ⟩={th['gamma']:.3f}、ℰ_i={th['E_i']:.4f}）；"
              f"维持所需 B_ω/B_e={hist[0]['beta_req']:.3f}；此后每步按剖面重算")
        if th["E_i"] > 1.0:
            print(f"[fwd] ⚠ 初态 ℰ_i={th['E_i']:.3f} > 1：该 (n_e,T_e) 与 Hugrass 理论"
                  f"不自洽（要求 n_e ≥ {th['ne_min']:.3e} m^-3）")

    t = 0.0
    t_total = float(args.steps) * args.dt
    phi_ref = float(args.hold_frac) * phi0 if args.hold else None
    for k in range(1, int(args.steps) + 1):
        ts = time.perf_counter()
        if args.hold:
            # 参考值线性地从 Φ_0 走到 hold_frac×Φ_0（frac=1 就是完全维持）
            phi_ref = phi0 * (1.0 - (1.0 - args.hold_frac) * ((t + args.dt) / t_total))
            # 闭环：由 Φ 误差反解 S_drive，再反解 β_ω（增益 1 = 一步到位）
            s_req = max(0.0, S0 + args.ctrl_gain * (phi_ref - phi) / args.dt)
            beta_w_step = float(np.sqrt(
                s_req / max(eta_cd_now * slip(x_slip) * S0, 1e-300)))
            if args.bw_max is not None:
                beta_w_step = min(beta_w_step, float(args.bw_max))
            drive = eta_cd_now * beta_w_step ** 2 * slip(x_slip) * S0
        else:
            beta_w_step = float(args.beta_w)
            drive = kappa_now * S0               # 驱动跟着当前耗散率与当前效率
        target = max(0.0, phi - (S0 - drive) * args.dt)
        guess = iz                               # 上一步的 I_ζ0 作首猜
        out, iz, phi_new, nev, err = s3.match_flux(
            target, guess, model, Il, out, args.step_tol,
            rtol=args.rtol, max_eval=args.max_eval, verbose=args.verbose)
        if out is None:
            print(f"[fwd] 第 {k} 步平衡不存在（Φ_target={target:.3e}），停在第 {k - 1} 步")
            break
        t += args.dt
        S0, spread = s3.flux_loss_rate(out["psiq"], pr, eta)
        phi = phi_new
        eta_cd_now, th = eta_cd_at(out)          # 新剖面的效率，供下一步用
        kappa_now = eta_cd_now * args.beta_w ** 2 * slip(x_slip)
        rec = s3._record(k, t, S0, spread, target, out, iz, phi, nev, err,
                         time.perf_counter() - ts)
        rec["drive"] = drive
        rec["Phi_ref"] = (float(phi_ref) if phi_ref is not None else float("nan"))
        tag(rec, eta_cd_now, th, out, beta_w_step)
        hist.append(rec)
        extra = (f"  β_w={beta_w_step:.3f}/{rec['beta_req']:.3f}" if args.hold else "")
        print(f"[fwd] {k:2d}  t={t * 1e3:7.4f} ms  Φ={phi:.4e}  "
              f"Iz0={iz / 1e3:7.2f} kA  Rs={rec['Rs'] * 1e2:5.2f} cm  "
              f"S0={S0:.4f}  drive={drive:.4f}  η_cd={eta_cd_now:.2f}  "
              f"err={err:.1e}{extra}  {rec['wall']:.1f}s/{nev}次")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.hold and hist:
        bw = np.array([h["beta_w"] for h in hist], dtype=float)
        bw_pk = float(np.nanmax(bw))
        line = (f"[fwd] 闭环预算：β_ω 峰值={bw_pk:.3f}"
                f" → B_ω 峰值={bw_pk * float(s['Be']) * 1e3:.2f} mT"
                f"（B_e={s['Be']} T）")
        if args.bw_max is not None:
            line += (f"；上限 {args.bw_max:g} → "
                     + ("够用" if bw_pk <= args.bw_max else "不够 ⚠"))
        print(line)
        prm = np.array([h["P_rmf"] for h in hist], dtype=float)
        if np.isfinite(prm).any():
            pk_kw = float(np.nanmax(prm)) / 1e3
            line2 = f"[fwd]            RMF 功率峰值={pk_kw:.2f} kW"
            if args.p_max is not None:
                line2 += (f"；上限 {args.p_max:g} kW → "
                          + ("够用" if pk_kw <= args.p_max else "不够 ⚠"))
            print(line2)
    # 理论版不覆盖原来的 forward.*（动图默认读 forward.npz），两条路径各写各的
    stem = ("forward" + ("_theory" if args.theory else "")
            + ("_hold" if args.hold else ""))
    keys = ("step", "t", "S0", "S0_spread", "Phi_target", "Phi", "Phi_err",
            "Izeta0", "Cs", "CI", "Rs", "Bz0", "beta", "W", "psi_wall",
            "gs_res", "n_eval", "wall", "drive",
            "eta_cd", "gamma", "E_i", "beta_req", "beta_w", "P_ohm", "P_rmf",
            "Phi_ref")
    # 逐步的 ψ 快照（供出动图用）：形状 (nstep+1, nR, nZ)
    psi_all = np.stack([np.asarray(h["psiq"], dtype=np.float32)
                        for h in hist], axis=0)
    np.savez_compressed(OUT_DIR / f"{stem}.npz",
                        **{k: np.asarray([h[k] for h in hist]) for k in keys},
                        psi_all=psi_all,
                        rr=np.asarray(s["rr"], dtype=float),
                        zz=np.asarray(s["zz"], dtype=float),
                        rl=np.asarray(s["rl"], dtype=float),
                        zl=np.asarray(s["zl"], dtype=float),
                        hl=np.asarray(s["hl"], dtype=float),
                        wl=np.asarray(s["wl"], dtype=float),
                        Rw=float(s["Rw"]), zw=float(s["zw"]),
                        Be=float(s["Be"]),
                        psi_last=np.asarray(hist[-1]["psiq"], dtype=float),
                        meta=json.dumps(dict(be=s["Be"], rs=s["Rs"],
                                             betas=s["betas"], te=args.te,
                                             eta=eta, eta_cd=args.eta_cd,
                                             beta_w=args.beta_w, f=args.f,
                                             ne=args.ne, kappa=kappa,
                                             theory=bool(args.theory),
                                             x_ref=args.x_ref,
                                             nu_ratio=args.nu_ratio)))
    print(f"[fwd] 产物：{OUT_DIR / f'{stem}.npz'}"
          f"（{len(hist)} 步、总 {sum(h['wall'] for h in hist):.1f} s）")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        nrow = 4 if args.hold else 3
        fig, ax = plt.subplots(nrow, 3, figsize=(14, 3.5 * nrow),
                               constrained_layout=True)
        if nrow == 3:
            ax = np.asarray(ax).reshape(3, 3)
        t_ms = np.array([h["t"] for h in hist]) * 1e3
        ax[0, 0].plot(t_ms, [h["Phi"] for h in hist], "o-")
        ax[0, 0].set_xlabel("t [ms]"); ax[0, 0].set_ylabel(r"$\Phi$ [Wb]")
        ax[0, 0].set_title("私通量（主状态量）")
        ax[0, 1].plot(t_ms, [h["S0"] for h in hist], "o-", label="耗散率 S0")
        ax[0, 1].plot(t_ms, [h["drive"] for h in hist], "s--",
                      label="驱动率 S_drive")
        ax[0, 1].legend(); ax[0, 1].set_xlabel("t [ms]")
        ax[0, 1].set_ylabel("[Wb/s]"); ax[0, 1].set_title("耗散率 vs 驱动率")
        ax[0, 2].plot(t_ms, [h["Rs"] * 1e2 for h in hist], "o-")
        ax[0, 2].set_xlabel("t [ms]"); ax[0, 2].set_ylabel(r"$R_s$ [cm]")
        ax[0, 2].set_title("分界面半径（FRC 尺寸）")
        ax[1, 0].plot(t_ms, [h["Izeta0"] / 1e3 for h in hist], "o-")
        ax[1, 0].set_xlabel("t [ms]"); ax[1, 0].set_ylabel(r"$I_{\zeta0}$ [kA]")
        ax[1, 0].set_title("线圈电流（通量执行器）")
        ax[1, 1].plot(t_ms, [h["Bz0"] for h in hist], "o-")
        ax[1, 1].set_xlabel("t [ms]"); ax[1, 1].set_ylabel(r"$B_{z0}$ [T]")
        ax[1, 1].set_title("轴上轴向场（反转场深度）")
        ax[1, 2].plot(t_ms, [h["beta"] for h in hist], "o-")
        ax[1, 2].set_xlabel("t [ms]"); ax[1, 2].set_ylabel(r"$\langle\beta\rangle$")
        ax[1, 2].set_title("分界面内体积平均比压")
        ax[2, 0].plot(t_ms, [h["eta_cd"] for h in hist], "o-")
        ax[2, 0].set_xlabel("t [ms]"); ax[2, 0].set_ylabel(r"$\eta_{cd}$")
        ax[2, 0].set_title("RMF 驱动效率"
                           + ("（理论值）" if args.theory else ""))
        ax[2, 1].plot(t_ms, [h["E_i"] for h in hist], "o-")
        ax[2, 1].axhline(1.0, color="gray", ls=":")
        ax[2, 1].set_xlabel("t [ms]")
        ax[2, 1].set_ylabel(r"$\mathcal{E}_i=1/\langle\gamma\rangle$")
        ax[2, 1].set_title("内部效率（须不大于 1）")
        ax[2, 2].plot(t_ms, [h["beta_req"] for h in hist], "o-",
                      label="所需幅度")
        if args.hold:
            ax[2, 2].plot(t_ms, [h["beta_w"] for h in hist], "s--",
                          label="实际施加（闭环）")
        else:
            ax[2, 2].axhline(args.beta_w, color="tab:red", ls="--",
                             label=rf"实际施加 {args.beta_w:g}")
        ax[2, 2].legend(fontsize=9)
        ax[2, 2].set_xlabel("t [ms]"); ax[2, 2].set_ylabel(r"$B_\omega/B_e$")
        ax[2, 2].set_title("维持所需的 RMF 幅度")
        if args.hold:
            bw = np.array([h["beta_w"] for h in hist])
            ax[3, 0].plot(t_ms, bw, "o-", label="实际施加")
            ax[3, 0].plot(t_ms, [h["beta_req"] for h in hist], "k--", lw=1,
                          label="所需")
            if args.bw_max is not None:
                ax[3, 0].axhline(args.bw_max, color="tab:red", ls=":", label="幅度上限")
            ax[3, 0].legend(fontsize=9)
            ax[3, 0].set_xlabel("t [ms]"); ax[3, 0].set_ylabel(r"$B_\omega/B_e$")
            ax[3, 0].set_title("幅度预算")
            ax[3, 1].plot(t_ms, np.array([h["P_rmf"] for h in hist]) / 1e3, "o-")
            if args.p_max is not None:
                ax[3, 1].axhline(args.p_max, color="tab:red", ls=":", label="功率上限")
                ax[3, 1].legend(fontsize=9)
            ax[3, 1].set_xlabel("t [ms]"); ax[3, 1].set_ylabel("P [kW]")
            ax[3, 1].set_title("RMF 功率")
            phi_h = np.array([h["Phi"] for h in hist])
            phi_ref_h = np.array([h["Phi_ref"] for h in hist])
            ax[3, 2].plot(t_ms, (phi_h - phi_ref_h) / phi_ref_h, "o-")
            ax[3, 2].axhline(0.0, color="gray", ls=":")
            ax[3, 2].set_xlabel("t [ms]")
            ax[3, 2].set_ylabel(r"$(\Phi-\Phi_{\rm ref})/\Phi_{\rm ref}$")
            ax[3, 2].set_title("轨迹跟踪误差")
        for a in ax.ravel():
            a.grid(alpha=0.3)
        fig.suptitle(f"设计点 A 正向演化：η_cd={args.eta_cd:g}、"
                     f"Bω/Be={args.beta_w:g}、κ={kappa:.3f}"
                     + ("（效率取理论值）" if args.theory else "")
                     + ("（闭环：托住通量）" if args.hold else ""))
        mstyle.style_figure(fig, fontsize=13.0, grid=mstyle.GRID_STYLE)
        fig.savefig(OUT_DIR / f"{stem}.png", dpi=110)
        print(f"[fwd] 产物：{OUT_DIR / f'{stem}.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[fwd] 画图跳过：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
