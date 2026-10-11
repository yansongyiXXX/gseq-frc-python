r"""设计点 A 的逆向反演：给定目标轨迹 \(\Phi_{\rm ref}(t)\)，反求执行器指令。

反演分两层
----------

**第一层（物理反演）**：把目标轨迹翻成所需的净驱动率。状态方程是

$$\dot\Phi=S_{\rm drive}-S_0(\Phi)\quad\Longrightarrow\quad
S_{\rm drive}^{\rm req}(t)=\dot\Phi_{\rm ref}+S_0(\Phi_{\rm ref})$$

其中 \(S_0(\Phi)\) 由正向演化（`tools/frc_evolve_design.py` 的产物）插值给出；
没有产物时退化成常数 \(S_0(\Phi_0)\)（脚本会打印用的是哪一档）。

**第二层（执行器反演）**：把 \(S_{\rm drive}^{\rm req}\) 分配到两个执行器上。

* RMF：\(S=\eta_{cd}\beta_\omega^2F(\omega\tau_e)S_0\Rightarrow
  B_\omega=B_e\sqrt{S_{\rm drive}^{\rm req}/(\eta_{cd}F S_0)}\)（只能为正）；
* 线圈：\(\Delta I=\Delta\Phi/g_I\)，\(g_I=\partial\Phi/\partial I_l\) 用**数值差分**
  实测（在名义电流附近多解一次平衡）。

可辨识性
--------
若只有一个观测量 \(\Phi\)，两个执行器的**瞬时**灵敏度向量是 1×2 的，秩恒为 1
（"线圈加一点、RMF 减一点"给出同一个 \(\Phi\)）——单时刻不可分。分开的办法有两条：

1. 用第二个观测量（\(R_s\) 或 \(B_z\) 剖面）：线圈改边界通量、RMF 改电流剖面，
   两者的**空间**响应不同；
2. 用时间信息：线圈对 \(\Phi\) 是即时作用，RMF 只作用在 \(\dot\Phi\) 上，
   两者的**时间签名**不同。

脚本默认按"RMF 管稳态、线圈管瞬态"分配，这是 2 号方案的工程实现。

用法
----

    python tools/frc_inverse_design.py                  # 默认：40% 降通量斜坡
    python tools/frc_inverse_design.py --target hold    # 只要维持
    python tools/frc_inverse_design.py --vmax 4500 --bw-max 0.03

产物：`outputs/design_point/inverse.npz`、`inverse.json` 与 `inverse.png`。
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

gcm = s3.gcm
OUT_DIR = ROOT / "outputs" / "design_point"
ME, QE = 9.1093837015e-31, 1.602176634e-19


def slip(x: float) -> float:
    r"""\(F(x)=x/(1+x^2)\)。"""
    return x / (1.0 + x ** 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", choices=("hold", "ramp"), default="ramp")
    ap.add_argument("--ramp-frac", type=float, default=0.40, help="斜坡下降比例")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--dt", type=float, default=2e-5)
    ap.add_argument("--te", type=float, default=30.0)
    ap.add_argument("--eta-cd", type=float, default=10.0)
    ap.add_argument("--f", type=float, default=2.0e6)
    ap.add_argument("--ne", type=float, default=1.41e20)
    ap.add_argument("--il-frac", type=float, default=0.02, help="差分步长（相对）")
    ap.add_argument("--vmax", type=float, default=450.0, help="电源电压上限 [V]")
    ap.add_argument("--coil-L", type=float, default=314.2e-6, help="回路自感 [H]")
    ap.add_argument("--bw-max", type=float, default=0.02, help="RMF 幅度上限 [T]")
    ap.add_argument("--il-max-frac", type=float, default=0.20, help="线圈电流最大偏移")
    args = ap.parse_args()

    s, model, il_scale, iz_scale = dp.build_design_model()
    be, rs = float(s["Be"]), float(s["Rs"])
    Il0 = np.asarray(s["Il"], dtype=float)
    eta = s3.spitzer_resistivity(args.te)
    tau_e = ME / (args.ne * QE ** 2 * eta)
    x_slip = 2 * np.pi * args.f * tau_e
    F = slip(x_slip)

    # ---- 初态（与正向脚本同一条暖启动路径）--------------------------------
    t0 = time.perf_counter()
    psi0, ci0 = dp.warm_start_guess(il_scale)
    out = gcm.solve(gcm.SRC_DIR, model=model, Il=Il0,
                    params={"Izeta0": float(s["Izeta0"])},
                    psi0=psi0, CI0=ci0)
    pr = model.problem(model.coil_flux(Il0)[3], 2)
    phi0 = float(out["diag"]["Phi_private_Wb"])
    S0_0, _ = s3.flux_loss_rate(out["psiq"], pr, eta)
    iz0 = float(s["Izeta0"])
    print(f"[inv] 初态（{time.perf_counter() - t0:.1f} s）：Φ0={phi0:.4e} Wb、"
          f"S0={S0_0:.4f} Wb/s、τ_Φ={phi0 / S0_0 * 1e3:.3f} ms")

    # ---- S0(Φ) 关系：优先用正向产物插值 -----------------------------------
    fw = OUT_DIR / "forward.npz"
    if fw.exists():
        d = np.load(fw)
        phi_tab, s0_tab = np.asarray(d["Phi"], float), np.asarray(d["S0"], float)
        order = np.argsort(phi_tab)
        phi_tab, s0_tab = phi_tab[order], s0_tab[order]
        src = "forward.npz 插值"
    else:
        phi_tab, s0_tab = np.array([phi0]), np.array([S0_0])
        src = "常数（没有 forward.npz）"
    print(f"[inv] S0(Φ)：{src}，覆盖 Φ/Φ0 ∈ "
          f"[{phi_tab.min() / phi0:.3f}, {phi_tab.max() / phi0:.3f}]")
    S0_of = lambda p: float(np.interp(p, phi_tab, s0_tab))  # noqa: E731

    # ---- 灵敏度 g_I = ∂Φ/∂I_l（数值差分，一次额外求解）-------------------
    ts = time.perf_counter()
    kw = dict(model=model, Il=Il0 * (1 + args.il_frac),
              params={"Izeta0": iz0}, step_tol=5e-3)
    out_hi = gcm.solve(gcm.SRC_DIR, psi0=out["psiq"], CI0=out["CI"],
                       psib0=out.get("psib"), step_ref=out.get("dpsi_step"), **kw)
    phi_hi = float(out_hi["diag"]["Phi_private_Wb"])
    g_I = (phi_hi - phi0) / (Il0[0] * args.il_frac)          # Wb/A
    print(f"[inv] 灵敏度：∂Φ/∂I_l={g_I:.3e} Wb/A"
          f"（I_l +{args.il_frac * 100:.1f}% → Φ {(phi_hi - phi0) / phi0 * 100:+.3f}%）；"
          f"{time.perf_counter() - ts:.1f} s")
    g_B = args.dt * args.eta_cd * 2 * 0.5 * S0_0 / be        # ∂Φ/∂B_w（单步）
    print(f"[inv] 单步增益：线圈 g_I·I0={g_I * Il0[0]:.3e} Wb、"
          f"RMF g_B·B_w={g_B * args.bw_max:.3e} Wb（Δt={args.dt * 1e6:.0f} μs）")
    J = np.array([[g_I * Il0[0], g_B * args.bw_max]])
    print(f"[inv] 可辨识性：单时刻雅可比 rank={int(np.linalg.matrix_rank(J))}"
          f"（1×2）→ 只有 Φ 一个观测量时两执行器不可分，需第二个观测量或时间签名")

    # ---- 目标轨迹与指令 ---------------------------------------------------
    dt, n = args.dt, int(args.steps)
    t = np.arange(n + 1) * dt
    if args.target == "hold":
        phi_ref = np.full(n + 1, phi0)
    else:
        phi_ref = phi0 * (1.0 - args.ramp_frac * (t / t[-1]))
    S0_ref = np.array([S0_of(p) for p in phi_ref[:-1]])
    S_req = np.diff(phi_ref) / dt + S0_ref

    with np.errstate(invalid="ignore", divide="ignore"):
        bw_raw = be * np.sqrt(np.maximum(S_req, 0.0) / (args.eta_cd * F * S0_ref))
    bw = np.clip(bw_raw, 0.0, args.bw_max)

    S_gap = S_req - args.eta_cd * (bw / be) ** 2 * F * S0_ref
    # 缺口由线圈补：ΔΦ_coil = g_I·ΔI，要它等于 S_gap·dt
    dI_raw = S_gap * dt / g_I if g_I != 0 else np.zeros_like(S_gap)
    dI_cap = min(args.vmax / args.coil_L * dt, args.il_max_frac * Il0[0])
    dI = np.clip(dI_raw, -dI_cap, dI_cap)
    I_cmd = Il0[0] + np.concatenate([[0.0], np.cumsum(dI)])

    print(f"[inv] 指令：B_w 峰值={bw.max() * 1e3:.2f} mT"
          f"（原始需求 {bw_raw.max() * 1e3:.2f} mT，上限 {args.bw_max * 1e3:.0f} mT）")
    print(f"[inv] 指令：ΔI 峰值={np.abs(dI).max():.1f} A（单步上限 {dI_cap:.1f} A）、"
          f"I ∈ [{I_cmd.min() / 1e3:.2f}, {I_cmd.max() / 1e3:.2f}] kA")

    # ---- 限幅后的残差：同一套 S0(Φ) 关系的 0D 复算 -------------------------
    phi_act = np.empty(n + 1)
    phi_act[0] = phi0
    for k in range(n):
        S0k = S0_of(phi_act[k])
        S_drv = args.eta_cd * (bw[k] / be) ** 2 * F * S0k + dI[k] * g_I / dt
        phi_act[k + 1] = max(0.0, phi_act[k] + (S_drv - S0k) * dt)
    err = np.abs(phi_act - phi_ref) / phi0
    print(f"[inv] 限幅后残差（0D 复算）：max={err.max() * 100:.3f}%、"
          f"rms={np.sqrt((err ** 2).mean()) * 100:.3f}%")
    if err.max() > 0.02:
        print("       → 超 2%：限幅卡住了轨迹，需更高 V_max / 更大 B_w 上限，"
              "或放宽目标")

    res = dict(be=be, rs=rs, phi0=phi0, S0_0=S0_0, eta=eta, x_slip=x_slip, F=F,
               eta_cd=args.eta_cd, g_I=g_I, g_B=g_B,
               rank_J=int(np.linalg.matrix_rank(J)), target=args.target,
               ramp_frac=args.ramp_frac, bw_max=args.bw_max,
               bw_peak=float(bw.max()), bw_raw_peak=float(bw_raw.max()),
               dI_peak_A=float(np.abs(dI).max()), dI_cap_A=float(dI_cap),
               I_min_A=float(I_cmd.min()), I_max_A=float(I_cmd.max()),
               resid_max=float(err.max()),
               resid_rms=float(np.sqrt((err ** 2).mean())), S0_source=src)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / "inverse.npz", t=t, phi_ref=phi_ref,
                        phi_act=phi_act, S_req=S_req, bw=bw, bw_raw=bw_raw,
                        dI=dI, I_cmd=I_cmd, err=err)
    with open(OUT_DIR / "inverse.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(f"[inv] 产物：{OUT_DIR / 'inverse.npz'}、{OUT_DIR / 'inverse.json'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tms = t * 1e3
        fig, ax = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        ax[0, 0].plot(tms, phi_ref, "k-", label=r"$\Phi_{ref}$")
        ax[0, 0].plot(tms, phi_act, "r--", label=r"$\Phi$ (clipped)")
        ax[0, 0].legend(); ax[0, 0].set_xlabel("t [ms]"); ax[0, 0].set_ylabel("[Wb]")
        ax[0, 0].set_title("tracking")
        ax[0, 1].plot(tms[1:], bw * 1e3, "o-")
        ax[0, 1].set_xlabel("t [ms]"); ax[0, 1].set_ylabel(r"$B_\omega$ [mT]")
        ax[1, 0].plot(tms, I_cmd / 1e3, "o-")
        ax[1, 0].set_xlabel("t [ms]"); ax[1, 0].set_ylabel(r"$I_l$ [kA]")
        ax[1, 1].semilogy(tms, np.maximum(err, 1e-8), "o-")
        ax[1, 1].set_xlabel("t [ms]"); ax[1, 1].set_ylabel(r"$|\Delta\Phi|/\Phi_0$")
        for a in ax.ravel():
            a.grid(alpha=0.3)
        fig.suptitle(rf"Inverse (design point A): {args.target}, "
                     rf"max err {err.max() * 100:.2f}%")
        fig.savefig(OUT_DIR / "inverse.png", dpi=110)
        print(f"[inv] 产物：{OUT_DIR / 'inverse.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[inv] 画图跳过：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
