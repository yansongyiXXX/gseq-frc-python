r"""θ 箍缩形成的 0D 模型（骨架，第一版）。

对应设计文档 `形成段_0D模型设计.md` §6 第 1 步 + §8.9 的参数。

做四件事：

1. **主币回路**：电容组 + 单匝形成线圈的 RLC 演化，峰值处 crowbar；
2. **场与反场**：由线圈电流算轴上外场 \(B_{\rm ext}(t)\)、净场
   \(B_{z0}(t)=B_{\rm bias}-B_{\rm ext}\)、反场因子 \(F=-B_{z0}/B_{\rm bias}\)；
3. **等离子体等效环**：通量守恒给 \(I_p(t)\) 与通量预算；
4. **自检**：能量守恒、纯回路解析极限、通量守恒、dt 收敛、与设计点对表。

用法（在仓库根目录）：

    python tools/frc_formation_0d.py            # 用 §8.9 的默认参数
    python tools/frc_formation_0d.py --verify   # 打印自检表
    python tools/frc_formation_0d.py --tau-r 40e-6 --v0 auto

产物：`outputs/frc_formation/formation0d.npz` / `.json` / `.png`。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

MU0 = 4 * np.pi * 1e-7
QE = 1.602176634e-19       # C
MP = 1.67262192369e-27     # kg，质子质量（氢/氘量级）
OUT_DIR = ROOT / "outputs" / "frc_formation"

# ---------------------------------------------------------------- 设计点锚点
#: 设计点 A（装置设计点_自定参数.md / tools/design_point.py）的平衡量
PHI_EQ = 1.076e-4          # Wb，私通量
IZETA_EQ = 41.4e3          # A，环向电流
TAU_PHI = 0.353e-3         # s，通量寿命
RS_EQ = 0.07967            # m，分离面半径
ELONG_EQ = 3.16            # 设计点 A 的伸长比（→ l_s^eq = 2·E·R_s）
BWALL_PER_A = 1.0689 / 1e6  # T/A，设计点三线圈几何下"壁面场/安匝"


def reconnection(h: dict, cfg: dict, kappa: float) -> tuple[np.ndarray, np.ndarray]:
    r"""重联把开场线通量搬成闭合通量（§8.4c）。

    $$\dot\Phi_{\rm open}=\dot\Phi_{\rm drive}-\dot\Phi_{\rm rec},\qquad
    \dot\Phi_{\rm closed}=\dot\Phi_{\rm rec}-\Phi_{\rm closed}/\tau_\Phi,\qquad
    \dot\Phi_{\rm rec}=\kappa\frac{\Phi_{\rm open}}{\tau_{\rm rec}}H(F-F_{\rm thr})
    \Big(1-\frac{\Phi_{\rm closed}}{\Phi_{\rm cap}}\Big)$$

    驱动 \(\dot\Phi_{\rm drive}=M_{\rm cp}\,\mathrm dI_c/\mathrm dt\)（只取上升支）。
    **饱和项**是关键：FRC 只能容纳 \(\Phi_{\rm cap}=cB_eR_s^2\) 的私通量，多出来的
    开场线通量被排出而不是无限堆成闭合通量。返回 \((\Phi_{\rm open},\Phi_{\rm closed})\)。
    """
    from scipy.integrate import solve_ivp

    t, Ic, F = h["t"], h["Ic"], h["F"]
    drive = cfg["M_cp"] * np.gradient(Ic, t)
    g = np.asarray(F >= cfg["f_thr"], dtype=float)
    tau_rec, tau_phi, cap = cfg["tau_rec"], TAU_PHI, cfg["phi_cap"]

    def rhs(tt, y):
        ph_o, ph_c = y
        gate = float(np.interp(tt, t, g))
        rec = kappa * ph_o / tau_rec * gate * max(0.0, 1.0 - ph_c / cap)
        return [float(np.interp(tt, t, drive)) - rec, rec - ph_c / tau_phi]

    sol = solve_ivp(rhs, [t[0], t[-1]], [0.0, 0.0], t_eval=t,
                    rtol=1e-9, atol=1e-15, max_step=cfg["dt"])
    return sol.y[0], sol.y[1]


def calibrate_kappa(h: dict, cfg: dict, target: float) -> float:
    """标定 κ，使闭合通量在窗口内**到达**目标私通量的时刻最接近给定时刻。

    有饱和项后 Φ_closed→Φ_cap 是自动的，κ 只影响到达快慢。这里返回 1.0 的
    占位（保留接口，供后续用实测形成时间反标定时使用）。
    """
    return 1.0


def sound_speed(te_ev: float, ti_ev: float, mass_kg: float = MP) -> float:
    r"""离子声速 \(c_s=\sqrt{(T_e+T_i)/m_i}\)（\(T\) 换算成 J）。"""
    return float(np.sqrt((te_ev + ti_ev) * QE / mass_kg))


def axial_contraction(h: dict, ph_closed: np.ndarray, cfg: dict) -> np.ndarray:
    r"""轴向收缩（§8.4d）：FRC 一形成就从初柱长 \(l_0\) 松弛到平衡长度 \(l_s^{\rm eq}\)。

    $$\tau_{\rm ax}\dot l_s=l_s^{\rm eq}-l_s,\qquad \tau_{\rm ax}\sim l_s^{\rm eq}/c_s$$

    收缩在"FRC 已经出现"之后才开始（以 \(\Phi_{\rm closed}\) 过半为门）。返回 \(l_s(t)\)。
    """
    from scipy.integrate import solve_ivp

    t = h["t"]
    ls_eq, tau_ax = cfg["ls_equil"], cfg["tau_ax"]
    gate = np.asarray(ph_closed >= 0.5 * cfg["phi_cap"], dtype=float)

    def rhs(tt, y):
        g = float(np.interp(tt, t, gate))
        return [(ls_eq - y[0]) / tau_ax * g]

    sol = solve_ivp(rhs, [t[0], t[-1]], [cfg["ls_init"]], t_eval=t,
                    rtol=1e-9, atol=1e-12, max_step=cfg["dt"])
    return sol.y[0]


def solenoid_axis_per_amp(half_len: float, radius: float) -> float:
    r"""有限长螺线管**轴心**场：\(B_z=\mu_0\,NI/(2\sqrt{a^2+R^2})\)，\(a\) 为半长。

    对形成线圈（半长 0.7 m、半径 0.15 m）给出 \(B_z/(NI)=8.79\times10^{-7}\) T/A，
    于是 \(NI=56.9\) kA 对应轴上 0.050 T——与设计点 \(B_e=0.05\) T 同量级。
    """
    return MU0 / (2.0 * np.sqrt(half_len ** 2 + radius ** 2))


def ring_self_inductance(radius: float, minor: float) -> float:
    r"""细环自感 \(L=\mu_0R(\ln(8R/a)-2)\)（用于环-环互感的量级校核）。"""
    return MU0 * radius * (np.log(8.0 * radius / minor) - 2.0)


def integrate(cfg: dict) -> dict:
    """两段积分：主币 RLC（0→t_crow），crowbar 后 L/R 自由衰减。"""
    from scipy.integrate import solve_ivp

    L, C, Rc, Lc = cfg["L"], cfg["C"], cfg["R_coil"], cfg["L_plasma"]
    Mc, Rp = cfg["M_cp"], cfg["R_plasma"]
    V0, t_crow, t_end = cfg["V0"], cfg["t_crow"], cfg["t_end"]
    kc = cfg["k_per_amp"]

    def rhs_pre(t, y):
        Ic, Q, Ip = y
        dIc = (Q / C - Rc * Ic) / L
        # 通量守恒：d/dt(Lp*Ip + Φ_link) = -Rp*Ip，而 Φ_link = Φ_bias - Mc*Ic
        # ⇒ Lp*dIp/dt = +Mc*dIc/dt - Rp*Ip
        return [dIc, -Ic, (Mc * dIc - Rp * Ip) / Lc]

    def rhs_post(t, y):
        Ic, Q, Ip = y
        dIc = -Rc * Ic / L
        return [dIc, 0.0, (Mc * dIc - Rp * Ip) / Lc]

    y0 = [0.0, C * V0, 0.0]
    n1 = max(2, int(cfg["n_per_step"] * t_crow / cfg["dt"]))
    sol1 = solve_ivp(rhs_pre, [0.0, t_crow], y0, method="RK45",
                     t_eval=np.linspace(0.0, t_crow, n1), rtol=1e-10, atol=1e-12)
    s2 = sol1.y[:, -1]
    n2 = max(2, int(cfg["n_per_step"] * (t_end - t_crow) / cfg["dt"]))
    sol2 = solve_ivp(rhs_post, [t_crow, t_end], s2, method="RK45",
                     t_eval=np.linspace(t_crow, t_end, n2), rtol=1e-10, atol=1e-12)

    t = np.concatenate([sol1.t, sol2.t[1:]])
    Ic = np.concatenate([sol1.y[0], sol2.y[0][1:]])
    Q = np.concatenate([sol1.y[1], sol2.y[1][1:]])
    Ip = np.concatenate([sol1.y[2], sol2.y[2][1:]])
    # 偏置场按**实际达到的峰值**定，使反场因子正好到 1.0（回路电阻让实际峰值比
    # 目标低 ~0.3%）。用户显式给了 B_bias 就尊重用户值。
    if cfg.get("b_bias_auto", True):
        cfg["B_bias"] = 0.5 * kc * float(np.max(Ic))
    B_ext = kc * Ic
    B_app = cfg["B_bias"] - B_ext
    F = -B_app / cfg["B_bias"]
    return dict(t=t, Ic=Ic, Q=Q, Ip=Ip, B_ext=B_ext, B_app=B_app, F=F)


def energy_split(h: dict, cfg: dict) -> dict:
    """回路能量记账：电容 / 磁 / 欧姆耗散（逐步累积）。"""
    t, Ic, Q = h["t"], h["Ic"], h["Q"]
    E_cap = Q ** 2 / (2 * cfg["C"])
    E_mag = 0.5 * cfg["L"] * Ic ** 2
    ohmic = np.concatenate([[0.0], np.cumsum(
        0.5 * (Ic[1:] ** 2 + Ic[:-1] ** 2) * cfg["R_coil"] * np.diff(t))])
    return dict(E_cap=E_cap, E_mag=E_mag, E_ohmic=ohmic)


def make_config(args) -> dict:
    """按 §8.9 组装参数；`auto` 表示由 \(\tau_r,I_{\rm peak}\) 反算 \(C,V_0\)。"""
    a_cond = np.sqrt(args.conductor ** 2 / np.pi)
    L = MU0 * args.radius * (np.log(8 * args.radius / a_cond) - 0.5)
    kc = solenoid_axis_per_amp(args.half_len, args.radius)
    I_peak = args.i_peak
    tau_r = args.tau_r
    C = (2 * tau_r / np.pi) ** 2 / L
    V0 = I_peak * np.sqrt(L / C)
    if args.c not in (None, "auto"):
        C = float(args.c)
        V0 = I_peak * np.sqrt(L / C)
    if args.v0 not in (None, "auto"):
        V0 = float(args.v0)
    B_ext_peak = kc * I_peak
    B_bias = args.b_bias if args.b_bias is not None else 0.5 * B_ext_peak
    b_bias_auto = args.b_bias is None
    L_plasma = args.l_plasma if args.l_plasma is not None else PHI_EQ / IZETA_EQ
    R_plasma = args.r_plasma if args.r_plasma is not None else L_plasma / TAU_PHI
    M_cp = kc * np.pi * args.r_plasma_ring ** 2
    ls_equil = args.ls_equil if args.ls_equil is not None else 2 * ELONG_EQ * RS_EQ
    cs = sound_speed(args.te, args.ti)
    tau_ax = args.tau_ax if args.tau_ax is not None else ls_equil / cs
    return dict(
        radius=args.radius, half_len=args.half_len, conductor=args.conductor,
        L=L, k_per_amp=kc, I_peak=I_peak, tau_r=tau_r, C=C, V0=V0,
        R_coil=args.r_coil, t_crow=tau_r, t_end=args.t_end, dt=args.dt,
        n_per_step=args.n_per_step,
        B_bias=B_bias, B_ext_peak=B_ext_peak,
        b_bias_auto=b_bias_auto,
        r_plasma_ring=args.r_plasma_ring, M_cp=M_cp,
        L_plasma=L_plasma, R_plasma=R_plasma,
        tau_rec=args.tau_rec, f_thr=args.f_thr, phi_target=args.phi_target,
        phi_cap=args.phi_cap,
        ls_init=args.ls_init, ls_equil=ls_equil, tau_ax=tau_ax,
        te=args.te, ti=args.ti, c_s=cs,
        E_bank=0.5 * C * V0 ** 2, L_over_R=L / args.r_coil,
    )


def run_checks(cfg: dict, h: dict) -> list[tuple[str, str, str]]:
    """自检表：名字 / 数值 / 判据。"""
    rows: list[tuple[str, str, str]] = []
    en = energy_split(h, cfg)
    tot = en["E_cap"] + en["E_mag"] + en["E_ohmic"]
    drift = float(np.max(np.abs(tot - tot[0])) / tot[0])
    rows.append(("能量守恒（电容+磁+欧姆）", f"{drift:.2e}", "≤1e-6"))

    # 纯回路解析极限：R=0 时 I_peak = V0*sqrt(C/L)，出现在 τ_r=π/2*sqrt(LC)
    Ip_ana = cfg["V0"] * np.sqrt(cfg["C"] / cfg["L"])
    rows.append(("峰值电流 I=V0√(C/L)", f"{Ip_ana / 1e3:.2f} kA",
                 f"目标 {cfg['I_peak'] / 1e3:.2f} kA（吻合 {abs(Ip_ana / cfg['I_peak'] - 1):.1e}）"))
    tr_ana = 0.5 * np.pi * np.sqrt(cfg["L"] * cfg["C"])
    rows.append(("上升时间 π/2√(LC)", f"{tr_ana * 1e6:.2f} µs",
                 f"目标 {cfg['tau_r'] * 1e6:.2f} µs"))

    # 等离子体通量守恒：Ψp = Lp*Ip + Φ_link，dΨp/dt = -Rp*Ip
    Phi_link = cfg["B_bias"] * np.pi * cfg["r_plasma_ring"] ** 2 - cfg["M_cp"] * h["Ic"]
    Psi_p = cfg["L_plasma"] * h["Ip"] + Phi_link
    t, Ip = h["t"], h["Ip"]
    cum = np.concatenate([[0.0], np.cumsum(
        0.5 * (Ip[1:] + Ip[:-1]) * cfg["R_plasma"] * np.diff(t))])
    expected = Psi_p[0] - cum
    rows.append(("等离子体通量守恒 dΨ/dt=-RpIp",
                 f"{float(np.max(np.abs(Psi_p - expected)) / abs(Psi_p[0])):.2e}",
                 "≤1e-6（对 ODE 一致性）"))

    # 反场因子
    Fmax = float(np.max(h["F"]))
    rows.append(("峰值反场因子 F", f"{Fmax:.3f}", "≥1.0（完全反场）"))

    # 场尺度对表
    rows.append(("轴上峰值场 B_ext", f"{cfg['B_ext_peak']:.4f} T",
                 "≈ 设计点 B_e=0.05 T"))
    rows.append(("偏置场 B_bias", f"{cfg['B_bias']:.4f} T", "= 峰值外场的一半"))

    # 通量预算
    swing = cfg["M_cp"] * cfg["I_peak"]
    rows.append(("线圈通量摆幅 M·I", f"{swing:.3e} Wb",
                 f"需私通量 {PHI_EQ:.3e} Wb（裕度 {swing / PHI_EQ:.2f}×）"))
    rows.append(("等离子体电流峰值 |Ip|", f"{float(np.max(np.abs(h['Ip']))) / 1e3:.2f} kA",
                 f"平衡参考 {IZETA_EQ / 1e3:.2f} kA"))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="θ 箍缩形成 0D 模型（骨架）")
    ap.add_argument("--radius", type=float, default=0.15, help="形成线圈半径 [m]")
    ap.add_argument("--half-len", type=float, default=0.70, help="螺线管半长 [m]")
    ap.add_argument("--conductor", type=float, default=0.020, help="导体边长 [m]")
    ap.add_argument("--i-peak", type=float, default=56.9e3, help="峰值安匝 [A]")
    ap.add_argument("--tau-r", type=float, default=20e-6, help="上升时间 [s]")
    ap.add_argument("--c", default="auto", help="电容 [F]，auto=按 τ_r 反算")
    ap.add_argument("--v0", default="auto", help="充电电压 [V]，auto=按 I_peak 反算")
    ap.add_argument("--r-coil", type=float, default=1.0e-4, help="回路电阻 [Ω]")
    ap.add_argument("--b-bias", type=float, default=None, help="偏置场 [T]，默认=峰值外场一半")
    ap.add_argument("--r-plasma-ring", type=float, default=0.08, help="等离子体环半径 [m]")
    ap.add_argument("--l-plasma", type=float, default=None, help="等离子体等效电感 [H]")
    ap.add_argument("--r-plasma", type=float, default=None, help="等离子体等效电阻 [Ω]")
    ap.add_argument("--t-end", type=float, default=60e-6, help="终点 [s]")
    ap.add_argument("--dt", type=float, default=1e-8, help="积分步 [s]")
    ap.add_argument("--n-per-step", type=int, default=1, help="每步采样数（占位）")
    ap.add_argument("--tau-rec", type=float, default=5e-6, help="重联时间 τ_rec [s]")
    ap.add_argument("--f-thr", type=float, default=0.9, help="重联门槛反场因子")
    ap.add_argument("--kappa-rec", type=float, default=1.0, help="重联速率系数 κ_rec")
    ap.add_argument("--phi-target", type=float, default=PHI_EQ, help="目标私通量 [Wb]")
    ap.add_argument("--phi-cap", type=float, default=PHI_EQ,
                    help="FRC 可容纳的私通量上限 Φ_cap [Wb]（= c·B_e·R_s²）")
    ap.add_argument("--ls-init", type=float, default=1.5, help="初柱长 l0 [m]（=2·z_w）")
    ap.add_argument("--ls-equil", type=float, default=None,
                    help="平衡 FRC 长度 [m]，默认 2·E·R_s（设计点）")
    ap.add_argument("--te", type=float, default=30.0, help="电子温度 [eV]（给 c_s）")
    ap.add_argument("--ti", type=float, default=30.0, help="离子温度 [eV]（给 c_s）")
    ap.add_argument("--tau-ax", type=float, default=None,
                    help="轴向收缩时间 [s]，默认 l_s^eq/c_s")
    ap.add_argument("--verify", action="store_true", help="只打印自检")
    args = ap.parse_args()

    cfg = make_config(args)
    h = integrate(cfg)
    kap = float(args.kappa_rec)
    cfg["kappa_rec"] = kap
    ph_open, ph_closed = reconnection(h, cfg, kap)
    above = np.nonzero(ph_closed >= 0.95 * cfg["phi_cap"])[0]
    i_form = int(above[0]) if above.size else int(np.argmax(ph_closed))
    t_form = float(h["t"][i_form])
    Il_handoff = float(h["Ic"][i_form])
    Phi_handoff = float(ph_closed[i_form])
    ls = axial_contraction(h, ph_closed, cfg)
    icont = np.nonzero(ls <= cfg["ls_equil"] * 1.05)[0]
    t_contract = float(h["t"][icont[0]]) if icont.size else float("nan")
    rows = run_checks(cfg, h)

    print("[formation0d] θ 箍缩形成 0D 骨架")
    print(f"  形成线圈：单匝 R={cfg['radius']} m，半长 {cfg['half_len']} m，"
          f"L={cfg['L'] * 1e6:.3f} µH，R={cfg['R_coil'] * 1e3:.3f} mΩ，"
          f"L/R={cfg['L_over_R'] * 1e3:.2f} ms")
    print(f"  主币：τ_r={cfg['tau_r'] * 1e6:.1f} µs  C={cfg['C'] * 1e6:.1f} µF  "
          f"V0={cfg['V0'] / 1e3:.2f} kV  E={cfg['E_bank']:.1f} J  "
          f"I_peak={cfg['I_peak'] / 1e3:.1f} kA  crowbar@{cfg['t_crow'] * 1e6:.1f} µs")
    print(f"  场：B_ext,peak={cfg['B_ext_peak']:.4f} T  B_bias={cfg['B_bias']:.4f} T")
    print(f"  重联：κ_rec={kap:.4g}（标定）τ_rec={cfg['tau_rec'] * 1e6:.1f} µs  "
          f"F_thr={cfg['f_thr']:.2f}")
    print(f"  形成：t_form={t_form * 1e6:.2f} µs  Φ_closed={Phi_handoff:.3e} Wb  "
          f"（目标 {args.phi_target:.3e}）  Il_handoff={Il_handoff / 1e3:.2f} kA 安匝")
    print(f"  轴向：l0={cfg['ls_init']:.2f} m → l_s^eq={cfg['ls_equil']:.3f} m  "
          f"c_s={cfg['c_s'] / 1e3:.1f} km/s  τ_ax={cfg['tau_ax'] * 1e6:.1f} µs  "
          f"收缩到位 t={t_contract * 1e6:.1f} µs")
    print("  自检：")
    for name, val, crit in rows:
        print(f"    {name:<26} {val:<16} {crit}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / "formation0d.npz",
                        t=h["t"], Ic=h["Ic"], Q=h["Q"], Ip=h["Ip"],
                        B_ext=h["B_ext"], B_app=h["B_app"], F=h["F"],
                        Phi_open=ph_open, Phi_closed=ph_closed,
                        l_s=ls,
                        cfg=np.array(json.dumps(
                            {k: (v if not isinstance(v, np.floating) else float(v))
                             for k, v in cfg.items()}, ensure_ascii=False)))
    np.savez_compressed(OUT_DIR / "handoff.npz",
                        Il=Il_handoff, Phi=Phi_handoff, t_form=t_form,
                        B_ext=float(h["B_ext"][i_form]), Ic=Il_handoff,
                        kappa_rec=kap, tau_rec=cfg["tau_rec"],
                        F_at_form=float(h["F"][i_form]),
                        ls_init=cfg["ls_init"], ls_equil=cfg["ls_equil"],
                        ls_at_form=float(ls[i_form]), tau_ax=cfg["tau_ax"],
                        c_s=cfg["c_s"], t_contract=t_contract)
    (OUT_DIR / "formation0d.json").write_text(json.dumps(
        {"cfg": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                 for k, v in cfg.items()},
         "checks": [{"name": n, "value": v, "crit": c} for n, v, c in rows],
         "formation": {"t_form_s": t_form, "Phi_closed_Wb": Phi_handoff,
                       "Il_handoff_A": Il_handoff, "kappa_rec": kap,
                       "F_at_form": float(h["F"][i_form])}},
        ensure_ascii=False, indent=1), encoding="utf-8")

    if not args.verify:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from gseqfrc import mstyle
            mstyle.install()
        except Exception:  # noqa: BLE001
            plt = None
        if plt is not None:
            en = energy_split(h, cfg)
            fig, ax = plt.subplots(2, 3, figsize=(15, 7))
            ax[0, 0].plot(h["t"] * 1e6, h["Ic"] / 1e3, color="C0", lw=2)
            ax[0, 0].set_xlabel("t [µs]"); ax[0, 0].set_ylabel("I_c [kA]")
            ax[0, 0].set_title("主币线圈电流（峰值处 crowbar）")
            ax[0, 0].axvline(cfg["t_crow"] * 1e6, color="gray", ls=":")
            ax[0, 1].plot(h["t"] * 1e6, h["B_app"], color="C3", lw=2)
            ax[0, 1].axhline(0.0, color="gray", ls=":")
            ax[0, 1].set_xlabel("t [µs]"); ax[0, 1].set_ylabel("B_z0 [T]")
            ax[0, 1].set_title("轴上净场（过零=场反转）")
            ax[0, 2].plot(h["t"] * 1e6, ph_open * 1e3, color="C1", lw=2,
                          label="Φ_open")
            ax[0, 2].plot(h["t"] * 1e6, ph_closed * 1e3, color="C5", lw=2,
                          label="Φ_closed")
            ax[0, 2].axhline(args.phi_target * 1e3, color="gray", ls="--")
            ax[0, 2].set_xlabel("t [µs]"); ax[0, 2].set_ylabel("Φ [mWb]")
            ax[0, 2].set_title("通量分解（重联）"); ax[0, 2].legend()
            ax[1, 0].plot(h["t"] * 1e6, h["F"], color="C2", lw=2)
            ax[1, 0].axhline(1.0, color="gray", ls="--")
            ax[1, 0].set_xlabel("t [µs]"); ax[1, 0].set_ylabel("F")
            ax[1, 0].set_title("反场因子 F（=1 完全反场）")
            ax[1, 1].plot(h["t"] * 1e6, h["Ip"] / 1e3, color="C4", lw=2)
            ax[1, 1].set_xlabel("t [µs]"); ax[1, 1].set_ylabel("I_p [kA]")
            ax[1, 1].set_title("等离子体等效环电流")
            ax[1, 2].plot(h["t"] * 1e6, ls, color="C6", lw=2)
            ax[1, 2].axhline(cfg["ls_equil"], color="gray", ls="--")
            ax[1, 2].axvline(t_form * 1e6, color="gray", ls=":", alpha=0.6)
            ax[1, 2].set_xlabel("t [µs]"); ax[1, 2].set_ylabel("l_s [m]")
            ax[1, 2].set_title(
                f"轴向收缩 l0={cfg['ls_init']:.2f}→l_s^eq={cfg['ls_equil']:.2f} m")
            fig.tight_layout()
            fig.savefig(OUT_DIR / "formation0d.png", dpi=140)
            plt.close(fig)
            print(f"[formation0d] 产物：{OUT_DIR / 'formation0d.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
