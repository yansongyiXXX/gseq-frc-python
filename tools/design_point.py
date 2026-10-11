r"""装置设计点：把"公开值当锚点、缺的参数自己定"这一步做成可复算的脚本。

设计点 A（PFRC-2 量级的 RMF 维持型 FRC）
---------------------------------------
输入（自定，依据见 `装置设计点_自定参数.md`）：

* 剖面：\(B_e=0.05\) T、\(R_s=0.08\) m、\(\beta_s=0.7\)（形状指数沿用 MRR 族）；
* 几何：沿用程序树 `icase=2` 那套线圈（壁 \(R_w=0.14\) m、半长 0.75 m、
  3 个线圈），它和 PFRC-2 中心室（内径 22.7 cm、长 83.8 cm）同一量级；
* 输运：\(T_e=30\) eV、\(Z=1\)、\(\ln\Lambda=10\)（Spitzer 电阻）。

输出：解出的平衡诊断量（私通量、分离面半径、轴上反场、比压、储能、耗散率、
寿命），以及由它们反推的工程量（线圈电流、\(L\)、\(R\)、电源电压、RMF 需求）。

用法：`python tools/design_point.py`
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
os.chdir(ROOT)

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

from gseqfrc import mpath  # noqa: E402

MU0 = 4 * np.pi * 1e-7
ME = 9.1093837015e-31
QE = 1.602176634e-19
SRC = ROOT / "GSEQ-FRC-open" / "second_paper" / "Fig6"
OUT_DIR = ROOT / "outputs" / "design_point"

# 基准算例（Fig6 默认几何、1 MA/线圈）在**中平面壁面**产生的真空场，
# 见 `修改记录.md` 第 24 条：B_ext = 1.0689 T。
B_WALL_PER_MA = 1.0689

mpath.addpath("second_paper/Fig6")
gcm = mpath.load("GSEQ_coil_MRR")
nk = gcm._load_nk_inner()


def spitzer_resistivity(te_ev: float, ln_lambda: float = 10.0, z: float = 1.0) -> float:
    r"""Spitzer 电阻率 [\(\Omega\cdot\)m]。"""
    return 5.2e-5 * z * ln_lambda / max(te_ev, 1e-6) ** 1.5


def collision_time(ne: float, eta: float) -> float:
    r"""电子—离子碰撞时间 \(\tau_e=m_e/(n_ee^2\eta)\) [s]。"""
    return ME / (ne * QE ** 2 * eta)


def self_inductance(R: float, area: float) -> float:
    r"""圆环自感 \(L=\mu_0R(\ln(8R/a)-0.5)\)，\(a=\sqrt{A/\pi}\)。"""
    a = np.sqrt(area / np.pi)
    return MU0 * R * (np.log(8.0 * R / a) - 0.5)


def loss_rate(psi2, pr, eta: float, k: int = 2):
    r"""\(S_0=2\pi\frac{\eta}{\mu_0}\Delta^*\psi|_{\text{O 点}}\) [Wb/s]（同第 21 条）。"""
    psif = np.asarray(psi2, dtype=float).ravel(order="F")
    dstar = -(pr.A @ psif) / pr.dZ2
    i0 = int(np.argmin(psif))
    nR = pr.nR
    iR, iZ = i0 % nR, i0 // nR
    vals = [float(dstar[(iR + a) + (iZ + b) * nR])
            for a in range(-k, k + 1) for b in range(-k, k + 1)
            if 1 <= iR + a <= nR - 2 and 1 <= iZ + b <= pr.nZ - 2]
    return 2 * np.pi * (eta / MU0) * float(np.median(vals))


def build_design_model(be=0.05, rs=0.08, betas=0.7,
                       be_ref=1.0, rs_ref=0.075, plasma_coupling=False):
    """按设计点 A 造一个 setup + MachineModel（几何沿用 `icase=2`）。

    两条标度律：

    * 线圈电流按磁通尺度 \(B_eR_s^2\)（否则 psi 尺度与压力剖面差两个量级，CI 环不收敛）；
    * 电流幅值 \(I_{\zeta0}\) 按 \(B_eR_s\)（同一族的两条相似律）。

    返回 `(s, model, il_scale, iz_scale)`。
    """
    s0 = gcm._setup(SRC)
    il_scale = (be * rs ** 2) / (be_ref * rs_ref ** 2)
    iz_scale = (be * rs) / (be_ref * rs_ref)
    s = dict(s0)
    s.update(Be=be, Rs=rs, betas=betas, Pm0=be ** 2 / (2 * MU0),
             Izeta0=float(s0["Izeta0"]) * iz_scale)
    s["Il"] = np.asarray(s0["Il"], dtype=float) * il_scale
    model = gcm.MachineModel(s, mutual=True, plasma_coupling=plasma_coupling)
    return s, model, il_scale, iz_scale


def warm_start_guess(il_scale, base_mat=None):
    r"""把基准解按磁通尺度缩放当暖启动初值 \(( \psi_0, CI_0)\)。

    冷启动直接解设计点在绝对容差下不收敛，必须走这条路径。
    """
    from scipy.io import loadmat
    p = Path(base_mat) if base_mat else (ROOT / "outputs" / "p2-Fig6" / "psiq.mat")
    m = loadmat(p, squeeze_me=True)
    return (np.asarray(m["psiq"], dtype=float).ravel(order="F") * il_scale,
            float(m["CIq"]))


def run_design(be=0.05, rs=0.08, betas=0.7, te=30.0, ln_lambda=10.0,
               coil_turns=20, coil_cond=20e-3 * 20e-3,
               be_ref=1.0, rs_ref=0.075, verbose=True):
    """解一个设计点平衡，并给出配套的工程量。"""
    s, model, il_scale, iz_scale = build_design_model(be, rs, betas, be_ref, rs_ref)
    s0 = gcm._setup(SRC)
    il_design = np.asarray(s["Il"], dtype=float)
    params = dict(Izeta0=float(s["Izeta0"]))
    psi0, ci0 = warm_start_guess(il_scale)

    t0 = time.perf_counter()
    out = gcm.solve(SRC, model=model, Il=il_design, params=params,
                    psi0=psi0, CI0=ci0)
    wall = time.perf_counter() - t0

    dg = out["diag"]
    psi2 = np.asarray(out["psiq"], dtype=float).reshape(s0["nR"], s0["nZ"], order="F")
    pr = model.problem(model.coil_flux(il_design)[3], 2,
                       params=nk.params_from_setup({**s0, **params}))
    eta = spitzer_resistivity(te, ln_lambda)
    S0 = loss_rate(psi2, pr, eta)
    phi = float(dg["Phi_private_Wb"])
    tau_phi = phi / S0 if S0 > 0 else float("nan")

    # ---- 线圈电流：电流（安匝）与边界真空场 ---------------------------------
    i_amp_turns = float(il_design[0])
    i_conductor = i_amp_turns / coil_turns
    b_wall_vac = B_WALL_PER_MA * il_scale

    # ---- 回路：圆环自感 + 铜导体电阻 ---------------------------------------
    r_coil = float(np.mean(s0["rl"]))
    L = coil_turns ** 2 * self_inductance(r_coil, coil_cond)
    R = coil_turns * 1.7e-8 * 2 * np.pi * r_coil / coil_cond
    # 目标：把绕组电流在 0.2 ms 内改变 10%（快速通量控制的量级）
    dt_ctrl, frac = 2e-4, 0.10
    v_req = L * (frac * i_conductor) / dt_ctrl

    # ---- 导体壳：要让壳不挡住 0.2 ms 的控制，壳时间常数需 ≲ dt_ctrl ----------
    sigma_cu = 5.9e7
    tau_per_m = MU0 * sigma_cu * float(s0["Rw"]) / 2.0     # tau = 该值 × 壳厚
    t_shell_for_ctrl = dt_ctrl / tau_per_m

    # ---- RMF：滑移最优处需要多大 B_w/B_e -----------------------------------
    ne_nominal = 6.83e10 * te ** 1.5 / (1.0 / (2 * np.pi * 2.0e6))
    tau_e = collision_time(ne_nominal, eta)
    f_opt = 1.0 / (2 * np.pi * tau_e)
    beta_req = {ec: float(1.0 / np.sqrt(ec * 0.5)) for ec in (1.0, 10.0, 100.0)}

    res = dict(
        params=dict(Be=be, Rs=rs, betas=betas, alpha=float(s0["alpha"]),
                    sigma=float(s0["sigma"]), n=float(s0["n"]),
                    Te=te, lnLambda=ln_lambda),
        solve=dict(wall_s=wall, C=float(out["C"]), CI=float(out["CI"]),
                   Be1_rel=float(dg.get("Be1_rel", float("nan"))),
                   Phi_private_Wb=phi, Rs_mid_m=float(dg.get("Rs_mid", np.nan)),
                   Bz_axis_T=float(dg.get("Bz_axis_mid", np.nan)),
                   beta_avg_sep=float(dg.get("beta_avg_sep", np.nan)),
                   elongation=float(dg.get("elongation", np.nan)),
                   xs_mid=float(dg.get("xs_mid", np.nan)),
                   W_sep_J=float(dg.get("W_sep", np.nan)),
                   Izeta_total_A=float(dg.get("Izeta_total", np.nan)),
                   Phi_over_BeRs2=float(dg.get("Phi_over_BeRs2", np.nan)),
                   psi_min=float(dg["psi_min"])),
        transport=dict(eta=eta, S0_Wb_s=S0, tau_phi_s=tau_phi,
                       ne_for_omega_tau1=ne_nominal, tau_e_s=tau_e, f_opt_Hz=f_opt),
        engineering=dict(coil_R_m=r_coil,
                         coil_cond_m2=coil_cond, coil_turns=coil_turns,
                         I_amp_turn_A=i_amp_turns, I_conductor_A=i_conductor,
                         B_wall_vacuum_T=b_wall_vac,
                         L_H=L, R_ohm=R, L_over_R_s=L / R,
                         V_req_200us_10pct_v=v_req,
                         shell_tau_per_m_s=tau_per_m,
                         shell_t_for_0p2ms_m=t_shell_for_ctrl),
        rmf=dict(beta_w_req=beta_req),
    )

    if verbose:
        print(f"[design] 解一次 {wall:.1f} s：C={out['C']:.6f} CI={out['CI']:.6f} "
              f"Be1/Be-1={dg.get('Be1_rel', np.nan):+.2e}")
        print(f"[design] 平衡：Phi={phi:.3e} Wb、Rs={dg.get('Rs_mid', np.nan) * 1e2:.2f} cm、"
              f"Bz_axis={dg.get('Bz_axis_mid', np.nan):+.4f} T、"
              f"<beta>_sep={dg.get('beta_avg_sep', np.nan):.3f}、"
              f"x_s={dg.get('xs_mid', np.nan):.3f}、"
              f"W_sep={dg.get('W_sep', np.nan):.1f} J")
        print(f"[design] 输运：eta={eta:.2e} Ohm*m、S0={S0:.4f} Wb/s、"
              f"tau_Phi=Phi/S0={tau_phi * 1e3:.3f} ms")
        print(f"[design] 线圈：安匝={i_amp_turns / 1e3:.1f} kA、"
              f"{coil_turns} 匝 → 导体电流={i_conductor / 1e3:.2f} kA、"
              f"中平面壁面真空场={b_wall_vac:.4f} T")
        print(f"[design] 回路：L={L * 1e6:.1f} uH、R={R * 1e3:.3f} mOhm、"
              f"L/R={L / R * 1e3:.1f} ms")
        print(f"[design] 电源：在 {dt_ctrl * 1e6:.0f} us 内改变 10% 导体电流需 "
              f"{v_req:.2f} V（电流源需能给出 {i_conductor / 1e3:.2f} kA）")
        print(f"[design] 壳：铜壳 tau≈{tau_per_m * 1e3:.2f} ms/mm 厚；"
              f"要在 0.2 ms 内不被壳屏蔽，壳厚需 ≲ {t_shell_for_ctrl * 1e6:.1f} um"
              f"（或开缝/主动补偿）")
        print(f"[design] RMF：使 omega*tau_e=1 的 n_e={ne_nominal:.2e} m^-3、"
              f"f={f_opt / 1e6:.3f} MHz")
        for ec, bw in beta_req.items():
            print(f"          eta_cd={ec:6.1f}  ->  B_w/B_e={bw:.3f}"
                  f"（B_w={bw * be * 1e3:.2f} mT @ B_e={be:.3f} T）")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "design_point.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    return res


if __name__ == "__main__":
    run_design()
    print(f"\n[design] 产物：{OUT_DIR / 'design_point.json'}")
