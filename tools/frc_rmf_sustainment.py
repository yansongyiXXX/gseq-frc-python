r"""Stage 5 = (b)：RMF 电流驱动 → 通量驱动率，以及"维持这个 FRC 需要多大 RMF"。

驱动模型（标准 RMF 标度，标定常数显式）
---------------------------------------
RMF（旋转磁场）对电子施加方位角转矩；在稳态，转矩与电阻性耗散平衡，
驱动率写成"耗散率的倍数"：

$$S_{\rm RMF}=\eta_{\rm cd}\,\beta_\omega^{2}\,F(\omega\tau_e)\;\frac{\Phi_0}{\tau_\eta},
\qquad \beta_\omega=\frac{B_\omega}{B_e},\qquad
F(x)=\frac{x}{1+x^{2}}$$

其中 \(\Phi_0/\tau_\eta=S_0\) 就是第 21/24 条量出来的**耗散率**（本算例 2.89 Wb/s），
\(F\) 是标准的"滑移因子"（\(\omega\tau_e=1\) 处最大，两端都掉），
\(\eta_{\rm cd}\) 是**唯一需要标定的无量纲效率**（默认 1；它把转矩→通量的
几何/穿透因子全兜住）。**维持条件**就是

$$\eta_{\rm cd}\,\beta_\omega^{2}F(\omega\tau_e)=1 .$$

为什么这么写（而不是拍一个具体系数）
------------------------------------
\(B_\omega^2\) 的标度、\(\omega\tau_e/(1+(\omega\tau_e)^2)\) 的滑移因子是 RMF 文献里
共识的部分；但"转矩 → 通量驱动率"的那一串几何/穿透因子各家写法不同，而本仓库里
没有 RMF 天线几何、也没有 \(n_e,T_e\) 剖面。所以这里**只把共识部分写死，把不共识的
部分集中成一个显式常数** \(\eta_{\rm cd}\)：一旦你给出实机的 RMF 参数（幅度、频率、
天线位置）或一组实验点的 \(\Phi\) 维持值，就能把 \(\eta_{\rm cd}\) 标定出来；
在标定之前，输出的是"**在标准标度下需要多少 RMF**"，量级是可靠的，系数不是。

\(B_\omega\) 还能从"刚性转子电流"这一侧交叉验证：高 \(\omega\tau_e\) 极限下 RMF 把
电子拖到与 RMF 图样共转（这正是 MRR 模型叫"刚性转子"的物理来源），
对应 \(J_\phi\simeq-n_ee\omega r\)；把这条代入 \(J_\zeta=R\,\mathrm{d}P/\mathrm{d}\psi\)
的同一套算子，就能从 \(\omega\) 反推所需的 \(B_\omega\)——`--crosscheck` 会打印这条
（需要 \(n_e\)，用 `--ne` 给）。

用法
----

    python tools/frc_rmf_sustainment.py                 # 标度表 + 维持边界图
    python tools/frc_rmf_sustainment.py --eta-cd 0.3    # 换效率
    python tools/frc_rmf_sustainment.py --omega-tau 0.5 # 指定滑移参数
    python tools/frc_rmf_sustainment.py --crosscheck --ne 1e19

理论版 η_cd（`--theory`，可选）
-------------------------------
按 Hugrass 1984 (Aust. J. Phys. 37, 509) 第 4 节的"内部效率"（原文 eq 33-35）把 η_cd
从待标定常数换成平衡剖面的泛函：
    γ(r)=(ν/ν0)(ωr/v_eθ)、⟨γ⟩=∫γηJ²2πr dr/∫ηJ²2πr dr、ℰ_i=1/⟨γ⟩≤1，
    η_cd^th=⟨γ⟩/(β_ref²F(x_ref))。
剖面默认取 `outputs/p2-Fig6/psiq1.mat` 里的 `Jzeta`（`--profile-mat` 可换，
`--profile-fresh` 现解一次）。需要 `--ne`。

产物：`outputs/frc_stage05/sustainment.npz` 与 `sustainment.png`。
"""

from __future__ import annotations

import argparse
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

from gseqfrc import mpath, mstyle  # noqa: E402

mstyle.install()   # 出图统一按 MATLAB 那一套（含中文字体回退，见 gseqfrc/mstyle.py）

# 效率内核（Hugrass 1984 eq 33-35）与正向演化共用，见 tools/rmf_efficiency.py
from rmf_efficiency import (  # noqa: E402
    E_CHARGE,
    M_ELECTRON,
    electron_collision_time,
    profile_integrals,
    required_beta,
    slip_factor,
    spitzer_resistivity,
    theory_eta_cd,
)

MU0 = 4 * np.pi * 1e-7
SRC = ROOT / "GSEQ-FRC-open" / "second_paper" / "Fig6"
OUT_DIR = ROOT / "outputs" / "frc_stage05"
DEFAULT_PROFILE_MAT = ROOT / "outputs" / "p2-Fig6" / "psiq1.mat"

mpath.addpath("second_paper/Fig6")
gcm = mpath.load("GSEQ_coil_MRR")
nk = gcm._load_nk_inner()


def midplane_profile(mat_path=DEFAULT_PROFILE_MAT, fresh=False, eta=None, te=50.0):
    r"""取中平面上的一维剖面 \((R,\psi,J_\zeta)\)。

    优先读已有产物里的 `Jzeta`（默认 `outputs/p2-Fig6/psiq1.mat`，由 `run.py p2-Fig6`
    生成）；`fresh=True` 或文件不存在时现解一次（p2-Fig6 默认口径，约十几秒）。
    """
    path = Path(mat_path)
    if not fresh and path.is_file():
        from scipy.io import loadmat

        d = loadmat(path)
        R = np.asarray(d["Rq"], float)[:, 0]
        Z = np.asarray(d["Zq"], float)[0, :]
        psi = np.asarray(d["psiq"], float)
        J = np.asarray(d["Jzeta"], float)
        i0 = int(np.argmin(np.abs(Z)))
        return R, psi[:, i0], J[:, i0], str(path)
    eta_used = spitzer_resistivity(te) if eta is None else float(eta)
    _S0, _Phi0, out = solve_loss_rate(eta_used, te)
    s = gcm._setup(gcm.SRC_DIR)
    R = np.asarray(s["rr"], float)
    Z = np.asarray(s["zz"], float)
    psi = np.asarray(out["psiq"], float)
    J = np.asarray(out["Jzeta"], float)
    i0 = int(np.argmin(np.abs(Z)))
    return R, psi[:, i0], J[:, i0], "现解初态（p2-Fig6 默认口径）"


def loss_rate_from_saved(out_dir: Path = ROOT / "outputs" / "frc_stage04"):
    """从 Stage 3/4 的产物里取耗散率 \(S_0\) 与初态通量 \(\Phi_0\)。

    优先用 `frc_stage04/history.npz`（含 S0），退回 `frc_stage03`。
    """
    for name in ("frc_stage04", "frc_stage03"):
        p = ROOT / "outputs" / name / "history.npz"
        if p.exists():
            d = np.load(p)
            if "S0" in d.files and d["S0"].size:
                return float(d["S0"][0]), float(d["Phi"][0]), name
    return None


def solve_loss_rate(eta: float, te: float):
    """没有产物时现算一次：解初态、取 \(\Phi_0\) 与 \(S_0=2\pi(\eta/\mu_0)\Delta^*\psi|_O\)。"""
    s = gcm._setup(gcm.SRC_DIR)
    model = gcm.MachineModel(s)
    out = gcm.solve(gcm.SRC_DIR, model=model)
    pr = model.problem(model.coil_flux(np.asarray(s["Il"], float))[3], 2)
    psi = np.asarray(out["psiq"], float)
    psif = psi.ravel(order="F")
    dstar = -(pr.A @ psif) / pr.dZ2
    i0 = int(np.argmin(psif))
    nR = pr.nR
    iR, iZ = i0 % nR, i0 // nR
    vals = [float(dstar[(iR + a) + (iZ + b) * nR])
            for a in (-2, -1, 0, 1, 2) for b in (-2, -1, 0, 1, 2)]
    S0 = 2 * np.pi * (eta / MU0) * float(np.median(vals))
    Phi0 = float(out["diag"]["Phi_private_Wb"])
    return S0, Phi0, out


def sustainment_table(S0: float, eta_cd: float, xs: np.ndarray) -> np.ndarray:
    r"""维持所需的 \(\beta_\omega=B_\omega/B_e\)：由 \(\eta_{\rm cd}\beta^2F=1\) 解出。

    \(\beta_{\rm req}=1/\sqrt{\eta_{\rm cd}F(x)}\)；\(F\) 小时它发散，这是物理的
    （滑移太大/太小都驱动不动）。
    """
    F = slip_factor(xs)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = 1.0 / np.sqrt(np.maximum(eta_cd * F, 1e-300))
    return beta


def rigid_rotor_ceiling(eta: float, ne: float, omega: float, r_O: float) -> tuple:
    r"""RMF 的**刚性转子上限**（第一性锚点，用来标定 \(\eta_{\rm cd}\)）。

    强 RMF（穿透、\(\omega\tau_e\gtrsim1\)）把电子拖到与 RMF 图样共转，电流密度被
    钉在 \(J_\phi^{RR}=n_ee\omega r\)——这正是 MRR 模型叫"刚性转子"的物理来源。
    用第 20/21 条同一套算子把它翻成通量驱动率：
    \(J_\phi=-\Delta^*\psi/(\mu_0r)\Rightarrow
    \Delta^*\psi^{RR}=\mu_0r^2n_ee\omega\Rightarrow
    S^{RR}=2\pi\eta n_ee\omega r_O^2\)（量纲 = V ✓）。

    返回 `(J_rr_A_m2, S_rr_Wb_s)`，其中 \(J\) 取 \(r_O\) 处的值。
    """
    e = 1.602176634e-19
    J_rr = ne * e * omega * r_O
    S_rr = 2 * np.pi * eta * ne * e * omega * r_O ** 2
    return J_rr, S_rr


def calibrate_eta_cd(eta: float, ne: float, S0: float, r_O: float,
                     beta_ref: float = 1.0, x_ref: float = 1.0) -> dict:
    r"""标定 \(\eta_{\rm cd}\)：让标度模型在**参考 RMF 条件**下等于刚性转子上限。

    参考条件取 \(B_\omega/B_e=\beta_{\rm ref}=1\)（强 RMF 的标准量级）与
    \(\omega\tau_e=x_{\rm ref}=1\)（滑移因子最大处）。于是

    $$\eta_{\rm cd}:=\frac{S^{RR}(\omega^*)}{\beta_{\rm ref}^2F(x_{\rm ref})S_0},
    \qquad \omega^*=x_{\rm ref}/\tau_e .$$

    这样模型里就没有自由常数了：唯一的外来输入是 \((n_e,T_e)\)（决定 \(\tau_e\) 与
    \(\eta\)）和 O 点半径 \(r_O\)。返回一个字典（含中间量，便于核对）。
    """
    tau_e = electron_collision_time(ne, eta)
    omega = x_ref / tau_e
    J_rr, S_rr = rigid_rotor_ceiling(eta, ne, omega, r_O)
    F = float(slip_factor(np.array([x_ref]))[0])
    eta_cd = S_rr / (beta_ref ** 2 * F * S0)
    return dict(tau_e=tau_e, omega=omega, J_rr=J_rr, S_rr=S_rr, F=F,
                eta_cd=eta_cd,
                beta_req_at_ref=float(1.0 / np.sqrt(eta_cd * F)))


def default_r_o(r_o_arg=None):
    """锚定口径的 O 点半径：显式给就用显式值；不给按 be-loop 初态 Rs=0.0569 的
    0.70 倍估（与 `--calibrate` 里原来那条口径一致）。"""
    return float(r_o_arg) if r_o_arg is not None else 0.70 * 0.0569


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--te", type=float, default=50.0, help="电子温度 [eV]（算 eta）")
    ap.add_argument("--eta", type=float, default=None, help="直接给电阻率 [Ohm*m]")
    ap.add_argument("--eta-cd", type=float, default=1.0,
                    help="RMF 驱动效率（唯一需要标定的无量纲常数，默认 1）")
    ap.add_argument("--omega-tau", type=float, default=None,
                    help="直接指定滑移参数 omega*tau_e（不给就扫一档）")
    ap.add_argument("--ne", type=float, default=None,
                    help="电子密度 [m^-3]，用于把 omega 换成频率、或做刚性转子交叉验证")
    ap.add_argument("--crosscheck", action="store_true",
                    help="打印「刚性转子电流」一侧的交叉验证（需要 --ne）")
    ap.add_argument("--calibrate", action="store_true",
                    help="用「刚性转子上限」标定 eta_cd（需要 --ne 与 O 点半径）")
    ap.add_argument("--r-o", type=float, default=None,
                    help="O 点半径 [m]；不给就按 0.70*Rs 估（MRR 族里 r_O/Rs≈0.70）")
    ap.add_argument("--fresh", action="store_true", help="不用已有产物，现解一次初态")
    ap.add_argument("--theory", action="store_true",
                    help="用 Hugrass 1984 的内部效率算理论版 eta_cd（需要 --ne），"
                         "并用它做维持边界")
    ap.add_argument("--nu-ratio", type=float, default=1.0,
                    help="有效/经典碰撞频率之比 nu/nu0（论文取 ~1，默认 1）")
    ap.add_argument("--profile-mat", default=str(DEFAULT_PROFILE_MAT),
                    help="含 Jzeta 的平衡产物（默认 outputs/p2-Fig6/psiq1.mat）")
    ap.add_argument("--profile-fresh", action="store_true",
                    help="不用现成产物，现解一次初态取中平面剖面")
    args = ap.parse_args()

    eta = (float(args.eta) if args.eta is not None
           else spitzer_resistivity(args.te))

    # 理论版 eta_cd（Hugrass 1984 内部效率口径）；不给 --theory 就退回原来的锚定常数
    eta_cd_used = args.eta_cd
    theory = None
    profile = None
    prof_src = ""
    anchor_eta_cd = None
    if args.theory:
        if not args.ne:
            print("[stage05] --theory 需要 --ne（v_eθ=J/(n_e e) 要用密度）；本次忽略。")
        else:
            R_mid, psi_mid, J_mid, prof_src = midplane_profile(
                args.profile_mat, fresh=args.profile_fresh, eta=eta, te=args.te)
            profile = profile_integrals(R_mid, psi_mid, J_mid)
            theory = theory_eta_cd(args.ne, eta, profile, nu_ratio=args.nu_ratio)
            eta_cd_used = theory["eta_cd"]

    got = None if args.fresh else loss_rate_from_saved()
    if got is not None:
        S0, Phi0, src = got
        tau_eta = Phi0 / S0
        print(f"[stage05] 耗散率取自 outputs/{src}/history.npz："
              f"S0={S0:.4f} Wb/s、Phi0={Phi0:.4e} Wb → Phi0/S0={tau_eta * 1e3:.4f} ms")
    else:
        t0 = time.perf_counter()
        S0, Phi0, _out = solve_loss_rate(eta, args.te)
        tau_eta = Phi0 / S0
        print(f"[stage05] 现算初态（{time.perf_counter() - t0:.1f} s）："
              f"S0={S0:.4f} Wb/s、Phi0={Phi0:.4e} Wb → Phi0/S0={tau_eta * 1e3:.4f} ms")
    print(f"[stage05] eta={eta:.3e} Ohm*m（T_e={args.te:g} eV）；"
          f"RMF 驱动效率 eta_cd={eta_cd_used:g}"
          + (f"（理论版；锚定值 {args.eta_cd:g}）" if theory is not None else ""))
    if args.ne:
        tau_e = electron_collision_time(args.ne, eta)
        print(f"[stage05] 由 n_e={args.ne:.3e} m^-3 反推 tau_e={tau_e:.3e} s"
              f" → omega*tau_e=1 对应 f={1.0 / (2 * np.pi * tau_e):.3e} Hz")

    if theory is not None:
        print("\n[stage05] 理论版 eta_cd（Hugrass 1984 内部效率，原文 eq 33-35）：")
        print(f"   剖面取自 {prof_src}")
        print(f"   Rs={profile['Rs'] * 1e2:.3f} cm，O 点 R={profile['R_O'] * 1e2:.3f} cm，"
              f"|J_zeta|峰值={profile['J_peak']:.3e} A/m2，"
              f"I1/I2={profile['I1'] / profile['I2']:.4e} m^3/A")
        print(f"   n_e={args.ne:.3e} m^-3、nu/nu0={args.nu_ratio:g} → "
              f"tau_e={theory['tau_e']:.3e} s、omega*={theory['omega']:.3e} rad/s "
              f"(f*={theory['omega'] / (2 * np.pi):.3e} Hz)")
        print(f"   <gamma>={theory['gamma']:.3f} → "
              f"eta_cd^th=<gamma>/(beta_ref^2 F(x_ref))={theory['eta_cd']:.3f}"
              f"（beta_ref=1、F(1)={theory['F']:g}）")
        ok = "满足" if theory["E_i"] <= 1.0 else "不满足 ⚠"
        print(f"   内部效率 E_i=1/<gamma>={theory['E_i']:.4f}（必须 ≤1：{ok}）；"
              f"隐含滑移 s=1-1/<gamma>={theory['slip']:.3f}")
        print(f"   自洽门槛：E_i≤1 要求 n_e ≥ {theory['ne_min']:.3e} m^-3")
        anchor_eta_cd = calibrate_eta_cd(eta, args.ne, S0, default_r_o(args.r_o))["eta_cd"]
        print(f"   同口径对照：锚定 η_cd（刚性转子上限）= {anchor_eta_cd:.3f}，"
              f"理论/锚定 = {eta_cd_used / anchor_eta_cd:.3f}")
        print(f"   维持边界改用 eta_cd^th={eta_cd_used:.3f}"
              f"（下表「锚定」列 = 同一 n_e 的 {anchor_eta_cd:.3f}）")

    xs = (np.array([args.omega_tau]) if args.omega_tau is not None
          else np.logspace(-2, 2, 9))
    beta_req = sustainment_table(S0, eta_cd_used, xs)
    beta_req_anchor = (None if anchor_eta_cd is None
                       else sustainment_table(S0, anchor_eta_cd, xs))
    print("\n[stage05] 维持边界（标准标度 eta_cd*beta_w^2*F=1）：")
    print("   omega*tau_e    F          beta_w=B_w/B_e    B_w[T](B_e=1T)   S_drive[Wb/s]"
          + ("    beta_w(锚定)" if beta_req_anchor is not None else ""))
    for i, (x, b) in enumerate(zip(xs, beta_req)):
        line = (f"   {x:9.3f}  {float(slip_factor(np.array([x]))[0]):8.4f}"
                f"       {b:8.3f}          {b:8.3f}         "
                f"{eta_cd_used * b ** 2 * float(slip_factor(np.array([x]))[0]) * S0:8.4f}")
        if beta_req_anchor is not None:
            line += f"        {beta_req_anchor[i]:8.3f}"
        print(line)

    i_best = int(np.argmin(beta_req))

    if args.calibrate:
        r_O = default_r_o(args.r_o)
        ne_list = ([args.ne] if args.ne else
                   [1e20, 3e20, 1e21, 3e21, 1e22])
        print(f"\n[stage05] 用刚性转子上限标定 eta_cd（参考条件 B_w/B_e=1、"
              f"omega*tau_e=1；r_O={r_O * 1e2:.3f} cm）")
        have_prof = profile is not None
        print("   n_e [m^-3]   tau_e [s]    omega* [rad/s]   f* [Hz]  "
              "J^RR [A/m2]   S^RR [Wb/s]   eta_cd(锚定)   B_w/B_e(最优)"
              + ("   <gamma>   eta_cd(理论)   E_i" if have_prof else ""))
        rows = []
        for ne in ne_list:
            c = calibrate_eta_cd(eta, ne, S0, r_O)
            rows.append((ne, c))
            line = (f"   {ne:9.2e}  {c['tau_e']:9.2e}  {c['omega']:11.3e}  "
                    f"{c['omega'] / (2 * np.pi):9.3e}  {c['J_rr']:10.3e}  "
                    f"{c['S_rr']:10.3e}  {c['eta_cd']:12.3f}   "
                    f"{c['beta_req_at_ref']:10.3f}")
            if have_prof:
                th = theory_eta_cd(ne, eta, profile, nu_ratio=args.nu_ratio)
                line += (f"   {th['gamma']:8.2f}   {th['eta_cd']:11.2f}   "
                         f"{th['E_i']:6.3f}")
            print(line)
        if have_prof:
            print("   注：eta_cd(理论) 取自 Hugrass 1984 的内部效率（同一份平衡剖面）；"
                  "E_i=1/<gamma> 必须 ≤1，E_i>1 的行说明该 n_e 与理论不自洽。")
        print("   注：J^RR 要与本平衡的电流密度量级比（本算例峰值 ~4.8e7 A/m2，"
              "be-loop 口径下 O 点处 ~1e7 量级）；S^RR 是它对应的通量驱动能力。")
        print("   标定后维持条件变成：B_w/B_e = 1/sqrt(eta_cd*F(omega*tau_e))，"
              "最优仍在 omega*tau_e=1。")
        ne_last, c_last = rows[-1]
        print(f"   → 用最后一行（n_e={ne_last:.2e}、eta_cd={c_last['eta_cd']:.3f}）"
              f"标定后的最优维持要求：B_w/B_e="
              f"{c_last['beta_req_at_ref']:.3f}"
              f"（f*={c_last['omega'] / (2 * np.pi):.3e} Hz），"
              f"驱动率 = eta_cd*beta^2*F*S0 = "
              f"{c_last['eta_cd'] * c_last['beta_req_at_ref'] ** 2 * c_last['F'] * S0:.3f}"
              f" Wb/s = 耗散率 ✓")

    print(f"\n[stage05] 最省 RMF 的滑移点：omega*tau_e={xs[i_best]:g}"
          f"（F 最大），此时需要 B_w/B_e={beta_req[i_best]:.3f}")
    if args.ne:
        tau_e = electron_collision_time(args.ne, eta)
        print(f"           对应频率 f = {xs[i_best] / (2 * np.pi * tau_e):.3e} Hz")
    print(f"           即：驱动率必须有 {S0:.3f} Wb/s（= 耗散率），"
          f"对应 B_w ≈ {beta_req[i_best]:.3f} T（B_e=1 T）")

    if args.crosscheck:
        if not args.ne:
            print("[stage05] 交叉验证需要 --ne（刚性转子电流要用密度）")
        else:
            e = 1.602176634e-19
            s = gcm._setup(gcm.SRC_DIR)
            Rw, zw = float(s["Rw"]), float(s["zw"])
            # 刚性转子电流 I = ∫ n e ω r dRdZ ≈ n e ω * (Rs^2/2)*(2Zs)
            Rs, Zs = 0.0569, 0.13          # be-loop 初态的 Rs（第 24 条）
            I_rr = args.ne * e * 1.0 * (Rs ** 2 / 2) * (2 * Zs)
            print(f"\n[stage05] 刚性转子交叉验证：J_phi≈-n e omega r → 单位 omega 的电流"
                  f" I/omega ≈ {I_rr:.3e} A*s")
            print(f"           本算例需要的驱动率 {S0:.3f} Wb/s；若把"
                  f" Phi ≈ 2pi|psi_min| 视作 L*I（L≈mu0*Rs），"
                  f"则所需 omega ≈ {S0 / (MU0 * Rs * I_rr):.3e} rad/s"
                  f"（该估计只给量级）")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    extra = {}
    if theory is not None:
        extra = dict(
            eta_cd_theory=np.array([theory["eta_cd"]]),
            gamma_theory=np.array([theory["gamma"]]),
            E_i_theory=np.array([theory["E_i"]]),
            ne_min=np.array([theory["ne_min"]]),
            beta_req_anchor=(beta_req_anchor if beta_req_anchor is not None
                             else np.array([])),
        )
    np.savez_compressed(OUT_DIR / "sustainment.npz",
                        omega_tau=xs, beta_req=beta_req, F=slip_factor(xs),
                        S0=np.array([S0]), Phi0=np.array([Phi0]),
                        eta=np.array([eta]), eta_cd=np.array([eta_cd_used]),
                        **extra)
    print(f"\n[stage05] 产物：{OUT_DIR / 'sustainment.npz'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
        xx = np.logspace(-2, 2, 400)
        ax[0].loglog(xx, slip_factor(xx), "k-")
        ax[0].axvline(1.0, color="gray", ls=":")
        ax[0].set_xlabel(r"$\omega\tau_e$")
        ax[0].set_ylabel(r"$F=\omega\tau_e/(1+(\omega\tau_e)^2)$")
        ax[0].set_title("滑移因子（ωτe=1 时最优）")
        for e_cd in (0.1, 0.3, 1.0):
            ax[1].loglog(xx, 1.0 / np.sqrt(np.maximum(e_cd * slip_factor(xx), 1e-300)),
                         label=rf"$\eta_{{cd}}={e_cd}$")
        if theory is not None:
            ax[1].loglog(xx, 1.0 / np.sqrt(np.maximum(eta_cd_used * slip_factor(xx), 1e-300)),
                         "k-", lw=2.2,
                         label=f"理论值 {eta_cd_used:.1f}")
        ax[1].axhline(1.0, color="gray", ls=":")
        ax[1].set_xlabel(r"$\omega\tau_e$")
        ax[1].set_ylabel(r"$B_\omega/B_e$")
        ax[1].set_title("维持所需的 RMF 幅度")
        ax[1].legend(fontsize=8)
        drv = np.array([0.0, 0.5, 1.0, 1.5]) * S0
        ax[2].plot(drv, drv - S0, "o-")
        ax[2].axhline(0, color="gray", ls=":")
        ax[2].set_xlabel(r"$S_{\rm drive}$ [Wb/s]")
        ax[2].set_ylabel(r"$\dot\Phi = S_{\rm drive}-S_0$  [Wb/s]")
        ax[2].set_title("净通量变化率（过零点 = 维持）")
        for a in ax:
            a.grid(alpha=0.3, which="both")
        fig.suptitle("RMF 维持边界（标准标度）")
        fig.savefig(OUT_DIR / "sustainment.png", dpi=110)
        print(f"[stage05] 产物：{OUT_DIR / 'sustainment.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[stage05] 画图跳过：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
