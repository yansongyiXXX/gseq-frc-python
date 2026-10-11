r"""Stage 0 + Stage 1：FRC 演化的**诊断量**与**剖面/幅值时变接口**（**修改版** 第 19 条）。

这个脚本只做两件事，都不引入新物理：

* **Stage 0**：逐步记录平衡的物理诊断量——私通量 \(\Phi=\psi_{sep}-\psi_O\)、
  中平面分离面半径 \(R_s\)、\(x_s=R_s/R_w\)、轴上反场深度 \(B_{z0}\)、极向伸长比、
  分界面以内的体积平均比压 \(\langle\beta\rangle\)、储能、环向电流……
  （口径与 1D 参考解的对照见下面「验收」一节。）
* **Stage 1**：让"决定平衡的两个量"随时间可给——除了线圈电流 \(I_l(t)\)，
  还能给剖面/幅值参数 \(I_{\zeta0}(t)\)、\(\beta_s(t)\)、\(B_e(t)\)…

所以这里跑出来的**不是**真正的演化（还没有通量扩散/回路方程/驱动模型），而是
"平衡族被外力推着走"的准静态序列：它能回答 Stage 3 之前必须知道的两件事——
**这套求解器在真实演化步长下稳不稳**、以及**平衡族里各量之间是什么关系**
（例如 \(\mathrm{d}\ln\Phi/\mathrm{d}\ln I_{\zeta0}\)），后者正是写通量衰减 ODE
时那个投影系数的实测值。

用法（在本仓库根目录）：

    python tools/frc_evolve_stage01.py                 # 默认：Izeta0 指数衰减
    python tools/frc_evolve_stage01.py --scenario ramp # 线圈电流上下不对称爬升
    python tools/frc_evolve_stage01.py --scenario both --steps 12 --dt 5e-5 --tau 1e-3
    python tools/frc_evolve_stage01.py --cold          # 不读保存的解，从冷启动起

产物：`outputs/frc_stage01/history.npz`（逐步诊断量 + ψ）与 `stage01.png`。

验收（2026-09-23，本机；详见 `修改记录.md` 第 19 条）
------------------------------------------------------
**① 诊断量口径**：把 `outputs/p2-Fig6/psiq.mat` 那份平衡的诊断量与同目录的 1D
参考解（`MRR1d.mat`，无限长圆柱、同一套 MRR 剖面）对照——私通量 \(\Phi\) 相差
−0.4%（\(1.890\) 对 \(1.897\times10^{-3}\) Wb）、\(R_s\) 相差 −0.5%
（0.07459 对 0.075 m）、轴上 \(B_{z0}\) 相差 −0.1%（−0.6320 对 −0.6325 T）、
分界面内 \(\langle\beta\rangle\) 相差 −4.8%（0.875 对 0.920，差在 2D 两端收缩）。
口径由此确认。

**② 接口稳不稳**：默认场景（\(I_{\zeta0}\propto e^{-t/1\,\mathrm{ms}}\)、
\(\mathrm{d}t=25\,\mu s\)、20 步）实测 0.49 s/步、总 9.9 s，每步 2–3 轮 CI、
残差 \(10^{-6}\) 一档，约束嵌入自洽到 \(2.6\times10^{-15}\)。

**③ 平衡族的关系（Stage 3 要用的实测系数）**：\(\Phi\propto I_{\zeta0}^{2.05}\)
（全量程；±10% 内近乎线性，斜率 1.01）、\(R_s\propto I_{\zeta0}^{1.1}\)，
而 \(\Phi/(B_eR_s^2)\) 全程只在 0.34→0.38 之间——**现有闭合里通量没有独立自由度**，
它被 CI 的压强平衡约束钉在 \(B_eR_s^2\) 上。要让通量自己会衰减/增长（Stage 3），
必须先松开这个约束。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

try:  # 控制台编码兜底：GBK 终端遇到 ζ/Φ/− 之类的字符不至于把脚本打断
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

from gseqfrc import mpath  # noqa: E402
from gseqfrc import mstyle  # noqa: E402

mstyle.install()   # 出图统一按 MATLAB 那一套（见 gseqfrc/mstyle.py）

OUT_DIR = ROOT / "outputs" / "frc_stage01"
MU0 = 4 * np.pi * 1e-7


def _initial_from_saved(gcm, model, mat_path: Path):
    """从保存的 Fig6 平衡造一个暖启动状态（与 `tools/evolve_demo.py` 同一套做法）。"""
    m = loadmat(mat_path, squeeze_me=True)
    psi0 = np.asarray(m["psiq"], dtype=float)
    CI0 = float(m["CIq"])
    psib_coil0 = model.coil_flux(model.s["Il"])[3]
    pr = model.problem(psib_coil0, 2)
    psif = psi0.ravel(order="F")
    C0 = pr.P["Izeta0"] / pr.jzmid_per_unit_C(psif)
    Jz = (pr.R.ravel(order="F") * pr.dp_per_unit_C(psif) * C0)[pr.int_idx]
    psib0 = CI0 * psib_coil0 + gcm.MU0 * CI0 * pr.dR * pr.dZ * (pr.Gsub.T @ Jz)
    return dict(psiq=psi0, psib=psib0, CI=CI0, C=C0, secant_state=None,
                dpsi_step=None)


def _spitzer_resistivity(te_ev: float, ln_lambda: float, z: float = 1.0) -> float:
    r"""Spitzer 电阻率（\(\Omega\cdot\mathrm{m}\)，NRL 等离子体手册那套系数）。"""
    return 5.2e-5 * z * ln_lambda / max(te_ev, 1e-6) ** 1.5


def _print_table(hist) -> None:
    head = ("step  t[ms]   Iz0[kA]  Izmid[kA] Iztot[kA]  C         CI        "
            "Phi[Wb]     Rs[cm]  Bz0[T]   beta_av W_sep[J]  res      nk[s]")
    print(head)
    print("-" * len(head))
    for h in hist:
        d = h["diag"] or {}
        print(f"{h['i']:4d}  {h['t']*1e3:6.3f}  "
              f"{d.get('Izeta0', float('nan')) / 1e3:9.2f}  "
              f"{d.get('Izeta_mid', float('nan')) / 1e3:9.2f}  "
              f"{d.get('Izeta_total', float('nan')) / 1e3:9.2f}  "
              f"{h['C']:9.6f} {h['CI']:9.6f} "
              f"{d.get('Phi_private_Wb', float('nan')):10.4e} "
              f"{d.get('Rs_mid', float('nan')) * 1e2:7.4f} "
              f"{d.get('Bz_axis_mid', float('nan')):+8.4f} "
              f"{d.get('beta_avg_sep', float('nan')):6.3f} "
              f"{d.get('W_sep', float('nan')):7.1f} "
              f"{h['gs_res']:8.1e} {h['wall']:6.2f}")


def _measure(hist) -> dict:
    """从 history 里量几件演化要用的事：通量衰减律、电流-通量敏感度、衰减时间。"""
    t = np.array([h["t"] for h in hist], dtype=float)
    Iz = np.array([(h["diag"] or {}).get("Izeta0", np.nan) for h in hist])
    Phi = np.array([(h["diag"] or {}).get("Phi_private_Wb", np.nan) for h in hist])
    Rs = np.array([(h["diag"] or {}).get("Rs_mid", np.nan) for h in hist])
    out: dict[str, float] = {}
    ok = np.isfinite(Phi) & (Phi > 0) & np.isfinite(Iz) & (Iz != 0)
    # 自洽性检查：约束嵌入保证 \(I_{\zeta,\rm mid}=I_{\zeta0}\)（应当到机器精度）
    Izm = np.array([(h["diag"] or {}).get("Izeta_mid", np.nan) for h in hist])
    if np.all(np.isfinite(Izm)) and np.any(Iz):
        out["mid_constraint_err"] = float(np.max(np.abs(Izm - Iz) / np.abs(Iz)))
    if ok.sum() >= 3:
        # dlnΦ/dlnIζ0：平衡族里"通量随电流幅值怎么变"的实测斜率。
        # 通量衰减 ODE 的右端项就差这个系数（Φ 的指数衰减 ↔ Iζ0 的指数衰减）。
        out["dlnPhi_dlnIz"] = float(
            np.polyfit(np.log(np.abs(Iz[ok])), np.log(Phi[ok]), 1)[0])
        # 一阶指数拟合 Φ(t) = Φ0 e^{-t/τΦ}（只在 Φ 单调时才有意义）
        if np.all(np.diff(Phi[ok]) < 0):
            out["tau_Phi_s"] = float(-1.0 / np.polyfit(t[ok], np.log(Phi[ok]), 1)[0])
        out["Phi_ratio"] = float(Phi[ok][-1] / Phi[ok][0])
        out["Iz_ratio"] = float(np.abs(Iz[ok][-1] / Iz[ok][0]))
        out["Rs_ratio"] = float(Rs[ok][-1] / Rs[ok][0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", choices=("decay", "ramp", "both"), default="decay",
                    help="decay=只让 Izeta0(t) 指数衰减；ramp=只动线圈电流；both=两者一起")
    ap.add_argument("--steps", type=int, default=8, help="演化步数")
    ap.add_argument("--dt", type=float, default=5e-5, help="时间步长 [s]")
    ap.add_argument("--tau", type=float, default=1e-3,
                    help="decay 场景的通量衰减时间 [s]（Izeta0 ∝ exp(-t/tau)）")
    ap.add_argument("--slope", type=float, default=5e-3,
                    help="ramp 场景每步的线圈电流相对斜率（上/中/下 = +1/+0.4/−1 倍）")
    ap.add_argument("--step-tol", type=float, default=5e-3,
                    help="freegsnke 式「每步相对」判据（见修改记录第 17 条）")
    ap.add_argument("--te", type=float, default=50.0, help="电子温度 [eV]（算 Spitzer 电阻参考值）")
    ap.add_argument("--ln-lambda", type=float, default=10.0, help="库仑对数")
    ap.add_argument("--cold", action="store_true", help="不读保存的 Fig6 解，冷启动")
    args = ap.parse_args()

    mpath.addpath("second_paper/Fig6")
    gcm = mpath.load("GSEQ_coil_MRR")
    s = gcm._setup(gcm.SRC_DIR)
    model = gcm.MachineModel(s)
    print(f"[stage01] 机器模型预计算 {model.build_s:.3f} s；"
          f"网格 {s['nR']}×{s['nZ']}；Izeta0={s['Izeta0'] / 1e3:.1f} kA")

    initial = None
    mat_path = ROOT / "outputs" / "p2-Fig6" / "psiq.mat"
    if not args.cold and mat_path.exists():
        initial = _initial_from_saved(gcm, model, mat_path)
        print(f"[stage01] 初值：{mat_path.name}（C={initial['C']:.6f}，"
              f"CI={initial['CI']:.6f}）")
    else:
        print("[stage01] 初值：冷启动")

    nt = int(args.steps)
    t = np.arange(nt) * float(args.dt)
    base = np.asarray(s["Il"], dtype=float)
    nl = base.size

    Il_traj = np.tile(base, (nt, 1))
    params_traj = None
    if args.scenario in ("ramp", "both"):
        # 特意上下不对称：上线圈升、下线圈降，破掉上下对称性
        pattern = np.array([+1.0, +0.4, -1.0])[:nl]
        Il_traj = base[None, :] * (1.0 + args.slope * pattern[None, :]
                                   * np.arange(nt)[:, None])
    if args.scenario in ("decay", "both"):
        Iz0 = float(s["Izeta0"])
        params_traj = {"Izeta0": Iz0 * np.exp(-t / float(args.tau))}
        print(f"[stage01] Izeta0(t)：{Iz0 / 1e3:.2f} → "
              f"{Iz0 * np.exp(-t[-1] / args.tau) / 1e3:.2f} kA"
              f"（τ={args.tau * 1e3:.2f} ms，共 {nt} 步）")

    t0 = time.perf_counter()
    hist, model = gcm.evolve(Il_traj, times=t, model=model, initial=initial,
                             step_tol=args.step_tol, params_traj=params_traj,
                             progress=True)
    wall = time.perf_counter() - t0

    print()
    _print_table(hist)

    meas = _measure(hist)
    print()
    print(f"[stage01] 总墙钟 {wall:.1f} s（{wall / nt:.2f} s/步）")
    if "mid_constraint_err" in meas:
        print(f"[stage01] 自洽性：max|Iζ,mid - Iζ0|/|Iζ0| = "
              f"{meas['mid_constraint_err']:.2e}（约束嵌入，应是机器精度）")
    if "dlnPhi_dlnIz" in meas:
        print(f"[stage01] 平衡族实测：dlnΦ/dlnIζ0 = {meas['dlnPhi_dlnIz']:.3f}"
              f"（Φ 末/初 = {meas['Phi_ratio']:.4f}，Iζ0 末/初 = {meas['Iz_ratio']:.4f}，"
              f"Rs 末/初 = {meas['Rs_ratio']:.4f}）")
        ph = [(h["diag"] or {}).get("Phi_over_BeRs2", np.nan) for h in hist]
        print(f"[stage01] 结构性结果：Φ/(Be·Rs^2) 从 {ph[0]:.4f} 变到 {ph[-1]:.4f}"
              f"（剖面形状被 CI 约束钉在 Be·Rs^2 上 → Φ 基本被 Rs 决定）")
    if "tau_Phi_s" in meas:
        rs = np.nanmean([(h["diag"] or {}).get("Rs_mid", np.nan) for h in hist])
        eta = _spitzer_resistivity(args.te, args.ln_lambda)
        tau_eta = MU0 * rs ** 2 / eta
        print(f"[stage01] 实测磁通衰减时间 τΦ = {meas['tau_Phi_s'] * 1e3:.3f} ms；"
              f"参考 mu0*Rs^2/eta (T_e={args.te:g} eV) = {tau_eta * 1e3:.3f} ms"
              f"（eta={eta:.2e} Ohm*m，Spitzer 量级估算）")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "history.npz",
        times=np.asarray([h["t"] for h in hist]),
        Il=np.asarray([h["Il"] for h in hist]),
        C=np.asarray([h["C"] for h in hist]),
        CI=np.asarray([h["CI"] for h in hist]),
        gs_res=np.asarray([h["gs_res"] for h in hist]),
        wall=np.asarray([h["wall"] for h in hist]),
        dpsi_step=np.asarray([h.get("dpsi_step") or np.nan for h in hist]),
        psiq=np.asarray([h["psiq"] for h in hist if h["psiq"] is not None]),
        diag_keys=np.array(sorted((hist[-1]["diag"] or {}).keys())),
        diag_vals=np.asarray([[(h["diag"] or {}).get(k, np.nan)
                               for k in sorted((hist[-1]["diag"] or {}).keys())]
                              for h in hist], dtype=float),
    )
    print(f"[stage01] 产物：{OUT_DIR / 'history.npz'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        keys = sorted((hist[-1]["diag"] or {}).keys())
        vals = np.asarray([[(h["diag"] or {}).get(k, np.nan) for k in keys]
                           for h in hist], dtype=float)
        get = lambda name: vals[:, keys.index(name)] if name in keys else None
        fig, ax = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        ax[0, 0].plot(t * 1e3, get("Phi_private_Wb"), "o-")
        ax[0, 0].set_ylabel(r"$\Phi_{private}$ [Wb]", color="tab:blue")
        axb = ax[0, 0].twinx()
        axb.plot(t * 1e3, -get("Izeta0") / 1e3, "s--", color="tab:red")
        axb.set_ylabel(r"$-I_{\zeta0}$ [kA]", color="tab:red")
        ax[0, 1].plot(t * 1e3, get("Rs_mid") * 1e2, "o-", label=r"$R_s$ [cm]")
        ax[0, 1].plot(t * 1e3, get("Zs_half") * 1e2, "s-", label=r"$Z_s$ [cm]")
        ax[0, 1].legend()
        ax[0, 2].plot(t * 1e3, get("Phi_over_BeRs2"), "o-")
        ax[0, 2].set_ylabel(r"$\Phi/(B_e R_s^2)$")
        ax[1, 0].plot(t * 1e3, get("Bz_axis_mid"), "o-")
        ax[1, 0].set_ylabel(r"$B_{z0}$ [T]")
        ax[1, 1].plot(t * 1e3, get("beta_avg_sep"), "o-", label=r"$\langle\beta\rangle$")
        axb2 = ax[1, 1].twinx()
        axb2.plot(t * 1e3, get("W_sep"), "s--", color="tab:red")
        axb2.set_ylabel(r"$W_{sep}$ [J]", color="tab:red")
        ax[1, 2].plot(t * 1e3, -get("Izeta_mid") / 1e3, "o-", label=r"$I_{\zeta,mid}$")
        ax[1, 2].plot(t * 1e3, -get("Izeta_total") / 1e3, "s--",
                      label=r"$\int J_\zeta\,dRdZ$")
        ax[1, 2].legend()
        for a in ax.ravel():
            a.set_xlabel("t [ms]")
            a.grid(alpha=0.3)
        fig.suptitle(f"FRC Stage 0+1（{args.scenario}）：准静态平衡序列 + 诊断量")
        mstyle.style_figure(fig, fontsize=13.0, grid=mstyle.GRID_STYLE)
        fig.savefig(OUT_DIR / "stage01.png", dpi=110)
        print(f"[stage01] 产物：{OUT_DIR / 'stage01.png'}")
    except Exception as exc:  # noqa: BLE001 - 画图失败不影响数值产物
        print(f"[stage01] 画图跳过：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
