r"""Stage 3：把通量衰减 ODE 接回二维自由边界求解器（**修改版** 第 21 条）。

模型（一阶：耗散率显式取当前平衡）
----------------------------------
时间演化只有一个状态量：**私通量** \(\Phi=2\pi|\psi_{\min}|\)（分离面取 \(\psi=0\)）。
每一步

1. 用**当前平衡**算耗散率

   $$S_0=2\pi\frac{\eta}{\mu_0}\,\Delta^*\psi\Big|_{\text{O 点}}$$

   （O 点处 \(\nabla\psi=0\)，所以 \(d\psi_{\min}/dt\) 就等于该点的局部值；
   \(S_0\) 只依赖 \(B_e\) 与剖面形状、与 \(R_s\) 无关——这是 Stage 2 验过的自相似性。）
2. 推进通量：\(\Phi^{n+1}=\max(0,\ \Phi^n-(S_0-S_{\rm drive})\,dt)\)。族闭合下这是
   **线性衰减**，寿命 \(\Phi_0/S_0\)；\(S_{\rm drive}\) 是外来的通量驱动率
   （RMF 的占位：只按 Wb/s 加通量，不含任何 RMF 物理）。
3. 用割线法找 \(I_{\zeta0}\) 使二维平衡的 \(\Phi\) 等于目标值（首猜用族关系
   \(I_{\zeta0}\propto\sqrt{\Phi}\)），每次试探都是一次暖启动的自由边界求解。

为什么状态量选 \(\Phi\) 而不是 \(I_{\zeta0}\)
--------------------------------------------
阻性耗散作用在**通量**上（Faraday + Ohm），电流是响应量。Stage 1/2 已经量到二维
序列里 \(\mathrm{d}\ln\Phi/\mathrm{d}\ln I_{\zeta0}\) 不是常数（0.67→3.04），把 ODE
写在 \(I_{\zeta0}\) 上等于凭空塞一个幂律；写在 \(\Phi\) 上，电流由平衡自洽给出。

用法（在本仓库根目录）
----------------------

    python tools/frc_evolve_stage03.py --verify          # 自检
    python tools/frc_evolve_stage03.py                   # 纯衰减
    python tools/frc_evolve_stage03.py --drive-rate 2.8  # 接近临界的驱动
    python tools/frc_evolve_stage03.py --te 100 --steps 20 --dt 5e-5

产物：`outputs/frc_stage03/history.npz` 与 `stage03.png`。
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
os.chdir(ROOT)

try:  # 控制台编码兜底
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

from gseqfrc import mpath  # noqa: E402
from gseqfrc import mstyle  # noqa: E402

mstyle.install()   # 出图统一按 MATLAB 那一套（见 gseqfrc/mstyle.py）

MU0 = 4 * np.pi * 1e-7
SRC = ROOT / "GSEQ-FRC-open" / "second_paper" / "Fig6"
OUT_DIR = ROOT / "outputs" / "frc_stage03"

mpath.addpath("second_paper/Fig6")
gcm = mpath.load("GSEQ_coil_MRR")
nk = gcm._load_nk_inner()


def spitzer_resistivity(te_ev: float, ln_lambda: float = 10.0, z: float = 1.0) -> float:
    """Spitzer 电阻率 [Ohm*m]。"""
    return 5.2e-5 * z * ln_lambda / max(te_ev, 1e-6) ** 1.5


def flux_loss_rate(psi2, pr, eta: float, k: int = 2):
    r"""由平衡算 \(S_0=2\pi\frac{\eta}{\mu_0}\Delta^*\psi|_{\text{O 点}}\) [Wb/s]。

    \(A=-k^2\Delta^*\)（Stage 2 对过），所以 \(\Delta^*\psi=-(A\psi)/dZ^2\)。
    O 点取 \(\psi\) 最小的格点；`k` 决定在附近取 \((2k+1)^2\) 个格点的**中位数**
    （抗单点抖动），同时返回邻域相对散布作为不确定度（这个量对"最近格点 vs 真 O 点"
    很敏感，实测散布约 ±10%——比拟合一个 τ 的不确定性小一个量级，够用）。
    """
    psif = np.asarray(psi2, dtype=float).ravel(order="F")
    dstar = -(pr.A @ psif) / pr.dZ2
    i0 = int(np.argmin(psif))
    if k <= 1:
        return 2 * np.pi * (eta / MU0) * float(dstar[i0]), 0.0
    nR = pr.nR
    iR, iZ = i0 % nR, i0 // nR
    vals = []
    for a in range(-k, k + 1):
        for b in range(-k, k + 1):
            ia, ib = iR + a, iZ + b
            if 1 <= ia <= nR - 2 and 1 <= ib <= pr.nZ - 2:
                vals.append(float(dstar[ia + ib * nR]))
    vals = np.asarray(vals)
    mid = float(np.median(vals))
    spread = float((vals.max() - vals.min()) / abs(mid)) if mid != 0 else float("nan")
    return 2 * np.pi * (eta / MU0) * mid, spread


def one_solve(*, model, Il, izeta0, prev, step_tol, verbose=False):
    """解一次二维自由边界平衡（暖启动），返回 `out`。"""
    kw = dict(model=model, Il=Il, inner="hybrid",
              params={"Izeta0": float(izeta0)}, step_tol=step_tol, verbose=verbose)
    if prev is None:
        return gcm.solve(gcm.SRC_DIR, **kw)
    return gcm.solve(gcm.SRC_DIR, psi0=prev["psiq"], CI0=prev["CI"],
                     psib0=prev.get("psib"), step_ref=prev.get("dpsi_step"), **kw)


def match_flux(target_phi, guess_iz, model, Il, prev, step_tol,
               rtol=2e-3, max_eval=4, verbose=False):
    """割线法找 \(I_{\zeta0}\) 使 \(\Phi_{\rm 2D}=\Phi_{\rm target}\)。

    返回 `(out, izeta0, phi, n_eval, err)`，`err=|Φ-Φ_target|/Φ_target`。
    首猜由调用方按族关系 \(\sqrt{\Phi}\) 给，多数步 1-2 次试探就够。
    """
    x_hist: list = []
    f_hist: list = []
    out_hist: list = []
    x = float(guess_iz)
    warm = prev
    for n in range(1, max_eval + 1):
        try:
            out = one_solve(model=model, Il=Il, izeta0=x, prev=warm,
                            step_tol=step_tol)
        except Exception as exc:  # noqa: BLE001
            # 反场区塌掉之后平衡本身不存在了（求解器会抛"起手状态无解"）——
            # 这不是数值故障，是模型适用域的边界，交给调用方收尾。
            if verbose:
                print(f"      [secant] eval {n} 失败：{type(exc).__name__}: {exc}")
            break
        phi = float(out["diag"]["Phi_private_Wb"])
        f = phi - target_phi
        x_hist.append(x); f_hist.append(f); out_hist.append(out)
        warm = out
        err = abs(f) / max(target_phi, 1e-30)
        if verbose:
            print(f"      [secant] eval {n}: Iz0={x / 1e3:9.2f} kA  "
                  f"Phi={phi:.6e}  target={target_phi:.6e}  err={err:.2e}")
        if err <= rtol:
            return out, x, phi, n, err
        if n == 1:
            scale = np.sqrt(max(target_phi, 1e-30) / max(phi, 1e-30))
            x = x_hist[0] * scale
        else:
            df = f_hist[-1] - f_hist[-2]
            dx = x_hist[-1] - x_hist[-2]
            if df == 0 or not np.isfinite(df):
                break
            x = x_hist[-1] - f_hist[-1] * dx / df
    if not f_hist:
        return None, x, float("nan"), 0, float("inf")
    n_best = int(np.argmin([abs(f) for f in f_hist]))
    return (out_hist[n_best], x_hist[n_best], target_phi + f_hist[n_best],
            len(f_hist), abs(f_hist[n_best]) / max(target_phi, 1e-30))


def _record(step, t, S0, spread, phi_target, out, iz, phi, n_eval, err, wall):
    """一步的记录；同时就是下一步的暖启动状态（`one_solve` 要的那几个键）。"""
    d = out["diag"]
    return dict(
        step=step, t=t, S0=S0, S0_spread=spread, Phi_target=phi_target,
        Phi=phi, Phi_err=err, Izeta0=iz, Cs=float(out["C"]),
        CI=float(out["CI"]), Rs=d["Rs_mid"], Bz0=d["Bz_axis_mid"],
        beta=d["beta_avg_sep"], W=d["W_sep"], psi_wall=d["psi_wall_mid"],
        gs_res=float(out["gs_res"]), n_eval=n_eval, wall=wall,
        psiq=out["psiq"], psib=out["psib"], secant_state=out.get("secant_state"),
        dpsi_step=out.get("dpsi_step"))


def run_evolution(*, steps, dt, eta, drive, model, step_tol=5e-3, rtol=2e-3,
                  max_eval=4, il_slope=0.0, verbose=False, progress=True,
                  drive_frac=None):
    """主循环：Φ → 割线求 Iz0 → 解平衡 → 记录。返回 `history`。"""
    s = model.s
    Il_base = np.asarray(s["Il"], dtype=float)
    nl = Il_base.size
    pattern = np.array([+1.0, +0.4, -1.0])[:nl] if il_slope else np.zeros(nl)

    t0 = time.perf_counter()
    # 第 0 步用**紧容差冷启动**（与第 17 条 `evolve` 的做法一致）：冷启动时
    # `step_tol` 没有上一时刻的 Δψ 作参照，判据会被"首轮大步长"放飞——实测会直接
    # 把内层推进停滞区（ψ_min=nan）。紧容差解出来之后，后面每步才用 step_tol 暖启动。
    out = one_solve(model=model, Il=Il_base, izeta0=float(s["Izeta0"]),
                    prev=None, step_tol=None)
    pr = model.problem(model.coil_flux(Il_base)[3], 2)
    phi = float(out["diag"]["Phi_private_Wb"])
    S0, spread = flux_loss_rate(out["psiq"], pr, eta)
    hist = [_record(0, 0.0, S0, spread, phi, out, float(s["Izeta0"]), phi, 0, 0.0,
                    time.perf_counter() - t0)]
    phi0 = phi
    t = 0.0
    if drive_frac is not None:            # 按初始耗散率的比例给驱动
        drive = float(drive_frac) * S0
    if progress:
        print(f"[stage03] 初态：Φ={phi:.4e} Wb  Iz0={float(s['Izeta0']) / 1e3:.2f} kA"
              f"  Rs={hist[0]['Rs'] * 1e2:.3f} cm  S0={S0:.4f} Wb/s"
              f"（邻域散布 {spread * 1e2:.2f}%）；Φ/S0 = {phi / S0 * 1e3:.4f} ms"
              + (f"；驱动 {drive:.4f} Wb/s" if drive else ""))
    for k in range(1, int(steps) + 1):
        ts = time.perf_counter()
        Il = Il_base * (1.0 + il_slope * pattern * k)
        pr = model.problem(model.coil_flux(Il)[3], 2)
        S0, spread = flux_loss_rate(hist[-1]["psiq"], pr, eta)
        phi_now = float(hist[-1]["Phi"])
        phi_target = max(0.0, phi_now - (S0 - drive) * dt)
        if phi_target <= 0.0:
            print(f"[stage03] t={t + dt:.4e} s：私通量已掉到 0"
                  f"（Φ={phi_now:.3e}，S0={S0:.3f} Wb/s），演化结束")
            break
        guess = float(hist[-1]["Izeta0"]) * np.sqrt(phi_target / phi_now)
        out, iz, phi, n_eval, err = match_flux(
            phi_target, guess, model, Il, hist[-1], step_tol, rtol, max_eval,
            verbose=verbose)
        if out is None:
            print(f"[stage03] t={t + dt:.4e} s：目标 Φ={phi_target:.4e} Wb 处"
                  f"**平衡已不存在**（反场区消失，求解器报「起手状态无解」）——"
                  f"FRC 到这里解体，演化结束。此时 Φ/Φ0="
                  f"{phi_target / phi0:.4f}，上一态 Iz0="
                  f"{hist[-1]['Izeta0'] / 1e3:.2f} kA、Rs="
                  f"{hist[-1]['Rs'] * 1e2:.3f} cm、Bz0={hist[-1]['Bz0']:+.4f} T")
            break
        t += dt
        wall = time.perf_counter() - ts
        rec = _record(k, t, S0, spread, phi_target, out, iz, phi, n_eval, err, wall)
        hist.append(rec)
        if progress:
            print(f"  [{k:2d}/{steps}] t={t * 1e3:7.4f} ms  Φ={phi:.4e} Wb"
                  f"（目标 {phi_target:.4e}，err {err:.1e}） Iz0={iz / 1e3:8.2f} kA"
                  f"  Rs={rec['Rs'] * 1e2:.3f} cm  Bz0={rec['Bz0']:+.4f} T"
                  f"  S0={S0:.3f}  试探 {n_eval}  {wall:.2f}s")
        if phi < 1e-3 * phi0:
            print("[stage03] Φ 已衰减到初值的 1e-3 以下，提前结束")
            break
        if err > 10 * rtol:
            print(f"[stage03] Φ 跟踪误差 {err:.1e} 已远超容差 {rtol:.1e}"
                  f"（多解/弱支边缘），演化在 t={t * 1e3:.4f} ms 停下")
            break
    return hist


def _load_1d_reference():
    """借 Stage 2 的工具算 1D 参照量（同一份 1D 平衡、同一套离散）。"""
    import importlib.util

    path = ROOT / "tools" / "frc_flux_decay_1d.py"
    spec = importlib.util.spec_from_file_location("gseqfrc._stage2_1d", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify(model, eta, args) -> None:
    """三项自检：(1) S0 与 1D 参照对照；(2) 步长收敛；(3) 驱动=耗散时 Φ 站住。"""
    print("[verify] (1) 二维平衡的 S0 对 1D 参照（Stage 2 同一套离散）")
    s = model.s
    Il_base = np.asarray(s["Il"], dtype=float)
    out = one_solve(model=model, Il=Il_base, izeta0=float(s["Izeta0"]), prev=None,
                    step_tol=None)
    pr = model.problem(model.coil_flux(Il_base)[3], 2)
    S2, spread = flux_loss_rate(out["psiq"], pr, eta)
    mod = _load_1d_reference()
    eq = mod.load_1d_equilibrium()
    r, psi0, par = eq["r"], eq["psi"], eq["par"]
    L1 = mod.build_operator(r)
    kO = int(np.argmin(psi0))
    S1 = 2 * np.pi * (eta / mod.MU0) * float((L1 @ psi0)[kO])
    phi_2d = float(out["diag"]["Phi_private_Wb"])
    phi_1d = 2 * np.pi * abs(psi0.min())
    print(f"    2D: Φ={phi_2d:.4e} Wb  S0={S2:.4f} Wb/s（O 点邻域散布 "
          f"{spread * 1e2:.2f}%）→ 寿命 Φ/S0={phi_2d / S2 * 1e3:.4f} ms")
    print(f"    1D: Φ={phi_1d:.4e} Wb  S0={S1:.4f} Wb/s → 寿命 "
          f"{phi_1d / S1 * 1e3:.4f} ms")
    print(f"    相对差：Φ {abs(phi_2d / phi_1d - 1) * 1e2:.2f}%，"
          f"S0 {abs(S2 / S1 - 1) * 1e2:.2f}%")

    print("[verify] (2) 步长敏感性（同样总时长：1 步 vs 2 步；"
          "注意 S0 会随平衡演化，两次轨迹不完全相同）")
    for dt in (args.dt, args.dt / 2):
        h = run_evolution(steps=int(round(args.dt / dt)), dt=dt, eta=eta, drive=0.0,
                          model=model, step_tol=args.step_tol, rtol=args.rtol,
                          max_eval=args.max_eval, progress=False)
        print(f"    dt={dt:.1e}: 末态 Φ={h[-1]['Phi']:.6e} Wb"
              f"（Φ/Φ0={h[-1]['Phi'] / h[0]['Phi']:.6f}），"
              f"Iz0={h[-1]['Izeta0'] / 1e3:.2f} kA，Rs={h[-1]['Rs'] * 1e2:.4f} cm")
    print("    （时间积分是一阶的；这里两次的差别还叠了 S0 沿轨迹的增长，"
          "所以是个敏感性量、不是纯收敛阶）")

    print("[verify] (3) 驱动 = 当前耗散率时 Φ 应站住（ODE 账面）")
    h = run_evolution(steps=2, dt=args.dt, eta=eta, drive=float(S2), model=model,
                      step_tol=args.step_tol, rtol=args.rtol,
                      max_eval=args.max_eval, progress=False)
    print(f"    drive=S0：Φ {h[0]['Phi']:.6e} -> {h[-1]['Phi']:.6e} Wb"
          f"（相对变化 {(h[-1]['Phi'] / h[0]['Phi'] - 1):+.2e}），"
          f"末端跟踪误差 {h[-1]['Phi_err']:.1e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=9,
                    help="步数（这个算例在 Φ 掉到 ~0.13Φ0、Rs~3 cm 时反场区就撑不住了"
                         "——再往后平衡不存在，程序会自己停下并说明）")
    ap.add_argument("--dt", type=float, default=3e-5, help="时间步 [s]")
    ap.add_argument("--te", type=float, default=50.0, help="电子温度 [eV]")
    ap.add_argument("--ln-lambda", type=float, default=10.0)
    ap.add_argument("--eta", type=float, default=None, help="直接给电阻率 [Ohm*m]")
    ap.add_argument("--drive-rate", type=float, default=0.0,
                    help="外部通量驱动率 S_drive [Wb/s]（RMF 占位）")
    ap.add_argument("--drive-frac", type=float, default=None,
                    help="按初始 S0 的比例给驱动（1.0 = 恰好维持）")
    ap.add_argument("--il-slope", type=float, default=0.0,
                    help="线圈电流每步相对斜率（上/中/下 = +1/+0.4/-1 倍）")
    ap.add_argument("--rtol", type=float, default=2e-3, help="割线法的 Φ 相对容差")
    ap.add_argument("--max-eval", type=int, default=4, help="每步最多几次试探求解")
    ap.add_argument("--step-tol", type=float, default=5e-3,
                    help="freegsnke 式每步相对判据（第 17 条）")
    ap.add_argument("--verify", action="store_true", help="只跑自检")
    args = ap.parse_args()

    eta = (float(args.eta) if args.eta is not None
           else spitzer_resistivity(args.te, args.ln_lambda))
    model = gcm.MachineModel(gcm._setup(gcm.SRC_DIR))
    print(f"[stage03] 机器模型 {model.build_s:.3f} s；eta={eta:.3e} Ohm*m"
          f"（T_e={args.te:g} eV）；网格 {model.s['nR']}x{model.s['nZ']}")
    if args.verify:
        verify(model, eta, args)
        return 0

    drive = float(args.drive_rate)
    hist = run_evolution(steps=args.steps, dt=args.dt, eta=eta, drive=drive,
                         model=model, step_tol=args.step_tol, rtol=args.rtol,
                         max_eval=args.max_eval, il_slope=args.il_slope,
                         verbose=args.max_eval > 4, drive_frac=args.drive_frac)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = [k for k in hist[-1] if k not in ("psiq", "psib", "secant_state")]
    np.savez_compressed(
        OUT_DIR / "history.npz",
        times=np.array([h["t"] for h in hist]),
        Phi=np.array([h["Phi"] for h in hist]),
        Phi_target=np.array([h["Phi_target"] for h in hist]),
        S0=np.array([h["S0"] for h in hist]),
        Izeta0=np.array([h["Izeta0"] for h in hist]),
        Rs=np.array([h["Rs"] for h in hist]),
        Bz0=np.array([h["Bz0"] for h in hist]),
        beta=np.array([h["beta"] for h in hist]),
        W=np.array([h["W"] for h in hist]),
        psi_wall=np.array([h["psi_wall"] for h in hist]),
        CI=np.array([h["CI"] for h in hist]),
        Cs=np.array([h["Cs"] for h in hist]),
        gs_res=np.array([h["gs_res"] for h in hist]),
        n_eval=np.array([h["n_eval"] for h in hist]),
        wall=np.array([h["wall"] for h in hist]),
        Phi_err=np.array([h["Phi_err"] for h in hist]),
        eta=np.array([eta]), drive=np.array([drive]), dt=np.array([args.dt]),
        psiq=np.array([h["psiq"] for h in hist]))
    print(f"\n[stage03] 产物：{OUT_DIR / 'history.npz'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        t = np.array([h["t"] for h in hist]) * 1e3
        ax[0, 0].plot(t, [h["Phi"] for h in hist], "o-", label=r"$\Phi$ 实际")
        ax[0, 0].plot(t, [h["Phi_target"] for h in hist], "k--", label=r"$\Phi$ 目标")
        ax[0, 0].set_ylabel(r"$\Phi$ [Wb]")
        ax[0, 0].legend()
        ax[0, 1].plot(t, -np.array([h["Izeta0"] for h in hist]) / 1e3, "o-")
        ax[0, 1].set_ylabel(r"$-I_{\zeta0}$ [kA]")
        ax[0, 2].plot(t, np.array([h["Rs"] for h in hist]) * 1e2, "o-")
        ax[0, 2].set_ylabel(r"$R_s$ [cm]")
        ax[1, 0].plot(t, [h["Bz0"] for h in hist], "o-")
        ax[1, 0].set_ylabel(r"$B_{z0}$ [T]")
        ax[1, 1].plot(t, [h["beta"] for h in hist], "o-")
        ax[1, 1].set_ylabel(r"$\langle\beta\rangle$")
        ax1 = ax[1, 2]
        ax1.plot(t, [h["S0"] for h in hist], "o-")
        ax1.set_ylabel(r"$S_0$ [Wb/s]")
        axb = ax1.twinx()
        axb.semilogy(t, np.maximum([h["Phi_err"] for h in hist], 1e-16), "s--",
                     color="tab:red")
        axb.set_ylabel("flux tracking error", color="tab:red")
        for a in ax.ravel():
            a.set_xlabel("t [ms]")
            a.grid(alpha=0.3)
        fig.suptitle("Stage 3: 2D forward evolution (flux-ODE closure)")
        mstyle.style_figure(fig, fontsize=13.0, grid=mstyle.GRID_STYLE)
        fig.savefig(OUT_DIR / "stage03.png", dpi=110)
        print(f"[stage03] 产物：{OUT_DIR / 'stage03.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[stage03] 画图跳过：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
