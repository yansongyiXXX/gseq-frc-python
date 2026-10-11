r"""连续整段放电：**通量当状态 + 每步解非线性 GS**（幅值闭合的第三代）。

为什么要换掉幅值闭合
--------------------
`tools/frc_full_discharge_2d.py` 里电流幅值是从链通量反解出来的：
\(I=(\Psi-\kappa a)/b\)。只要电流分布的形状由反馈决定，这条反解就会正反馈
发散——`--closed` 把电流硬限制在闭合面内时，形状变小 → \(b\) 变小 →
幅值暴涨 → 反转冲到壁面（实测 \(|\psi_{\min}|=8.9\times10^{-3}\)，设计点
\(1.7\times10^{-5}\)）。**瓶颈在幅值闭合，不在几何。**

本脚本换掉它：

* **状态量是 FRC 的闭合私通量 \(\Phi\)**（Wb），不再是电流：

  $$\frac{d\Phi}{dt}=S_{\rm rec}-(1-\eta_{\rm cd}\beta_\omega^2F)\,S_0$$

  其中 \(S_0=2\pi\frac{\eta}{\mu_0}\Delta^*\psi\big|_O\) 由当前平衡直接算（第 21 条
  口径），\(S_{\rm rec}\) 是重联源（限在形成窗口内，反场过阈值才开，容量饱和）；

* 每步解一次**非线性 GS**：剖面用 MRR 的 \(P'(\psi)\) 形式、归一化尺度取当前解的
  \(S=|\psi_O|/c_0\)（不预设 \((B_e,R_s)\)），幅值 \(C\) 用割线**反解到
  \(\Phi_{\rm target}\)**。非线性用欠松弛 Picard 收；

* 电流幅值 \(I_\phi=\int J_\zeta\,dRdZ\) 是**读出来的**，不再经过 \(b\) 反解。

于是"通量"是自变量、"几何"是解出来的，正反馈的发散路径被切断。

用法：

    python tools/frc_full_discharge_gs.py                # 约 10–20 分钟
    python tools/frc_full_discharge_gs.py --gif

产物：`outputs/frc_formation/full_discharge_gs.{npz,png,gif}`。

跑到哪一步（2026-10-09）
-----------------------

* 修掉一个真错误：重联源的**容量**原本取了线圈通量摆幅（~1e-3 Wb），
  应该是 **FRC 自己能容纳的私通量** \(cB_eR_s^2=1.08\times10^{-4}\) Wb。
  改过来之后状态通量落在 6.5e-5（容量的 60%）。
* **但幅值反解撞到一堵墙**：在这个族里 \(\Phi(C)\) 是**阈值型**的——
  电流不到反转阈值时 \(\Phi\equiv0\)，一到就跳到 \(\gtrsim5\times10^{-4}\) Wb，
  中间没有过渡。所以割线/三分怎么找都落不到设计点的 \(1.08\times10^{-4}\)，
  最终退化成"最接近的一次"，结果是零电流的真空场。
* **根因**：把归一化尺度换成动态的 \(S=|\psi_O|/c_0\) 之后，剖面变成
  **自相似**的——形状固定、尺度跟着解走。这意味着 **FRC 的尺寸不再由剖面控制**，
  而是由线圈场 + 那个自相似形状共同决定，结果总是偏大。
  换句话说：MRR 原来的 \((B_e,R_s)\) 是**两个**独立参数（尺度 + 幅值），
  我把它压成了一个，于是失去了尺寸这个控制量。
* **下一步**：把 \(R_s\) 恢复成独立参数（剖面宽度），让幅值/\(B_e\) 去跟通量状态挂钩，
  即回到两参数族 \((\text{尺度},\text{幅值})\)——那正是设计文档 §8.10/§8.13
  反复用到的 \(\Phi=cB_eR_s^2\) 的二维对应物。

两参数族做出来之后（2026-10-09 追加）
------------------------------------

新增三个开关：`--psi-scale`（固定剖面 ψ 尺度 \(S\)）、`--rs-target`（外层再调
\(S\) 使 \(R_s\) 命中目标）、`--family-fromflux`（\(S=\Phi/2\pi c_0\) 由通量定）。
三条支路的实测：

| 支路 | 幅值反解 | FRC 形状 |
| --- | --- | --- |
| 自相似 \(S=|\psi_O|/c_0\) | **不收敛**（阈值型：0 或过大） | 形状对：伸长比 2.56（设计 2.82），但通量大 6 倍 → \(R_s=12.5\) cm |
| 固定 \(S=3.2\times10^{-4}\) | **收敛** ✓（\(I_\phi=-19.7\) kA） | 针状：\(R_s=1.64\) cm、\(l_s=37.5\) cm、伸长比 11.4 |
| \(S=\Phi/2\pi c_0\) | 不收敛 | 无反转 |

**结论要修正**：上一轮说"要恢复两参数族"，这个判断被实验部分推翻——
自相似那条支路的**形状其实已经对了**（伸长比接近设计点），它唯一的毛病是
**通量幅值**（6 倍）。两参数族真正的收益是**让幅值反解变成良态**（固定 \(S\) 之后
\(I_\phi\) 真的解出来了），代价是形状退化成针。

也就是说：在这个闭合里，**"幅值可解"和"形状正确"目前不落在同一个参数区域**。
要两者兼得，缺的仍然是力平衡那一环——让**压强**当状态、由它定幅值和形状，
而不是靠"给定剖面 + 反解幅值"。
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

import design_point as dp                                          # noqa: E402
from frc_formation_2d import build_shape, shape_from_psi, make_gif  # noqa: E402
from frc_full_discharge_2d import lcfs_level                        # noqa: E402

OUT = ROOT / "outputs" / "frc_formation"
MU0 = dp.MU0
C_FAMILY = 0.337                      # Φ = c·B_e·R_s²（Stage 1/2 实测）
C0 = C_FAMILY / (2 * np.pi)           # |ψ_O| = c0·B_e·R_s²

#: 重联源的**容量**是 FRC 自己能容纳的私通量 \(cB_eR_s^2\)（设计点 A 的
#: \(0.337\times0.05\times0.08^2=1.078\times10^{-4}\) Wb），
#: **不是**线圈的通量摆幅（~1e-3 Wb）——后者会把 FRC 撑到 10 倍大。
PHI_CAP_DESIGN = C_FAMILY * 0.05 * 0.08 ** 2


def mr_shape(psi, S, alpha, sigma, n):
    r"""无量纲 MRR 剖面 \(\mathrm dP/\mathrm d\psi\)（自变量 \(\psi/S\)）。"""
    x = np.clip(np.asarray(psi, float) / max(S, 1e-30), -60.0, 60.0)
    u = np.maximum(1.0 + sigma * x, 1e-12)
    return np.exp(-alpha * x) * u ** (n - 1.0) * (-alpha * u + n * sigma)


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--t-fire", type=float, default=20e-6)
    ap.add_argument("--t-end", type=float, default=0.62e-3)
    ap.add_argument("--n-fine", type=int, default=150)
    ap.add_argument("--n-coarse", type=int, default=200)
    ap.add_argument("--rp", type=float, default=0.08)
    ap.add_argument("--wr", type=float, default=0.02)
    ap.add_argument("--zpow", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=31.5)
    ap.add_argument("--sigma", type=float, default=8.31)
    ap.add_argument("--n-mrr", type=float, default=2.0)
    ap.add_argument("--gs-iters", type=int, default=8, help="Picard 迭代次数")
    ap.add_argument("--gs-w", type=float, default=0.5, help="Picard 欠松弛权重")
    ap.add_argument("--psi-scale", type=float, default=0.0,
                    help=r"剖面 psi 尺度 S=B_e R_s^2 [Wb/2pi]；>0 时**固定**它"
                         r"（两参数族的第二个参数），=0 时用自相似的 S=|psi_O|/c0")
    ap.add_argument("--rs-target", type=float, default=0.0,
                    help="若 >0：外层再调 S 使 R_s 命中该值 [m]（两参数同时用上）")
    ap.add_argument("--family-fromflux", action="store_true",
                    help=r"每步把 ψ 尺度取成 \(S=\Phi/(2\pi c_0)\)（由通量状态定，"
                         r"不用解的深度迭代）——两参数由族关系绑定")
    ap.add_argument("--amp-iters", type=int, default=4, help="幅值割线迭代次数")
    ap.add_argument("--rec", type=float, default=1.0, help="重联源系数")
    ap.add_argument("--tau-rec", type=float, default=5e-6)
    ap.add_argument("--f-thr", type=float, default=0.9)
    ap.add_argument("--rec-cap", type=float, default=0.0,
                    help="重联容量 [Wb]；<=0 时取 FRC 的通量容量 c·B_e·R_s²")
    ap.add_argument("--capture", type=float, default=1.0,
                    help="起步阶段（空间剖面）的捕获效率")
    ap.add_argument("--te", type=float, default=30.0)
    ap.add_argument("--drive", type=float, default=None,
                    help="RMF 驱动率；缺省取 discharge.json 里的值")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--name", default="full_discharge_gs")
    args = ap.parse_args()

    f0 = np.load(OUT / "formation0d.npz", allow_pickle=True)
    t0d = np.asarray(f0["t"], float)
    Ic0 = np.asarray(f0["Ic"], float)
    ls0d = np.asarray(f0["l_s"], float)
    dsc = json.loads((OUT / "discharge.json").read_text(encoding="utf-8"))
    trk = dsc["track"]
    t_d = np.array([r["t"] for r in trk], float)
    Il_d = np.array([r["Il"] for r in trk], float)
    rmf = dsc.get("rmf") or {}
    drive = float(rmf.get("drive_frac", 0.0) if args.drive is None else args.drive)

    print("[gs] 建机器模型 …")
    s, model, _i, _z = dp.build_design_model()
    nR, nZ = s["nR"], s["nZ"]
    Z0 = s["Z0"] - 1
    rr = np.asarray(s["rr"], float)
    zz = np.asarray(s["zz"], float)
    R2d = np.asarray(s["R"], float)
    Rflat = R2d.ravel(order="F")
    Rinv2d = np.where(R2d > 0, 1.0 / np.maximum(R2d, 1e-12), 0.0)
    dR, dZ = float(s["dR"]), float(s["dZ"])

    Il_pk = float(np.max(Ic0))
    _a, _b, _c, psib_pk = model.coil_flux(np.full(3, Il_pk))
    psib_pk = np.asarray(psib_pk, float).ravel()
    pr = model.problem(psib_pk, order=2)
    lu = pr.lu_full()
    print(f"[gs] 网格 {nR}x{nZ}，内部点 {pr.int_idx.size}")

    def solve(src_flat, psib_vec):
        pr.set_psib(psib_vec)
        rhs = np.array(pr.bnd_val)
        rhs[pr.int_idx] = pr.dZ2 * MU0 * pr.Rint ** 2 * src_flat[pr.int_idx]
        return lu.solve(rhs)

    def psib_of(src_flat):
        Jz_int = Rflat[pr.int_idx] * src_flat[pr.int_idx]
        return MU0 * pr.dR * pr.dZ * (pr.Gsub.T @ Jz_int)

    psi_coil_pk = solve(np.zeros(pr.N), psib_pk).reshape(nR, nZ, order="F")
    B_coil_pk = 2.0 * psi_coil_pk[1, Z0] / dR ** 2
    B_bias = 0.5 * abs(B_coil_pk)
    psi_bias = B_bias * R2d ** 2 / 2.0
    eta = 3.16e-6 * (30.0 / args.te) ** 1.5
    kap = float(args.capture)

    def loss_rate(psi2):
        r"""\(S_0=2\pi\frac{\eta}{\mu_0}\Delta^*\psi|_O\)（第 21 条口径）。"""
        psif = np.asarray(psi2, float).ravel(order="F")
        dstar = -(pr.A @ psif) / pr.dZ2
        i0 = int(np.argmin(psif))
        iR0, iZ0 = i0 % nR, i0 // nR
        vals = [float(dstar[(iR0 + a) + (iZ0 + b) * nR])
                for a in range(-2, 3) for b in range(-2, 3)
                if 1 <= iR0 + a <= nR - 2 and 1 <= iZ0 + b <= nZ - 2]
        return 2 * np.pi * (eta / MU0) * (float(np.median(vals)) if vals else 0.0)

    # ---------------- 时间网格与线圈时序 ----------------
    tf = args.t_fire
    tt = np.concatenate([np.linspace(0.0, tf, args.n_fine, endpoint=False),
                         np.linspace(tf, args.t_end, args.n_coarse)])
    nt = tt.size
    Il_hist = np.zeros((nt, 3))
    for n, t in enumerate(tt):
        Il_hist[n] = (np.interp(t, t0d, Ic0) if t < tf
                      else [np.interp(t - tf, t_d, Il_d[:, k]) for k in range(3)])
    # 三线圈单位电流的边界通量（线性叠加）
    pb_unit = np.zeros((3, psib_pk.size))
    for k in range(3):
        _p, _b2, _c2, pb = model.coil_flux(np.eye(3)[k])
        pb_unit[k] = np.asarray(pb, float).ravel()
    print(f"[gs] {nt} 步，t_end={tt[-1] * 1e3:.3f} ms，RMF drive={drive:.3f}")

    rec = {k: np.zeros(nt) for k in
           ("Idrive", "Bz0", "Fdrive", "Phi", "Rs", "Iphi", "zc", "ls",
            "S0", "C_amp", "gs_iters")}
    psi_frames = np.zeros((nt, nR, nZ))
    Phi = 0.0
    C_amp = 1.0e6
    psi_prev = None
    t0 = time.perf_counter()

    def gs_solve(C, psi0, S_fix=None):
        r"""非线性 GS：给定幅值 \(C\) 与 ψ 尺度，Picard + 欠松弛解出 \(\psi\)。

        `S_fix` 给了就用固定的 \(S\)（两参数族）；不给就用自相似的
        \(S=|\psi_O|/c_0\)——后者的毛病是剖面跟着解走，FRC 尺寸不再受控。
        """
        psi = psi0.copy()
        it_used = 0
        for it in range(args.gs_iters):
            it_used = it + 1
            po = float(psi.min())
            if S_fix is not None:
                S = S_fix
            else:
                S = abs(po) / C0 if po < -1e-12 else max(0.25 * B_bias, 1e-30)
            src = np.where(R2d > 0,
                           C * mr_shape(psi, S, args.alpha, args.sigma,
                                        args.n_mrr), 0.0)
            sf = src.ravel(order="F")
            new = solve(sf, psib_coil_n + psib_of(sf)).reshape(
                nR, nZ, order="F") + psi_bias
            psi = (1.0 - args.gs_w) * psi + args.gs_w * new
        return psi, it_used

    def solve_amp(Phi_t, C_start, psi0, S_fix=None):
        r"""反解幅值 \(C\)，使 \(2\pi|\psi_{\min}|\) 尽量接近 \(\Phi_t\)。

        注意 \(\Phi(C)\) 是**阈值型**的：电流不到反转阈值时 \(\Phi\equiv0\)，
        一到就跳到一个不小的值，中间没有连续过渡。所以割线/三分都可能踩空，
        这里改成"记录最接近的一次"，避免崩到零电流。
        """
        C = float(C_start)
        psi, it_used = gs_solve(C, psi0, S_fix)
        Ph = 2 * np.pi * max(0.0, -float(psi.min()))
        best = (abs(Ph - Phi_t), C, psi)
        # 先把量级调到同一档（通量随 C 单调，×3 / ÷3 很快到位）
        for _ in range(10):
            if Ph > 2.0 * Phi_t:
                C /= 3.0
            elif Ph < 0.5 * Phi_t:
                C *= 3.0
            else:
                break
            psi, it_used = gs_solve(C, psi, S_fix)
            Ph = 2 * np.pi * max(0.0, -float(psi.min()))
            if abs(Ph - Phi_t) < best[0]:
                best = (abs(Ph - Phi_t), C, psi)
        _, C, psi = best
        Ph = 2 * np.pi * max(0.0, -float(psi.min()))
        if Ph <= 1e-30:
            return psi, C, it_used
        C_pre, f_pre = None, None
        for _ in range(args.amp_iters):
            psi, it_used = gs_solve(C, psi, S_fix)
            Ph = 2 * np.pi * max(0.0, -float(psi.min()))
            err = Ph - Phi_t
            if abs(err) <= 2e-2 * Phi_t:
                break
            if C_pre is None:
                Cn = C * Phi_t / max(Ph, 1e-30)
            else:
                dC, df = C - C_pre, err - f_pre
                Cn = (C - err * dC / df) if df != 0 else C * Phi_t / max(Ph, 1e-30)
            C_pre, f_pre = C, err
            lo, hi = sorted((0.2 * C, 5.0 * C))
            C = float(np.clip(Cn, lo, hi))
        return psi, C, it_used

    a_prev = None
    S_cur = float(args.psi_scale) if args.psi_scale > 0 else None
    for n, t in enumerate(tt):
        Il = Il_hist[n]
        psib_coil_n = (pb_unit[0] * Il[0] + pb_unit[1] * Il[1]
                       + pb_unit[2] * Il[2])
        psi_coil_n = solve(np.zeros(pr.N), psib_coil_n).reshape(nR, nZ, order="F")
        psi_ext = psi_coil_n + psi_bias
        bcoil = B_coil_pk * (float(np.sum(Il)) / (3.0 * Il_pk))
        fdrv = abs(bcoil) / B_bias - 1.0 if B_bias else 0.0

        # 驱动通量摆幅（用空间剖面当权重，只为给重联源定容量）
        g_w = build_shape(s, args.rp, args.wr,
                          0.5 * float(np.interp(min(t, t0d[-1]), t0d, ls0d)),
                          args.zpow)
        a_now = 2 * np.pi * float(np.sum(psi_ext * g_w) * dR * dZ)
        if a_prev is None:
            a_prev = a_now
        it_used = 0

        if n == 0:
            # 起步：没有反场，先用空间电流薄片把反转做出来
            psi = psi_ext
            Phi = 0.0
        else:
            S0 = loss_rate(psi_prev)
            dt = tt[n] - tt[n - 1]
            # 重联源：只在形成窗口内、反场过阈值才开，容量饱和
            Srec = 0.0
            if args.rec > 0.0 and t <= tf and fdrv >= args.f_thr:
                phi_cap = args.rec_cap if args.rec_cap > 0 else PHI_CAP_DESIGN
                Srec = (args.rec * max(phi_cap - Phi, 0.0) / args.tau_rec)
            Phi = max(0.0, Phi + (Srec - (1.0 - (drive if t >= tf else 0.0))
                                  * S0) * dt)
            if Phi > 0.0:
                S_use = S_cur
                if args.family_fromflux:
                    S_use = Phi / (2 * np.pi * C0)   # S 由通量状态定
                psi, C_amp, it_used = solve_amp(Phi, C_amp, psi_prev, S_use)
                # 外层：用第二个参数 S 把 R_s 拉向目标（两参数族）
                if args.rs_target > 0 and S_cur is not None:
                    for _ in range(2):
                        neg0 = np.nonzero(psi[:, Z0] < 0.0)[0]
                        Rs_now = float(rr[neg0[-1]]) if neg0.size else float("nan")
                        if not np.isfinite(Rs_now) or Rs_now <= 0:
                            break
                        if abs(Rs_now - args.rs_target) <= 0.05 * args.rs_target:
                            break
                        S_cur *= (args.rs_target / Rs_now) ** 2
                        psi, C_amp, it_used = solve_amp(Phi, C_amp, psi, S_cur)
            else:
                psi = psi_ext

        psi_frames[n] = psi
        psi_prev = psi

        # 读电流：J_zeta = R·src
        po = float(psi.min())
        S = abs(po) / C0 if po < -1e-12 else 0.0
        if S > 0:
            src = np.where(R2d > 0, C_amp * mr_shape(psi, S, args.alpha,
                                                     args.sigma, args.n_mrr), 0.0)
            Jz = R2d * src
            Iphi = float(np.sum(Jz[1:-1, 1:-1]) * dR * dZ)
        else:
            Iphi = 0.0

        _lvl, comp = lcfs_level(psi)
        core = comp if comp is not None else (psi < 0.0)
        neg = np.nonzero(core[:, Z0])[0]
        Rs = float(rr[neg[-1]]) if neg.size else float("nan")
        cols = np.nonzero(core.any(axis=0))[0]
        if cols.size:
            zlo, zhi = float(zz[cols].min()), float(zz[cols].max())
            rec["zc"][n], rec["ls"][n] = 0.5 * (zlo + zhi), zhi - zlo
        else:
            rec["zc"][n] = rec["ls"][n] = np.nan
        rec["Idrive"][n] = np.sum(Il) / 3.0
        rec["Bz0"][n] = 2.0 * psi[1, Z0] / dR ** 2
        rec["Fdrive"][n] = fdrv
        rec["Phi"][n] = 2 * np.pi * max(0.0, -po)
        rec["Rs"][n] = Rs
        rec["Iphi"][n] = Iphi
        rec["S0"][n] = loss_rate(psi)
        rec["C_amp"][n] = C_amp
        rec["gs_iters"][n] = it_used
        if n % 25 == 0 or n == nt - 1:
            rs_txt = "-" if not np.isfinite(Rs) else f"{Rs * 1e2:.2f}"
            print(f"[gs] t={t * 1e6:8.2f} us  Il={np.mean(Il) / 1e3:6.2f} kA  "
                  f"Iphi={Iphi / 1e3:+8.2f} kA  Bz0={rec['Bz0'][n]:+.4f} T  "
                  f"Phi={rec['Phi'][n]:.3e} Wb  Rs={rs_txt} cm  "
                  f"ls={rec['ls'][n] * 1e2:.1f} cm  zc={rec['zc'][n] * 1e2:+.2f}"
                  f"  [状态 Phi={Phi:.3e} f_drv={fdrv:+.3f} C={C_amp:.3g}]")
    wall = time.perf_counter() - t0
    print(f"\n[gs] 完成：{nt} 步，用时 {wall / 60:.1f} min")
    print(f"[gs] 末态：Iphi={rec['Iphi'][-1] / 1e3:+.2f} kA  "
          f"Phi={rec['Phi'][-1]:.3e} Wb  Rs={rec['Rs'][-1] * 1e2:.2f} cm  "
          f"ls={rec['ls'][-1] * 1e2:.2f} cm  zc={rec['zc'][-1] * 1e2:+.2f} cm")
    print(f"[gs] 通量 Φ_end/Φ_peak = {rec['Phi'][-1] / max(rec['Phi'].max(), 1e-30):.3f}")

    np.savez_compressed(
        OUT / f"{args.name}.npz", t=tt, psi=psi_frames[:, ::2, ::2].astype(np.float32),
        rr=rr[::2], zz=zz[::2], **rec, Il=Il_hist,
        rl=np.asarray(s["rl"], float), zl=np.asarray(s["zl"], float),
        hl=np.asarray(s["hl"], float), wl=np.asarray(s["wl"], float),
        Rw=float(s["Rw"]), zw=float(s["zw"]), B_bias=B_bias,
        B_coil_pk=float(B_coil_pk), t_fire=tf)
    if args.gif:
        try:
            make_gif(OUT / f"{args.name}.gif", psi_frames, rr, zz, tt, rec, s,
                     B_bias, nn=min(70, nt), fps=args.fps)
            print(f"[gs] 动图：{OUT / (args.name + '.gif')}")
        except Exception as exc:  # noqa: BLE001
            print(f"[gs] 动图跳过：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
