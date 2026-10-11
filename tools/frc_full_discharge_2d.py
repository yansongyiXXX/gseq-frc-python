r"""**一条连续时间线**的二维磁通演化：θ 箍缩形成 → 平移 → 压缩 → 平顶。

为什么另起一个脚本
------------------
`tools/make_full_chain_animation.py` 那条"全链"是**拼接**的：形成段是
`frc_formation_2d`（给定剖面 + 通量守恒），交接是一次平衡求解（本来只是
*验证* 0D 末态合法，不是状态传递），放电段又是每步解平衡的另一套模型——
三者状态表示都不同，接缝处必然跳变。那不是一条连续演化。

本脚本把**同一套状态**一次跑完整个 0.6 ms：

* 状态量：等离子体环向电流 \(I_\phi\)、它链到的总磁通 \(\Psi_\phi\)、
  以及电流分布形状 \(g(R,Z)\)；
* 场：每步解一次**线性** G-S（\(\Delta^*\psi=-\mu_0R\,J_\zeta\)，剖面给定 →
  一次 LU），线圈 Green 函数 / G-S 张量 / 网格与平衡求解器完全同一套；
* 幅值：通量守恒 \(\Psi_\phi=\kappa a(t)+b(t)I_\phi\)、\(d\Psi_\phi/dt=-R_\phi I_\phi\)；
* **RMF**：把耗散按 \(\big(1-\eta_{\rm cd}\beta_\omega^2F\big)\) 打折
  （与主干 `\Phi^{n+1}=\Phi^n-(S_0-S_{\rm RMF})\Delta t` 同一口径）；
* **剖面**：闭合区出现后投影到磁面上（同 `frc_formation_2d` 的 hybrid）。

**唯一的外加输入是线圈电流时序 \( \mathbf I_l(t)\)**——它就是真实的控制量，
不需要任何"交接"：

* 形成段（\(t<t_{\rm fire}\)）：主币 RLC 波形，三线圈同值（取自
  `formation0d.npz` 的 \(I_c(t)\)）；
* 其余时间：直接插值 `discharge.json` 里**已经跑出来的**线圈时序
  （平移的差模、压缩的中心线圈、平顶保持）。

**这条线是连续的，但不要指望它和文档里的放电数字对得上**：形成段这套闭合
（给定剖面 + 通量守恒）给出的电流比设计点平衡大 ~2 倍，FRC 会偏大；RMF 驱动
也只用同一个 \(\eta_{\rm cd}\) 标度。它的价值是"**同一状态、同一模型、一次跑完**"，
而不是复现某一条曲线。

用法：

    python tools/frc_full_discharge_2d.py                 # 约 6–10 分钟
    python tools/frc_full_discharge_2d.py --t-end 0.62 --gif

已经跑到哪一步（2026-10-09）
---------------------------

* `--profile spatial`：FRC 形成后大约 70 µs 就熄灭（柱长跟 0D 收缩 → 自感涨
  2.4 倍 → 通量守恒下电流掉到 −30 kA）。
* `--profile fb`（力平衡剖面，MRR 的 \(P'(\psi)\)，尺度取 \(|\psi_O|/c_0\)）
  **＋ `--rec`（重联源，限在形成窗口内）**：FRC **活满 0.62 ms**
  （\(\Phi_{\rm end}/\Phi_{\rm peak}=1.00\)），磁面光滑。
* 重联源还有一个可测量的作用：**降低捕获效率门槛**——不加 `--rec` 时
  `--capture` 低于 ~0.75 就压根没有闭合区；加了之后 0.15 也能起来。
* **几何诊断要用闭合判据，不能看 \(\psi<0\)**：`lcfs_level()` 找"含 O 点、
  不碰壁也不碰端面"的最大连通块。之前用 \(\psi<0\) 量出 \(l_s\approx148\) cm
  是**测量 bug**——把端部分散的数值毛刺也算进去了。按连通块量，本组参数下
  反场区是**闭合**的：\(R_s\approx12.5\) cm、\(l_s\approx64\) cm、\(z_c\approx+2.6\) cm
  （设计点：8 cm / 45 cm / +3 cm）。
* **`--closed`（把电流硬限制在闭合面内）会失稳**：形状被压小 → 自感 \(b\) 变小 →
  通量守恒反解出的电流暴涨 → 反转冲到壁面。这是个**负面结果**，说明瓶颈在
  **幅值闭合**（\(I=(\Psi-\kappa a)/b\)）而不在几何——只要形状由反馈决定，
  这个反解就会正反馈发散。

产物：`outputs/frc_formation/full_discharge_2d.{npz,png,gif}`。
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

import design_point as dp                                    # noqa: E402
from frc_formation_2d import build_shape, shape_from_psi, make_gif   # noqa: E402

OUT = ROOT / "outputs" / "frc_formation"
MU0 = dp.MU0

#: 族自相似常数 \(c=\Phi/(B_eR_s^2)\)（Stage 1/2 实测 0.337），
#: 用来把"当前解的 \(| \psi_O|\)"换算成力平衡剖面需要的尺度 \(B_eR_s^2\)。
C_FAMILY = 0.337


#: 4-连通（不用 8-连通）：对角相连会让闭合区"贴着壁渗出去"，把闭合判据弄错
_ST4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


def lcfs_level(psi, nlev: int = 22):
    r"""找**最后闭合磁面**（LCFS）：含 O 点、且**不碰计算域边界**的最大连通块。

    判据是几何的，不是看 \(\psi\) 的符号：\(\psi<0\) 只说明"场反了"，
    不说明磁面闭合。开场线的磁面会一直延伸到端部平面——那种情况下
    \(\psi=0\) 的等值线不闭合，**不是**分离面。

    做法：在 \([\psi_{\min},0]\) 上二分。对每个水平 \(\ell\)，取
    \(\{\psi\le\ell\}\) 里含 O 点的 4-连通块；它只要碰到 \(R=R_w\) 的壁
    或 \(Z=\pm z_w\) 的端面就算"开放"（\(R=0\) 是几何对称轴，不算边界）。
    返回 \((\ell_{\rm LCFS}, \text{闭合区掩码})\)；完全没有闭合面时返回 `(None, None)`。
    """
    from scipy import ndimage

    psi = np.asarray(psi, float)
    pmin = float(psi.min())
    if not np.isfinite(pmin) or pmin >= 0.0:
        return None, None
    iO = np.unravel_index(int(np.argmin(psi)), psi.shape)
    bnd = np.zeros(psi.shape, bool)
    bnd[-1, :] = True          # r = Rw 壁
    bnd[:, 0] = True           # z = -zw 端面
    bnd[:, -1] = True          # z = +zw 端面

    def comp_at(lvl):
        lab, _n = ndimage.label(psi <= lvl, structure=_ST4)
        k = int(lab[iO])
        if k == 0:
            return None
        c = lab == k
        return None if (c & bnd).any() else c

    c0 = comp_at(pmin)
    if c0 is None:
        return None, None
    lo, hi = pmin, 0.0
    best = (pmin, c0)
    for _ in range(nlev):
        mid = 0.5 * (lo + hi)
        c = comp_at(mid)
        if c is None:
            hi = mid
        else:
            lo, best = mid, (mid, c)
    return best


def shape_fb(s, psi, alpha: float, sigma: float, n: float,
             c0: float = C_FAMILY / (2 * np.pi), u_floor: float = 1e-12,
             closed_only: bool = False):
    r"""**力平衡剖面**：\(J_\zeta\propto R\,P'(\psi)\)，函数形式取 MRR，
    但归一化尺度用**当前解**而不是预设的 \((B_e,R_s)\)。

    MRR 的 \(\mathrm dP/\mathrm d\psi\) 写成无量纲形式（\(\hat\psi=\psi/(B_eR_s^2)\)）：

    $$F(\hat\psi)=e^{-\alpha\hat\psi}\big(1+\sigma\hat\psi\big)^{n-1}
    \big[-\alpha(1+\sigma\hat\psi)+n\sigma\big]$$

    **关键**：把 \(B_eR_s^2\) 换成 \(S=|\psi_O|/c_0\)（\(c_0=c/2\pi\)，族自相似常数）。
    这样剖面是**自相似**的——形状由 \((\alpha,\sigma,n)\) 定，尺度由解本身定，
    于是 \(R_s,l_s\) 是**解出来的**，不是外面塞进去的。这就是"让电流剖面跟着
    力平衡走"。闭合区还没出现时返回 `None`（由调用方退回空间剖面）。
    """
    psi = np.asarray(psi, float)
    psi_o = float(psi.min())
    scale = float(np.abs(psi).max())
    if (not np.isfinite(psi_o) or scale <= 0.0 or psi_o >= -1e-6 * scale
            or int((psi < 0.0).sum()) < 20):
        return None
    mask = None
    if closed_only:
        _lvl, mask = lcfs_level(psi)
        if mask is None:
            return None
    S = abs(psi_o) / max(c0, 1e-12)
    x = np.clip(psi / S, -60.0, 60.0)
    u = np.maximum(1.0 + sigma * x, u_floor)
    F = np.exp(-alpha * x) * u ** (n - 1.0) * (-alpha * u + n * sigma)
    R = np.asarray(s["R"], float)
    w = R * np.maximum(-F, 0.0)          # 只保留主（抗磁）电流层
    if mask is not None:
        w = np.where(mask, w, 0.0)       # 电流只跑在**闭合**磁面上
        if not (w > 0).any():
            return None
    tot = float(w.sum()) * float(s["dR"]) * float(s["dZ"])
    if not np.isfinite(tot) or tot <= 0.0:
        return None
    return w / tot


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--t-fire", type=float, default=20e-6,
                    help="形成段结束、接管放电线圈时序的时刻 [s]")
    ap.add_argument("--t-end", type=float, default=0.62e-3, help="终点 [s]")
    ap.add_argument("--n-fine", type=int, default=140, help="形成段步数")
    ap.add_argument("--n-coarse", type=int, default=220, help="其后步数")
    ap.add_argument("--rp", type=float, default=0.08)
    ap.add_argument("--wr", type=float, default=0.02)
    ap.add_argument("--zpow", type=float, default=4.0)
    ap.add_argument("--zp-mode", choices=("ls", "fixed"), default="ls",
                    help="电流柱轴向半长：ls=跟 0D 的 l_s(t)（含轴向收缩）；"
                         "fixed=固定在 --zp-value")
    ap.add_argument("--zp-value", type=float, default=0.25,
                    help="zp-mode=fixed 时的半长 [m]")
    ap.add_argument("--x0", type=float, default=0.50)
    ap.add_argument("--wx", type=float, default=0.80)
    ap.add_argument("--shape-relax", type=int, default=2)
    ap.add_argument("--shape-w", type=float, default=0.5)
    ap.add_argument("--blend-steps", type=int, default=25)
    ap.add_argument("--capture", type=float, default=1.0)
    ap.add_argument("--profile", choices=("spatial", "hybrid", "fb"),
                    default="spatial",
                    help="spatial=空间电流剖面；hybrid=按 psi 的高斯投影；"
                         "fb=力平衡剖面 J∝R·P'(psi)（MRR 形式，尺度取当前解）")
    ap.add_argument("--alpha", type=float, default=31.5, help="MRR alpha（fb 用）")
    ap.add_argument("--sigma", type=float, default=8.31, help="MRR sigma（fb 用）")
    ap.add_argument("--n-mrr", type=float, default=2.0, help="MRR n（fb 用）")
    ap.add_argument("--rec", type=float, default=0.0,
                    help="重联源系数 kappa_rec（0=关；见 --tau-rec）")
    ap.add_argument("--tau-rec", type=float, default=5e-6, help="重联时间 [s]")
    ap.add_argument("--f-thr", type=float, default=0.9,
                    help="重联阈值：驱动反场因子 F_drive 超过它才开")
    ap.add_argument("--rec-cap", type=float, default=0.0,
                    help="重联源的容量 Phi_cap [Wb]；<=0 时取驱动通量摆幅")
    ap.add_argument("--closed", action="store_true",
                    help="把电流限制在**最后闭合磁面**以内（fb 用）。关掉时按 "
                         "psi<0 给电流——那样反场区会一路铺到腔室端部")
    ap.add_argument("--te", type=float, default=30.0)
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--name", default="full_discharge_2d")
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
    drive = float(rmf.get("drive_frac", 0.0))

    # ---------------- 机器模型 ----------------
    print("[full2d] 建机器模型 …")
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
    print(f"[full2d] 网格 {nR}x{nZ}，内部点 {pr.int_idx.size}")

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
    print(f"[full2d] |B_coil|peak={abs(B_coil_pk):.5f} T  B_bias={B_bias:.5f} T  "
          f"eta={eta:.3e}  RMF drive={drive:.3f}")

    # ---------------- 时间网格：形成段细、其后粗 ----------------
    tf = args.t_fire
    t_fine = np.linspace(0.0, tf, args.n_fine, endpoint=False)
    t_coarse = np.linspace(tf, args.t_end, args.n_coarse)
    tt = np.concatenate([t_fine, t_coarse])
    nt = tt.size

    # ---------------- 线圈电流时序（唯一外加输入）----------------
    Il_hist = np.zeros((nt, 3))
    for n, t in enumerate(tt):
        if t < tf:
            Il_hist[n] = np.interp(t, t0d, Ic0)
        else:
            Il_hist[n] = [np.interp(t - tf, t_d, Il_d[:, k]) for k in range(3)]
    print(f"[full2d] 时间线 {nt} 步（细 {args.n_fine} + 粗 {args.n_coarse}），"
          f"t_end={tt[-1] * 1e3:.3f} ms")
    print(f"[full2d] 线圈电流：形成末 {Il_hist[args.n_fine - 1] / 1e3} kA → "
          f"末态 {Il_hist[-1] / 1e3} kA")

    # ---------------- 时间推进 ----------------
    psi_frames = np.zeros((nt, nR, nZ))
    rec = {k: np.zeros(nt) for k in
           ("Idrive", "Bz0", "Fdrive", "Phi", "Rs", "Iphi", "Lphi", "Rphi",
            "Psi", "zc", "ls")}

    Iphi = 0.0
    Psi = None
    a0 = None
    psi_prev = None
    g_carry = None
    k_closed = None
    n_hybrid = 0
    t0 = time.perf_counter()
    for n, t in enumerate(tt):
        Il = Il_hist[n]
        # 线圈边界通量按各线圈线性叠加
        psib_coil_n = np.zeros_like(psib_pk)
        for k in range(3):
            _p, _b2, _c2, pb = model.coil_flux(
                np.eye(3)[k] * Il[k])
            psib_coil_n += np.asarray(pb, float).ravel()
        psi_coil_n = solve(np.zeros(pr.N), psib_coil_n).reshape(nR, nZ, order="F")
        psi_ext = psi_coil_n + psi_bias

        # 电流柱轴向半长
        if args.zp_mode == "fixed":
            zp_now = args.zp_value
        else:
            zp_now = 0.5 * float(np.interp(min(t, t0d[-1]), t0d, ls0d))
        g_sp = build_shape(s, args.rp, args.wr, zp_now, args.zpow)
        g = g_carry if g_carry is not None else g_sp

        def coeffs(gg):
            src_f = (gg * Rinv2d).ravel(order="F")
            psi_u = solve(src_f, psib_of(src_f)).reshape(nR, nZ, order="F")
            aa = 2.0 * np.pi * float(np.sum(psi_ext * gg) * dR * dZ)
            bb = 2.0 * np.pi * float(np.sum(psi_u * gg) * dR * dZ)
            rr_ = eta * float(np.sum(gg ** 2 * 2 * np.pi * R2d) * dR * dZ)
            return aa, bb, rr_, psi_u

        a, b, Rphi, psi_u = coeffs(g)
        if not (np.isfinite(a) and np.isfinite(b) and b > 0):
            g = g_sp
            a, b, Rphi, psi_u = coeffs(g_sp)

        # RMF：耗散按 (1-drive) 打折（形成段不施加）
        drv = drive if t >= tf else 0.0
        if n == 0:
            Psi = kap * a
            a0 = a
        else:
            Psi = Psi - (1.0 - drv) * Rphi * Iphi * (tt[n] - tt[n - 1])
        # 重联源：端部重联把"开场线通量"转成"闭合通量"，由等离子体电流承载
        # → 电流幅值上升。在这套状态量里它表现为链通量的一个消耗项
        # （Ψ 减小 ⇒ I=(Ψ-κa)/b 更负 ⇒ 闭合区更深）。速率限在 τ_rec，
        # 反场过阈值才开，容量取驱动通量摆幅（与 0D §8.4c 同一结构）。
        # 重联是**形成期**的机制（端部重联在 µs 尺度上完成），所以除了反场阈值，
        # 还要限在形成窗口内；否则同样的速率作用 0.6 ms 会把电流灌爆。
        if args.rec > 0.0 and n > 0 and psi_prev is not None and t <= tf:
            phi_closed = 2.0 * np.pi * max(0.0, -float(psi_prev.min()))
            phi_cap = (args.rec_cap if args.rec_cap > 0
                       else abs(a - a0))
            bcoil_n = B_coil_pk * (float(np.sum(Il)) / (3.0 * Il_pk))
            fdrv = abs(bcoil_n) / B_bias - 1.0 if B_bias else 0.0
            gate = 1.0 if fdrv >= args.f_thr else 0.0
            eps = (args.rec * max(phi_cap - phi_closed, 0.0) / args.tau_rec
                   * gate)
            Psi = Psi - eps * (tt[n] - tt[n - 1])
        Iphi = (Psi - kap * a) / b

        # 剖面：hybrid=按 ψ 的高斯；fb=力平衡 J∝R·P'(ψ)；spatial=直接跳过
        def profile_of(psiv):
            if args.profile == "hybrid":
                return shape_from_psi(s, psiv, args.x0, args.wx)
            if args.profile == "fb":
                return shape_fb(s, psiv, args.alpha, args.sigma, args.n_mrr,
                                closed_only=args.closed)
            return None

        g_new = profile_of(psi_coil_n + Iphi * psi_u + psi_bias)
        if g_new is not None and k_closed is None:
            k_closed = n
        lam = 0.0 if (g_new is None or k_closed is None) else \
            min(1.0, (n - k_closed + 1) / max(1, args.blend_steps))
        w_eff = args.shape_w * lam
        moved = False
        for _ in range(args.shape_relax):
            if g_new is None or w_eff <= 0.0:
                break
            g_mix = (1.0 - w_eff) * g + w_eff * g_new
            a2, b2, _r2, psi_u2 = coeffs(g_mix)
            if not (np.isfinite(b2) and b2 > 0):
                break
            g, a, b, psi_u = g_mix, a2, b2, psi_u2
            moved = True
            g_new = profile_of(psi_coil_n + Iphi * psi_u + psi_bias)
        if moved:
            n_hybrid += 1
            Psi = kap * a + b * Iphi
            g_carry = g
        else:
            g_carry = None

        psi = psi_coil_n + Iphi * psi_u + psi_bias
        psi_frames[n] = psi
        psi_prev = psi

        # ---- 诊断 ----
        mid = psi[:, Z0]
        bz0 = 2.0 * psi[1, Z0] / dR ** 2
        bcoil = B_coil_pk * (np.sum(Il) / (3.0 * Il_pk))
        # 几何诊断按**闭合区**量（有 LCFS 就用它；否则退回 psi<0）
        _lvl, comp = lcfs_level(psi)
        core = comp if comp is not None else (psi < 0.0)
        neg = np.nonzero(core[:, Z0])[0]
        Rs = float(rr[neg[-1]]) if neg.size else float("nan")
        cols = np.nonzero(core.any(axis=0))[0]
        if cols.size:
            zlo, zhi = float(zz[cols].min()), float(zz[cols].max())
            rec["zc"][n] = 0.5 * (zlo + zhi)
            rec["ls"][n] = zhi - zlo
        else:
            rec["zc"][n] = rec["ls"][n] = np.nan
        rec["Idrive"][n] = np.sum(Il) / 3.0
        rec["Bz0"][n] = bz0
        rec["Fdrive"][n] = abs(bcoil) / B_bias - 1.0 if B_bias else np.nan
        rec["Phi"][n] = 2 * np.pi * max(0.0, -float(psi.min()))
        rec["Rs"][n] = Rs
        rec["Iphi"][n] = Iphi
        rec["Lphi"][n] = b
        rec["Rphi"][n] = Rphi
        rec["Psi"][n] = Psi
        if n % 25 == 0 or n == nt - 1:
            rs_txt = "-" if not np.isfinite(Rs) else f"{Rs * 1e2:.2f}"
            print(f"[full2d] t={t * 1e6:8.2f} us  Il={np.mean(Il) / 1e3:6.2f} kA  "
                  f"Iphi={Iphi / 1e3:+8.2f} kA  Bz0={bz0:+.4f} T  "
                  f"Phi={rec['Phi'][n]:.3e} Wb  Rs={rs_txt} cm  "
                  f"zc={rec['zc'][n] * 1e2:+.2f}  ls={rec['ls'][n] * 1e2:.2f} cm")
    wall = time.perf_counter() - t0

    print(f"\n[full2d] 完成：{nt} 步，用时 {wall / 60:.1f} min")
    print(f"[full2d] 末态：Iphi={rec['Iphi'][-1] / 1e3:+.2f} kA  "
          f"Phi={rec['Phi'][-1]:.3e} Wb  Rs={rec['Rs'][-1] * 1e2:.2f} cm  "
          f"zc={rec['zc'][-1] * 1e2:+.2f} cm  ls={rec['ls'][-1] * 1e2:.2f} cm")
    print(f"[full2d] 剖面投影用了 {n_hybrid}/{nt} 步")
    print(f"[full2d] 通量 Φ_end/Φ_peak = "
          f"{rec['Phi'][-1] / np.nanmax(rec['Phi']):.3f}")

    # ψ 逐帧存全分辨率太重（92 MB），按 2:1 抽稀存 float32
    np.savez_compressed(
        OUT / f"{args.name}.npz", t=tt,
        psi=psi_frames[:, ::2, ::2].astype(np.float32),
        rr=rr[::2], zz=zz[::2], **rec,
        Il=Il_hist, rl=np.asarray(s["rl"], float),
        zl=np.asarray(s["zl"], float), hl=np.asarray(s["hl"], float),
        wl=np.asarray(s["wl"], float), Rw=float(s["Rw"]), zw=float(s["zw"]),
        B_bias=B_bias, B_coil_pk=float(B_coil_pk), t_fire=tf,
        cfg=json.dumps(dict(t_fire=tf, t_end=args.t_end, drive=drive,
                            capture=args.capture), ensure_ascii=False))
    if args.gif:
        try:
            make_gif(OUT / f"{args.name}.gif", psi_frames, rr, zz, tt, rec, s,
                     B_bias, nn=min(70, nt), fps=args.fps)
            print(f"[full2d] 动图：{OUT / (args.name + '.gif')}")
        except Exception as exc:  # noqa: BLE001
            print(f"[full2d] 动图跳过：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
