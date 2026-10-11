r"""θ 箍缩形成的**二维**场演化（路线 2：不做平衡求解，只推进通量）。

为什么要有这一步
----------------
`tools/frc_formation_0d.py` 把形成段做成 0D 集总模型，只能给出**轴上**的场与
标量时间线，画不出磁面。而现成的二维求解器每步都要解一个**平衡**，这在形成段
不成立（还没有闭合 FRC 时它会收敛到真空态、\(R_s\to\) nan，见 `修改记录.md`
第 38 条）。

本模块走第三条路：**把"每步解一个平衡"这条要求去掉，只推进通量场。**

模型
----
坐标与算符完全复用程序树（`nk_inner.Problem`）：内部满足

$$\Delta^*\psi=-\mu_0 R\,J_\zeta,\qquad
\Delta^*=R\frac{\partial}{\partial R}\frac1R\frac{\partial}{\partial R}
+\frac{\partial^2}{\partial Z^2}$$

自由边界的边界通量由线圈 Green 函数（`MachineModel.coil_flux`）与等离子体电流
的 G-S 张量 `Gsub` 给出，与平衡求解器用的是同一套。

**关键差别**：等离子体电流剖面 \(\hat g(R,Z;t)\) 是**给定的**（不是由
\(P(\psi)\) 反解出来的），所以方程对 \(\psi\) **线性**——一次 LU 求解即可，
不需要牛顿迭代，也不要求闭合区存在。

三个部件：

1. **主币回路**：直接取 `formation0d.npz` 的线圈电流波形 \(I_c(t)\)
   （四分之一正弦上升 + crowbar），保证与 0D 模型同一驱动。
2. **偏置场**：均匀轴向场 \(\psi_{\rm bias}=+B_{\rm bias}R^2/2\)
   （它满足 \(\Delta^*\psi=0\)，是合法真空场，可解析叠加）。
   取 \(B_{\rm bias}=\tfrac12|B_{\rm coil}^{\rm axis}|_{\rm peak}\)，
   于是**净外加场** \(B_{\rm app}=B_{\rm bias}-B_{\rm coil}\) 在峰值处正好翻号到
   \(-B_{\rm bias}\)（与 0D §8.9 同一口径，驱动反场因子 \(F_{\rm drive}\) 到 1）。
   偏置与线圈**同向**，靠线圈更强把净场压过去——这一步决定了 \(\psi=0\)
   在开场线阶段就落在求解域内部，分离面判据才有意义。
3. **等离子体电流**：由**通量守恒**推进，不预设幅值：

   $$\Psi_\phi=\kappa\,a(t)+b(t)\,I_\phi,\qquad
   \frac{d\Psi_\phi}{dt}=-R_\phi I_\phi$$

   其中三个系数**全部由二维几何数值算出**（这是相对 0D 的实质升级——
   0D 里 \(L_p,R_p\) 是人为占位）：

   * \(a(t)=2\pi\int\psi_{\rm ext}\,\hat g\,dRdZ\)：外场（线圈＋偏置）链到等离子体的通量；
   * \(b(t)=2\pi\int\psi_{\rm self}\,\hat g\,dRdZ\)：等离子体自感（单位电流的场）；
   * \(R_\phi=\eta\int \hat g^2\,2\pi R\,dRdZ\)：等效电阻
     （与 \(R=\eta\,2\pi R_p/A\) 同口径，是欧姆耗散 \(P_\Omega=\int\eta J^2dV\) 的等价量）。

   轴向剖面用 0D 的 \(l_s(t)\)（轴向收缩直接进模型），径向中心取 0D 的
   \(R_p\)。**径向内爆动力学不在本模型内**（与 0D 的限制相同，见文末）。

   \(\kappa\) 是**捕获效率**（设计文档 §2.4 的 \(\kappa_{\rm trap}\) 占位量）：
   只有穿过导电等离子体的那部分外场通量参与守恒，其余穿过开场线不参与。
   \(\kappa=1\) 是"全部捕获"的上限；取小一点则等离子体电流与私通量同比例减小。

判据与产物
----------
每步记录：\(B_{z0}\)（轴上净场）、\(F_{\rm drive}\) 与 \(F_{\rm 2d}\)、
私通量 \(\Phi=2\pi|\psi_{\min}|\)、分离面半径 \(R_s\)、伸长比，以及逐步的
\(\psi(R,Z)\)。

用法：

    python tools/frc_formation_2d.py                 # 约 1 分钟
    python tools/frc_formation_2d.py --gif           # 顺带出一条动图
    python tools/frc_formation_2d.py --rp 0.08 --wr 0.015 --te 30

产物：`outputs/frc_formation/formation2d.{npz,png}`（`--gif` 再加 `.gif`）。

两种剖面（`--profile`）
-----------------------

* **`spatial`（默认）**：电流剖面是空间中给定的电流层（\(R_p,w_R,z_p\) 由 0D 给）。
  简单稳健，但分离面会描着这个外轮廓走——**平顶剖面给出方方正正的磁面**。
* **`hybrid`**：闭合区一出现，就把电流剖面**投影到磁面上**（
  `shape_from_psi`，\(\hat\psi=\psi/\psi_O\)，还带真实剖面必须有的 \(R\) 因子）。
  每步内的投影**保持电流 \(I_\phi\) 不变**、只让分布松弛到磁面上（这一步代表
  重联/剖面再分布），投影完再重算链通量 \(\Psi_\phi\)。这是 1D 参照模型里
  "family 投影"的二维对应物。

  **它解决了什么**：磁面变成光滑嵌套的穹顶（不再是方角）。
  **它没解决什么**：幅值。投影后的剖面自感 \(b\) 比空间剖面小 ~4 倍，同样的
  通量守恒会驱动更大的电流；加上投影剖面本身"每安培产生的私通量"更高，
  末态 FRC 会一直胀到贴壁（\(R_s\to R_w\)）。也就是说：**形状自洽了，尺寸还不自洽**。
  要把尺寸也对上，需要真实的 \(\mathrm dP/\mathrm d\psi\) 形状——而那就是回到
  MRR（要预设 \((B_e,R_s)\)），或者再加一条幅值标定。这是路线 2 的真实边界。
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
sys.path.insert(0, str(ROOT / "tools"))
os.chdir(ROOT)
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

import design_point as dp  # noqa: E402

OUT = ROOT / "outputs" / "frc_formation"
MU0 = dp.MU0


def build_shape(s, rp: float, wr: float, zp: float,
                zpow: float = 4.0) -> np.ndarray:
    """归一化电流剖面 g（∫g dRdZ = 1），形状 (nR, nZ)，order='F' 语义。

    轴向用 \(\exp[-(Z/z_p)^{\rm zpow}]\)：`zpow=4` 是**平顶**剖面（中间几乎不变、
    两端陡降），`zpow=2` 是高斯。**分离面的形状基本跟着这个剖面走**——
    平顶剖面会给出方方正正的闭合面，这是假设的产物，不是物理。
    """
    R = np.asarray(s["R"], float)
    Z = np.asarray(s["Z"], float)
    dR, dZ = float(s["dR"]), float(s["dZ"])
    zp = float(np.clip(zp, 0.05, 0.95 * float(s["zw"])))
    g = np.exp(-((R - rp) / wr) ** 2 - np.abs(Z / zp) ** zpow)
    g = np.where(R > 0, g, 0.0)
    return g / (g.sum() * dR * dZ)


def shape_from_psi(s, psi, x0: float, wx: float):
    r"""按**归一化通量坐标**给的电流剖面（hybrid 模式用）。

    令 \(\hat\psi=\psi/\psi_O\)（\(\psi_O\) 是 O 点的 \(\psi<0\)）：
    \(\hat\psi=1\) 在 O 点、\(0\) 在分离面、\(<0\) 在开场线区。于是

    $$g\propto R\;\exp\left[-\left(\frac{\hat\psi-x_0}{w_x}\right)^2\right]$$

    电流**是磁通的函数**，所以电流的等值线天然与磁面重合——磁面自动光滑嵌套，
    这是空间剖面做不到的。归一化仍取 \(\int g\,dRdZ=1\)（总电流由通量守恒给），
    所以这个模式**不假设力平衡、也不需要预设的 \((B_e,R_s)\)**。

    **那个 \(R\) 因子不能省**：真实剖面的电流是 \(J_\zeta=R\,\mathrm dP/\mathrm d\psi\)，
    峰值靠近分离面正是被 \(R\) 顶上来的（设计点平衡里 \(|\mathrm dP/\mathrm d\psi|\)
    在芯部其实相当平，中平面电流层却在中 \(R\approx7.2\) cm 处有尖峰）。
    少了 \(R\)，电流会往轴心堆，FRC 会缩成一根针。

    闭合区还没出现（\(\psi_{\min}\ge0\)）时返回 `None`，由调用方退回空间剖面。
    """
    psi = np.asarray(psi, float)
    psi_o = float(psi.min())
    scale = float(np.abs(psi).max())
    # 反场区要"解析得出来"才用通量坐标：既要有相对幅值，也要有足够多的格点，
    # 否则 1e-18 量级的数值噪声也会被当成 O 点、把 psi/psi_O 放大到无意义。
    if (not np.isfinite(psi_o) or scale <= 0.0
            or psi_o >= -1e-6 * scale or int((psi < 0.0).sum()) < 20):
        return None
    x = np.clip(psi / psi_o, -4.0, 1.5)
    R = np.asarray(s["R"], float)
    w = R * np.exp(-((x - x0) / wx) ** 2)
    w = np.where(R > 0, w, 0.0)
    tot = float(w.sum()) * float(s["dR"]) * float(s["dZ"])
    if not np.isfinite(tot) or tot <= 0:
        return None
    return w / tot


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nt", type=int, default=160, help="时间步数")
    ap.add_argument("--t-end", type=float, default=40e-6, help="终点 [s]")
    ap.add_argument("--rp", type=float, default=0.08, help="等离子体柱半径 [m]")
    ap.add_argument("--wr", type=float, default=0.02, help="电流层径向宽度 [m]")
    ap.add_argument("--zpow", type=float, default=4.0,
                    help="轴向剖面指数：4=平顶（默认），2=高斯")
    ap.add_argument("--profile", choices=("spatial", "hybrid"), default="spatial",
                    help="spatial=空间电流剖面（默认）；hybrid=闭合区出现后改用"
                         "按磁通坐标定义的剖面（磁面自洽光滑）")
    ap.add_argument("--x0", type=float, default=0.50,
                    help="hybrid：电流峰在归一化通量坐标上的位置（1=O 点，0=分离面）")
    ap.add_argument("--wx", type=float, default=0.80,
                    help="hybrid：通量坐标上的剖面宽度")
    ap.add_argument("--shape-relax", type=int, default=2,
                    help="hybrid：每步内层形状欠松弛迭代次数")
    ap.add_argument("--shape-w", type=float, default=0.5,
                    help="hybrid：内层形状欠松弛权重")
    ap.add_argument("--blend-steps", type=int, default=30,
                    help="hybrid：闭合区出现后，投影权重在这么多步内从 0 升到 1"
                         "（避免剖面切换时电流跳变）")
    ap.add_argument("--te", type=float, default=30.0, help="电子温度 [eV]")
    ap.add_argument("--capture", type=float, default=1.0,
                    help="捕获效率 kappa（1=全部外场通量参与守恒）")
    ap.add_argument("--scale-flux", type=float, default=1.0,
                    help="把偏置场按比例缩放（默认 1）")
    ap.add_argument("--gif", action="store_true", help="顺带出一条动图")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--name", default="formation2d")
    args = ap.parse_args()

    # ---------------- 0D 的驱动波形与轴向长度 ----------------
    f0 = np.load(OUT / "formation0d.npz", allow_pickle=True)
    t0d = np.asarray(f0["t"], float)
    Ic0 = np.asarray(f0["Ic"], float)
    ls0d = np.asarray(f0["l_s"], float)

    # ---------------- 机器模型（设计点 A 几何）----------------
    print("[f2d] 建机器模型（线圈 Green 函数 + G-S 张量）…")
    s, model, _il_scale, _iz = dp.build_design_model()
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
    psib_coil_pk = np.asarray(psib_pk, float).ravel()
    pr = model.problem(psib_coil_pk, order=2)
    lu = pr.lu_full()
    print(f"[f2d] 网格 {nR}x{nZ}，内部点 {pr.int_idx.size}，LU 已分解")

    def solve(src_flat, psib_vec):
        """线性 G-S 一次求解：源 + 边界通量 → psi（整张网格，order='F'）。"""
        pr.set_psib(psib_vec)
        rhs = np.array(pr.bnd_val)
        rhs[pr.int_idx] = pr.dZ2 * MU0 * pr.Rint ** 2 * src_flat[pr.int_idx]
        return lu.solve(rhs)

    def psib_of(src_flat):
        """等离子体电流对边界通量的贡献（同一份 Gsub 张量）。"""
        Jz_int = Rflat[pr.int_idx] * src_flat[pr.int_idx]
        return MU0 * pr.dR * pr.dZ * (pr.Gsub.T @ Jz_int)

    psi_coil_pk = solve(np.zeros(pr.N), psib_coil_pk).reshape(nR, nZ, order="F")
    B_coil_pk = 2.0 * psi_coil_pk[1, Z0] / dR ** 2
    B_bias = 0.5 * abs(B_coil_pk) * args.scale_flux
    kap = float(args.capture)
    psi_bias = B_bias * R2d ** 2 / 2.0
    print(f"[f2d] 线圈峰值轴场 |B_coil|={abs(B_coil_pk):.5f} T  →  "
          f"B_bias={B_bias:.5f} T（偏置与线圈同向，取 0.5×峰值，"
          f"使净外加场在峰值处翻号）")

    eta = 3.16e-6 * (30.0 / args.te) ** 1.5      # Spitzer 的相对标度
    print(f"[f2d] T_e={args.te:g} eV  →  eta={eta:.3e} ohm·m")

    # ---------------- 时间推进 ----------------
    tt = np.linspace(0.0, args.t_end, args.nt)
    Ic = np.interp(tt, t0d, Ic0)
    ls = np.interp(tt, t0d, ls0d)
    psi_frames = np.zeros((args.nt, nR, nZ))
    rec = {k: np.zeros(args.nt) for k in
           ("Idrive", "Bz0", "Fdrive", "F2d", "Phi", "Rs", "elong",
            "Iphi", "Lphi", "Rphi", "Psi")}

    Iphi = 0.0
    Psi = None
    n_hybrid = 0
    g_carry = None
    k_closed = None

    def coeffs(g):
        """给定电流剖面 g：返回 (a, b, R_phi, psi_u)。

        用线性叠加：总场 = psi_coil + I_phi·psi_u + psi_bias，
        所以每评估一次剖面只要**一次** LU 求解（`psi_coil_n` 每步一次）。
        """
        src_f = (g * Rinv2d).ravel(order="F")
        psi_u = solve(src_f, psib_of(src_f)).reshape(nR, nZ, order="F")
        a = 2.0 * np.pi * float(np.sum(psi_ext * g) * dR * dZ)
        b = 2.0 * np.pi * float(np.sum(psi_u * g) * dR * dZ)
        Rphi = eta * float(np.sum(g ** 2 * 2 * np.pi * R2d) * dR * dZ)
        return a, b, Rphi, psi_u

    for n, t in enumerate(tt):
        # ---- 本步的线圈场（线性缩放）与外场 ----
        psib_coil_n = psib_coil_pk * (Ic[n] / Il_pk)
        psi_coil_n = solve(np.zeros(pr.N), psib_coil_n).reshape(nR, nZ, order="F")
        psi_ext = psi_coil_n + psi_bias

        # ---- 剖面：空间剖面兜底；hybrid 在闭合区出现后把剖面往磁面上投影 ----
        g_sp = build_shape(s, args.rp, args.wr, 0.5 * float(ls[n]), args.zpow)
        g = g_carry if g_carry is not None else g_sp

        a, b, Rphi, psi_u = coeffs(g)
        if not (np.isfinite(a) and np.isfinite(b) and b > 0):
            g = g_sp
            a, b, Rphi, psi_u = coeffs(g_sp)

        if n == 0:
            Psi = kap * a                      # 初始等离子体电流为 0
        else:
            Psi = Psi - Rphi * Iphi * (tt[n] - tt[n - 1])
        Iphi = (Psi - kap * a) / b             # 通量守恒反解电流

        # ---- hybrid 的**剖面投影**（对应 1D 参照模型里的 family 投影）----
        # 物理上这是重联/剖面再分布：**电流 \(I_\phi\) 保持不变**，电流分布
        # 松弛到当前磁面上；链通量会因此变化，所以投影完要重算 \(Psi\)。
        # 反过来做（保持 \(Psi\) 不变去改形状）会让电流跳变甚至反号——
        # 因为两种形状的 \(a,b\) 差很多，拿旧的 \(Psi\) 去配新的 \((a,b)\)
        # 得到的 \(I=(Psi-\kappa a)/b\) 没有物理意义。
        if args.profile == "hybrid":
            moved = False
            g_new = shape_from_psi(s, psi_coil_n + Iphi * psi_u + psi_bias,
                                   args.x0, args.wx)
            if g_new is not None and k_closed is None:
                k_closed = n                       # 闭合区首次出现的那一步
            lam = 0.0 if (g_new is None or k_closed is None) else \
                min(1.0, (n - k_closed + 1) / max(1, args.blend_steps))
            w_eff = args.shape_w * lam
            for _ in range(args.shape_relax):
                if g_new is None or w_eff <= 0.0:
                    break
                g_mix = (1.0 - w_eff) * g + w_eff * g_new
                a2, b2, _R2, psi_u2 = coeffs(g_mix)
                if not (np.isfinite(b2) and b2 > 0):
                    break
                g, a, b, psi_u = g_mix, a2, b2, psi_u2
                moved = True
                g_new = shape_from_psi(s, psi_coil_n + Iphi * psi_u + psi_bias,
                                       args.x0, args.wx)
            if moved:
                n_hybrid += 1
                Psi = kap * a + b * Iphi       # 再分布后的链通量
                g_carry = g
            else:
                g_carry = None                 # 还没有闭合区 → 退回空间剖面

        psi = psi_coil_n + Iphi * psi_u + psi_bias
        psi_frames[n] = psi

        # ---- 诊断 ----
        mid = psi[:, Z0]
        bz0 = 2.0 * psi[1, Z0] / dR ** 2
        bcoil = B_coil_pk * (Ic[n] / Il_pk)
        neg = np.nonzero(mid < 0.0)[0]
        Rs = float(rr[neg[-1]]) if neg.size else float("nan")
        pmin = float(psi.min())
        elong = float("nan")
        if neg.size:
            from scipy import ndimage
            lab, nl = ndimage.label(psi < 0.0,
                                    structure=np.array([[0, 1, 0], [1, 1, 1],
                                                        [0, 1, 0]]))
            if nl:
                sizes = np.bincount(lab.ravel(), minlength=nl + 1)[1:]
                core = lab == (int(np.argmax(sizes)) + 1)
                iz = np.nonzero(core.any(axis=0))[0]
                zh = float(np.max(np.abs(zz[iz])))
                if np.isfinite(Rs) and Rs > 0:
                    elong = zh / Rs
        rec["Idrive"][n] = Ic[n]
        rec["Bz0"][n] = bz0
        rec["Fdrive"][n] = abs(bcoil) / B_bias - 1.0 if B_bias else np.nan
        rec["F2d"][n] = (-bz0 / bcoil) if bcoil else np.nan
        rec["Phi"][n] = 2 * np.pi * max(0.0, -pmin)
        rec["Rs"][n] = Rs
        rec["elong"][n] = elong
        rec["Iphi"][n] = Iphi
        rec["Lphi"][n] = b
        rec["Rphi"][n] = Rphi
        rec["Psi"][n] = Psi
        if n % 20 == 0 or n == args.nt - 1:
            rs_txt = "-" if not np.isfinite(Rs) else f"{Rs * 1e2:.2f}"
            print(f"[f2d] t={t * 1e6:6.2f} us  Ic={Ic[n] / 1e3:6.2f} kA  "
                  f"Iphi={Iphi / 1e3:+7.2f} kA  Bz0={bz0:+.4f} T  "
                  f"F_drive={rec['Fdrive'][n]:5.2f}  Phi={rec['Phi'][n]:.3e} Wb  "
                  f"Rs={rs_txt} cm")

    _mid = psi_frames[:, :, Z0]
    _sc = np.abs(psi_frames).max(axis=(1, 2))
    closed_mid = (_mid.min(axis=1) < -1e-6 * _sc) & (_sc > 0)
    if np.any(closed_mid):
        k = int(np.argmax(closed_mid))
        print(f"[f2d] 首次出现闭合区：t={tt[k] * 1e6:.2f} us，"
              f"此时 F_drive={rec['Fdrive'][k]:.3f}、Iphi={rec['Iphi'][k] / 1e3:+.2f} kA")
    else:
        print("[f2d] 注意：整段都没有出现闭合区（psi 始终不小于 0）")
    print(f"[f2d] 捕获效率 kappa={kap:g}")
    if args.profile == "hybrid":
        print(f"[f2d] hybrid：{n_hybrid}/{args.nt} 步用磁通面剖面"
              f"（x0={args.x0:g}, wx={args.wx:g}, "
              f"内层 {args.shape_relax} 次欠松弛 w={args.shape_w:g}）")
    print(f"[f2d] 末态：Iphi={rec['Iphi'][-1] / 1e3:+.2f} kA  "
          f"Phi={rec['Phi'][-1]:.3e} Wb  Rs={rec['Rs'][-1] * 1e2:.2f} cm  "
          f"elong={rec['elong'][-1]:.2f}")
    print(f"[f2d] 数值自检：L_phi={rec['Lphi'][0]:.4e} H   "
          f"R_phi={rec['Rphi'][0]:.4e} ohm   L/R={rec['Lphi'][0] / rec['Rphi'][0]:.3e} s")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / f"{args.name}.npz", t=tt, psi=psi_frames, rr=rr, zz=zz, **rec,
        rl=np.asarray(s["rl"], float), zl=np.asarray(s["zl"], float),
        hl=np.asarray(s["hl"], float), wl=np.asarray(s["wl"], float),
        Rw=float(s["Rw"]), zw=float(s["zw"]), B_bias=B_bias,
        B_coil_pk=float(B_coil_pk), eta=eta, rp=args.rp, wr=args.wr,
        cfg=json.dumps(dict(nt=args.nt, t_end=args.t_end, te=args.te,
                            scale_flux=args.scale_flux, rp=args.rp, wr=args.wr,
                            profile=args.profile, zpow=args.zpow,
                            x0=args.x0, wx=args.wx),
                       ensure_ascii=False))

    try:
        draw_summary(OUT / f"{args.name}.png", tt, rec)
        print(f"[f2d] 产物：{OUT / (args.name + '.png')}")
    except Exception as exc:  # noqa: BLE001
        print(f"[f2d] 出图跳过：{exc}")
    if args.gif:
        try:
            make_gif(OUT / f"{args.name}.gif", psi_frames, rr, zz, tt, rec, s,
                     B_bias, nn=min(60, args.nt), fps=args.fps)
            print(f"[f2d] 动图：{OUT / (args.name + '.gif')}")
        except Exception as exc:  # noqa: BLE001
            print(f"[f2d] 动图跳过：{exc}")
    return 0


def _style():
    import matplotlib
    matplotlib.use("Agg")
    from gseqfrc import mstyle
    mstyle.install()
    return matplotlib


def draw_summary(path, tt, rec):
    _style()
    import matplotlib.pyplot as plt
    from gseqfrc import mstyle
    fig, ax = plt.subplots(2, 3, figsize=(14.5, 7.0))
    x = tt * 1e6
    ax[0, 0].plot(x, rec["Idrive"] / 1e3, color=mstyle.BLUE, lw=2)
    ax[0, 0].set_ylabel("I_c [kA]")
    ax[0, 0].set_title("主币线圈电流（取 0D 波形）")
    ax[0, 1].plot(x, rec["Iphi"] / 1e3, color=mstyle.RED, lw=2)
    ax[0, 1].set_ylabel("I_phi [kA]")
    ax[0, 1].set_title("环向电流（通量守恒反解）")
    ax[0, 2].plot(x, rec["Bz0"], color=mstyle.BLACK, lw=2)
    ax[0, 2].axhline(0.0, color="0.6", lw=0.9)
    ax[0, 2].set_ylabel("B_z0 [T]")
    ax[0, 2].set_title("轴上净场（过零即场反转）")
    ax[1, 0].plot(x, rec["Fdrive"], color=mstyle.GREEN, lw=2, label="F_drive")
    ax[1, 0].plot(x, rec["F2d"], color=mstyle.MAGENTA, lw=2, ls="--",
                  label="F_2d")
    ax[1, 0].axhline(1.0, color="0.5", ls=":", lw=1.2)
    ax[1, 0].set_ylabel("F")
    ax[1, 0].set_title("反场因子")
    ax[1, 0].legend(fontsize=9, frameon=False)
    ax[1, 1].plot(x, rec["Phi"] * 1e3, color=mstyle.BLUE, lw=2)
    ax[1, 1].set_ylabel("Phi [mWb]")
    ax[1, 1].set_title("私通量（闭合区出现后 > 0）")
    ax[1, 2].plot(x, rec["Rs"] * 1e2, color=mstyle.RED, lw=2)
    ax[1, 2].set_ylabel("R_s [cm]")
    ax[1, 2].set_title("中平面分离面半径")
    for a in ax.ravel():
        a.set_xlabel("t [us]")
        a.grid(True, **mstyle.GRID_STYLE)
        mstyle.style_axes(a, fontsize=12, fontweight="bold", linewidth=1.4)
    fig.suptitle("theta-pinch formation: 2D flux evolution without equilibrium solve",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_gif(path, psi_frames, rr, zz, tt, rec, s, B_bias, nn=60, fps=8.0):
    matplotlib = _style()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from gseqfrc import mstyle
    from make_flux_animation import write_gif
    idx = np.unique(np.linspace(0, psi_frames.shape[0] - 1, nn).astype(int))
    rl = np.asarray(s["rl"], float)
    zl = np.asarray(s["zl"], float)
    hl = np.asarray(s["hl"], float)
    wl = np.asarray(s["wl"], float)
    Rw, zw = float(s["Rw"]), float(s["zw"])
    pmin = float(psi_frames.min())
    pmax = float(psi_frames.max())
    lt = max(abs(pmin), 1e-9)
    norm = matplotlib.colors.SymLogNorm(
        lt, linscale=3.0, vmin=1.02 * min(pmin, -lt),
        vmax=1.02 * max(pmax, lt), base=10)
    Zg, Rg = np.meshgrid(zz, rr, indexing="ij")
    fig = plt.figure(figsize=(12.0, 6.4))
    axm = fig.add_axes([0.055, 0.415, 0.765, 0.515])
    axp = fig.add_axes([0.850, 0.415, 0.015, 0.515])
    axa = fig.add_axes([0.060, 0.140, 0.215, 0.150])
    axb = fig.add_axes([0.350, 0.140, 0.215, 0.150])
    axc = fig.add_axes([0.640, 0.140, 0.215, 0.150])
    sm = plt.cm.ScalarMappable(cmap="parula", norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=axp)
    cb.set_label(r"$\psi$ [Wb/2$\pi$]", fontsize=10.5, fontweight="bold")
    cb.ax.tick_params(labelsize=9)
    info = fig.text(0.5, 0.040, "", ha="center", va="bottom", fontsize=10.5)
    x = tt * 1e6
    frames = []
    for k in idx:
        for a in (axm, axa, axb, axc):
            a.clear()
        p2 = psi_frames[k]
        axm.pcolormesh(Zg, Rg, p2.T, cmap="parula", norm=norm, shading="auto")
        axm.contour(Zg, Rg, p2.T, levels=[0.0], colors=mstyle.RED,
                    linewidths=2.0, linestyles="--")
        for r_, z_, h_, w_ in zip(rl, zl, hl, wl):
            axm.add_patch(Rectangle((z_ - w_ / 2, r_ - h_ / 2), w_, h_,
                                    fill=False, ec="0.15", lw=1.6))
        axm.plot([-zw, zw], [Rw, Rw], color="k", lw=1.8)
        axm.set_xlim(-zw, zw)
        axm.set_ylim(0, 0.17)
        axm.set_xlabel(r"$z$ [m]")
        axm.set_ylabel(r"$r$ [m]")
        rs_txt = "-" if not np.isfinite(rec["Rs"][k]) else f"{rec['Rs'][k] * 1e2:.2f}"
        axm.set_title(f"theta-pinch formation (2D, 无平衡求解)   "
                      f"t = {tt[k] * 1e6:.2f} us   "
                      f"Phi = {rec['Phi'][k] * 1e3:.4f} mWb   Rs = {rs_txt} cm",
                      fontsize=11.5)
        axa.plot(x[:k + 1], rec["Iphi"][:k + 1] / 1e3, color=mstyle.RED, lw=2)
        axa.set_ylabel("I_phi [kA]")
        axa.set_title("环向电流", fontsize=10)
        axb.plot(x[:k + 1], rec["Bz0"][:k + 1], color=mstyle.BLACK, lw=2)
        axb.axhline(0.0, color="0.6", lw=0.8)
        axb.set_ylabel("B_z0 [T]")
        axb.set_title("轴上净场", fontsize=10)
        axc.plot(x[:k + 1], rec["Phi"][:k + 1] * 1e3, color=mstyle.BLUE, lw=2)
        axc.set_ylabel("Phi [mWb]")
        axc.set_title("私通量", fontsize=10)
        for a in (axa, axb, axc):
            a.set_xlabel("t [us]")
        for a in (axm, axa, axb, axc):
            mstyle.style_axes(a, fontsize=10.5, fontweight="bold",
                              linewidth=1.3, force=True)
        info.set_text(f"I_c = {rec['Idrive'][k] / 1e3:.2f} kA    "
                      f"I_phi = {rec['Iphi'][k] / 1e3:+.2f} kA    "
                      f"B_z0 = {rec['Bz0'][k]:+.4f} T    "
                      f"F_drive = {rec['Fdrive'][k]:.2f}    "
                      f"B_bias = {B_bias:.4f} T")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    write_gif(frames, path, fps)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
