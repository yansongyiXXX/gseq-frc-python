r"""Stage 4：线圈回路 + 导体壳——把边界条件也变成状态量（**修改版** 第 22 条）。

思路（关键的一招）
------------------
把**导体壳当作额外的线圈**，直接附加到 `s` 的线圈表后面（`rl/zl/hl/wl/Il/nl`），
于是程序树里的每一样都能原样复用：

* `_coil_green_basis` / `coil_flux`：壳电流对边界通量的贡献（壳在 \(R=R_w\) 上，
  \(\psi_b\) 自动包含它）；
* `_coil_mutual_matrix`：线圈—线圈、线圈—壳、壳—壳互感（对角线仍由丝状电流
  奇异，须另给自感）；
* `_coil_grid_green`：壳↔等离子体网格耦合（回路方程里 \(\mathcal{M}_{l,y}\dot I_y\)
  那一项）。

程序树一个字都不用改。

回路方程
--------
\(M\dot I+RI=V-\dot\Psi_{\rm pl}\)，其中
\(\Psi_{{\rm pl},l}=\sum_y \mathcal{M}_{l,y}I_y\)、\(I_y=J_\zeta\,dRdZ\)。
隐式欧拉（\(\Psi_{\rm pl}\) 用族关系 \(\propto\sqrt{\Phi}\) 外推一步，一阶）：

$$(M+\Delta t\,R)\,I^{n+1}=M\,I^n+\Delta t\,V-\Delta\Psi_{\rm pl}$$

等离子体那边沿用第 21 条的通量 ODE（状态量 \(\Phi\)），每步再用割线法把
\(I_{\zeta0}\) 调到目标通量。壳的 \(V\equiv0\)；线圈的 \(V\) 由用户给（默认给
\(V=RI_0\)，即"恰好维持初始电流"，等价于第 21 条的固定电流极限）。

自感/电阻是要标定的量
---------------------
自感用 freegsnke 的圆环公式 \(L=\mu_0R(\ln(8R/a)-0.5)\)，等效导体半径
\(a=\sqrt{\text{截面}/\pi}\)；电阻用 \(R=\rho\,2\pi R/A\)。**这两条都是几何占位**——
真实线圈的匝数、绕法、导体截面、以及壳的厚度/材料都要按实机给（`--coil-L`、
`--coil-R`、`--shell-rho`、`--shell-t`）。程序会把算出来的 \(L/R\) 时间打印出来，
和 \(\Phi\) 的衰减时间一比就知道这些输入重不重要。

**路线 A：壳环通量不动点（`--shell-loop`，**修改版** 新增）**
----------------------------------------------------------
原实现里壳电流由回路方程算出、再当线圈喂进平衡，而求解器的边界通量
\(\psi_b\) 始终是"输出"——没有任何一步要求它回到冻结值。`--shell-loop` 加一层
与 CI 环 / \(B_e\) 环同构的外层迭代：**未知量是壳环电流，约束是每个壳环所在面的
边界通量等于冻结值**（理想壳恒等于 \(t=0\) 的值；电阻壳按
\(\mathrm{d}\Psi/\mathrm{d}t=-RI\) 松弛）。雅可比取线圈 Green 基里壳那几列，
等离子体那部分对壳电流的依赖靠迭代收进去。每步额外打印逐环通量与
\(\varepsilon_l=\lvert\Psi_l(t)-\Psi_l(0)\rvert/\lvert\Psi_l(0)\rvert\)。

用法
----

    python tools/frc_evolve_stage04.py --verify            # 自检（约 3 分钟）
    python tools/frc_evolve_stage04.py                     # 线圈电压维持 + 铜壳
    python tools/frc_evolve_stage04.py --shell none        # 和 Stage 3 同一条极限
    python tools/frc_evolve_stage04.py --shell ideal       # 理想导体壳（通量冻结）
    python tools/frc_evolve_stage04.py --flux-norm be-loop --shell ideal --shell-loop
    python tools/frc_evolve_stage04.py --coupled           # 步内回路↔平衡迭代到自洽
    python tools/frc_evolve_stage04.py --coil-L 1e-6,1e-6,1e-6

产物：`outputs/frc_stage04/history.npz` 与 `stage04.png`。
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

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

from gseqfrc import mpath  # noqa: E402
from gseqfrc import mstyle  # noqa: E402

mstyle.install()   # 出图统一按 MATLAB 那一套（见 gseqfrc/mstyle.py）

MU0 = 4 * np.pi * 1e-7
SRC = ROOT / "GSEQ-FRC-open" / "second_paper" / "Fig6"
OUT_DIR = ROOT / "outputs" / "frc_stage04"

mpath.addpath("second_paper/Fig6")
gcm = mpath.load("GSEQ_coil_MRR")
nk = gcm._load_nk_inner()


def spitzer_resistivity(te_ev: float, ln_lambda: float = 10.0, z: float = 1.0) -> float:
    """Spitzer 电阻率 [Ohm*m]。"""
    return 5.2e-5 * z * ln_lambda / max(te_ev, 1e-6) ** 1.5


def self_inductance(R: float, a: float) -> float:
    """圆环自感 \(L=\mu_0R(\ln(8R/a)-0.5)\)（freegsnke `self_ind_circular_loop` 同式）。

    `a` 是等效导体半径；线圈表里的矩形截面用 \(a=\sqrt{A/\pi}\) 折算——
    **这是占位**，真实线圈要按绕组标定。
    """
    return MU0 * R * (np.log(8.0 * R / a) - 0.5)


def shell_ring_nodes(s, z_shell):
    """壳环中心 \((R_w,z_l)\) 对应的**边界数组下标**（用来取 \(\psi_b\)）。

    边界点坐标在 `s["Rbnd"]`/`s["Zbnd"]` 里；取距离 \((R_w,z_l)\) 最近的那一个，
    它一定落在 \(r=R_w\) 那一列上。返回长度 `n_shell` 的整数下标数组。
    """
    Rb = np.asarray(s["Rbnd"], dtype=float).ravel()
    Zb = np.asarray(s["Zbnd"], dtype=float).ravel()
    Rw = float(s["Rw"])
    out = [int(np.argmin(np.hypot(Rb - Rw, Zb - z)))
           for z in np.atleast_1d(np.asarray(z_shell, dtype=float))]
    return np.asarray(out, dtype=int)


def build_machine(s0, *, shell="resistive", n_shell=4, shell_t=5e-3,
                  shell_rho=1.7e-8, coil_area=None, coil_L=None, coil_R=None,
                  plasma_coupling=True, verbose=True):
    """构造"线圈 + 导体壳"的机器模型（壳 = 附加线圈）。

    返回 dict：`s`（扩展后的 setup）、`model`、`M`（含自感的互感矩阵）、
    `Rvec`、`n_active`、`n_shell`、索引切片、以及自检用的中间量。
    """
    s = dict(s0)
    rl = np.atleast_1d(np.asarray(s0["rl"], dtype=float)).copy()
    zl = np.atleast_1d(np.asarray(s0["zl"], dtype=float)).copy()
    hl = np.atleast_1d(np.asarray(s0["hl"], dtype=float)).copy()
    wl = np.atleast_1d(np.asarray(s0["wl"], dtype=float)).copy()
    Il = np.atleast_1d(np.asarray(s0["Il"], dtype=float)).copy()
    n_active = rl.size
    Rw, zw = float(s0["Rw"]), float(s0["zw"])

    if shell != "none":
        dz = 2 * zw / n_shell
        zs = np.linspace(-zw + dz / 2, zw - dz / 2, n_shell)
        rl = np.concatenate([rl, np.full(n_shell, Rw)])
        zl = np.concatenate([zl, zs])
        hl = np.concatenate([hl, np.full(n_shell, shell_t)])
        wl = np.concatenate([wl, np.full(n_shell, dz)])
        Il = np.concatenate([Il, np.zeros(n_shell)])
    n_shell = int(rl.size - n_active)
    z_shell = (np.asarray(zl[n_active:], dtype=float).copy() if n_shell
               else np.zeros(0, dtype=float))

    s["rl"], s["zl"], s["hl"], s["wl"], s["Il"] = rl, zl, hl, wl, Il
    s["nl"] = int(rl.size)
    # **修改版**（路线 A）：每个壳环中心对应的边界点下标——冻结约束就写在这些点上
    shell_nodes = (shell_ring_nodes(s, z_shell) if n_shell
                   else np.zeros(0, dtype=int))
    t0 = time.perf_counter()
    model = gcm.MachineModel(s, mutual=True, plasma_coupling=plasma_coupling)
    build_s = time.perf_counter() - t0

    # ---- 自感（对角线）与电阻 ----
    if coil_area is None:
        area = hl * wl                     # 矩形截面（占位）
    else:
        area = np.full(rl.size, float(coil_area))
    a_eq = np.sqrt(area / np.pi)
    L_self = self_inductance(rl, a_eq)
    if coil_L is not None:
        vals = np.asarray([float(x) for x in str(coil_L).split(",")], dtype=float)
        L_self[:vals.size] = vals
    Rvec = 1.7e-8 * 2 * np.pi * rl / area
    if coil_R is not None:
        vals = np.asarray([float(x) for x in str(coil_R).split(",")], dtype=float)
        Rvec[:vals.size] = vals
    if n_shell:
        Rvec[n_active:] = shell_rho * 2 * np.pi * rl[n_active:] / area[n_active:]
        if shell == "ideal":
            Rvec[n_active:] = 1e-12

    M = np.array(model.Mcoil, dtype=float, copy=True)
    np.fill_diagonal(M, L_self)
    if verbose:
        print(f"[stage04] 机器：{n_active} 个线圈 + {n_shell} 个壳环"
              f"（壳: {'无' if shell == 'none' else shell}），"
              f"模型构建 {build_s:.1f} s")
        print("          自感 L [H] = " + np.array2string(L_self, precision=3))
        print("          电阻 R [Ohm] = " + np.array2string(Rvec, precision=3))
        print("          L/R [ms]   = " + np.array2string(L_self / Rvec * 1e3,
                                                           precision=3))
    return dict(s=s, model=model, M=M, Rvec=Rvec, n_active=int(n_active),
                n_shell=int(n_shell), build_s=build_s,
                z_shell=z_shell, shell_nodes=shell_nodes)


def flux_loss_rate(psi2, pr, eta: float, k: int = 2):
    r"""同第 21 条：\(S_0=2\pi\frac{\eta}{\mu_0}\Delta^*\psi|_{\text{O 点}}\) [Wb/s]。"""
    psif = np.asarray(psi2, dtype=float).ravel(order="F")
    dstar = -(pr.A @ psif) / pr.dZ2
    i0 = int(np.argmin(psif))
    nR = pr.nR
    iR, iZ = i0 % nR, i0 // nR
    vals = []
    for a in range(-k, k + 1):
        for b in range(-k, k + 1):
            ia, ib = iR + a, iZ + b
            if 1 <= ia <= nR - 2 and 1 <= ib <= pr.nZ - 2:
                vals.append(float(dstar[ia + ib * nR]))
    mid = float(np.median(vals))
    spread = float((np.max(vals) - np.min(vals)) / abs(mid)) if mid else float("nan")
    return 2 * np.pi * (eta / MU0) * mid, spread


def plasma_flux(out, model):
    """等离子体在线圈/壳上的磁链 \(\Psi_{{\rm pl},l}=\sum_y\mathcal{M}_{l,y}I_y\) [Wb]。

    \(I_y=J_\zeta\,dR\,dZ\)（内部点，A）；`model.coil_grid` 就是
    `_coil_grid_green` 出来的 \(\mathcal{M}_{l,y}\)（H）。
    """
    Jz = np.asarray(out["Jzeta"], dtype=float).ravel(order="F")
    dR, dZ = float(out["dR"]), float(out["dZ"])
    Iy = (Jz * dR * dZ)[model.int_lin]
    if model.coil_grid is None:            # 没建耦合时不参与回路
        return np.zeros(model.nl)
    return model.coil_grid @ Iy


def coil_edge_field(model, Il):
    """线圈（真空）在中平面壁面处的 \(B_z\)——也就是"真实外场" \(B_{\rm ext}\)。

    用 `coil_flux` 给出的边界点上的 \(B_z\)（与 `figure_6_2` 打印的
    "B_z at midplane in vacuum" 是同一个量）。物理口径下剖面参考场应当取它，
    而不是沿用 1D 拟合时的 \(B_e=1\)。
    """
    s = model.s
    _psiw, _Br, Bz, _psib = model.coil_flux(Il)
    nR, nZ, Z0 = s["nR"], s["nZ"], s["Z0"] - 1
    # 边界枚举（`_setup` 的 bnd_lin）里，r=Rw 那一列对应 jrb=nR、jzb=j
    # 边界点的顺序见 `_setup`：下边、上边、再右边（r=Rw, j=2..nZ-1）
    nb = s["nb"]
    j_lo, j_hi = nR - 1, nb - (nR - 1)          # 右列在边界数组里的下标范围
    seg = np.arange(j_lo, j_hi)                 # 对应 jzb = 2..nZ-1 → z=zz[j]
    jz = np.arange(1, nZ - 1)
    k = int(seg[np.argmin(np.abs(jz - (s["Z0"] - 1)))])
    return float(abs(Bz[k])), Bz


def advance_circuit(M, Rvec, I, dt, V, dPsi_pl):
    """隐式欧拉解回路方程 \((M+\Delta tR)I^{n+1}=MI^n+\Delta tV-\Delta\Psi_{\rm pl}\)。"""
    A = M + dt * np.diag(Rvec)
    b = M @ I + dt * V - dPsi_pl
    return np.linalg.solve(A, b)


def one_solve(*, model, Il, izeta0, prev, step_tol, ci_fixed=None, be=None,
              be_loop=False):
    pr = {"Izeta0": float(izeta0)}
    if be is not None:
        pr["Be"] = float(be)
    kw = dict(model=model, Il=Il, inner="hybrid", params=pr, step_tol=step_tol,
              ci_fixed=ci_fixed, be_loop=be_loop)
    if prev is None:
        return gcm.solve(gcm.SRC_DIR, **kw)
    return gcm.solve(gcm.SRC_DIR, psi0=prev["psiq"], CI0=prev["CI"],
                     psib0=prev.get("psib"), step_ref=prev.get("dpsi_step"), **kw)


def match_flux(target_phi, guess_iz, model, Il, prev, step_tol,
               rtol=2e-3, max_eval=4, ci_fixed=None, be=None, be_loop=False):
    """割线法：找 \(I_{\zeta0}\) 使二维平衡的 \(\Phi\) 等于目标（同第 21 条）。"""
    xs, fs, outs = [], [], []
    x, warm = float(guess_iz), prev
    for n in range(1, max_eval + 1):
        try:
            out = one_solve(model=model, Il=Il, izeta0=x, prev=warm,
                            step_tol=step_tol, ci_fixed=ci_fixed, be=be,
                            be_loop=be_loop)
        except Exception:  # noqa: BLE001 - 反场区塌掉后平衡不存在
            break
        phi = float(out["diag"]["Phi_private_Wb"])
        f = phi - target_phi
        xs.append(x); fs.append(f); outs.append(out)
        warm = out
        if abs(f) / max(target_phi, 1e-30) <= rtol:
            return out, x, phi, n, abs(f) / max(target_phi, 1e-30)
        if n == 1:
            x = xs[0] * np.sqrt(max(target_phi, 1e-30) / max(phi, 1e-30))
        else:
            df, dx = fs[-1] - fs[-2], xs[-1] - xs[-2]
            if df == 0 or not np.isfinite(df):
                break
            x = xs[-1] - fs[-1] * dx / df
    if not fs:
        return None, x, float("nan"), 0, float("inf")
    k = int(np.argmin([abs(f) for f in fs]))
    return outs[k], xs[k], target_phi + fs[k], len(fs), abs(fs[k]) / max(target_phi, 1e-30)


def substep_phi(phi0, s0_0, k, drive, dt, n):
    """**驱动项子步**（第 37 条）：把 \(\mathrm{d}\Phi/\mathrm{d}t=-(S_0(\Phi)-\text{drive})\)
    用 \(n\) 段细步积分，局部模型 \(S_0(\Phi)=S_0^{(0)}+k(\Phi-\Phi^{(0)})\)。

    \(k\) 由最近两个平衡态样本估出来（`ds0/dphi`）；没有两个样本时取 \(k=0\)
    （退化为把 \(S_0\) 冻结在步首的显式积分）。细步只做标量算术，**不需要解平衡**——
    这正是它比"全局小 dt"便宜的地方。
    """
    if n <= 1:
        return max(0.0, float(phi0) - (float(s0_0) - drive) * dt)
    h = dt / n
    phi = float(phi0)
    kk = 0.0 if k is None else float(k)
    for _ in range(int(n)):
        s0 = float(s0_0) + kk * (phi - float(phi0))
        phi = max(0.0, phi - (s0 - drive) * h)
    return phi


def shell_freeze_mask(M, Rv, n_active, n_shell, dt, cfl=0.5):
    """哪些壳环还值得当"理想壳"冻结：\(L/R\) 相对时间步够长才行。

    冻结目标的显式更新 \(\Delta\psi=-\Delta t\,R\,I/(2\pi)\) 在
    \(\Delta t\,R/L\gtrsim1\) 时不稳——实测 \(L/R=9.4\ \mu s<\Delta t=30\ \mu s\)
    那组 4 步内就把壳电流推到 \(+3.7\) MA、平衡崩掉。所以
    \(\Delta t\,R/L>\text{cfl}\) 的环退回**回路方程**（隐式欧拉，任何 \(\Delta t\)
    都稳），不再假装它冻得住。
    """
    if n_shell <= 0:
        return np.zeros(0, dtype=bool)
    L = np.abs(np.diag(M)[n_active:n_active + n_shell])
    R = np.abs(np.asarray(Rv, dtype=float)[n_active:n_active + n_shell])
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = dt * R / np.maximum(L, 1e-300)
    theta = np.where(np.isfinite(theta), theta, 0.0)
    return theta <= float(cfl)


def shell_flux_loop(*, model, machine, Il, psi_target, phi_target, guess_iz, prev,
                    step_tol, rtol, max_eval, ci_fixed=None, be=None,
                    be_loop=False, tol=1e-3, max_iter=3, relax=0.7, progress=True,
                    mask=None):
    """**路线 A**：解壳环电流，使各壳环所在面的边界通量等于冻结值。

    与外层 CI 环 / \(B_e\) 环同构：未知量是壳环电流，约束是
    \(\psi_b(\text{环 }l)=\psi_{\rm target,l}\)（\(\psi\) 单位 Wb/2π）。

    * 雅可比 \(J_{lm}=\partial\psi_b(l)/\partial I_{{\rm shell},m}\) 取线圈 Green 基
      里壳那几列（电流到边界通量的线性部分）；等离子体对壳电流的依赖是非线性的，
      靠外层迭代 + 欠松弛 `relax` 收进去。
    * 每次迭代内部仍用 `match_flux` 把 \(\Phi\) 调到目标（壳电流一动，等离子体
      平衡就得重解）。

    返回 `(out, I, iz, phi, n_eval, phi_err, shell_resid)`；`out is None` 表示平衡
    已不存在。`I` 是收敛后的全导体电流（含壳）。

    `mask` 给出**哪些壳环还值得冻结**（见 `shell_freeze_mask`）：`False` 的那些环
    保持回路方程给出的电流、不参与约束迭代（\(L/R\) 小于时间步时本来也冻不住）。
    """
    nodes = np.asarray(machine["shell_nodes"], dtype=int)
    n_active = machine["n_active"]
    n_sh = machine["n_shell"]
    mask = (np.ones(n_sh, dtype=bool) if mask is None
            else np.asarray(mask, dtype=bool).copy())
    rows = np.nonzero(mask)[0]
    basis = model.basis
    J = np.zeros((rows.size, rows.size))
    for a, m in enumerate(rows):
        j = n_active + int(m)
        nr, nz = basis["meta"][j]
        J[a, :] = (MU0 * np.asarray(basis["psiw_sum"][j],
                                    dtype=float)[nodes[rows]] / (nr * nz))
    I = np.array(Il, dtype=float, copy=True)
    tgt = np.asarray(psi_target, dtype=float)[rows]
    # 判据按**逐环相对**（与逐环 ε_l 同一口径）：每环用自己的冻结通量当分母
    tgt_scale = np.maximum(np.abs(tgt), 1e-30)
    out, iz = None, float(guess_iz)
    phi, n_eval, phi_err = float("nan"), 0, float("inf")
    resid = float("inf")
    if rows.size == 0:                     # 没有环值得冻结：只解平衡、不动壳
        out, iz, phi, n_eval, phi_err = match_flux(
            phi_target, iz, model, I, prev, step_tol, rtol, max_eval,
            ci_fixed=ci_fixed, be=be, be_loop=be_loop)
        if out is None:
            return None, I, float("nan"), float("nan"), 0, float("inf"), float("inf")
        return out, I, iz, phi, n_eval, phi_err, 0.0
    for k in range(max(1, int(max_iter)) + 1):
        out, iz, phi, n_eval, phi_err = match_flux(
            phi_target, iz, model, I, prev, step_tol, rtol, max_eval,
            ci_fixed=ci_fixed, be=be, be_loop=be_loop)
        if out is None:
            return None, I, float("nan"), float("nan"), 0, float("inf"), float("inf")
        psi_ring = np.asarray(out["psib"], dtype=float).ravel()[nodes[rows]]
        r = psi_ring - tgt
        resid = (float(np.max(np.abs(r) / tgt_scale)) if rows.size
                 else float("inf"))
        if progress:
            print(f"        壳环 {k + 1}/{max_iter + 1}: "
                  f"max|Δψ_ring|/|ψ_ring| = {resid:.2e}   I_shell = "
                  f"{np.array2string(I[n_active:] / 1e3, precision=2)} kA")
        if resid <= tol or k == max_iter:
            break
        try:
            dI = -np.linalg.solve(J, r)
        except np.linalg.LinAlgError:
            dI = -np.linalg.lstsq(J, r, rcond=None)[0]
        prev = out                       # 暖启动：上一层解直接喂给下一次试探
        I[n_active + rows] += relax * dI
    return out, I, iz, phi, n_eval, phi_err, resid


def coupled_step(*, model, machine, M, Rv, dt, V, I_prev, Psi_pl_prev, dPsi0,
                 phi_target, guess_iz, prev, step_tol, rtol, max_eval,
                 ci_fixed=None, be=None, be_loop=False, use_shell_loop=False,
                 psi_target=None, shell_mask=None, shell_tol=2e-4, shell_max=3,
                 shell_relax=0.7,
                 tol=5e-3, max_iter=4, relax=0.7, progress=True):
    """**同一时间步内**把"回路推进"和"平衡求解"迭代到自洽。

    原来的一步是"用上一步外推的 \(\Delta\Psi_{\rm pl}\) 算一次回路，再解平衡"——
    两者不闭合（实测外推误差 3.8%）。这里改成块不动点：

    $$I^{(k)}\ \to\ \text{平衡解}\ \to\ \Psi_{\rm pl}^{(k)}\ \to\
    I^{(k+1)}=\text{回路}(I^n,\ \Delta\Psi_{\rm pl}^{(k)})$$

    迭代到 \(\Delta\Psi_{\rm pl}\) 不再变化。**收敛判据就是回路方程残差**：
    由 \((M+\Delta tR)I^{n+1}=MI^n+\Delta tV-\Delta\Psi_{\rm pl}\) 可知，
    "用的 \(\Delta\Psi_{\rm pl}\)"与"实测的"之差就等于方程残差，两者代数上相等。

    `use_shell_loop=True` 时，壳环那几行由冻结约束（`shell_flux_loop`）决定，
    回路只负责线圈行；残差也只在线圈行上量（理想壳的壳行由约束接管）。

    返回 `(out, I, iz, phi, n_eval, phi_err, shell_resid, dPsi_used, n_iter,
    resid)`；`out is None` 表示平衡已不存在。
    """
    n_active = machine["n_active"]
    dPsi = np.array(dPsi0, dtype=float, copy=True)
    I = np.array(I_prev, dtype=float, copy=True)
    out, iz = None, float(guess_iz)
    phi, n_eval, phi_err, shell_resid = float("nan"), 0, float("inf"), 0.0
    dPsi_used, change, k = dPsi.copy(), float("inf"), 0
    for k in range(max(1, int(max_iter)) + 1):
        I = advance_circuit(M, Rv, I_prev, dt, V, dPsi)
        if use_shell_loop:
            out, I, iz, phi, n_eval, phi_err, shell_resid = shell_flux_loop(
                model=model, machine=machine, Il=I, psi_target=psi_target,
                phi_target=phi_target, guess_iz=iz, prev=prev,
                step_tol=step_tol, rtol=rtol, max_eval=max_eval,
                ci_fixed=ci_fixed, be=be, be_loop=be_loop,
                tol=shell_tol, max_iter=shell_max, relax=shell_relax,
                progress=False, mask=shell_mask)
        else:
            out, iz, phi, n_eval, phi_err = match_flux(
                phi_target, iz, model, I, prev, step_tol, rtol, max_eval,
                ci_fixed=ci_fixed, be=be, be_loop=be_loop)
        if out is None:
            return (None, I, float("nan"), float("nan"), 0, float("inf"),
                    0.0, dPsi, k, float("inf"))
        rows = slice(0, n_active) if use_shell_loop else slice(None)
        dPsi_new = plasma_flux(out, model) - Psi_pl_prev
        change = float(np.max(np.abs(dPsi_new[rows] - dPsi[rows]))
                       / max(float(np.max(np.abs(dPsi_new[rows]))), 1e-30))
        if progress:
            print(f"        整体迭代 {k + 1}/{max_iter + 1}: "
                  f"回路残差 = {change:.2e}")
        dPsi_used = dPsi.copy()
        dPsi = dPsi + relax * (dPsi_new - dPsi)
        prev, guess_iz = out, iz
        if change <= tol or k == max_iter:
            break
    return (out, I, iz, phi, n_eval, phi_err, shell_resid, dPsi_used,
            k + 1, change)


def newton_step(*, model, machine, M, Rv, dt, V, I_prev, Psi_pl_prev, I_guess,
                 dPsi0, guess_iz, phi_target, prev, step_tol, ci_fixed=None, be=None,
                 be_loop=False, use_shell_loop=False, psi_target=None, shell_mask=None,
                 implicit_phi=False, s0_func=None, phi_prev=None, drive=0.0,
                 s0_ref=None,
                 fd_theta=True,
                 tol=5e-3, max_iter=6, damping=1.0, line_search=3,
                 progress=True):
    """**整体牛顿步**：把"回路 + 平衡 + 壳约束"写进同一个残差向量一次解。

    未知量 \(x=[I,\ \theta]\)（\(\theta=I_{\zeta0}\)，共 \(n+1\) 个）；
    残差 \(F(x)\)（同样 \(n+1\) 个）：

    * 前 \(n_{\rm active}\) 行是线圈回路方程
      \((M+\Delta tR)I-MI^n-\Delta tV+\Delta\Psi_{\rm pl}=0\)；
    * 壳那 \(n_{\rm shell}\) 行：`--shell-loop` 打开时用**冻结约束**
      \(\psi_b(\text{环})=\psi_{\rm target}\)，否则用壳自己的回路方程；
    * 最后一行是通量匹配 \((\Phi-\Phi_{\rm target})/\Phi_{\rm target}=0\)。

    **雅可比用现成的互感矩阵**：线圈行 \(\partial F/\partial I=M+\Delta tR\)
    （`_coil_mutual_matrix` 出来的那支）；壳约束行的灵敏度是线圈 Green 基在
    壳环边界点上的值（也是现成的解析量）；\(\theta\) 那一列用族关系
    \(\Phi\propto I_{\zeta0}^2\) 给初值。之后用 Broyden 秩一更新补上
    "等离子体对电流的隐式响应"这一块（不额外花求解）。

    每次迭代只解**一次**平衡（不像 `coupled_step` 里还套一层割线），
    带线搜索；失败时调用方会退回块迭代。
    """
    n = int(np.shape(M)[0])
    n_active = machine["n_active"]
    n_sh = machine["n_shell"]
    nodes = np.asarray(machine["shell_nodes"], dtype=int)
    shell_mask = (np.ones(n_sh, dtype=bool) if shell_mask is None
                  else np.asarray(shell_mask, dtype=bool))
    frozen = np.nonzero(shell_mask)[0]
    basis = model.basis
    Rdiag = dt * np.asarray(Rv, dtype=float)
    A = np.array(M, dtype=float, copy=True)
    A[np.diag_indices(n)] += Rdiag
    # 壳约束行对电流的灵敏度：∂ψ_b(node)/∂I_j（Green 基，解析）
    GB = np.zeros((max(n_sh, 0), n))
    for j in range(n):
        nr, nz = basis["meta"][j]
        GB[:, j] = MU0 * np.asarray(basis["psiw_sum"][j],
                                    dtype=float)[nodes] / (nr * nz)
    tgt_abs = (np.maximum(np.abs(np.asarray(psi_target, dtype=float)), 1e-30)
               if n_sh else np.zeros(0))
    # 行的归一口径：线圈行按**本步通量变化量**归（不是按线圈自感磁链，
    # 那会把回路方程放松到没约束）；壳行/Φ 行本来就是相对量。
    pscale = max(float(np.max(np.abs(dPsi0))) if np.size(dPsi0) else 0.0,
                 1e-6 * float(np.max(np.abs(M @ I_prev))), 1e-30)
    phtgt = max(abs(phi_target), 1e-30)

    def residual(x):
        I = np.asarray(x[:n], dtype=float)
        th = float(x[n])
        if not np.isfinite(th) or abs(th) < 1e-9:
            return None
        try:
            out = one_solve(model=model, Il=I, izeta0=th, prev=prev,
                            step_tol=step_tol, ci_fixed=ci_fixed, be=be,
                            be_loop=be_loop)
        except Exception as exc:  # noqa: BLE001 - 平衡可能不存在
            if progress and not getattr(residual, "_warned", False):
                print(f"        [牛顿] 平衡求解失败：{type(exc).__name__}: {exc}")
                residual._warned = True
            out = None
        if out is None:
            return None
        Psi_pl = plasma_flux(out, model)
        phi = float(out["diag"]["Phi_private_Wb"])
        psib = np.asarray(out["psib"], dtype=float).ravel()
        CI = float(out["CI"])
        coil_b = np.asarray(out["psib_coil"], dtype=float).ravel()
        psi_pl_b = (psib - CI * coil_b) / CI if CI else psib
        F = np.zeros(n + 1)
        F[:n] = ((M + np.diag(Rdiag)) @ I - (M @ I_prev + dt * V)
                 + (Psi_pl - Psi_pl_prev)) / pscale
        if use_shell_loop and n_sh:
            if frozen.size:
                F[n_active + frozen] = (
                    (psib[nodes[frozen]]
                     - np.asarray(psi_target, float)[frozen]) / tgt_abs[frozen])
        s0 = None
        if implicit_phi and s0_func is not None:
            # **修改版**（第 36 条）：Φ 也隐式——(Φ−Φ^n)/Δt + S0(Φ) − drive = 0
            s0 = float(s0_func(out["psiq"]))
            F[n] = ((phi - phi_prev) / dt + s0 - drive) / s0_ref
        else:
            F[n] = (phi - phi_target) / phtgt
        return out, F, Psi_pl, psi_pl_b, phi, s0

    def jacobian(x, Psi_pl, psi_pl_b, phi, s0=None):
        B = np.zeros((n + 1, n + 1))
        B[:n, :n] = A / pscale
        if use_shell_loop and n_sh:
            if frozen.size:
                B[n_active + frozen, :n] = GB[frozen] / tgt_abs[frozen, None]
        th = float(x[n])
        if abs(th) < 1e-9:            # 本算例 I_zeta0 是负的，不能只判 th<=0
            th = 1e-9 if th >= 0 else -1e-9
        B[:n_active, n] = (2.0 * np.asarray(Psi_pl, float)[:n_active] / th) / pscale
        if use_shell_loop and n_sh:
            if frozen.size:
                B[n_active + frozen, n] = (2.0 * psi_pl_b[nodes[frozen]]
                                           / th) / tgt_abs[frozen]
        if implicit_phi and s0 is not None:
            B[n, n] = ((2.0 * phi / th) / dt + 2.0 * s0 / th) / s0_ref
        else:
            B[n, n] = 2.0 * phi / th / phtgt
        return B

    x = np.concatenate([np.asarray(I_guess, dtype=float).copy(),
                        [float(guess_iz)]])
    res = residual(x)
    if res is None:
        return (None, np.asarray(I_guess, dtype=float), float("nan"),
                float("nan"), 0, float("inf"), float("inf"),
                np.zeros(n), 0)
    out, F, Psi_pl, psi_pl_b, phi, s0 = res
    B = jacobian(x, Psi_pl, psi_pl_b, phi, s0)
    n_eval, it, resid = 1, 0, float(np.max(np.abs(F)))
    if fd_theta:
        # **修改版**（第 36 条）：θ 那一列整体用一次有限差分量出来。
        # 解析估计（∂Φ/∂θ≈2Φ/θ、∂S0/∂θ≈2S0/θ）在隐式 Φ 下不够准，
        # 牛顿残差会振荡（实测 4e-2→4e-2→2e-2→1.7e-2→3.9e-2）而不收敛。
        delta = 1e-3 * max(abs(float(x[n])), 1.0)
        x_p = np.array(x, dtype=float, copy=True)
        x_p[n] += delta
        res_p = residual(x_p)
        n_eval += 1
        if res_p is not None:
            B[:, n] = (res_p[1] - F) / delta
    for it in range(1, max(1, int(max_iter)) + 1):
        try:
            dx = np.linalg.solve(B, -F)
        except np.linalg.LinAlgError:
            dx = -np.linalg.lstsq(B, -F, rcond=None)[0]
        if not np.all(np.isfinite(dx)):
            break
        base = float(np.max(np.abs(F)))
        best = None
        for ls in range(max(0, int(line_search)) + 1):
            w = float(damping) / (2 ** ls)
            x_new = x + w * dx
            res_new = residual(x_new)
            if res_new is None:
                continue
            n_eval += 1
            F_new = res_new[1]
            f_new = float(np.max(np.abs(F_new)))
            if best is None or f_new < best[0]:
                best = (f_new, x_new, res_new, w)
            if f_new <= 0.7 * base:
                break
        if best is None:
            break
        f_new, x_new, res_new, _w = best
        dxs = x_new - x
        # Broyden 秩一更新：补上等离子体对电流的隐式响应
        B = B + np.outer(F_new - F - B @ dxs, dxs) / max(float(dxs @ dxs), 1e-30)
        x, F = x_new, F_new
        out, _, Psi_pl, psi_pl_b, phi, s0 = res_new
        resid = f_new
        if progress:
            print(f"        牛顿 {it}/{max_iter}: max|F| = {resid:.2e}")
        if resid <= tol:
            break
    iz = float(x[n])
    I = np.asarray(x[:n], dtype=float)
    err = (abs(F[n]) if implicit_phi and s0 is not None
           else abs(phi - phi_target) / phtgt)
    # 返回"回路等效用的 ΔΨ"：由 (M+ΔtR)I = MI^n + ΔtV − ΔΨ 反解，
    # 这样调用方按 |实测 ΔΨ − 用的 ΔΨ| 量出来的就是真正的方程残差
    dPsi_used = (M @ I_prev + dt * V) - (M + np.diag(Rdiag)) @ I
    return out, I, iz, phi, n_eval, err, resid, dPsi_used, it


def newton_step_flux(*, model, machine, M, Rv, dt, V, I_prev, Psi_pl_prev,
                     psi_s_prev, I_guess, guess_iz, dPsi0, phi_target, prev,
                     step_tol, ci_fixed=None, be=None, be_loop=False,
                     implicit_phi=False, s0_func=None, phi_prev=None, drive=0.0,
                     s0_ref=None,
                     fd_theta=True,
                     tol=5e-3, max_iter=6, damping=1.0, line_search=3,
                     progress=True):
    """整体牛顿步 + **把壳的冻结通量 \(\psi_s\) 本身作为未知量**。

    未知量 \(x=[I,\ \theta,\ \psi_s]\)（\(n+1+n_{\rm shell}\) 个，与残差同维）：

    * 线圈行（\(n_{\rm active}\)）：回路方程，同第 33 条；
    * **壳链行**（\(n_{\rm shell}\)）：\(\psi_b(\text{环})=\psi_s\)，
      即"平衡看到的边界通量 = 壳的通量状态"；
    * **壳松弛行**（\(n_{\rm shell}\)）：隐式欧拉写
      \(\mathrm{d}\psi/\mathrm{d}t=-RI/(2\pi)\)，即
      \(\psi_s-\psi_s^{\,n}+\Delta t\,R\,I_{\rm shell}/(2\pi)=0\)；
    * \(\Phi\) 行：通量匹配。

    这样做的好处：理想壳（\(R\to0\)）给出 \(\psi_s=\psi_s^{\,n}\)**精确**守恒，
    不再有显式更新的漂移；电阻壳是**隐式**的，\(\Delta t\,R/L\) 再大也稳，
    不需要第 34 条那个 `--shell-freeze-cfl` 切换。
    """
    n = int(np.shape(M)[0])
    n_active = machine["n_active"]
    n_sh = machine["n_shell"]
    nodes = np.asarray(machine["shell_nodes"], dtype=int)
    basis = model.basis
    nrow = n_active + 2 * n_sh + 1
    ncol = n + 1 + n_sh
    Rdiag = dt * np.asarray(Rv, dtype=float)
    A = np.array(M, dtype=float, copy=True)
    A[np.diag_indices(n)] += Rdiag
    GB = np.zeros((n_sh, n))
    for j in range(n):
        nr, nz = basis["meta"][j]
        GB[:, j] = MU0 * np.asarray(basis["psiw_sum"][j],
                                    dtype=float)[nodes] / (nr * nz)
    Rsh = np.asarray(Rv, dtype=float)[n_active:n_active + n_sh]
    psi_s_prev = np.asarray(psi_s_prev, dtype=float)
    s_scale = np.maximum(np.abs(psi_s_prev), 1e-30)
    pscale = max(float(np.max(np.abs(dPsi0))) if np.size(dPsi0) else 0.0,
                 1e-6 * float(np.max(np.abs(M @ I_prev))), 1e-30)
    phtgt = max(abs(phi_target), 1e-30)
    rep = slice(n_active, n_active + n_sh)          # 壳链行
    rer = slice(n_active + n_sh, n_active + 2 * n_sh)  # 壳松弛行

    def residual(x):
        I = np.asarray(x[:n], dtype=float)
        th = float(x[n])
        psis = np.asarray(x[n + 1:], dtype=float)
        if not np.isfinite(th) or abs(th) < 1e-9:
            return None
        try:
            out = one_solve(model=model, Il=I, izeta0=th, prev=prev,
                            step_tol=step_tol, ci_fixed=ci_fixed, be=be,
                            be_loop=be_loop)
        except Exception as exc:  # noqa: BLE001
            if progress and not getattr(residual, "_warned", False):
                print(f"        [隐式壳] 平衡求解失败：{type(exc).__name__}: {exc}")
                residual._warned = True
            out = None
        if out is None:
            return None
        Psi_pl = plasma_flux(out, model)
        phi = float(out["diag"]["Phi_private_Wb"])
        psib = np.asarray(out["psib"], dtype=float).ravel()
        CI = float(out["CI"])
        coil_b = np.asarray(out["psib_coil"], dtype=float).ravel()
        psi_pl_b = (psib - CI * coil_b) / CI if CI else psib
        F = np.zeros(nrow)
        F[:n_active] = ((M + np.diag(Rdiag)) @ I - (M @ I_prev + dt * V)
                        + (Psi_pl - Psi_pl_prev))[:n_active] / pscale
        F[rep] = (psib[nodes] - psis) / s_scale
        F[rer] = (psis - psi_s_prev + dt * Rsh * I[n_active:] / (2 * np.pi)) / s_scale
        s0 = None
        if implicit_phi and s0_func is not None:
            # **修改版**（第 36 条）：Φ 也隐式——(Φ−Φ^n)/Δt + S0(Φ) − drive = 0
            s0 = float(s0_func(out["psiq"]))
            F[-1] = ((phi - phi_prev) / dt + s0 - drive) / s0_ref
        else:
            F[-1] = (phi - phi_target) / phtgt
        return out, F, Psi_pl, psi_pl_b, phi, s0

    def jacobian(x, Psi_pl, psi_pl_b, phi, s0=None):
        B = np.zeros((nrow, ncol))
        B[:n_active, :n] = A[:n_active, :] / pscale
        th = float(x[n])
        if abs(th) < 1e-9:
            th = 1e-9 if th >= 0 else -1e-9
        B[:n_active, n] = (2.0 * np.asarray(Psi_pl, float)[:n_active]
                           / th) / pscale
        B[rep, :n] = GB / s_scale[:, None]
        B[rep, n] = (2.0 * psi_pl_b[nodes] / th) / s_scale
        for m in range(n_sh):
            B[n_active + m, n + 1 + m] = -1.0 / s_scale[m]
        for m in range(n_sh):
            B[n_active + n_sh + m, n_active + m] = (dt * Rsh[m]
                                                    / (2 * np.pi)) / s_scale[m]
            B[n_active + n_sh + m, n + 1 + m] = 1.0 / s_scale[m]
        if implicit_phi and s0 is not None:
            B[-1, n] = ((2.0 * phi / th) / dt + 2.0 * s0 / th) / s0_ref
        else:
            B[-1, n] = 2.0 * phi / th / phtgt
        return B

    x = np.concatenate([np.asarray(I_guess, dtype=float).copy(),
                        [float(guess_iz)], psi_s_prev.copy()])
    res = residual(x)
    if res is None:
        return (None, np.asarray(I_guess, dtype=float), float("nan"),
                float("nan"), 0, float("inf"), float("inf"),
                np.zeros(n), 0, psi_s_prev.copy())
    out, F, Psi_pl, psi_pl_b, phi, s0 = res
    B = jacobian(x, Psi_pl, psi_pl_b, phi, s0)
    n_eval, it, resid = 1, 0, float(np.max(np.abs(F)))
    if fd_theta:
        # **修改版**（第 36 条）：θ 那一列整体用一次有限差分量出来。
        # 解析估计（∂Φ/∂θ≈2Φ/θ、∂S0/∂θ≈2S0/θ）在隐式 Φ 下不够准，
        # 牛顿残差会振荡而不收敛（实测 3.8e-2→4.2e-2→2.3e-2→…→3.3e-2）。
        delta = 1e-3 * max(abs(float(x[n])), 1.0)
        x_p = np.array(x, dtype=float, copy=True)
        x_p[n] += delta
        res_p = residual(x_p)
        n_eval += 1
        if res_p is not None:
            B[:, n] = (res_p[1] - F) / delta
    for it in range(1, max(1, int(max_iter)) + 1):
        try:
            dx = np.linalg.solve(B, -F)
        except np.linalg.LinAlgError:
            dx = -np.linalg.lstsq(B, -F, rcond=None)[0]
        if not np.all(np.isfinite(dx)):
            break
        base = float(np.max(np.abs(F)))
        best = None
        for ls in range(max(0, int(line_search)) + 1):
            w = float(damping) / (2 ** ls)
            x_new = x + w * dx
            if x_new[n + 1:].min() <= 0 and psi_s_prev.min() > 0:
                pass                        # 通量可为负（本算例 Ψ_pl 为负），不拦
            res_new = residual(x_new)
            if res_new is None:
                continue
            n_eval += 1
            F_new = res_new[1]
            f_new = float(np.max(np.abs(F_new)))
            if best is None or f_new < best[0]:
                best = (f_new, x_new, res_new, w)
            if f_new <= 0.7 * base:
                break
        if best is None:
            break
        f_new, x_new, res_new, _w = best
        dxs = x_new - x
        B = B + np.outer(F_new - F - B @ dxs, dxs) / max(float(dxs @ dxs), 1e-30)
        x, F = x_new, F_new
        out, _, Psi_pl, psi_pl_b, phi, s0 = res_new
        resid = f_new
        if progress:
            print(f"        隐式壳牛顿 {it}/{max_iter}: max|F| = {resid:.2e}")
        if resid <= tol:
            break
    iz = float(x[n])
    I = np.asarray(x[:n], dtype=float)
    psi_s_new = np.asarray(x[n + 1:], dtype=float)
    err = (abs(F[-1]) if implicit_phi and s0 is not None
           else abs(phi - phi_target) / phtgt)
    dPsi_used = (M @ I_prev + dt * V) - (M + np.diag(Rdiag)) @ I
    return (out, I, iz, phi, n_eval, err, resid, dPsi_used, it, psi_s_new)


def verify_circuit(machine, args) -> None:
    """(a)(b)：纯回路自检——自由衰减的 L/R 时间、稳态 V=RI、能量平衡。"""
    M, Rv = machine["M"], machine["Rvec"]
    n = M.shape[0]
    I0 = np.asarray(machine["s"]["Il"], dtype=float)
    dt = 1e-5
    print("[verify] (a) 纯回路自由衰减：I(t) 应满足 L/R 时间（无等离子体项）")
    for j in range(n):
        L, R = M[j, j], Rv[j]
        I = np.zeros(n); I[j] = 1.0
        for _ in range(int(2e-3 / dt)):
            I = advance_circuit(M, Rv, I, dt, np.zeros(n), np.zeros(n))
        ratio = I[j] / 1.0
        print(f"    第 {j} 个导体：L={L:.3e} H  R={R:.3e} Ohm  L/R={L / R * 1e3:8.3f} ms"
              f"  2 ms 后 I/I0={ratio:.4f}（解析 exp(-2ms/(L/R)) 见下）"
              f"  解析={np.exp(-2e-3 / (L / R)):.4f}")
    print("    （注意：M 非对角，孤立衰减只在耦合弱时接近单指数；"
          "这里能对上说明回路矩阵与阻抗的符号/量级没问题）")

    print("[verify] (b) 稳态：V=RI0 时电流应站住（无等离子体项）")
    V = Rv * I0
    I = I0.copy()
    for _ in range(200):
        I = advance_circuit(M, Rv, I, dt, V, np.zeros(n))
    # 壳的初始电流是 0，不能用 |I0| 当分母
    den = max(float(np.max(np.abs(I0))), 1e-30)
    print(f"    max|I-I0| / max|I0| = {np.max(np.abs(I - I0)) / den:.2e}")

    print("[verify] (c) 能量平衡：Δ(½IᵀMI) = VᵀI·dt − IᵀRI·dt（一路积 1 ms）")
    I = I0.copy(); e_supply = e_res = 0.0
    E0 = 0.5 * I @ M @ I
    for _ in range(100):
        e_supply += float(V @ I) * dt
        e_res += float(I @ (Rv * I)) * dt
        I = advance_circuit(M, Rv, I, dt, V, np.zeros(n))
    E1 = 0.5 * I @ M @ I
    scale = max(abs(e_supply), abs(e_res), abs(E1 - E0), 1e-30)
    print(f"    供给 VᵀI·dt = {e_supply:+.6e} J；耗散 IᵀRI·dt = {e_res:+.6e} J；"
          f"ΔE 储能 = {E1 - E0:+.6e} J")
    print(f"    闭合差 ΔE-(供给−耗散) = {E1 - E0 - (e_supply - e_res):.3e} J，"
          f"相对最大能量项 {abs(E1 - E0 - (e_supply - e_res)) / scale:.1e}"
          f"（压到机器精度量级即通过；稳态下 ΔE 本身 ~0，不能用它当分母）")


def run(*, args, machine, eta, drive=0.0, voltage="ri", steps=None, dt=None,
        progress=True):
    """主循环：回路 →（新边界通量）→ 平衡（割线调 Iz0 匹配目标 Φ）→ 记录。"""
    s, model, M, Rv = machine["s"], machine["model"], machine["M"], machine["Rvec"]
    I = np.asarray(s["Il"], dtype=float).copy()
    I0 = I.copy()
    n_active = machine["n_active"]
    steps = int(args.steps if steps is None else steps)
    dt = float(args.dt if dt is None else dt)
    V = Rv * I0 if voltage == "ri" else np.zeros_like(I)

    def record(step, t, S0, spread, phi_target, out, iz, phi, n_eval, err, wall,
               I, Psi_pl, dE):
        d = out["diag"]
        return dict(
            step=step, t=t, S0=S0, S0_spread=spread, Phi=phi, Phi_target=phi_target,
            Phi_err=err, Izeta0=iz, I=np.array(I, copy=True),
            I_shell=(float(np.sum(I[n_active:])) if machine["n_shell"] else 0.0),
            Psi_pl=np.array(Psi_pl, copy=True), E=0.5 * float(I @ M @ I), dE=dE,
            Rs=d["Rs_mid"], Bz0=d["Bz_axis_mid"], beta=d["beta_avg_sep"],
            W=d["W_sep"], psi_wall=d["psi_wall_mid"], CI=float(out["CI"]),
            gs_res=float(out["gs_res"]), n_eval=n_eval, wall=wall,
            psiq=out["psiq"], psib=out["psib"], secant_state=out.get("secant_state"),
            dpsi_step=out.get("dpsi_step"))

    t0 = time.perf_counter()
    be_loop = (args.flux_norm == "be-loop")
    ci_fixed = (1.0 if (args.flux_norm in ("physical", "be-loop")
                        or args.be_from_coils) else None)
    be = None
    if args.be_from_coils:
        be, _Bz = coil_edge_field(model, I)
        print(f"[stage04] 物理口径：剖面参考场取线圈给出的真实外场 "
              f"B_ext={be:.4f} T（原来是 B_e={float(s['Be']):.4f} T）")
    out = one_solve(model=model, Il=I, izeta0=float(s0_izeta0(s)), prev=None,
                    step_tol=None, ci_fixed=ci_fixed, be=be, be_loop=be_loop)
    pr = model.problem(model.coil_flux(I)[3], 2)
    phi = float(out["diag"]["Phi_private_Wb"])
    S0, spread = flux_loss_rate(out["psiq"], pr, eta)
    Psi_pl = plasma_flux(out, model)
    # **修改版**（路线 A）：壳环所在面的冻结通量与逐环误差
    nodes = (np.asarray(machine["shell_nodes"], dtype=int) if machine["n_shell"]
             else np.zeros(0, dtype=int))
    psi_ring0 = (np.asarray(out["psib"], dtype=float).ravel()[nodes]
                 if machine["n_shell"] else np.zeros(0, dtype=float))
    use_shell_loop = (bool(getattr(args, "shell_loop", False))
                      and machine["n_shell"] > 0)
    # **修改版**（第 34 条）：L/R 小于时间步的壳环冻不住，退回回路行
    shell_freeze = (shell_freeze_mask(M, Rv, n_active, machine["n_shell"], dt,
                                      float(getattr(args, "shell_freeze_cfl", 0.5)))
                    if (use_shell_loop and machine["n_shell"])
                    else np.zeros(machine["n_shell"], dtype=bool))
    psi_target = psi_ring0.copy()
    # **修改版**（第 35 条）：壳通量状态（隐式松弛用）
    flux_state_mode = (bool(getattr(args, "shell_flux_state", False))
                       and use_shell_loop and machine["n_shell"] > 0)
    psi_s_state = psi_ring0.copy()
    hist = [record(0, 0.0, S0, spread, phi, out, float(s0_izeta0(s)), phi, 0, 0.0,
                   time.perf_counter() - t0, I, Psi_pl, 0.0)]
    hist[0]["Psi_ring"] = psi_ring0.copy()
    hist[0]["eps_ring"] = np.zeros_like(psi_ring0)
    hist[0]["psi_target_ring"] = psi_target.copy()
    hist[0]["shell_resid"] = 0.0
    hist[0]["circ_resid"] = 0.0
    hist[0]["coupled_iter"] = 0
    phi0 = phi
    # **修改版**（第 37 条）：已接受状态的 (Φ, S0) 样本，供驱动项子步估局部斜率
    phi_trk, s0_trk = [float(phi)], [float(S0)]
    t = 0.0
    if progress:
        print(f"[stage04] 初态：Φ={phi:.4e} Wb  Rs={hist[0]['Rs'] * 1e2:.3f} cm  "
              f"psi_wall={hist[0]['psi_wall']:.4e}  S0={S0:.4f} Wb/s")
        if machine["n_shell"]:
            print(f"          等离子体在壳上的磁链 Ψ_pl(壳) = "
                  f"{np.array2string(Psi_pl[n_active:], precision=3)} Wb")
            print(f"          壳环冻结通量 ψ_ring0 = "
                  f"{np.array2string(psi_ring0, precision=4)}（Wb/2π）"
                  f"{'   [壳环通量不动点：开]' if use_shell_loop else ''}")
            if use_shell_loop:
                print(f"          冻结掩码（dt*R/L ≤ {args.shell_freeze_cfl}）："
                      f"{np.array2string(shell_freeze)}"
                      f"   L/R [ms] = {np.array2string(np.abs(np.diag(M)[n_active:]) / np.maximum(np.abs(Rv[n_active:]), 1e-300) * 1e3, precision=3)}")
    for k in range(1, steps + 1):
        ts = time.perf_counter()
        S0, spread = flux_loss_rate(hist[-1]["psiq"], pr, eta)
        phi_now = float(hist[-1]["Phi"])
        phi_trk.append(phi_now)
        s0_trk.append(S0)
        phi_target = max(0.0, phi_now - (S0 - drive) * dt)
        if phi_target <= 0.0:
            print(f"[stage04] t={t + dt:.4e} s：Φ 已归零，结束")
            break
        # 等离子体磁链的一步外推。**别用 sqrt(Φ) 的族关系**：Ψ_pl 依赖电流分布，
        # 实测它比 sqrt(Φ) 掉得快得多（每步比值 0.78 vs 0.96）——用错了会让壳的
        # 补偿电流小 5 倍（第一版就是这么错的）。改用**上一步实测的变化率**外推；
        # 第一步没有历史，退化为族关系。
        Psi_prev = hist[-1]["Psi_pl"]
        if len(hist) >= 3:
            old = hist[-2]["Psi_pl"]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(np.abs(old) > 1e-12, Psi_prev / old, 1.0)
            ratio = np.where(np.isfinite(ratio), ratio, 1.0)
        else:
            ratio = np.full_like(Psi_prev, np.sqrt(phi_target / phi_now))
        Psi_pl_next = Psi_prev * ratio
        dPsi = Psi_pl_next - hist[-1]["Psi_pl"]     # 迭代初猜 / 基线用的外推值
        E0 = 0.5 * float(I @ M @ I)
        guess = float(hist[-1]["Izeta0"]) * np.sqrt(phi_target / phi_now)
        shell_resid, coupled_iter = 0.0, 0
        dPsi_used = np.array(dPsi, dtype=float, copy=True)
        if voltage == "hold":          # 理想电源：电流被强行钉住（第 21 条极限）
            I = I0.copy()
            dPsi = np.zeros_like(dPsi)
            dPsi_used = dPsi.copy()
            if use_shell_loop:
                out, I, iz, phi, n_eval, err, shell_resid = shell_flux_loop(
                    model=model, machine=machine, Il=I, psi_target=psi_target,
                    phi_target=phi_target, guess_iz=guess, prev=hist[-1],
                    step_tol=args.step_tol, rtol=args.rtol,
                    max_eval=args.max_eval, ci_fixed=ci_fixed, be=be,
                    be_loop=be_loop, tol=args.shell_loop_tol,
                    max_iter=args.shell_loop_max,
                    relax=args.shell_loop_relax, progress=progress,
                    mask=shell_freeze)
            else:
                out, iz, phi, n_eval, err = match_flux(
                    phi_target, guess, model, I, hist[-1], args.step_tol,
                    args.rtol, args.max_eval, ci_fixed=ci_fixed, be=be,
                    be_loop=be_loop)
        elif bool(getattr(args, "coupled", False)):
            # **修改版**（第 32 条）：同一时间步内把回路推进与平衡求解迭代到自洽
            (out, I, iz, phi, n_eval, err, shell_resid, dPsi_used,
             coupled_iter, _cpl_resid) = coupled_step(
                model=model, machine=machine, M=M, Rv=Rv, dt=dt, V=V,
                I_prev=hist[-1]["I"], Psi_pl_prev=hist[-1]["Psi_pl"],
                dPsi0=dPsi, phi_target=phi_target, guess_iz=guess,
                prev=hist[-1], step_tol=args.step_tol, rtol=args.rtol,
                max_eval=args.max_eval, ci_fixed=ci_fixed, be=be,
                be_loop=be_loop, use_shell_loop=use_shell_loop,
                psi_target=psi_target, shell_mask=shell_freeze,
                shell_tol=args.shell_loop_tol,
                shell_max=args.shell_loop_max,
                shell_relax=args.shell_loop_relax, tol=args.coupled_tol,
                max_iter=args.coupled_max, relax=args.coupled_relax,
                progress=progress)
        elif flux_state_mode:
            # **修改版**（第 35 条）：壳通量作为未知量，隐式松弛 dΨ/dt=-RI
            # **修改版**（第 36 条）：`--implicit-flux` 时再套一层**外层修正**，
            # 把 Φ 的目标迭代到与 S0(Φ^{n+1}) 自洽（隐式欧拉），见文件头说明。
            I_pred = advance_circuit(M, Rv, hist[-1]["I"], dt, V, dPsi)
            n_corr = (int(getattr(args, "flux_corrector", 2))
                      if args.implicit_flux else 0)
            # **修改版**（第 37 条）：驱动项子步——用最近两个 (Φ,S0) 样本的局部斜率
            n_sub = max(1, int(getattr(args, "substeps", 1)))
            k_lin = None
            if n_sub > 1 and len(phi_trk) >= 2:
                dphi_ = phi_trk[-1] - phi_trk[-2]
                if abs(dphi_) > 1e-12:
                    k_lin = (s0_trk[-1] - s0_trk[-2]) / dphi_
            if n_sub > 1:
                phi_target = substep_phi(phi_now, S0, k_lin, drive, dt, n_sub)
            for _c in range(n_corr + 1):
                (out, I, iz, phi, n_eval, err, _nres, dPsi_used, coupled_iter,
                 psi_s_state) = newton_step_flux(
                    model=model, machine=machine, M=M, Rv=Rv, dt=dt, V=V,
                    I_prev=hist[-1]["I"], Psi_pl_prev=hist[-1]["Psi_pl"],
                    psi_s_prev=psi_s_state, I_guess=I_pred, guess_iz=guess,
                    dPsi0=dPsi, phi_target=phi_target, prev=hist[-1],
                    step_tol=args.step_tol, ci_fixed=ci_fixed, be=be,
                    be_loop=be_loop, tol=args.newton_tol,
                    max_iter=args.newton_max, damping=args.newton_damp,
                    line_search=args.newton_line_search,
                    implicit_phi=bool(getattr(args, "implicit_flux_row", False)),
                    s0_func=lambda psi2: flux_loss_rate(psi2, pr, eta)[0],
                    phi_prev=phi_now, drive=drive,
                    s0_ref=max(abs(S0), abs(drive), 1e-30),
                    progress=progress)
                if out is None or _c == n_corr:
                    break
                # 用新平衡的 S0 重算隐式目标；修正量落进 S0 自身的噪声底就停
                s0_new, s0_spread = flux_loss_rate(out["psiq"], pr, eta)
                if n_sub > 1:
                    dphi_ = phi - phi_now
                    k_loc = (k_lin if abs(dphi_) < 1e-12
                             else (s0_new - S0) / dphi_)
                    phi_new = substep_phi(phi_now, S0, k_loc, drive, dt, n_sub)
                else:
                    phi_new = max(0.0, phi_now - (s0_new - drive) * dt)
                rel = abs(phi_new - phi_target) / max(phi_target, 1e-30)
                if rel <= max(float(args.flux_corr_tol), float(s0_spread)):
                    break
                if progress:
                    print(f"        Φ 修正 {_c + 1}/{n_corr}: ΔΦ/Φ = {rel:.2e}"
                          f"（S0 噪声底 {s0_spread:.1e}）")
                phi_target = phi_new
                guess = iz
                I_pred = np.array(I, dtype=float, copy=True)
            shell_resid = 0.0
            if out is None and progress:
                print("        隐式壳牛顿失败，退回整体块迭代")
            if out is None:
                (out, I, iz, phi, n_eval, err, shell_resid, dPsi_used,
                 coupled_iter, _cpl_resid) = coupled_step(
                    model=model, machine=machine, M=M, Rv=Rv, dt=dt, V=V,
                    I_prev=hist[-1]["I"], Psi_pl_prev=hist[-1]["Psi_pl"],
                    dPsi0=dPsi, phi_target=phi_target, guess_iz=guess,
                    prev=hist[-1], step_tol=args.step_tol, rtol=args.rtol,
                    max_eval=args.max_eval, ci_fixed=ci_fixed, be=be,
                    be_loop=be_loop, use_shell_loop=use_shell_loop,
                    psi_target=psi_target, shell_mask=shell_freeze,
                    shell_tol=args.shell_loop_tol,
                    shell_max=args.shell_loop_max,
                    shell_relax=args.shell_loop_relax, tol=args.coupled_tol,
                    max_iter=args.coupled_max, relax=args.coupled_relax,
                    progress=progress)
        elif bool(getattr(args, "newton", False)):
            # **修改版**（第 33 条）：回路 + 平衡 + 壳约束 合成一个整体牛顿步
            I_pred = advance_circuit(M, Rv, hist[-1]["I"], dt, V, dPsi)
            (out, I, iz, phi, n_eval, err, _nres, dPsi_used,
             coupled_iter) = newton_step(
                model=model, machine=machine, M=M, Rv=Rv, dt=dt, V=V,
                I_prev=hist[-1]["I"], Psi_pl_prev=hist[-1]["Psi_pl"],
                I_guess=I_pred, guess_iz=guess, phi_target=phi_target,
                dPsi0=dPsi,
                prev=hist[-1], step_tol=args.step_tol, ci_fixed=ci_fixed,
                be=be, be_loop=be_loop, use_shell_loop=use_shell_loop,
                psi_target=psi_target, shell_mask=shell_freeze, tol=args.newton_tol,
                max_iter=args.newton_max, damping=args.newton_damp,
                line_search=args.newton_line_search,
                implicit_phi=bool(args.implicit_flux),
                s0_func=lambda psi2: flux_loss_rate(psi2, pr, eta)[0],
                phi_prev=phi_now, drive=drive,
                s0_ref=max(abs(S0), abs(drive), 1e-30),
                progress=progress)
            shell_resid = 0.0
            if out is None and progress:
                print("        牛顿步失败，退回整体块迭代")
            if out is None:
                (out, I, iz, phi, n_eval, err, shell_resid, dPsi_used,
                 coupled_iter, _cpl_resid) = coupled_step(
                    model=model, machine=machine, M=M, Rv=Rv, dt=dt, V=V,
                    I_prev=hist[-1]["I"], Psi_pl_prev=hist[-1]["Psi_pl"],
                    dPsi0=dPsi, phi_target=phi_target, guess_iz=guess,
                    prev=hist[-1], step_tol=args.step_tol, rtol=args.rtol,
                    max_eval=args.max_eval, ci_fixed=ci_fixed, be=be,
                    be_loop=be_loop, use_shell_loop=use_shell_loop,
                    psi_target=psi_target, shell_mask=shell_freeze,
                    shell_tol=args.shell_loop_tol,
                    shell_max=args.shell_loop_max,
                    shell_relax=args.shell_loop_relax, tol=args.coupled_tol,
                    max_iter=args.coupled_max, relax=args.coupled_relax,
                    progress=progress)
        else:
            I = advance_circuit(M, Rv, hist[-1]["I"], dt, V, dPsi)
            dPsi_used = np.array(dPsi, dtype=float, copy=True)
            if use_shell_loop:
                out, I, iz, phi, n_eval, err, shell_resid = shell_flux_loop(
                    model=model, machine=machine, Il=I, psi_target=psi_target,
                    phi_target=phi_target, guess_iz=guess, prev=hist[-1],
                    step_tol=args.step_tol, rtol=args.rtol,
                    max_eval=args.max_eval, ci_fixed=ci_fixed, be=be,
                    be_loop=be_loop, tol=args.shell_loop_tol,
                    max_iter=args.shell_loop_max,
                    relax=args.shell_loop_relax, progress=progress,
                    mask=shell_freeze)
            else:
                out, iz, phi, n_eval, err = match_flux(
                    phi_target, guess, model, I, hist[-1], args.step_tol,
                    args.rtol, args.max_eval, ci_fixed=ci_fixed, be=be,
                    be_loop=be_loop)
        dE = 0.5 * float(I @ M @ I) - E0
        if out is None:
            print(f"[stage04] t={t + dt:.4e} s：平衡已不存在（反场区消失），演化结束。"
                  f"此时 Φ/Φ0={phi_target / phi0:.4f}")
            break
        # **修改版**（第 38 条）：平衡消失的判定要看**物理诊断量**，不能只看求解器
        # 有没有抛异常——反场区撑不住时它会"收敛"到真空态（Φ=0、Rs 变 nan），
        # 旧代码会把这一步照记不误。
        _diag = out["diag"]
        if (not np.isfinite(float(_diag.get("Rs_mid", np.nan)))
                or phi <= 1e-6 * phi0):
            print(f"[stage04] t={t + dt:.4e} s：平衡已不存在"
                  f"（Φ/Φ0={phi / phi0:.4f}，Rs 诊断量非有限），演化结束。")
            break
        t += dt
        Psi_pl = plasma_flux(out, model)
        psi_ring = (np.asarray(out["psib"], dtype=float).ravel()[nodes]
                    if machine["n_shell"] else np.zeros(0, dtype=float))
        eps_ring = (np.abs(psi_ring - psi_ring0)
                    / np.maximum(np.abs(psi_ring0), 1e-30)
                    if machine["n_shell"] else np.zeros(0, dtype=float))
        if flux_state_mode:      # 用隐式壳通量状态量报告冻结误差
            psi_ring = np.asarray(psi_s_state, dtype=float).copy()
            eps_ring = np.abs(psi_ring - psi_ring0) / np.maximum(
                np.abs(psi_ring0), 1e-30)
        pr = model.problem(model.coil_flux(I)[3], 2)
        rec = record(k, t, S0, spread, phi_target, out, iz, phi, n_eval, err,
                     time.perf_counter() - ts, I, Psi_pl, dE)
        rec["Psi_ring"] = psi_ring
        rec["eps_ring"] = eps_ring
        rec["psi_target_ring"] = psi_target.copy()
        rec["shell_resid"] = shell_resid
        # 回路方程残差：本步"用的 ΔΨ_pl"与"实测的 ΔΨ_pl"之差（代数上二者相等）
        rows = (slice(0, n_active) if use_shell_loop else slice(None))
        d_act = Psi_pl - hist[-1]["Psi_pl"]
        d_use = np.asarray(dPsi_used, dtype=float)
        circ_resid = float(np.max(np.abs(d_act[rows] - d_use[rows]))
                           / max(float(np.max(np.abs(d_act[rows]))), 1e-30))
        rec["circ_resid"] = circ_resid
        rec["coupled_iter"] = coupled_iter
        # 预测 vs 实测：壳/线圈接收到的等离子体磁链变了多少（一致性诊断）
        denom = max(float(np.max(np.abs(hist[-1]["Psi_pl"]))), 1e-30)
        rec["Psi_pl_pred_err"] = float(
            np.max(np.abs(Psi_pl_next - Psi_pl)) / denom)
        hist.append(rec)
        if progress:
            ring_txt = ("" if not eps_ring.size else
                        "  ε_ring=[" + ", ".join(f"{v:.1e}" for v in eps_ring) + "]")
            print(f"  [{k:2d}/{steps}] t={t * 1e3:7.4f} ms  Φ={phi:.4e}"
                  f"（目标 {phi_target:.4e}） Iz0={iz / 1e3:8.2f} kA"
                  f"  Rs={rec['Rs'] * 1e2:.3f} cm  Bz0={rec['Bz0']:+.4f} T"
                  f"  Ι_coil/kA=[{', '.join(f'{v / 1e3:.2f}' for v in I[:n_active])}]"
                  f"  I_shell={rec['I_shell'] / 1e3:+.2f} kA"
                  f"  ψw={rec['psi_wall']:.4e}{ring_txt}"
                  f"  回路残差={circ_resid:.1e}"
                  f"（迭代 {coupled_iter}）  Ψpl预测误差={rec['Psi_pl_pred_err']:.1e}"
                  f"  {rec['wall']:.2f}s")
        # **修改版**（路线 A）：目标环通量按壳的电阻松弛（理想壳 R≈0，几乎不动）
        if use_shell_loop and args.shell != "ideal":
            if not flux_state_mode:      # 通量状态模式由隐式松弛接管
                upd = np.zeros_like(psi_target)
                upd[shell_freeze] = (-dt * Rv[n_active:][shell_freeze]
                                     * I[n_active:][shell_freeze] / (2 * np.pi))
                psi_target = psi_target + upd
        if phi < 1e-3 * phi0:
            print("[stage04] Φ 已衰减到 1e-3 以下，提前结束")
            break
        if err > 10 * args.rtol:
            print(f"[stage04] Φ 跟踪误差 {err:.1e} 超过容差，在 t={t * 1e3:.4f} ms 停下")
            break
    return hist


def s0_izeta0(s):
    """从 setup 取初始 \(I_{\zeta0}\)。"""
    return float(s.get("Izeta0", -451325.7))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--dt", type=float, default=3e-5, help="时间步 [s]")
    ap.add_argument("--te", type=float, default=50.0, help="电子温度 [eV]")
    ap.add_argument("--eta", type=float, default=None, help="直接给电阻率 [Ohm*m]")
    ap.add_argument("--shell", choices=("none", "resistive", "ideal"),
                    default="resistive", help="导体壳：无 / 有限电阻 / 理想")
    ap.add_argument("--flux-norm", choices=("ci", "physical", "be-loop"),
                    default="ci",
                    help="磁链口径：ci=沿用压强平衡归一化（第 19–22 条）；"
                         "physical=钉住 CI=1（压强平衡变预测）；"
                         "be-loop=钉住 CI=1 并把 B_e 当未知量反解（物理磁链+压强平衡，"
                         "第 24 条的 C 路线）")
    ap.add_argument("--be-from-coils", action="store_true",
                    help="把剖面参考场 B_e 换成线圈真空场在中平面壁面的值"
                         "（真实外场），同时自动钉住 CI=1——物理口径的完整版")
    ap.add_argument("--n-shell", type=int, default=4, help="壳离散成几个环")
    ap.add_argument("--shell-t", type=float, default=5e-3, help="壳厚（径向）[m]")
    ap.add_argument("--shell-rho", type=float, default=1.7e-8, help="壳电阻率 [Ohm*m]")
    ap.add_argument("--shell-loop", action="store_true",
                    help="路线 A：壳环通量不动点——把冻结约束当约束施加到平衡上")
    ap.add_argument("--shell-loop-tol", type=float, default=1e-3,
                    help="壳环通量残差容差（相对初值）")
    ap.add_argument("--shell-loop-max", type=int, default=3,
                    help="壳环外层迭代次数上限")
    ap.add_argument("--shell-loop-relax", type=float, default=0.7,
                    help="壳环电流更新的欠松弛因子")
    ap.add_argument("--shell-freeze-cfl", type=float, default=0.5,
                    help="壳环冻结的稳定性上限：dt*R/L 超过它就不再施加冻结约束"
                         "（退回回路方程；壳已经近似透明）")
    ap.add_argument("--shell-flux-state", action="store_true",
                    help="把壳的冻结通量作为未知量、用隐式欧拉写电阻松弛"
                         "（dΨ/dt=-RI）——任意 dt·R/L 都稳，不需要 freeze-cfl"
                         "（需与 --shell-loop 同时使用）")
    ap.add_argument("--implicit-flux", action="store_true",
                    help="等离子体通量也隐式：用外层修正把 Φ 目标迭代到与 "
                         "S0(Φ^{n+1}) 自洽（不再用步首 S0 显式推目标）"
                         "（配合 --shell-flux-state）")
    ap.add_argument("--flux-corrector", type=int, default=2,
                    help="隐式 Φ 的外层修正次数上限")
    ap.add_argument("--flux-corr-tol", type=float, default=2e-2,
                    help="隐式 Φ 外层修正的相对收敛判据（默认与 S0 的噪声底取大者）")
    ap.add_argument("--implicit-flux-row", action="store_true",
                    help="[实验] 把隐式 Φ 直接写进牛顿残差行（不用外层修正）。"
                         "实测在近维持配置下会被 S0 的 O 点噪声卡住，收敛不到 1e-3，"
                         "保留是为了复现这个负面结果")
    ap.add_argument("--substeps", type=int, default=1,
                    help="驱动项子步：把 dΦ/dt=-(S0-drive) 在步内用 N 段细步积分"
                         "（局部 S0(Φ) 线性模型，不再解平衡），N=1 为关")
    ap.add_argument("--tag", type=str, default=None,
                    help="产物文件名前缀（多组对照时避免互相覆盖）")
    ap.add_argument("--coupled", action="store_true",
                    help="同一时间步内把回路推进与平衡求解迭代到自洽"
                         "（消掉 ΔΨ_pl 的一步外推）")
    ap.add_argument("--coupled-tol", type=float, default=5e-3,
                    help="回路方程残差容差（相对实测 ΔΨ_pl）")
    ap.add_argument("--coupled-max", type=int, default=4,
                    help="步内整体迭代次数上限")
    ap.add_argument("--coupled-relax", type=float, default=0.7,
                    help="步内 ΔΨ_pl 更新的欠松弛因子")
    ap.add_argument("--newton", action="store_true",
                    help="整体牛顿步：回路 + 平衡 + 壳约束写进同一残差，"
                         "互感矩阵当雅可比（Broyden 补等离子体响应）")
    ap.add_argument("--newton-tol", type=float, default=5e-3,
                    help="牛顿残差的收敛判据（相对量）")
    ap.add_argument("--newton-max", type=int, default=6,
                    help="牛顿迭代次数上限")
    ap.add_argument("--newton-damp", type=float, default=1.0,
                    help="牛顿步长因子（<1 为欠松弛）")
    ap.add_argument("--newton-line-search", type=int, default=3,
                    help="线搜索折半次数上限")
    ap.add_argument("--coil-L", type=str, default=None,
                    help="逗号分隔的自感 [H]（覆盖几何估计；前 nl 个）")
    ap.add_argument("--coil-R", type=str, default=None,
                    help="逗号分隔的电阻 [Ohm]（覆盖几何估计）")
    ap.add_argument("--coil-area", type=float, default=None, help="统一导体截面积 [m^2]")
    ap.add_argument("--voltage", choices=("ri", "zero", "hold"), default="ri",
                    help="线圈供电：ri=V=RI0（维持）；zero=V=0（短路）；hold=理想钉住")
    ap.add_argument("--drive-rate", type=float, default=0.0, help="外部通量驱动 [Wb/s]")
    ap.add_argument("--no-plasma-coupling", action="store_true",
                    help="不建线圈↔等离子体耦合（省 ~30 s，回路只受自身驱动）")
    ap.add_argument("--rtol", type=float, default=2e-3)
    ap.add_argument("--max-eval", type=int, default=4)
    ap.add_argument("--step-tol", type=float, default=5e-3)
    ap.add_argument("--verify", action="store_true", help="只跑回路自检（很快）")
    args = ap.parse_args()

    eta = (float(args.eta) if args.eta is not None
           else spitzer_resistivity(args.te))
    s0 = gcm._setup(gcm.SRC_DIR)
    print(f"[stage04] eta={eta:.3e} Ohm*m（T_e={args.te:g} eV）；"
          f"线圈↔等离子体耦合：{'关' if args.no_plasma_coupling else '开'}")
    machine = build_machine(
        s0, shell=args.shell, n_shell=args.n_shell, shell_t=args.shell_t,
        shell_rho=args.shell_rho, coil_area=args.coil_area, coil_L=args.coil_L,
        coil_R=args.coil_R, plasma_coupling=not args.no_plasma_coupling)

    if args.verify:
        verify_circuit(machine, args)
        return 0

    hist = run(args=args, machine=machine, eta=eta, drive=args.drive_rate,
               voltage=args.voltage)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{args.tag}_" if args.tag else ""
    if machine["n_shell"]:
        e_max = max(float(np.max(h["eps_ring"])) for h in hist)
        w0, w1 = float(hist[0]["psi_wall"]), float(hist[-1]["psi_wall"])
        print(f"\n[stage04] 小结：n_shell={machine['n_shell']}  "
              f"shell_loop={'on' if args.shell_loop else 'off'}  "
              f"max ε_ring={e_max:.3e}  "
              f"中平面 ψ_wall 漂移={(w1 - w0) / abs(w0):+.2%}")
    if len(hist) > 1:
        r_max = max(float(h["circ_resid"]) for h in hist[1:])
        it_max = max(int(h["coupled_iter"]) for h in hist)
        print(f"[stage04] 小结：coupled={'on' if args.coupled else 'off'}  "
              f"max 回路方程残差={r_max:.2e}  步内迭代最多 {it_max} 次")
    np.savez_compressed(
        OUT_DIR / f"{stem}history.npz",
        times=np.array([h["t"] for h in hist]),
        Phi=np.array([h["Phi"] for h in hist]),
        Phi_target=np.array([h["Phi_target"] for h in hist]),
        S0=np.array([h["S0"] for h in hist]),
        Izeta0=np.array([h["Izeta0"] for h in hist]),
        I=np.array([h["I"] for h in hist]),
        Psi_pl=np.array([h["Psi_pl"] for h in hist]),
        E=np.array([h["E"] for h in hist]),
        dE=np.array([h["dE"] for h in hist]),
        Rs=np.array([h["Rs"] for h in hist]),
        Bz0=np.array([h["Bz0"] for h in hist]),
        beta=np.array([h["beta"] for h in hist]),
        W=np.array([h["W"] for h in hist]),
        psi_wall=np.array([h["psi_wall"] for h in hist]),
        CI=np.array([h["CI"] for h in hist]),
        gs_res=np.array([h["gs_res"] for h in hist]),
        n_eval=np.array([h["n_eval"] for h in hist]),
        wall=np.array([h["wall"] for h in hist]),
        Phi_err=np.array([h["Phi_err"] for h in hist]),
        eta=np.array([eta]), dt=np.array([args.dt]),
        n_active=np.array([machine["n_active"]]),
        n_shell=np.array([machine["n_shell"]]),
        shell_nodes=np.asarray(machine["shell_nodes"], dtype=int),
        shell_loop=np.array([bool(args.shell_loop)]),
        Psi_ring=np.array([h["Psi_ring"] for h in hist]),
        eps_ring=np.array([h["eps_ring"] for h in hist]),
        shell_resid=np.array([h["shell_resid"] for h in hist]),
        circ_resid=np.array([h["circ_resid"] for h in hist]),
        coupled_iter=np.array([h["coupled_iter"] for h in hist]),
        coupled=np.array([bool(args.coupled)]),
        newton=np.array([bool(args.newton)]),
        shell_flux_state=np.array([bool(args.shell_flux_state)]),
        implicit_flux=np.array([bool(args.implicit_flux)]),
        shell=np.array([args.shell]), voltage=np.array([args.voltage]),
        psiq=np.array([h["psiq"] for h in hist]))
    print(f"\n[stage04] 产物：{OUT_DIR / f'{stem}history.npz'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        t = np.array([h["t"] for h in hist]) * 1e3
        na = machine["n_active"]
        ax[0, 0].plot(t, [h["Phi"] for h in hist], "o-")
        ax[0, 0].set_ylabel(r"$\Phi$ [Wb]")
        ax[0, 1].plot(t, -np.array([h["Izeta0"] for h in hist]) / 1e3, "o-")
        ax[0, 1].set_ylabel(r"$-I_{\zeta0}$ [kA]")
        ax[0, 2].plot(t, [h["psi_wall"] for h in hist], "o-")
        ax[0, 2].set_ylabel(r"$\psi_{\rm wall}$ [Wb/$2\pi$]")
        Iarr = np.array([h["I"] for h in hist])
        for j in range(Iarr.shape[1]):
            lab = f"coil{j}" if j < na else f"shell{j - na}"
            ax[1, 0].plot(t, Iarr[:, j] / 1e3, "o-", label=lab)
        ax[1, 0].set_ylabel("I [kA]")
        ax[1, 0].legend(fontsize=8)
        ax[1, 1].plot(t, np.array([h["Rs"] for h in hist]) * 1e2, "o-")
        ax[1, 1].set_ylabel(r"$R_s$ [cm]")
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
        fig.suptitle(f"Stage 4: coil circuit + shell ({args.shell}, V={args.voltage})")
        mstyle.style_figure(fig, fontsize=13.0, grid=mstyle.GRID_STYLE)
        fig.savefig(OUT_DIR / f"{stem}stage04.png", dpi=110)
        print(f"[stage04] 产物：{OUT_DIR / f'{stem}stage04.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[stage04] 画图跳过：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
