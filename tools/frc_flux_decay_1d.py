r"""Stage 2：一维 (r,t) 刚性转子 FRC 的**通量衰减参照模型**（**修改版** 第 20 条）。

为什么要有这一步
----------------
第 19 条把二维自由边界求解器接成了"准静态平衡序列"，并且量出：在那个闭合里
\(\mathrm{d}\ln\Phi/\mathrm{d}\ln I_{\zeta0}\approx2.05\)、而
\(\Phi/(B_eR_s^2)\) 几乎不变——也就是说**通量没有独立自由度**。要判断这是
"模型本来就该这样"还是"闭合造成的假象"，需要一个**不依赖二维求解器**的参照：
直接解二维那条方程在 1D 圆柱下的形式，即通量扩散

$$\frac{\partial\psi}{\partial t}=\frac{\eta}{\mu_0}\,\Delta^{*}\psi,
\qquad \Delta^{*}\psi=\frac{\partial^2\psi}{\partial r^2}
-\frac{1}{r}\frac{\partial\psi}{\partial r}$$

（符号约定与程序树一致：\(\psi\) 的单位是 \(\mathrm{Wb}/(2\pi)\)、
\(B_z=\frac1r\partial_r\psi\)、\(\mu_0J_\phi=-\Delta^*\psi/r\)，于是
\(\partial_t\psi=-\eta rJ_\phi\)，电流与通量一起衰减。）

两个模型，差一步"投影"
------------------------
* `free`：**纯扩散**。psi 按上式演化，剖面 \(P(\psi)\) 只是跟着 psi 取值。
  它是"没有快弛豫"的极限，会逐渐偏离刚性转子族——偏离量由
  \(\mathcal{R}=\Delta^*\psi+\mu_0r^2\,\mathrm{d}P/\mathrm{d}\psi\) 度量
  （平衡解上它为零）。
* `family`：**扩散 + 每步投影回刚性转子族**。这是"快弛豫远快于扩散"的极限，
  也是二维那条准静态序列在 1D 里的对应物。

族为什么能投影得这么干净
--------------------------
MRR 族在 \((B_e,R_s)\) 上**自相似**：把 \(\psi=B_eR_s^2f(x)\)、\(x=r/R_s\) 代进
GS，\(B_e\)、\(R_s\) 全部约掉，只剩

$$f''-\frac{f'}{x}=-\frac{\beta_s}{2}x^2\,g'(f),$$

所以 \(\Phi/(B_eR_s^2)=2\pi|f_{\min}|\) **只是剖面形状参数（\(\beta_s,\alpha,\sigma,n\)）
的函数**，与 \(B_e\)、\(R_s\) 无关——第 19 条里"\(\Phi/(B_eR_s^2)\) 几乎不变"
由此得到解释：它不是 CI 闭合的假象，而是这个平衡族的自相似性。
`family` 模式的投影就是：保 \(\Phi\)、按 \(R_s^2=\Phi/(cB_e)\) 定新尺度、再把形状
换成族里的形状 \(f\)。\((1.865R_s\) 以外的真空尾在数值上精确等于
\(a+\tfrac12x^2\)，拟合残差 \(6\times10^{-12}\)，所以外插是解析的。）

`free` 与 `family` 是同一物理问题的两个极限，二者的差别回答了第 19 条留下的
问题：*在那个闭合里，通量到底是被什么钉住的。*

离散与验收
----------
径向离散**与二维求解器的径向部分逐字同式**（保守形式
\(L\psi_i=\frac{r_i}{dr^2}[(\psi_{i+1}-\psi_i)/r_{i+1/2}
+(\psi_{i-1}-\psi_i)/r_{i-1/2}]\)，它等于 \(\Delta^*\) 到二阶；已用二维矩阵的行
系数逐项对过）。边界：\(\psi(0)=0\)；外边界默认 \(\psi(R_w)\) 固定
（**通量守恒器**，理想导体壳）。时间推进默认隐式欧拉（一阶），`--scheme cn`
换成 Crank–Nicolson（二阶），两者的差可以当时间步收敛的检查。

`--verify` 跑一套自检：

1. 初值上 \(\Delta^*\psi\) 与 \(-\mu_0r\cdot(r\,\mathrm{d}P/\mathrm{d}\psi)\) 的相对差；
2. 初值的 \(I_{\zeta0}=2\pi\int r^2\mathrm{d}P/\mathrm{d}\psi\,dr\) 与 `MRR1d.mat` 的对照；
3. 通量守恒器模式下 \(\psi(R_w)\) 的时间漂移（应到机器精度）；
4. 隐式欧拉 vs Crank–Nicolson 的时间步收敛。

用法（在本仓库根目录）：

    python tools/frc_flux_decay_1d.py                 # free + family
    python tools/frc_flux_decay_1d.py --verify        # 只跑自检
    python tools/frc_flux_decay_1d.py --steps 80 --dt 2.5e-5
    python tools/frc_flux_decay_1d.py --te 30         # 换电子温度（Spitzer 电阻）

产物：`outputs/frc_stage02/history.npz` 与 `stage02.png`；若存在
`outputs/frc_stage01/history.npz`，还会打印与二维序列的逐项对照。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

try:  # 控制台编码兜底（GBK 终端遇到希腊字母不至于打断脚本）
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

from gseqfrc import mpath  # noqa: E402
from gseqfrc import mstyle  # noqa: E402

mstyle.install()   # 出图统一按 MATLAB 那一套（见 gseqfrc/mstyle.py）

MU0 = 4 * np.pi * 1e-7
SRC = ROOT / "GSEQ-FRC-open" / "second_paper" / "Fig6"
OUT_DIR = ROOT / "outputs" / "frc_stage02"

#: 与二维求解器共用同一份剖面公式（单一来源，避免两处各写一份）
mpath.addpath("second_paper/Fig6")
_GCM = mpath.load("GSEQ_coil_MRR")
_NK = _GCM._load_nk_inner()


def spitzer_resistivity(te_ev: float, ln_lambda: float = 10.0, z: float = 1.0) -> float:
    """Spitzer 电阻率 [Ohm*m]（NRL 手册那套系数，量级够用）。"""
    return 5.2e-5 * z * ln_lambda / max(te_ev, 1e-6) ** 1.5


def load_1d_equilibrium(path: Path = SRC / "MRR1d.mat") -> dict:
    r"""读 `MRR1d.mat`（论文附带的 1D 平衡），换算成物理单位。

    `rr1d` 的单位是 \(R_s\)、`psi1d` 的单位是 \(B_eR_s^2\)（都验过：换算后
    max P 正好等于 \(P_{m0}=B_e^2/2\mu_0\)，且
    \(2\pi\int r^2\mathrm{d}P/\mathrm{d}\psi\,dr\) 与文件里的 `Izeta0` 差
    \(2\times10^{-6}\)）。
    """
    m = loadmat(path, squeeze_me=True)
    Be, Rs, Rw = float(m["Be"]), float(m["Rs"]), float(m["Rw"])
    par = dict(betas=float(m["betas"]), alpha=float(m["alpha"]),
               sigma=float(m["sigma"]), n=float(m["n"]), Be=Be, Rs=Rs,
               Pm0=Be ** 2 / (2 * MU0), Izeta0=float(m["Izeta0"]))
    r = np.asarray(m["rr1d"], dtype=float) * Rs
    psi = np.asarray(m["psi1d"], dtype=float) * Be * Rs ** 2
    x_tab = np.asarray(m["rr1d"], dtype=float)
    f_tab = np.asarray(m["psi1d"], dtype=float)
    # 真空尾 f = a + x²/2（数值上精确：末 5 点最小二乘给出 b=0.500000、残差 1e-12）
    f_tail_a = float(f_tab[-1] - 0.5 * x_tab[-1] ** 2)
    return dict(r=r, psi=psi, par=par, Rw=Rw, Izeta0_file=float(m["Izeta0"]),
                j1d=np.asarray(m["j1d"], dtype=float),
                pr1d=np.asarray(m["pr1d"], dtype=float),
                Bz1d=np.asarray(m["Bz1d"], dtype=float),
                psi1d_raw=f_tab, x_tab=x_tab, f_tab=f_tab, f_tail_a=f_tail_a,
                #: \(\Phi=2\pi BeR_s^2|f_{\min}|\) —— 族的自相似常数
                c_family=2 * np.pi * abs(float(f_tab.min())))


def build_operator(r: np.ndarray) -> sp.csc_matrix:
    r"""\(\Delta^*=\partial_{rr}-\frac1r\partial_r\) 的径向离散（**与二维求解器
    的径向部分同一套系数**，保守形式，二阶）。

    内部点：\(L\psi_i=\frac{r_i}{dr^2}\left[\frac{\psi_{i+1}-\psi_i}{r_{i+1/2}}
    +\frac{\psi_{i-1}-\psi_i}{r_{i-1/2}}\right]\)。

    第 0 行与第 \(n-1\) 行**是零行**——这一点很关键：它是"演化算子"而不是
    "解 Dirichlet 问题用的算子"，边界点的 \(\partial_t\psi\) 必须为 0
    （轴上 \(\psi\equiv0\)、壁上 \(\psi\equiv\psi_w\) 由初值自带，
    通量守恒器就是靠这条自动保持）。写成单位行会让每步把边界值除以
    \((1-a)\) 而缓慢漂移。
    """
    n = r.size
    dr = r[1] - r[0]
    Rp = r[1:-1] + 0.5 * dr
    Rm = r[1:-1] - 0.5 * dr
    c = r[1:-1] / dr ** 2
    rows = np.concatenate([np.arange(1, n - 1)] * 3)
    cols = np.concatenate([np.arange(1, n - 1), np.arange(0, n - 2),
                           np.arange(2, n)])
    vals = np.concatenate([-c * (1.0 / Rp + 1.0 / Rm), c / Rm, c / Rp])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()


class FluxDiffusion1D:
    r"""一维通量扩散 \(\partial_t\psi=\frac{\eta}{\mu_0}L\psi\)（\(L\) 即
    `build_operator`）。

    隐式欧拉（默认）：\((I-aL)\psi^{n+1}=\psi^n\)，\(a=\frac{\eta}{\mu_0}dt\)；
    `scheme="cn"` 用 Crank–Nicolson。矩阵只分解一次（\(a\) 不随时间变）。
    """

    def __init__(self, r, psi0, eta, wall="conserver", scheme="ie"):
        self.r = np.asarray(r, float)
        self.psi = np.asarray(psi0, float).copy()
        self.eta = float(eta)
        self.wall = wall
        self.scheme = scheme
        self.L = build_operator(self.r)
        self.psi_w0 = float(self.psi[-1])
        self._cache: dict[float, object] = {}

    def _factor(self, a: float):
        if a not in self._cache:
            n = self.r.size
            I = sp.identity(n, format="csc")
            if self.scheme == "ie":
                self._cache[a] = splu((I - a * self.L).tocsc())
            else:
                self._cache[a] = (splu((I - 0.5 * a * self.L).tocsc()),
                                  (I + 0.5 * a * self.L).tocsc())
        return self._cache[a]

    def step(self, dt: float) -> None:
        """推进一个时间步（同一个 `dt` 重复使用时会复用分解）。"""
        a = round(self.eta / MU0 * float(dt), 24)
        fac = self._factor(a)
        if self.scheme == "ie":
            lu, rhs = fac, self.psi
        else:
            lu, M2 = fac
            rhs = M2 @ self.psi
        self.psi = lu.solve(rhs)


def family_shape(eq, x):
    r"""族里的形状函数 \(f(x)=\psi/(B_eR_s^2)\)，\(x=r/R_s\)。

    表内用 1D 平衡的采样点插值；表外（\(x>x_{\max}\)）用**精确的真空尾**
    \(f=a+\tfrac12x^2\)（数值上拟合残差 \(6\times10^{-12}\)，物理上就是
    \(B_z=B_e\) 的真空场）。给 \(x=0\) 时用插值端点值（\(\psi(0)=0\)）。
    """
    x_tab, f_tab = eq["x_tab"], eq["f_tab"]
    a = eq["f_tail_a"]
    out = np.where(x <= x_tab[-1], np.interp(x, x_tab, f_tab),
                   a + 0.5 * x ** 2)
    return out


def project_to_family(psi, r, eq):
    r"""把 psi 投影回自相似 MRR 族：**保私通量**、按 Φ 定新尺度、形状换回 \(f\)。

    返回 `(psi_new, Rs_new, drift, par_new)`：

    * `drift = ||psi_new-psi||₂/||psi||₂` 是"这一步扩散把状态推离族多远"的度量；
    * `par_new` 是把剖面参数里的 \(R_s\) 换成新的 \(R_s^{new}\) 之后的那一份——
      **这一步不能省**：MRR 剖面的 \(\psi\) 尺度就是 \(B_eR_s^2\)
      （\(\Rightarrow\) 族在 \((B_e,R_s)\) 上自相似），投影换了尺度就必须把剖面
      参数一起换，否则 \(P(\psi)\) 是在用旧的尺度取值，诊断量全错
      （实测会让 \(I_{\zeta0}\) 反而变大 8 倍）。

    在 t=0 上这是恒等映射（网格与采样点重合，插值精确），可以当自检用。
    """
    par = eq["par"]
    Phi = 2 * np.pi * max(0.0, -float(np.min(psi)))
    if Phi <= 0:
        return np.zeros_like(psi), 0.0, 0.0, dict(par)
    Rs_new = float(np.sqrt(Phi / (eq["c_family"] * par["Be"])))
    f = family_shape(eq, r / Rs_new)
    new = par["Be"] * Rs_new ** 2 * f
    den = float(np.linalg.norm(psi))
    drift = float(np.linalg.norm(new - psi) / den) if den > 0 else 0.0
    par_new = dict(par)
    par_new["Rs"] = Rs_new
    return new, Rs_new, drift, par_new


def diagnostics_1d(r, psi, par, L=None) -> dict:
    """一维诊断量，口径与第 19 条的 `frc_diagnostics` 一致（去掉 Z 方向）。"""
    n = r.size
    dr = r[1] - r[0]
    nan = float("nan")
    psi_min = float(psi.min())
    dP = _NK.profile_dp_frc(psi, par)
    P = _NK.profile_p_frc(psi, par)
    neg = np.nonzero(psi < 0.0)[0]
    Rs = nan
    if neg.size:
        i1 = int(neg[-1])
        if i1 + 1 < n and psi[i1 + 1] != psi[i1]:
            Rs = float(r[i1] + dr * psi[i1] / (psi[i1] - psi[i1 + 1]))
        else:
            Rs = float(r[i1])
    Pmax = float(np.nanmax(P))
    Be1 = float(np.sqrt(max(2 * MU0 * Pmax, 0.0)))
    W = float(np.nansum(P[1:-1] * 2 * np.pi * r[1:-1]) * dr)
    Izeta0 = float(2 * np.pi * np.sum((r ** 2 * dP)[1:-1]) * dr)
    out = dict(
        t=nan, psi_min=psi_min, psi_wall=float(psi[-1]),
        Phi_private_Wb=2 * np.pi * max(0.0, -psi_min),
        Rs_mid=Rs, xs_mid=(Rs / r[-1] if np.isfinite(Rs) else nan),
        Bz_axis_mid=float(2.0 * psi[1] / dr ** 2),
        Bz_edge_mid=float((psi[-1] - psi[-2]) / dr / r[-1]),
        Pmax=Pmax, Be1=Be1, Be1_rel=(Be1 / par["Be"] - 1.0),
        W_sep=W, Izeta0=Izeta0, proj_drift=0.0,
        Rs_param=float(par["Rs"]),
        Phi_over_BeRs2=(2 * np.pi * max(0.0, -psi_min) / (par["Be"] * Rs ** 2)
                        if np.isfinite(Rs) and Rs > 0 else nan),
    )
    if L is not None:
        # 只在“有电流”的区域量：远 SOL 里 Δ*ψ 与 r²dP/dψ 都是零附近的数值噪声，
        # 不设阈值的话残差会被那里的噪声主导（实测基底 5e-2 全来自 r>1.6Rs）。
        src = MU0 * r ** 2 * dP
        R = np.asarray(L @ psi).ravel()
        R[1:-1] += (MU0 * r[1:-1] ** 2 * dP[1:-1])
        den = float(np.max(np.abs(src[1:-1])))
        mask = np.abs(src) > 1e-2 * den
        mask[0] = mask[-1] = False
        out["offfamily"] = float(np.max(np.abs(R[mask])) / den) if den > 0 else nan
    return out


def run_mode(r, psi0, par, eta, mode, steps, dt, scheme="ie", eq=None):
    """跑一个模式，返回 `(逐步诊断量, 逐步 psi 剖面)`。"""
    solver = FluxDiffusion1D(r, psi0, eta, scheme=scheme)
    L = solver.L
    hist, psis = [], []
    psi = psi0.copy()
    t = 0.0
    d0 = diagnostics_1d(r, psi, par, L)
    d0["t"] = 0.0
    hist.append(d0)
    psis.append(psi.copy())
    Phi0 = max(float(d0["Phi_private_Wb"]), 1e-30)
    for _ in range(int(steps)):
        solver.step(dt)
        t += dt
        psi = solver.psi
        drift = 0.0
        par_now = par
        if mode == "family":
            psi, _Rs, drift, par_now = project_to_family(psi, r, eq)
            solver.psi = psi
        d = diagnostics_1d(r, psi, par_now, L)
        d["t"] = t
        d["proj_drift"] = drift
        if mode == "family":
            d["Rs_param"] = par_now["Rs"]
        hist.append(d)
        psis.append(psi.copy())
        if d["Phi_private_Wb"] < 1e-3 * Phi0:   # 私通量已经衰减到网格分辨不出
            break
    return hist, psis


def _slopes(hist, floor: float = 0.05) -> dict:
    r"""从历史里量：τΦ、dlnΦ/dlnIζ0、dlnR_s/dlnIζ0、末/初比。

    拟合只用 \(\Phi/\Phi_0\ge\)`floor` 的那一段：再往后私通量已经衰减到网格
    分辨不出（\(\psi_{\min}\sim10^{-15}\)），把这些点带进对数拟合会把斜率
    拉偏（实测能把 2.0 拉成 1.5）。`ratio` 类的量仍用整段。
    """
    t = np.array([h["t"] for h in hist], dtype=float)
    Phi = np.array([h["Phi_private_Wb"] for h in hist], dtype=float)
    Iz = np.array([h["Izeta0"] for h in hist], dtype=float)
    Rs = np.array([h["Rs_mid"] for h in hist], dtype=float)
    out: dict = {}
    ok = (np.isfinite(Phi) & (Phi > 0) & np.isfinite(Iz) & (Iz != 0)
          & np.isfinite(Rs))
    if ok.sum() >= 3:
        tk, Phik, Izk, Rsk = t[ok], Phi[ok], Iz[ok], Rs[ok]
        out["tau_Phi"] = float(-1.0 / np.polyfit(tk, np.log(Phik), 1)[0])
        out["Phi_ratio"] = float(Phik[-1] / Phik[0])
        out["Iz_ratio"] = float(np.abs(Izk[-1] / Izk[0]))
        out["Rs_ratio"] = float(Rsk[-1] / Rsk[0])
        fit = Phik >= floor * Phik[0]
        if fit.sum() >= 3:
            out["slope_span"] = float(Phik[fit][0] / Phik[fit][-1])
            out["dlnPhi_dlnIz"] = float(
                np.polyfit(np.log(np.abs(Izk[fit])), np.log(Phik[fit]), 1)[0])
            out["dlnRs_dlnIz"] = float(
                np.polyfit(np.log(np.abs(Izk[fit])), np.log(Rsk[fit]), 1)[0])
    return out


def verify(par, r, psi0, eta) -> None:
    """四项自检（见模块 docstring）。"""
    print("[verify] (1) 初值上的力平衡：Delta*psi 对 -mu0*r*(r dP/dpsi)")
    L = build_operator(r)
    Lpsi = np.asarray(L @ psi0).ravel()
    dP = _NK.profile_dp_frc(psi0, par)
    Jop = -Lpsi / (MU0 * r)
    Jpr = r * dP
    sel = slice(1, r.size - 1)
    m = np.abs(Jpr[sel]) > 1e-3 * np.abs(Jpr).max()
    d = np.abs(Jop[sel] - Jpr[sel])[m]
    print(f"    电流峰值 {np.abs(Jpr).max():.3e} A/m^2；"
          f"峰值附近最大相对差 {np.max(d) / np.abs(Jpr).max():.2e}")
    print("[verify] (2) 初值的 Izeta0 = 2*pi*int r^2 (dP/dpsi) dr 对 MRR1d.mat")
    print(f"    由剖面算出 {diagnostics_1d(r, psi0, par)['Izeta0']:.1f} A；"
          f"文件里 {load_1d_equilibrium()['Izeta0_file']:.1f} A"
          f"（文件记的是 +，程序树里 `_setup` 会取负号，这里与二维同号）")
    print("[verify] (3) 通量守恒器：psi(R_w) 的时间漂移")
    s = FluxDiffusion1D(r, psi0, eta, wall="conserver")
    for _ in range(20):
        s.step(5e-5)
    print(f"    psi(Rw) 相对漂移 {abs(s.psi[-1] / psi0[-1] - 1.0):.3e}")
    print("[verify] (4) 时间步收敛：隐式欧拉(1 阶) vs Crank-Nicolson(2 阶)")
    # 用 ψ 的 L2 范数量差（Φ=|ψ_min| 是点量，主要反映空间离散误差，看不出时间阶数）
    for dt in (2e-4, 1e-4, 5e-5):
        n = max(1, int(round(2e-4 / dt)))
        sa = FluxDiffusion1D(r, psi0, eta, scheme="ie")
        sb = FluxDiffusion1D(r, psi0, eta, scheme="cn")
        for _ in range(n):
            sa.step(dt)
            sb.step(dt)
        rel = float(np.linalg.norm(sa.psi - sb.psi) / np.linalg.norm(sb.psi))
        print(f"    dt={dt:.1e}: ||psi_IE-psi_CN||/||psi_CN|| = {rel:.3e}")
    print("    （IE 一阶、CN 二阶：dt 减半时这个差应大致减半）")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("free", "family", "both"), default="both",
                    help="free=纯扩散；family=每步投影回刚性转子族（二维那条闭合）")
    ap.add_argument("--steps", type=int, default=15,
                    help="步数（默认覆盖约 3 个私通量衰减时间；Φ 掉到 1e-3 以下会自动提前停）")
    ap.add_argument("--dt", type=float, default=2.5e-5, help="时间步 [s]")
    ap.add_argument("--te", type=float, default=50.0,
                    help="电子温度 [eV]（算 Spitzer 电阻）")
    ap.add_argument("--ln-lambda", type=float, default=10.0)
    ap.add_argument("--eta", type=float, default=None,
                    help="直接给电阻率 [Ohm*m]，覆盖 --te")
    ap.add_argument("--scheme", choices=("ie", "cn"), default="ie",
                    help="时间推进：隐式欧拉 / Crank-Nicolson")
    ap.add_argument("--verify", action="store_true", help="只跑自检")
    args = ap.parse_args()

    eq = load_1d_equilibrium()
    r, psi0, par = eq["r"], eq["psi"], eq["par"]
    eta = (float(args.eta) if args.eta is not None
           else spitzer_resistivity(args.te, args.ln_lambda))
    d0 = diagnostics_1d(r, psi0, par)
    Rs0 = float(d0["Rs_mid"])
    tau_eta = MU0 * Rs0 ** 2 / eta
    print(f"[stage02] 1D 平衡：{r.size} 点，r in [0,{r[-1] * 1e2:.3f}] cm；"
          f"R_s={Rs0 * 1e2:.3f} cm，Phi={d0['Phi_private_Wb']:.4e} Wb，"
          f"Izeta0={d0['Izeta0'] / 1e3:.1f} kA")
    print(f"[stage02] eta={eta:.3e} Ohm*m（T_e={args.te:g} eV）；"
          f"电阻时间 mu0*Rs^2/eta = {tau_eta * 1e3:.3f} ms")
    # 通量**耗散率**：O 点处 grad psi=0，所以 d(psi_min)/dt 就是局部值
    # (eta/mu0)*Delta*psi。族在 (Be,Rs) 上自相似 ⇒ 这个量 ∝ Be、与 Rs 无关，
    # 于是 **Φ 随时间线性衰减**：Phi(t)=max(0, Phi0 - S0*t)。这就是 Stage 3 的闭合。
    L0 = build_operator(r)
    S0_rate = 2 * np.pi * (eta / MU0) * float((L0 @ psi0)[int(np.argmin(psi0))])
    print(f"[stage02] 通量耗散率 S0 = 2*pi*(eta/mu0)*Delta*psi|_O = {S0_rate:.4f} Wb/s；"
          f"族闭合下 Φ 线性衰减，寿命 Phi0/S0 = "
          f"{d0['Phi_private_Wb'] / S0_rate * 1e3:.4f} ms")

    if args.verify:
        verify(par, r, psi0, eta)
        return 0

    modes = ("free", "family") if args.mode == "both" else (args.mode,)
    all_hist: dict[str, list] = {}
    all_psis: dict[str, list] = {}
    for mode in modes:
        t0 = time.perf_counter()
        hist, psis = run_mode(r, psi0, par, eta, mode, args.steps, args.dt,
                              args.scheme, eq=eq)
        wall = time.perf_counter() - t0
        all_hist[mode] = hist
        all_psis[mode] = psis
        sl = _slopes(hist)
        print(f"\n[stage02] {mode}：{args.steps} 步 x {args.dt * 1e6:g} us"
              f"（{hist[-1]['t'] * 1e3:.3f} ms = {hist[-1]['t'] / tau_eta:.2f} tau_eta），"
              f"墙钟 {wall:.2f} s")
        print(f"    Phi: {hist[0]['Phi_private_Wb']:.4e} -> "
              f"{hist[-1]['Phi_private_Wb']:.4e} Wb"
              f"（{sl.get('Phi_ratio', float('nan')):.4f}）；"
              f"Izeta0: {hist[0]['Izeta0'] / 1e3:.1f} -> "
              f"{hist[-1]['Izeta0'] / 1e3:.1f} kA"
              f"（{sl.get('Iz_ratio', float('nan')):.4f}）")
        print(f"    R_s: {hist[0]['Rs_mid'] * 1e2:.3f} -> "
              f"{hist[-1]['Rs_mid'] * 1e2:.3f} cm；"
              f"B_z0: {hist[0]['Bz_axis_mid']:+.4f} -> "
              f"{hist[-1]['Bz_axis_mid']:+.4f} T；"
              f"psi_wall: {hist[0]['psi_wall']:.4e} -> {hist[-1]['psi_wall']:.4e}")
        print(f"    实测 tau_Phi = {sl.get('tau_Phi', float('nan')) * 1e3:.3f} ms"
              f" = {sl.get('tau_Phi', float('nan')) / tau_eta:.3f} x (mu0 Rs^2/eta)；"
              f"dlnPhi/dlnIz0 = {sl.get('dlnPhi_dlnIz', float('nan')):.3f}；"
              f"dlnRs/dlnIz0 = {sl.get('dlnRs_dlnIz', float('nan')):.3f}")
        print(f"    （tau_Phi 只是描述性的指数拟合；族闭合下 Φ 是**线性**衰减、"
              f"速率恒为 S0 = {S0_rate:.4f} Wb/s —— dt→0 才收敛到这条直线）")
        if "offfamily" in hist[0]:
            print(f"    偏离刚性转子族（平衡解上应为 0）："
                  f"{hist[0].get('offfamily', float('nan')):.2e} -> "
                  f"{hist[-1].get('offfamily', float('nan')):.2e}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = sorted(k for k in all_hist[modes[0]][-1].keys() if k != "psi_profile")
    payload: dict[str, np.ndarray] = {}
    for mode in modes:
        payload[f"{mode}_keys"] = np.array(keys)
        payload[f"{mode}_vals"] = np.asarray(
            [[h.get(k, np.nan) for k in keys] for h in all_hist[mode]], dtype=float)
        payload[f"{mode}_r"] = r
        payload[f"{mode}_psi"] = np.asarray(all_psis[mode], dtype=float)
    payload["eta"] = np.array([eta])
    payload["tau_eta"] = np.array([tau_eta])
    payload["Rs0"] = np.array([Rs0])
    payload["r"] = r
    payload["psi0"] = psi0
    np.savez_compressed(OUT_DIR / "history.npz", **payload)
    print(f"\n[stage02] 产物：{OUT_DIR / 'history.npz'}")

    p2 = ROOT / "outputs" / "frc_stage01" / "history.npz"
    if p2.exists():
        d = np.load(p2)
        k2 = [str(x) for x in d["diag_keys"]]
        v2 = d["diag_vals"]
        iz2 = v2[:, k2.index("Izeta0")]
        ph2 = v2[:, k2.index("Phi_private_Wb")]
        rs2 = v2[:, k2.index("Rs_mid")]
        sl2 = float(np.polyfit(np.log(np.abs(iz2)), np.log(ph2), 1)[0])
        # 端点指数（首末两点的 Φ ∝ Iz^n 指数）比最小二乘斜率更贴族关系
        end2 = float(np.log(ph2[-1] / ph2[0]) / np.log(abs(iz2[-1] / iz2[0])))
        print("\n[stage02] 与二维准静态序列（Stage 1）对照：")
        print(f"    2D（固定线圈电流、给 Izeta0(t) 波形）：拟合 dlnPhi/dlnIz0 = {sl2:.3f}，"
              f"端点指数 = {end2:.3f}，"
              f"Rs 末/初 = {rs2[-1] / rs2[0]:.4f}，Phi 末/初 = {ph2[-1] / ph2[0]:.4f}")
        for mode in modes:
            sl = _slopes(all_hist[mode])
            h = all_hist[mode]
            end = float(np.log(h[-1]["Phi_private_Wb"] / h[0]["Phi_private_Wb"])
                        / np.log(abs(h[-1]["Izeta0"] / h[0]["Izeta0"])))
            print(f"    1D {mode:6s}（真解扩散方程）："
                  f"拟合 {sl.get('dlnPhi_dlnIz', float('nan')):.3f}，"
                  f"端点指数 = {end:.3f}，"
                  f"Rs 末/初 = {sl.get('Rs_ratio', float('nan')):.4f}，"
                  f"Phi 末/初 = {sl.get('Phi_ratio', float('nan')):.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        for mode in modes:
            h = all_hist[mode]
            t = np.array([x["t"] for x in h]) / tau_eta
            ax[0, 0].plot(t, [x["Phi_private_Wb"] for x in h], "o-", label=mode)
            ax[0, 1].plot(t, np.array([x["Rs_mid"] for x in h]) * 1e2, "o-", label=mode)
            ax[0, 2].plot(t, [x["Bz_axis_mid"] for x in h], "o-", label=mode)
            ax[1, 0].plot(t, -np.array([x["Izeta0"] for x in h]) / 1e3, "o-", label=mode)
            ax[1, 1].plot(t, [x["Phi_over_BeRs2"] for x in h], "o-", label=mode)
            if "offfamily" in h[0]:
                ax[1, 2].semilogy(t, [x.get("offfamily", np.nan) for x in h], "o-",
                                  label=mode + " (off-family)")
        ax[0, 0].set_ylabel(r"$\Phi_{private}$ [Wb]")
        ax[0, 1].set_ylabel(r"$R_s$ [cm]")
        ax[0, 2].set_ylabel(r"$B_{z0}$ [T]")
        ax[1, 0].set_ylabel(r"$-I_{\zeta0}$ [kA]")
        ax[1, 1].set_ylabel(r"$\Phi/(B_eR_s^2)$")
        ax[1, 2].set_ylabel("off-family residual")
        for a in ax.ravel():
            a.set_xlabel(r"$t/\tau_\eta$   ($\tau_\eta=\mu_0R_s^2/\eta$)")
            a.grid(alpha=0.3)
            if a.get_legend_handles_labels()[0]:
                a.legend(fontsize=8)
        fig.suptitle("Stage 2: 1D flux-decay reference (free vs re-projected to rigid rotor)")
        mstyle.style_figure(fig, fontsize=13.0, grid=mstyle.GRID_STYLE)
        fig.savefig(OUT_DIR / "stage02.png", dpi=110)
        print(f"[stage02] 产物：{OUT_DIR / 'stage02.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[stage02] 画图跳过：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
