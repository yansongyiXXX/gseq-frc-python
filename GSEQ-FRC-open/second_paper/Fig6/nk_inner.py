"""Fig6 自由边界求解器的**快速内层**：FreeGSNKE 式配方（从 accel/nk_recipe_probe.py 机械抽取）。

只搬被验证过的那套配方，探索模式与被证伪的 Newton-Krylov 变体都没有搬
（来历、消融实验与独立复核见 `accel/README.md` 的「FreeGSNKE 的配方能不能搬到 FRC 上」，
合并进本树的过程见 `修改记录.md` 第 7 条）。

配方三要素，缺一不可：

1. **约束内嵌**：每一步都用 \\(C=I_{\\zeta0}/j_{z,\\rm mid}(\\psi)\\) 定幅度，于是
   \\(J_{z,\\rm mid}/I_{\\zeta0}\\equiv1\\)——"弱等离子体支"（只带 38% 电流）从构造上
   就被排除，不需要外加强支保护；
2. **IMEX 方向**：源项取当前迭代量的值，耦合项用一次稀疏直接解精确传播，
   而不是逐点松弛那种每轮只走 1% 的推进；
3. **容差 1.2 的线搜索**：允许残差单步最多涨 20%（freegsnke `forward_solve` 的原始判据），
   迭代因此能越过残差暂时上升的鞍区；严格下降会卡在 \\(2\\times10^{-3}\\) 处。

外加**混合回退**（`solve_inner_hybrid`）：新配方一旦停滞，就把控制权交给原文那套逐点松弛
（慢但收敛域大），所以代价有界——最坏退化成原文的内层求解，但不会崩。这一条是必需的：
没有它时换初值（×0.7 / ×1.3）会让外层 CI 一路推到上限、解退化。

**它是 `GSEQ_coil_MRR` 现在的默认内层**（`inner="hybrid"`）。原文那条逐点松弛仍然保留，
用 `inner="relax"` 打开——那条能逐位复现冻结基准，是做对照时的参照物。
"""

from __future__ import annotations

import time
from math import sqrt

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

MU0 = 4 * np.pi * 1e-7

#: 收敛判据：freegsnke 式的"幅值相对残差" max|F| / max|psi|，见 res_range
TOL_RANGE = 1e-5

#: **修改版**（第 19 条 / Stage 1）：`Problem.P` 里这几个量是"可以随时间变"的剖面/
#: 闭合参数。网格、离散算子 `A`、Green 张量 `G2`、LU 分解都与它们无关，所以时间演化
#: 里每步只需改 `Problem.P`（`set_params`），不必重建 `Problem`。
PROFILE_KEYS = ("betas", "alpha", "sigma", "n", "Be", "Rs", "Pm0", "Izeta0")


def profile_dp_frc(psi, P):
    r"""MRR 剖面 \(\mathrm{d}P/\mathrm{d}\psi\)（**修改版** 第 19 条：从
    `Problem.dp_per_unit_C` 抽出来，让诊断/演化共用同一份算式；表达式与运算顺序
    逐字不变）。
    """
    ap = P["alpha"] / (P["Be"] * P["Rs"] ** 2)
    bp = P["sigma"] / (P["Be"] * P["Rs"] ** 2)
    psi = np.asarray(psi)
    u = bp * psi + 1.0
    return (P["Pm0"] * P["betas"] * np.exp(-ap * psi) * u ** (P["n"] - 1)
            * (-ap * u + P["n"] * bp))


def profile_p_frc(psi, P):
    r"""MRR 剖面 \(P(\psi)=C P_{m0}\beta_s e^{-\alpha\psi/(B_eR_s^2)}
    (1+\sigma\psi/(B_eR_s^2))^n\) 里**不含 \(C\)** 的那部分（诊断/储能用）。
    """
    ap = P["alpha"] / (P["Be"] * P["Rs"] ** 2)
    bp = P["sigma"] / (P["Be"] * P["Rs"] ** 2)
    psi = np.asarray(psi)
    u = bp * psi + 1.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return P["Pm0"] * P["betas"] * np.exp(-ap * psi) * u ** P["n"]


def params_from_setup(s, params=None):
    r"""从 setup 字典取出剖面参数，再叠加 `params` 覆盖。

    `params` 只认 `PROFILE_KEYS`；给了 `Be` 而没给 `Pm0` 时按 \(P_{m0}=B_e^2/2\mu_0\)
    自动重算（与 `_setup` 里的定义一致）。
    """
    P = {k: s[k] for k in PROFILE_KEYS}
    if params:
        unknown = sorted(set(params) - set(PROFILE_KEYS))
        if unknown:
            raise ValueError(f"未知的剖面参数 {unknown}；可用的是 {list(PROFILE_KEYS)}")
        P.update({k: float(v) for k, v in params.items()})
        if "Be" in params and "Pm0" not in params:
            P["Pm0"] = P["Be"] ** 2 / (2 * MU0)
    return P


class Problem:
    """Fig6 网格上的固定边界非线性 G-S 问题（含自由边界所需的 Green 函数）。"""

    def __init__(self, s, psib_coil, G2, order: int = 2) -> None:
        """**修改版**：setup 与 Green 张量由调用方（`GSEQ_coil_MRR.solve`）传入。

        原来这份是实验脚本，自己从头 `_setup` 一遍、再把线圈 Green 函数与 GrzJz
        张量重算一遍（约 1.3 s）。接进程序树之后，这三样在求解器里已经算好了，
        直接借用即可——数值完全一样（同一个 setup、同一张张量）。

        `order`：2（默认，原文那套 3 点/二阶）或 4（**修改版**新增的四阶 5 点，
        用于"粗网格 + 高阶"的对比实验）。
        """
        self.s = s
        self.order = int(order)
        self.nR, self.nZ = s["nR"], s["nZ"]
        self.N = self.nR * self.nZ
        self.R, self.Z = np.asarray(s["R"]), np.asarray(s["Z"])
        self.rr, self.zz = np.asarray(s["rr"]), np.asarray(s["zz"])
        self.dR, self.dZ, self.dR2, self.dZ2 = s["dR"], s["dZ"], s["dR2"], s["dZ2"]
        self.Z0 = s["Z0"]
        self.jrb, self.jzb = np.asarray(s["jrb"]), np.asarray(s["jzb"])
        self.psib_coil = np.asarray(psib_coil).ravel()
        self.maxpsiw = float(np.abs(self.psib_coil).max())
        # **修改版**（第 19 条）：剖面参数单独一份、可随演化每步覆盖（`set_params`）。
        self.P = params_from_setup(s)
        self.rw, self.zw = s["rw"], s["zw"]

        ii, jj = np.meshgrid(np.arange(1, self.nR - 1), np.arange(1, self.nZ - 1),
                             indexing="ij")
        self.int_idx = (ii + jj * self.nR).ravel()
        self.Rint = self.R.ravel(order="F")[self.int_idx]
        self.int_mask = np.zeros(self.N, dtype=bool)
        self.int_mask[self.int_idx] = True
        self.jr_mid = int(self.jrb[self.Z0 - 1]) - 1

        self.A = self._build_operator()
        self.A_int = self.A[self.int_idx][:, self.int_idx].tocsc()
        self.G2 = np.asarray(G2)
        self.Gsub = self.G2[self.int_idx]          # (内部点 × 边界点)
        self.bnd_lin = np.asarray(s["bnd_lin"]).ravel()
        self.psi_init = self._initial_psi()
        self.set_psib(np.zeros(self.bnd_lin.size))
        self._luA = None

    def lu_A(self):
        """线性 G-S 算子的因子分解（freegsnke 的残差预条件就用它）。"""
        if self._luA is None:
            self._luA = splu(self.A_int)
        return self._luA

    # ---- 剖面参数（**修改版** 第 19 条：时间演化的入口）---------------------
    def set_params(self, params=None):
        r"""就地更新剖面/闭合参数，返回更新后的 `self.P`。

        时间演化里每秒（每步）只需要改这几个标量：网格、`A`（含 4 阶开关）、
        `Gsub`、LU 分解、`bnd_lin`、`int_idx` 都与它们无关，所以**不必重建
        `Problem`**、也不必重新分解算子。覆盖是**累积**的（只改传进来的那些键），
        所以"第二步只调 `Izeta0`"不会把上一步调过的 `Be` 冲回原值。

        只认 `PROFILE_KEYS`；给 `Be` 而不给 `Pm0` 时按 \(P_{m0}=B_e^2/2\mu_0\)
        自动重算 `Pm0`。`psi_init`/`lu_A` 不动（它们与剖面参数无关）。
        """
        if not params:
            return self.P
        unknown = sorted(set(params) - set(PROFILE_KEYS))
        if unknown:
            raise ValueError(f"未知的剖面参数 {unknown}；可用的是 {list(PROFILE_KEYS)}")
        self.P.update({k: float(v) for k, v in params.items()})
        if "Be" in params and "Pm0" not in params:
            self.P["Pm0"] = self.P["Be"] ** 2 / (2 * MU0)
        return self.P

    # ---- 几何与算子 --------------------------------------------------------
    def _build_operator(self):
        if getattr(self, "order", 2) == 4:
            return self._build_operator_o4()
        return self._build_operator_o2()

    def _build_operator_o2(self):
        nR, nZ, dR = self.nR, self.nZ, self.dR
        Rv = self.rr[1:nR - 1]
        c = self.dZ2 * Rv / self.dR2
        d0 = 2 + c * (1 / (Rv + 0.5 * dR) + 1 / (Rv - 0.5 * dR))
        ncol = nZ - 2
        lint = (np.arange(1, nR - 1)[:, None]
                + np.arange(1, nZ - 1)[None, :] * nR).ravel()
        rows = np.concatenate([lint] * 5)
        cols = np.concatenate([lint, lint - 1, lint + 1, lint - nR, lint + nR])
        vals = np.concatenate([np.repeat(d0, ncol),
                               -np.repeat(c / (Rv - 0.5 * dR), ncol),
                               -np.repeat(c / (Rv + 0.5 * dR), ncol),
                               -np.ones(lint.size), -np.ones(lint.size)])
        bnd = np.unique(np.concatenate([np.arange(0, nZ) * nR,
                                        np.arange(0, nZ) * nR + (nR - 1),
                                        np.arange(0, nR),
                                        np.arange(0, nR) + (nZ - 1) * nR]))
        rows = np.concatenate([rows, bnd])
        cols = np.concatenate([cols, bnd])
        vals = np.concatenate([vals, np.ones(bnd.size)])
        return sp.csr_matrix((vals, (rows, cols)), shape=(self.N, self.N)).tocsc()

    def _build_operator_o4(self):
        """**修改版**：四阶（每方向 5 点）离散，与 freegsnke 的 `gs_operator_order=4` 同一思路。

        约定与二阶版完全一样：\\(A\\psi = k^2\\mu_0R^2\\,\\mathrm{d}P\\)，其中
        \\(A = k^2(\\psi_{RR}-\\psi_R/R+\\psi_{ZZ})\\)（这里 \\(k=\\mathrm{d}Z\\)）。
        轴向、径向都用 5 点四阶公式；**紧贴边界的第一圈内部点退回二阶**（那里的
        5 点模板会越界到边界外）。边界行仍是 Dirichlet 行（对角 1）。
        """
        nR, nZ, dR, dZ = self.nR, self.nZ, self.dR, self.dZ
        i_int = np.arange(1, nR - 1)
        j_int = np.arange(1, nZ - 1)
        ncol = j_int.size
        Rv = np.asarray(self.rr)[i_int]
        k2 = self.dZ2
        av = k2 / (12 * dR ** 2)
        c2 = k2 * Rv / dR ** 2

        # ---- 径向系数（按行）----
        a_m2 = np.zeros(i_int.size); a_m1 = np.zeros(i_int.size)
        a_0 = np.zeros(i_int.size); a_p1 = np.zeros(i_int.size)
        a_p2 = np.zeros(i_int.size)
        loc = (i_int >= 2) & (i_int <= nR - 3)          # 能用 i±2 的行
        # 径向：A_r = k²(ψ_R/R − ψ_RR) = −k²ψ_RR + k²ψ_R/R
        #   ψ_RR 用 5 点四阶、ψ_R 用 5 点四阶，R 取该行的半径
        a_m2[loc] = av * (1.0 + dR / Rv[loc])
        a_m1[loc] = av * (-16.0 - 8.0 * dR / Rv[loc])
        a_0[loc] = av * (30.0)
        a_p1[loc] = av * (-16.0 + 8.0 * dR / Rv[loc])
        a_p2[loc] = av * (1.0 - dR / Rv[loc])
        a_0[~loc] = c2[~loc] * (1.0 / (Rv[~loc] - 0.5 * dR) + 1.0 / (Rv[~loc] + 0.5 * dR))
        a_m1[~loc] = -c2[~loc] / (Rv[~loc] - 0.5 * dR)
        a_p1[~loc] = -c2[~loc] / (Rv[~loc] + 0.5 * dR)

        # ---- 轴向系数（按列）----
        b_0 = np.full(ncol, 2.0); b_1 = np.full(ncol, -1.0); b_2 = np.zeros(ncol)
        jok = (j_int >= 2) & (j_int <= nZ - 3)           # 能用 j±2 的列
        b_0[jok] = 30.0 / 12.0
        b_1[jok] = -16.0 / 12.0
        b_2[jok] = 1.0 / 12.0

        ii, jj = np.meshgrid(i_int, j_int, indexing="ij")
        lin = (ii + jj * nR).ravel()
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []

        def put(di: int, dj: int, value, mask=None) -> None:
            r = lin
            c = lin + di + dj * nR
            v = np.broadcast_to(np.asarray(value, dtype=float), (i_int.size, ncol)).ravel()
            if mask is not None:
                keep = np.broadcast_to(mask, (i_int.size, ncol)).ravel()
                r, c, v = r[keep], c[keep], v[keep]
            rows.append(r); cols.append(c); vals.append(v)

        ones = np.ones((i_int.size, ncol), dtype=bool)
        put(0, 0, a_0[:, None] + b_0[None, :])
        put(+1, 0, a_p1[:, None]); put(-1, 0, a_m1[:, None])
        put(0, +1, b_1[None, :]); put(0, -1, b_1[None, :])
        put(+2, 0, a_p2[:, None], mask=loc[:, None] & ones)
        put(-2, 0, a_m2[:, None], mask=loc[:, None] & ones)
        put(0, +2, b_2[None, :], mask=ones & jok[None, :])
        put(0, -2, b_2[None, :], mask=ones & jok[None, :])

        bnd = np.unique(np.concatenate([np.arange(0, nZ) * nR,
                                        np.arange(0, nZ) * nR + (nR - 1),
                                        np.arange(0, nR),
                                        np.arange(0, nR) + (nZ - 1) * nR]))
        rows.append(bnd); cols.append(bnd); vals.append(np.ones(bnd.size))
        return sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.N, self.N)).tocsc()

    def _initial_psi(self):
        R, Z = self.R, self.Z
        blob = 0.01 * (0.1 - np.exp(-(((R / self.rw - 0.2) ** 2 / 0.3 ** 2)
                                      + (Z / self.zw) ** 2 / 0.4 ** 2))) * self.maxpsiw
        return np.asarray(blob).ravel(order="F")

    def set_psib(self, psib):
        """设定 Dirichlet 边界数据（轴=0，盒子边=psib）。"""
        self.psib = np.asarray(psib).ravel()
        v = np.zeros(self.N)
        v[np.arange(0, self.nZ) * self.nR] = 0.0
        v[self.bnd_lin] = self.psib
        self.bnd_val = v

    def enforce_bc(self, psi):
        out = np.array(psi, dtype=float)
        out[~self.int_mask] = self.bnd_val[~self.int_mask]
        return out

    # ---- 剖面 --------------------------------------------------------------
    def dp_per_unit_C(self, psi, shape="frc"):
        P = self.P
        psi = np.asarray(psi)
        if shape == "frc":
            # **修改版**（第 19 条）：算式抽到 `profile_dp_frc`，与诊断/演化共用
            # 同一份（表达式与运算顺序逐字不变）。
            return profile_dp_frc(psi, P)
        psi_min = float(np.min(psi))
        if psi_min >= -1e-12:
            return np.zeros_like(psi)
        ph = np.clip(psi / psi_min, 0.0, 1.0)          # 0 在分离面、1 在芯部
        return P["Pm0"] * P["betas"] * (1.0 - ph ** 2) * (psi <= 0.0)

    def _mid_idx(self):
        j = self.Z0 - 1
        return np.arange(0, self.jr_mid) + j * self.nR

    def jzmid_per_unit_C(self, psi, shape="frc", dP=None):
        # **修改版**：`dP` 可以由调用方传进来复用（同一份 ψ 上重复算它是纯浪费；
        # 数值完全一样，因为函数是确定性的）。
        if dP is None:
            dP = self.dp_per_unit_C(psi, shape)
        Rf = self.R.ravel(order="F")
        idx = self._mid_idx()
        return float(np.sum(2 * np.pi * (Rf * dP)[idx] * Rf[idx] * self.dR))

    def jzmid(self, psi, C, shape="frc", dP=None):
        """中平面电流（带幅度）：判定"强支/弱支"的量。"""
        return self.jzmid_per_unit_C(psi, shape, dP=dP) * C

    def amplitude(self, psi, mode, shape, C_target, lam, dP=None):
        if mode == "frozen":
            return C_target * lam, True
        j = self.jzmid_per_unit_C(psi, shape, dP=dP)
        if not np.isfinite(j) or abs(j) < 1e-8 * abs(self.P["Izeta0"]):
            return np.nan, False                     # 真空支：约束无法满足
        return lam * self.P["Izeta0"] / j, True

    # ---- 残差与辅助量 ------------------------------------------------------
    def residual(self, psi, mode, shape, C_target, lam=1.0, with_dP=False):
        """残差；`with_dP=True` 时把中间量 `dP` 一并返回，供线搜索复用。

        **修改版**去重：原来这里先算一遍 `dP`，`amplitude` 里又算一遍（`jzmid`），
        线搜索还会第三次调用——实测 15927 次/次求解、占约 4 s。现在一次算完往下传。
        """
        dP = self.dp_per_unit_C(psi, shape)
        C, ok = self.amplitude(psi, mode, shape, C_target, lam, dP=dP)
        if not ok:
            return (None, np.nan, False, None) if with_dP else (None, np.nan, False)
        rhs = np.array(self.bnd_val)
        rhs[self.int_idx] = self.dZ2 * MU0 * self.Rint ** 2 * dP[self.int_idx] * C
        F = (self.A @ psi - rhs)[self.int_idx]
        return (F, C, True, dP) if with_dP else (F, C, True)

    def jacobian_diag(self, psi, C):
        P = self.P
        ap = P["alpha"] / (P["Be"] * P["Rs"] ** 2)
        bp = P["sigma"] / (P["Be"] * P["Rs"] ** 2)
        psi = np.asarray(psi)
        u = bp * psi + 1.0
        df = (-ap * np.exp(-ap * psi) * u ** (P["n"] - 1) * (-ap * u + P["n"] * bp)
              + np.exp(-ap * psi) * (P["n"] - 1) * bp * u ** (P["n"] - 2)
              * (-ap * u + P["n"] * bp)
              + np.exp(-ap * psi) * u ** (P["n"] - 1) * (-ap * bp))
        return (self.dZ2 * MU0 * self.Rint ** 2 * C * P["Pm0"] * P["betas"]
                * np.asarray(df).ravel()[self.int_idx])

    def psib_from_psi(self, psi, C, CI=1.0):
        """自由边界：由内部电流算出边界通量（原文 [C] 那一步）。"""
        Jz = (self.R.ravel(order="F") * self.dp_per_unit_C(psi) * C)[self.int_idx]
        return CI * self.psib_coil + MU0 * CI * self.dR * self.dZ * (self.Gsub.T @ Jz)

    def vacuum_state(self, psib):
        """零电流解 A ψ = 0（内部）+ 给定 Dirichlet 数据：延拓的天然起点。"""
        self.set_psib(psib)
        return splu(self.A).solve(np.array(self.bnd_val))

    def lu_full(self):
        """整张网格的因子分解（含 Dirichlet 行）。"""
        if not hasattr(self, "_luF") or self._luF is None:
            self._luF = splu(self.A)
        return self._luF

    def rhs_full(self, psi, mode, shape, C_target, lam=1.0):
        """给定 ψ 时的右端项（源项 + Dirichlet），返回 (rhs, C, ok)。"""
        dP = self.dp_per_unit_C(psi, shape)
        C, ok = self.amplitude(psi, mode, shape, C_target, lam, dP=dP)
        if not ok:
            return None, np.nan, False
        rhs = np.array(self.bnd_val)
        rhs[self.int_idx] = self.dZ2 * MU0 * self.Rint ** 2 * dP[self.int_idx] * C
        return rhs, C, True

def frc_diagnostics(s, psi, C, CI=None, P=None):
    r"""FRC 平衡的物理诊断量（**修改版** 第 19 条 / Stage 0）。

    约定与求解器完全一致：`psi` 的单位是 \(\mathrm{Wb}/(2\pi)\)（即
    \(B_z=\frac{1}{R}\partial_R\psi\)），分离面取 \(\psi=0\)，轴上 \(\psi=0\)、
    芯部 \(\psi<0\)（反场区）。所以

    * **私通量** \(\Phi_{\rm private}=\psi_{\rm sep}-\psi_O=2\pi\,|\psi_{\min}|\)（Wb）；
    * **中平面分离面半径** \(R_s\)：\(\psi\) 沿 \(Z=Z_0\) 的最外侧零点（线性内插）；
    * 反场区的**连通域**给出截面积 \(A\)、等效半径 \(\sqrt{A/\pi}\) 与伸长比。

    全部返回标量，便于逐步写进 `history`（也便于直接 np.savez）。
    诊断只做 \(O(N)\) 的数组运算，代价可忽略。

    注意：\(\mathrm{d}P/\mathrm{d}\psi\) **一律在这里按剖面公式重算**（单位 C），
    不从外面传——`_solve_fast_inner` 手里那份是**已经乘过 C 的**，传进来会再乘一次
    C（实测会把 \(I_{\zeta,\rm mid}\) 放大 \(C\) 倍，正好可以当自洽性检查用）。
    """
    nR, nZ, Z0 = s["nR"], s["nZ"], s["Z0"] - 1
    rr = np.asarray(s["rr"], dtype=float)
    zz = np.asarray(s["zz"], dtype=float)
    dR, dZ = float(s["dR"]), float(s["dZ"])
    R = np.asarray(s["R"], dtype=float)
    psi2 = np.asarray(psi, dtype=float).reshape(nR, nZ, order="F")
    psi_flat = psi2.ravel(order="F")
    if P is None:
        P = params_from_setup(s)
    dP2 = profile_dp_frc(psi_flat, P).reshape(nR, nZ, order="F")
    # 压强剖面（储能/比压/分离面以内的能量都要它；先算好，下面几块共用）
    Ps2 = (C * profile_p_frc(psi_flat, P)).reshape(nR, nZ, order="F")

    nan = float("nan")
    out = dict(
        psi_min=float(psi2.min()), psi_max=float(psi2.max()), psi_sep=0.0,
        psi_wall_mid=float(psi2[-1, Z0]),
        Rw=float(rr[-1]), CI=(float(CI) if CI is not None else nan),
        Izeta0=float(P["Izeta0"]), Be=float(P["Be"]), betas=float(P["betas"]),
        alpha=float(P["alpha"]), sigma=float(P["sigma"]), n=float(P["n"]),
        Rs_model=float(P["Rs"]),
    )

    # ---- 私通量（分离面在 psi=0）------------------------------------------
    out["Phi_private_Wb"] = 2 * np.pi * max(0.0, -float(psi2.min()))

    # ---- 反场区的连通域：等效半径、半长、伸长比 ---------------------------
    from scipy import ndimage

    A_sep = Rs_area = Zs_half = elong = Volume_sep = W_sep = nan
    R_core_min = R_core_max = nan
    mask = psi2 < 0.0
    if bool(mask.any()):
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
        lab, nlab = ndimage.label(mask, structure=structure)
        if nlab:
            # 用 bincount 数每块的点数（`scipy.ndimage.sum` 在 1.15 之后改名
            # `sum_labels`，这里避开版本差异）
            sizes = np.bincount(lab.ravel(), minlength=nlab + 1)[1:]
            core = lab == (int(np.argmax(sizes)) + 1)
            iR, iZ = np.nonzero(core)
            A_sep = float(core.sum()) * dR * dZ
            Rs_area = float(np.sqrt(A_sep / np.pi))
            Zs_half = float(np.max(np.abs(zz[iZ])))
            R_core_min, R_core_max = float(rr[iR].min()), float(rr[iR].max())
            Volume_sep = float(np.sum(2 * np.pi * R[iR, iZ]) * dR * dZ)
            W_sep = float(np.sum(Ps2[iR, iZ] * 2 * np.pi * R[iR, iZ]) * dR * dZ)

    # ---- 中平面分离面半径（最外侧 psi=0 的零点）--------------------------
    psi_mid = psi2[:, Z0]
    Rs_mid = nan
    neg = np.nonzero(psi_mid < 0.0)[0]
    if neg.size:
        i1 = int(neg[-1])
        if i1 + 1 < nR and psi_mid[i1 + 1] != psi_mid[i1]:
            Rs_mid = float(rr[i1] + dR * psi_mid[i1]
                           / (psi_mid[i1] - psi_mid[i1 + 1]))
        else:
            Rs_mid = float(rr[i1])
    if np.isfinite(Rs_mid) and np.isfinite(Zs_half):
        elong = float(Zs_half / Rs_mid)

    # ---- 中平面磁场（$B_z=\frac{1}{R}\partial_R\psi$）---------------------
    Bz_axis = Bz_edge = nan
    if nR >= 3:
        Bz_axis = float(2.0 * psi_mid[1] / dR ** 2)          # 轴上：psi ~ ½Bz R²
        Bz_edge = float((psi_mid[-1] - psi_mid[-2]) / dR / rr[-1])

    # ---- 压强 / 储能 / 电流 ----------------------------------------------
    ii, jj = np.meshgrid(np.arange(1, nR - 1), np.arange(1, nZ - 1),
                         indexing="ij")
    int_lin = (ii + jj * nR).ravel()
    Rf = R.ravel(order="F")
    Rint = Rf[int_lin]
    Pmax_mid = float(np.nanmax(Ps2[:, Z0]))
    Be1 = float(np.sqrt(max(2 * MU0 * Pmax_mid, 0.0)))
    W_stored = float(np.nansum(Ps2.ravel(order="F")[int_lin] * 2 * np.pi * Rint)
                     * dR * dZ)
    Volume = float(np.sum(2 * np.pi * Rint) * dR * dZ)
    #: 全计算域平均比压（含分界面外的低压层，**不是**文献里那个 \(\langle\beta\rangle\)，
    #: 只是给个"能量摊在整盒子里"的参考值）
    beta_avg_domain = (W_stored / Volume) / (P["Be"] ** 2 / (2 * MU0)) \
        if Volume else nan
    #: 等离子体（分界面以内 = 反场区）体积平均比压——FRC 文献里 \(\langle\beta\rangle\)
    #: 的通常口径是"对等离子体体积平均"，所以这一项才是能和文献比的量。
    #: 注意两个"储能"不同：`W_stored` 是全计算域积分，`W_sep` 只积在分界面以内
    #: （本模型的 \(P(\psi=0)=\beta_s P_{m0}\neq0\)，分界面外还有一层压强，
    #: 所以全域名积分会偏大；比压要用 `W_sep`，否则会算出 \(\beta>1\) 的假值）。
    beta_avg_sep = (W_sep / Volume_sep) / (P["Be"] ** 2 / (2 * MU0)) \
        if Volume_sep and np.isfinite(Volume_sep) and Volume_sep > 0 else nan
    Izeta_total = float(C * np.nansum((Rf * dP2.ravel(order="F"))[int_lin])
                        * dR * dZ)
    mid_idx = np.arange(nR) + Z0 * nR
    Izeta_mid = float(2 * np.pi * C
                      * np.sum((Rf * dP2.ravel(order="F"))[mid_idx] * Rf[mid_idx])
                      * dR)

    out.update(
        Rs_mid=Rs_mid, Rs_area=Rs_area, Zs_half=Zs_half, elongation=elong,
        A_sep=A_sep, R_core_min=R_core_min, R_core_max=R_core_max,
        Volume_sep=Volume_sep,
        xs_mid=(Rs_mid / rr[-1] if np.isfinite(Rs_mid) else nan),
        Bz_axis_mid=Bz_axis, Bz_edge_mid=Bz_edge,
        Bz_axis_over_Be=(Bz_axis / P["Be"] if np.isfinite(Bz_axis) else nan),
        Pmax_mid=Pmax_mid, Be1=Be1,
        Be1_rel=(Be1 / P["Be"] - 1.0 if P["Be"] else nan),
        W_stored=W_stored, Volume=Volume, beta_avg_domain=beta_avg_domain,
        W_sep=W_sep, beta_avg_sep=beta_avg_sep,
        Izeta_total=Izeta_total, Izeta_mid=Izeta_mid,
        Phi_private_mid_Wb=2 * np.pi * max(0.0, -float(psi_mid.min())),
        Phi_over_BeRs2=(out["Phi_private_Wb"] / (P["Be"] * Rs_mid ** 2)
                        if np.isfinite(Rs_mid) and Rs_mid > 0 else nan),
    )
    return out


def res_l2(F):
    return float(np.linalg.norm(F))


def res_range(psi, F):
    return float((F.max() - F.min()) / (psi.max() - psi.min()))


def decay_rate(hist, n=3):
    """最近 n 段样本的几何衰减比（**修改版**新增）。

    用来回答"这一层还在下降，还是进了平台"：返回值 >1 表示每段都在降，
    ≈1 表示进平台、再迭代也换不来分辨率。nan 表示样本不够或数据无效。
    判据的用法是**外推**：按当前速率把残差压到目标还需要多少段/多少轮，
    再和剩余预算比——而不是写死一个段数或轮数上限。
    """
    h = [float(v) for v in hist if np.isfinite(v) and v > 0.0]
    if len(h) < 2:
        return float("nan")
    k = min(n, len(h) - 1)
    ratio = h[-1 - k] / h[-1]
    return float(ratio ** (1.0 / k))


def anderson_next(X, G, F, depth: int = 3, reg: float = 1e-10,
                  max_ratio: float = 2.0, min_ratio: float = 1e-3,
                  hull_pad: float = 0.2):
    """**修改版**（第 18 条）：Anderson 外推一步（Walker & Ni 2011 的 type-II 形式）。

    用于加速中层的边界通量不动点 \\(\\psi_b \\leftarrow \\Phi(\\psi_b)\\)，
    不改变不动点，只改"下一步从哪出发"。

    参数
    ----
    X, G, F : 历史列表（最旧在前）
        `X` 是输入（上一步的 psib），`G = Φ(X)` 是对应输出（内层解完之后由 ψ
        反算的 psib），`F = G - X` 是残差。每次 Anderson 外推只多用一次内层解，
        所以"省的是迭代轮数"。
    depth : 最多用多少段差分（m=0 时退化成普通不动点，用于保证可复现）。
    reg   : 正规化系数（残差大小差异大，正规化避免 A^T A 病态）。
    max_ratio : 外推步长相对普通步长的上限，防止把 psib 甩到不可行区。
    min_ratio : **只在残差平滑收缩时外推**——最近两步的残差比必须落在
                (min_ratio, 1) 内；否则退回普通不动点。冷启动的大步长区
                映射不平滑（内层会停滞），实测不做这个判断时步数会 1473→2524。
    hull_pad  : 外推结果逐点夹到历史 `G` 的凸包 ± pad×包宽内，保证 psib 不越界。
    """
    x_k, g_k, f_k = X[-1], G[-1], F[-1]
    m = min(int(depth), len(F) - 1)
    if m <= 0:
        return np.array(g_k, dtype=float)
    # 只在"平滑收缩"时加速：残差不降（或降得异常快）就先走普通不动点。
    if len(F) >= 2:
        n_old = float(np.linalg.norm(F[-2]))
        n_new = float(np.linalg.norm(F[-1]))
        if n_old > 0.0:
            ratio = n_new / n_old
            if not (float(min_ratio) < ratio < 1.0):
                return np.array(g_k, dtype=float)
    Fs = [np.asarray(v, dtype=float).ravel() for v in F[-(m + 1):]]
    Gs = [np.asarray(v, dtype=float).ravel() for v in G[-(m + 1):]]
    # Δf_i = f_{k-m+i} - f_{k-m+i-1}
    dF = np.stack([Fs[i + 1] - Fs[i] for i in range(m)], axis=1)
    AtA = dF.T @ dF
    lam = float(reg) * max(float(np.trace(AtA)) / m, 1e-300) + 1e-300
    try:
        gamma = np.linalg.solve(AtA + lam * np.eye(m), dF.T @ f_k)
    except np.linalg.LinAlgError:
        return np.array(g_k, dtype=float)
    # α_0 = γ_1；α_i = γ_{i+1} - γ_i；α_m = 1 - γ_m
    alpha = np.empty(m + 1)
    alpha[0] = gamma[0]
    for i in range(1, m):
        alpha[i] = gamma[i] - gamma[i - 1]
    alpha[m] = 1.0 - gamma[m - 1]
    x_acc = alpha @ np.stack(Gs, axis=0)
    # 安全阀：外推步长不许比普通步长（g_k - x_k）大太多
    step = x_acc - x_k
    step_plain = g_k - x_k
    ns = float(np.linalg.norm(step))
    npl = float(np.linalg.norm(step_plain))
    if npl > 0.0 and ns > float(max_ratio) * npl:
        x_acc = x_k + step * (float(max_ratio) * npl / ns)
    # 逐点凸包保护：psib 是物理边界通量，外推不许离开历史范围太远
    Gm = np.stack(Gs, axis=0)
    lo, hi = Gm.min(axis=0), Gm.max(axis=0)
    pad = float(hull_pad) * (hi - lo)
    x_acc = np.clip(x_acc, lo - pad, hi + pad)
    if not np.all(np.isfinite(x_acc)):
        return np.array(g_k, dtype=float)
    return x_acc


def _relax_rounds(pr, psi, psib, mode, shape, C_target, lam, rounds, w=0.99,
                  c_control=True):
    """原文的**内层迭代**（逐点松弛 + 1%/轮的 C 控制器）跑若干轮，用作回退。

    这是原文 `GSEQ_coil_MRR.m` 内层循环的逐字对应（`C=wc*C+(1-wc)*Ctmp*Izeta0/Jztotal_mid`），
    所以它的收敛域和无条件稳定性与原文一致——回退的目的就是把解拉回原文能收敛的区域。
    起手的 C 取自当前状态（内嵌幅度），之后按原文的控制器演化。
    """
    pr.set_psib(psib)
    psi = pr.enforce_bc(psi)
    rc = pr.rr[1:pr.nR - 1][:, None]
    coef_r, rc_p, rc_m = rc / pr.dR2, rc + 0.5 * pr.dR, rc - 0.5 * pr.dR
    MU0_rc2, half_dZ2 = MU0 * rc ** 2, 0.5 * pr.dZ2
    if mode == "frozen":
        C = C_target * lam
    else:
        C, ok = pr.amplitude(psi, "embedded", shape, C_target, lam)
        if not ok:
            C = C_target * lam
    Ctmp = C
    dP = np.zeros((pr.nR, pr.nZ))
    for _ in range(rounds):
        dP = (C * pr.dp_per_unit_C(psi, shape)).reshape(pr.nR, pr.nZ, order="F")
        if c_control and mode == "embedded":
            jz = float(np.sum(2 * np.pi * (pr.R * dP)[: pr.jr_mid, pr.Z0 - 1]
                              * pr.R[: pr.jr_mid, pr.Z0 - 1] * pr.dR))
            if np.isfinite(jz) and abs(jz) > 1e-6 * abs(pr.P["Izeta0"]):
                C = 0.99 * C + 0.01 * Ctmp * pr.P["Izeta0"] / jz
                Ctmp = C
        p0 = psi.reshape(pr.nR, pr.nZ, order="F")
        new = w * p0[1:-1, 1:-1] + half_dZ2 * (
            coef_r * ((p0[2:, 1:-1] - p0[1:-1, 1:-1]) / rc_p
                      - (p0[1:-1, 1:-1] - p0[:-2, 1:-1]) / rc_m)
            + (p0[1:-1, 2:] + p0[1:-1, :-2]) / pr.dZ2
            + MU0_rc2 * dP[1:-1, 1:-1]) * (1 - w)
        p1 = p0.copy()
        p1[1:-1, 1:-1] = new
        p1[0, :] = 0.0
        p1.ravel(order="F")[pr.bnd_lin] = pr.psib
        psi = p1.ravel(order="F")
    return psi
def solve_imex(pr, psib, mode, shape, C_target, lam=1.0, psi0=None, omega=1.0,
               kappa=None, limiter="exp-active", max_rel_step=0.2,
               tol_range=TOL_RANGE, max_iter=300, verbose=False, jz_floor=None,
               jz_monotone=False, accept_ratio=1.0,
               fallback_rounds=0, fallback_max=0, shrink_retry=20):
    """半隐式（源项隐式、耦合精确）：ψ ← ψ + ω (A⁻¹ rhs(ψ) − ψ)。

    ω=1 就是"每步精确解一次线性系统"（原文 [D] 一节里的无阻尼 Picard，据说一步就炸）。
    限步模式：
      None          ：限步之后仍加一层"残差必须下降"的线搜索（把发散变成阻尼）
      'exp'         ：均匀指数限步 |α_eff·Δψ| ≤ κ（外区也被压住）
      'local'       ：逐点 |Δψ_i| ≤ κ·A_ii/|D_i|（源项弱的地方放大步）
      'exp-active'  ：换变量 s = e^{-α_eff ψ} 的用法——活跃区（ψ<0）按"指数相对变化
                      ≤ κ"限步，即 |Δψ| ≤ ln(1+κ)/α_eff；外区（ψ≥0，源项已是 e^{-246}
                      量级）仍按 ψ 的普通上限走。这样换变量只作用在真正需要它的地方。
    所有模式都带"残差必须下降"的折半线搜索（最多 12 次），保证不发散。
    给 `jz_floor` 时再加一条硬判据（**强支保护**）：凡使 |Jzmid| 掉到
    jz_floor·|Izeta0| 以下的步一律拒绝——把"弱等离子体支"（只带 38% 电流）从
    搜索空间里剔出去，同时保留线搜索的稳定性。

    给 `fallback_rounds` 时再加**回退保护**：线搜索一旦停滞，就做 fallback_rounds 轮
    原文的 1% 逐点松弛（无条件稳定，收敛域大但慢），再回到 IMEX；最多回退
    `fallback_max` 次。这样把两条线的收敛域并起来——快速段用 IMEX，卡住时用松弛。

    `shrink_retry`（**修改版**新增，默认 20）：线搜索连续减半 12 次仍没找到可接受的步时，
    不立刻放弃，而是照 freegsnke 的做法**按残差比缩步再试**——只要新残差不再变差就接受
    这一步（`reduce_by = 旧残差/新残差`，不行再乘 0.75 继续缩）。它对应 freegsnke
    `GSstaticsolver` 里那句 "Increase in residual, update reduction triggered."：
    残差涨了也把缩小的那一步走掉，而不是停手。
    """
    pr.set_psib(psib)
    psi = pr.enforce_bc(pr.psi_init if psi0 is None else psi0)
    lu = pr.lu_full()
    alpha_eff = pr.P["alpha"] / (pr.P["Be"] * pr.P["Rs"] ** 2)
    F, C, ok = pr.residual(psi, mode, shape, C_target, lam)
    if not ok:
        return dict(psi=psi, C=C, iters=0, F=None, rng=np.inf, hist=[])
    cur = res_range(psi, F)
    jz0 = pr.jzmid(psi, C, shape)
    best_psi, best_cur = psi.copy(), cur
    hist, omegas, jz_hist = [cur], [], [jz0 / pr.P["Izeta0"]]
    stall = "迭代上限"
    n_fb = 0
    n_shrink = 0
    n_fb_rounds: list[int] = []
    k = 0
    for k in range(1, max_iter + 1):
        rhs, C, ok = pr.rhs_full(psi, mode, shape, C_target, lam)
        if not ok:
            break
        delta = lu.solve(rhs)[pr.int_idx] - psi[pr.int_idx]
        del_psi = float(np.max(psi) - np.min(psi))
        if kappa is None:
            step = delta
        elif limiter == "exp":
            dmax = float(np.max(np.abs(delta)))
            step = delta * min(1.0, kappa / (alpha_eff * dmax)) if dmax > 0 else delta
        elif limiter == "local":
            D = pr.jacobian_diag(psi, C)
            lim = np.minimum(kappa * pr.A_int.diagonal() / np.maximum(np.abs(D), 1e-12),
                             max_rel_step * del_psi)
            step = np.clip(delta, -lim, lim)
        else:                                   # exp-active
            lim = np.full(pr.int_idx.size, max_rel_step * del_psi)
            neg = psi[pr.int_idx] < 0.0
            lim[neg] = np.log1p(kappa) / alpha_eff
            step = np.clip(delta, -lim, lim)
        step *= omega
        a, accepted = 1.0, False
        trial = np.empty_like(psi)      # **修改版**：试步缓冲复用，避免每次 copy
        for _ in range(12):
            trial[:] = psi
            trial[pr.int_idx] += a * step
            # with_dP：把剖面 dP 一起拿回来，下面判电流分支时直接复用（不再重算）
            Ft, Ct, okt, dPt = pr.residual(trial, mode, shape, C_target, lam,
                                           with_dP=True)
            if okt and np.all(np.isfinite(Ft)):
                c_t = res_range(trial, Ft)
                jz_t = pr.jzmid(trial, Ct, shape, dP=dPt)
                jz_ok = (jz_floor is None
                         or abs(jz_t) >= jz_floor * abs(pr.P["Izeta0"]))
                if jz_monotone:
                    jz_ok = jz_ok and abs(jz_t) <= abs(jz0) + 1e-12
                if c_t < accept_ratio * cur and jz_ok:
                    # 必须复制：trial 是复用的缓冲，直接别名会让下一轮改到 psi 本身
                    psi = trial.copy()
                    F, C, cur = Ft, Ct, c_t
                    omegas.append(a)
                    jz_now = pr.jzmid(psi, C, shape, dP=dPt)
                    jz_hist.append(jz_now / pr.P["Izeta0"])
                    jz0 = jz_now
                    accepted = True
                    break
            a *= 0.5

        if not accepted and shrink_retry > 0:
            # ---------------------------------------------------------------
            # 缩步重试（**修改版**：借鉴 freegsnke 的做法，见修改记录第 8 条）
            #
            # freegsnke 的逐步逻辑是：算出 update 之后先按 max_rel_update_size
            # 限步；若残差涨了就**拒绝并缩步**（reduce_by = 旧残差/新残差），
            # 然后**照样把这一步走掉**——它从不因为"线搜索没找到可接受的步"而
            # 放弃迭代。我们原来在这一点上直接停手，把控制权交给 20000 轮松弛兜底，
            # 于是 CI2 那几步每次都跑满兜底（实测占全量时间的 65%）。
            #
            # 这里照它的口径补上：按残差比缩步、逐步再缩，只要新残差不再变差
            # 电流下界不再参与——约束内嵌已经把幅度钉在 Jzmid/Izeta0 = 1 上，
            # 消融实验（accel/README 的 U12）也表明那个下界不是必需的。
            #
            # 注意它**是无条件**接受缩小后的那一步的（只要残差算得出来），
            # 不要求残差下降——这也是它不停滞的关键。这里照做：先找容忍范围内
            # （c_t ≤ accept_ratio·cur）的步；找不到就取"残差最小的那一步"。
            # ---------------------------------------------------------------
            reduce_by = 0.75
            best_trial = None
            for _ in range(shrink_retry):
                trial = psi.copy()
                trial[pr.int_idx] += a * reduce_by * step
                Ft, Ct, okt = pr.residual(trial, mode, shape, C_target, lam)
                if okt and np.all(np.isfinite(Ft)):
                    c_t = res_range(trial, Ft)
                    if c_t <= accept_ratio * cur:
                        psi, F, C, cur = trial, Ft, Ct, c_t
                        omegas.append(a * reduce_by)
                        jz_hist.append(pr.jzmid(psi, C, shape) / pr.P["Izeta0"])
                        jz0 = pr.jzmid(psi, C, shape)
                        accepted = True
                        n_shrink += 1
                        break
                    if best_trial is None or c_t < best_trial[0]:
                        best_trial = (c_t, trial, Ft, Ct)
                    # 残差涨了：按"旧/新"的比例缩（freegsnke 的 reduce_by）
                    if c_t > 0:
                        reduce_by = min(reduce_by, max(cur / c_t, 0.75)) * 0.75
                else:
                    reduce_by *= 0.75
            if not accepted and best_trial is not None:
                # 容忍范围内一步都没找到：像 freegsnke 那样把已经缩过的、
                # 残差最小的那一步走掉（best-so-far 由调用方兜住）。
                c_t, trial, Ft, Ct = best_trial
                psi, F, C, cur = trial, Ft, Ct, c_t
                omegas.append(a * reduce_by)
                jz_hist.append(pr.jzmid(psi, C, shape) / pr.P["Izeta0"])
                jz0 = pr.jzmid(psi, C, shape)
                accepted = True
                n_shrink += 1

        if not accepted:
            if fallback_rounds > 0 and n_fb < fallback_max:
                # 回退：做若干轮原文松弛（无条件稳定），再回到 IMEX
                psi = _relax_rounds(pr, psi, psib, mode, shape, C_target, lam,
                                    fallback_rounds)
                F2, C2, ok2 = pr.residual(psi, mode, shape, C_target, lam)
                if not ok2 or not np.all(np.isfinite(F2)):
                    stall = "回退后仍落在无效区"
                    break
                F, C = F2, C2
                cur = res_range(psi, F)
                jz0 = pr.jzmid(psi, C, shape)
                hist.append(cur)
                if cur < best_cur:
                    best_cur, best_psi = cur, psi.copy()
                n_fb += 1
                n_fb_rounds.append(fallback_rounds)
                continue
            stall = "线搜索找不到既降残差、又守住电流下界的步"
            break
        hist.append(cur)
        if cur < best_cur:
            best_cur, best_psi = cur, psi.copy()
        if verbose and k % 10 == 0:
            print(f"      it={k:4d} 步长因子={a:.3e} 幅值相对残差={cur:.3e} "
                  f"ψ_min={psi.min():+.3e} Jzmid/Izeta0="
                  f"{pr.jzmid(psi, C, shape) / pr.P['Izeta0']:.4f}")
        if cur < tol_range:
            stall = "收敛"
            break
    use = best_psi
    Fu, Cu, oku = pr.residual(use, mode, shape, C_target, lam)
    return dict(psi=use, C=Cu, iters=k, F=Fu if oku else None, rng=best_cur,
                hist=hist, omega_mean=float(np.mean(omegas)) if omegas else np.nan,
                jz_hist=jz_hist, stall=stall, n_fallback=n_fb, n_shrink=n_shrink,
                fallback_rounds=n_fb_rounds)
def solve_inner_hybrid(pr, psib, C_target, inner_tol=1e-6, inner_max=300,
                       fb_rounds=200, fb_max=8, relax_max=20000, warm=None,
                       verbose=False, relax_chunk=500, relax_extrap=True):
    """混合内层求解器：先跑新配方（快）；**不收敛就交给原文松弛**（慢但稳）。

    这样保证外层循环每次拿到的都是一个"按原文判据收敛了的"内层解——外层那两层
    （itout 的边界通量不动点、CI 的归一化 secant）依赖的正是这个反馈；一旦内层
    返回未收敛的中间态，反馈就断了，CI 会一路走到 0（这就是 R3/R4 崩掉的机理）。
    代价有界：最坏情况退化成原文的 inner solve，不会更慢。
    """
    t0 = time.perf_counter()
    r = solve_imex(pr, psib, "embedded", "frc", C_target, lam=1.0, psi0=warm,
                   tol_range=inner_tol, max_iter=inner_max, jz_floor=0.8,
                   accept_ratio=1.2, fallback_rounds=fb_rounds,
                   fallback_max=fb_max, verbose=verbose)
    if r.get("stall") == "收敛":
        r["path"] = "IMEX"
        r["wall"] = time.perf_counter() - t0
        return r
    # 兜底：原文的内层迭代（逐点松弛 + 1%/轮 C 控制器），每 relax_chunk 轮查一次判据
    psi, rounds = r["psi"], 0
    F, C, ok = pr.residual(psi, "embedded", "frc", C_target, 1.0)
    hist_r = [res_range(psi, F) if ok else np.inf]
    extrap: dict = {}
    while rounds < relax_max and ok and res_range(psi, F) > inner_tol:
        psi = _relax_rounds(pr, psi, psib, "embedded", "frc", C_target, 1.0,
                            relax_chunk)
        rounds += relax_chunk
        F, C, ok = pr.residual(psi, "embedded", "frc", C_target, 1.0)
        hist_r.append(res_range(psi, F) if ok else np.inf)
        # **修改版**：这一层能不能交付——按实测衰减速率外推，预测压到判据还要
        # 多少轮；若超出剩余预算，就不要再烧了（实测这条松弛的速率稳在
        # 1.032/500 轮，压到 1e-5 要 ~62 000 轮，而预算写死 20 000，
        # 于是每次都跑满 2 万轮、交出一个比目标差 20 倍的残差）。
        if relax_extrap and len(hist_r) >= 3:
            rate = decay_rate(hist_r, 3)
            if np.isfinite(rate) and rate > 1.0:
                need = relax_chunk * np.log(hist_r[-1] / inner_tol) / np.log(rate)
                extrap = dict(relax_rate=rate, relax_need=float(need))
                # 判据就是"外推 + 剩余预算"：按当前速率压到判据还需要的轮数，
                # 加上已经跑掉的，若超过预算就停——它本来也到不了。
                # （参考数：速率 1.032/500 轮时从 5e-4 压到 1e-5 要 ~62 000 轮，
                # 预算 20 000，所以实测每次都跑满 2 万轮、交出比目标差 20 倍的残差。）
                if rounds + need > relax_max:
                    extrap["relax_stop"] = (
                        f"外推判死：速率 {rate:.4f}/{relax_chunk} 轮，压到 "
                        f"{inner_tol:g} 还需约 {need:,.0f} 轮 > 预算 {relax_max:,}")
                    break
    extrap["relax_hist"] = hist_r
    r.update(psi=psi, C=C, F=F if ok else None,
             rng=res_range(psi, F) if ok else np.inf, path="原文松弛兜底",
             relax_rounds=rounds, wall=time.perf_counter() - t0, **extrap)
    return r


def full_case(pr, C_base, psi_base=None, ntout=30, CI_tol=5e-4, itCI_max=200,
              inner_tol=1e-6, inner_max=300, jz_floor=0.8, accept_ratio=1.2,
              out_tol=2e-9, verbose=True, fb_rounds=0, fb_max=0, dci_max=None,
              mid_plateau_stop=False, psi0=None, CI0=1.0, secant_state=None,
              step_tol=None, step_ref=None, step_tol_floor=1e-5,
              mid_floor_rel=3e-5, psib_coil=None, psib0=None,
              mid_accel="none", mid_depth=3, mid_reg=1e-10, mid_max_ratio=2.0,
              ci_fixed=None, be_loop=False):
    """完整三层循环：外层 CI（secant）+ itout（边界通量不动点）+ 新内层求解器。

    内层用"约束内嵌 + IMEX + 容差线搜索"（见 accel/README.md 的 U 系列），
    所以原文那个 1%/轮的 C 控制器被约束内嵌取代——内层每个迭代点上
    Jzmid 都严格等于目标电流。

    `psi_base` 只是实验脚本用来打印"与冻结基准差多少"的对照量；接进程序树之后
    传 None 即可（不再拿基准作比较，那时是实验脚本的活儿）。

    `dci_max`（**修改版**新增，默认 None=关，**负面结果，别开**）：把外层 CI 的
    单步变化限成 ±dci_max，即"目标分几步爬过去"。实测见修改记录第 15 条：
    诊断对了（CI 的第一步 −19.6% 确实导致紧跟的 itout 首轮要 298 步），但药方
    错了——多一轮 CI 就多付 30 轮 itout，摊开后反而触发松弛兜底，全量 11.1 → 30.8 s。

    `mid_plateau_stop`（**修改版**新增，默认 **False=关**）：中层的"平台判据"。
    开成 True 时，用**两个 5 轮窗口的中位数比**判趋势（单点速率不行：实测默认
    配置下 dpsimout 逐轮上下摆 ±3 倍，单点判据会在噪声上误触发，把 CI1 从 30 轮
    砍到 21 轮，紧跟的 CI2 内层就 stall 到 NaN）。默认关的理由见修改记录第 16 条：
    这一层实测每轮仍在真降，**没有可停的空转轮**，真正的空转在兜底那一层。

    ---- **修改版**：时间演化用的三个新入口（见修改记录第 17 条）----

    * `psi0` / `CI0`：**暖启动**。给上一时刻的 ψ（二维数组或列优先展平）与外层
      归一化因子 CI，直接从这里起步，不再从 `pr.psi_init` 冷启动。
    * `psib0`：上一时刻的**边界通量**。必须和 `psi0` 一起给：`enforce_bc` 会把
      ψ 的边界行改成 `pr.bnd_val`，不给 `psib0` 的话边界会被清成上次遗留/零值，
      第一步就会产生一个全幅的假变化（实测把本步变化估成 4.4e-2 = 整个幅值，
      判据随之被放飞到 5e-3，解直接跑偏）。
    * `secant_state`：上一层留下的割线点 `(CI_a, g_a)`（或一串点），让外层
      secant 第一步就有斜率，省掉冷启动那两次试探。
    * `step_tol`：打开 freegsnke 式**「每步相对」判据**（`step_tol=None` 时走
      原来的 `out_tol` 路径，逐位不变）。目标是把相对残差压到

          target = step_tol × (本时间步的磁通变化 / ψ 幅值)

      这和 freegsnke 的 `rtol_NK = working_relative_tol_GS * d_plasma_psi_step`
      同一口径（它的 ψ 是归一化的，所以 d_plasma_psi_step 就是相对变化）。
      `step_ref` 给上一层实测的步变化；不给时先用 `inner_tol` 跑一轮，
      再用实测的 `max(ψ-ψ_warm) - min(ψ-ψ_warm)` 定尺度。

      两个地板是必须的：内层求解器能交付的相对精度是 `step_tol_floor`
      （默认 1e-5，见修改记录第 7 条），中层判据不能比它更严，否则 dpsimout
      永远压不到判据以下、每轮必然跑满 `ntout`（实测默认配置 120 次内层调用
      全是 30 轮上限，就是这么来的）。`mid_floor_rel` 是中层自身的可达地板
      （默认 3e-5 × ψ 幅值）。
    * `psib_coil`：本步的线圈边界通量（`MachineModel.coil_flux(Il)` 给），
      时间演化时每步线圈电流都不同，需要就地更新 `pr`。

    ---- **修改版**（第 18 条）：中层边界通量不动点加速 ----

    `mid_accel="anderson"` 时，中层 \\(\\psi_b\\leftarrow\\Phi(\\psi_b)\\)
    用 `anderson_next`（深度 `mid_depth`、正规化 `mid_reg`、步长安全阀
    `mid_max_ratio`）外推下一步的 `psib`；`"none"`（默认）走原来的
    Picard/逐轮重算路径，**逐位不变**。每次外推仍然只花一次内层解，
    省的是迭代轮数。

    **限制**：加速只在 `step_tol` 打开（即"每步相对"判据、时间演化暖启动）
    时生效；`step_tol=None`（冷启动/紧容差）时自动忽略——那一段 \(\\Phi\) 不
    平滑，实测 Anderson 会把解推到 NaN（见修改记录第 18 条）。另外内层一旦
    在外推点上停滞，本 CI 轮立刻停止加速、清历史、回到普通不动点。

    ---- **修改版**（第 23 条）：把 \(CI\) 钉住（供"物理磁链口径"用）----

    `ci_fixed` 给定时，外层**只跑一轮且不做归一化**：\(CI\) 恒等于给定值
    （一般取 1），边界通量 \(\psi_b=CI(\psi_{\rm coil,b}+\psi_{\rm pl,b})\) 就等于
    各电流的**物理**贡献之和。这样线圈/导体壳的回路方程（写在物理磁链上）与平衡
    求解器用的是同一套口径，壳的"通量冻结"才能作用到平衡上。
    压强平衡量 \(B_{e1}=\sqrt{2\mu_0\max P}\) 仍然算出来，但只作为**诊断量**
    （返回字典里的 `Be1`），不再用来改 \(CI\)。默认 `None` ⇒ 与原来逐位相同。

    ---- **修改版**（第 24 条）：把归一化搬到 \(B_e\) 上（"\(B_e\) 环"）----

    `be_loop=True` 时，外层改的**不是 \(CI\) 而是剖面参考场 \(B_e\)**：
    \(CI\equiv1\)（磁链physical），\(B_e\) 用与 CI 环**完全同构**的带保护 secant
    求 \(B_{e1}=B_e\)，每次更新都走 `pr.set_params({"Be": …})`（于是
    \(P_{m0}=B_e^2/2\mu_0\) 同步改），然后**重跑中/内层**（源项 ∝ \(B_e\)，必须重解）。
    这样"物理磁链"和"压强平衡"同时成立——代价是平衡族由 \((B_e,R_s)\) 的联合解定出。
    步长按 ±30% 夹住（与 CI 环同一道保险），首猜用 \(B_e\leftarrow B_{e1}\)。
    """
    from math import sqrt
    t0 = time.perf_counter()
    if psib_coil is not None:
        # **修改版**：线圈几何固定、只有电流变时，边界线圈通量每步重算一次即可，
        # 不必重建 Problem（Green 张量与算子都与电流无关）。
        pr.psib_coil = np.asarray(psib_coil, dtype=float).ravel()
        pr.maxpsiw = float(np.abs(pr.psib_coil).max())
    if psib0 is not None:
        # 暖启动的第一步：先把上一时刻的边界通量装回去，enforce_bc 才不会被清零
        pr.set_psib(np.asarray(psib0, dtype=float).ravel())
    if psi0 is None:
        psi = pr.enforce_bc(pr.psi_init)
    else:
        psi = pr.enforce_bc(np.asarray(psi0, dtype=float).ravel(order="F"))
    #: 时间步起点：freegsnke 式"每步相对"判据的参照（见修改记录第 17 条）
    psi_warm = psi.copy()

    # ---------------------------------------------------------------
    # 初值兜底（**修改版**：借鉴 freegsnke 的 "Initial guess ... failed,
    # trying to scale" 那一段，见修改记录第 8 条）
    #
    # 它在正式迭代前先试算一次残差；算不出来就把初始 ψ 放大 25%（除以 0.8）
    # 重试，最多 10 次，再不行才换默认初始化。我们原来没有这一层——初值一旦
    # 落在"电流归一化没法定义"的区域（换初值 ×0.7 / ×1.3 那两种情形）就直接崩。
    # ---------------------------------------------------------------
    ok0, F0 = False, None
    for _ in range(10):
        try:
            F0, _C0, ok0 = pr.residual(psi, "embedded", "frc", C_base)
            ok0 = bool(ok0) and bool(np.all(np.isfinite(F0)))
        except Exception:  # noqa: BLE001 - 兜底就是为了接住这些
            ok0 = False
        if ok0:
            break
        psi = pr.enforce_bc(psi / 0.8)
    if not ok0:
        # 缩放救不回来（例如初值全是 0）：退回脚本自己那套默认初始化
        # （freegsnke 同样有这一步：`Default plasma_psi initialisation ... invoked`）
        psi = pr.enforce_bc(pr._initial_psi())
        try:
            F0, _C0, ok0 = pr.residual(psi, "embedded", "frc", C_base)
            ok0 = bool(ok0) and bool(np.all(np.isfinite(F0)))
        except Exception:  # noqa: BLE001
            ok0 = False
    if not ok0:
        raise RuntimeError("起手状态无解：缩放 10 次、退回默认初始化后仍算不出有效残差")
    Be = pr.P["Be"]
    Be1 = 1.1 * Be
    # **修改版**（第 23/24 条）：`ci_fixed` 给定时 CI 被钉住；`be_loop` 走物理口径
    # （CI≡1），把压强平衡的归一化交给 B_e 环。
    if be_loop:
        ci_fixed = 1.0 if ci_fixed is None else float(ci_fixed)
    CI = float(ci_fixed) if ci_fixed is not None else float(CI0)
    CI_hist, g_hist, outer, inner_iters = [], [], [], []
    # **修改版**：把上一层留下的割线点接进来，外层 secant 第一步就有斜率
    if secant_state is not None:
        if len(secant_state) and np.isscalar(secant_state[0]):
            CI_hist.append(float(secant_state[0]))
            g_hist.append(float(secant_state[1]))
        else:
            for pair in secant_state:
                CI_hist.append(float(pair[0]))
                g_hist.append(float(pair[1]))
    # ---- **修改版**：freegsnke 式「每步相对」判据（步骤 1）----
    # target_res_range = step_tol × (dpsi_step / psi_amp)；见上面 docstring。
    step_mode = (step_tol is not None) and (float(step_tol) > 0.0)
    psi_amp_warm = float(psi_warm.max() - psi_warm.min()) if step_mode else 0.0
    dpsi_step = None
    if step_mode and step_ref is not None and np.isfinite(step_ref) and step_ref > 0:
        dpsi_step = float(step_ref)
    # 中层判据的地板：即便本步变化很小（或 step_ref 来自一个"几乎没动"的上一小步，
    # 例如演化开始时 Il 没变的那一步），也不能把判据压到内层交付精度以下——否则
    # 中层永远达不到、必然跑满 ntout（实测 90 次调用 = 3×30 就是这么来的）。
    out_tol_floor_abs = (float(mid_floor_rel) * psi_amp_warm) if step_mode else 0.0
    out_tol_eff = (max(float(step_tol) * dpsi_step, out_tol_floor_abs)
                   if (step_mode and dpsi_step) else float(out_tol))
    inner_tol_eff = float(inner_tol)
    if step_mode and dpsi_step is not None and psi_amp_warm > 0:
        inner_tol_eff = max(float(step_tol) * dpsi_step / psi_amp_warm,
                            float(step_tol_floor))
    #: 逐层"还在降 / 进平台"的诊断（**修改版**新增，见修改记录第 16 条）
    diag = dict(mid_plateau=None, ci_stuck=None, ci_rate=None)
    gabs_hist: list[float] = []
    stuck_hist: list[float] = []
    #: **修改版**（第 24 条）：B_e 环自己的割线历史
    be_hist: list[float] = []
    bg_hist: list[float] = []
    itCI = 0
    # **修改版**（第 24 条）：`be_loop` 时收敛判据放进循环体（用**新鲜**的 Be1 判），
    # 因为它的更新是 dead-beat 的（Be←Be1），顶部的陈旧判据会被"一步满足"而跳过重解。
    while (be_loop or abs(Be1 / Be - 1.0) > CI_tol) and itCI < itCI_max:
        itCI += 1
        dpsimout, psiout = 1e9, psi.copy()
        dps_hist: list[float] = []
        itout = 0
        # ---- **修改版**（第 18 条）：中层加速的状态 ----
        # 把中层的迭代量从 ψ 换成边界通量 psib：Φ(psib) = 由 Solve(psib) 的 ψ
        # 反算出来的 psib。不动点不变，但 Anderson 只作用在 511 维的 psib 上，
        # 最小二乘很便宜，而每次外推仍然只花一次内层解。
        # **只在"每步相对"判据（时间演化暖启动）下启用**：冷启动/紧容差的大步长区
        # 里 Φ 不平滑（内层频繁停滞），实测 Anderson 会把解推到 NaN（即使有凸包
        # 保护）。那一段的判据本来也不可达，加速没有意义。
        accel_mid = (mid_accel == "anderson") and step_mode
        if accel_mid:
            j1 = pr.jzmid_per_unit_C(psi)
            C = pr.P["Izeta0"] / j1
            Jz = pr.R.ravel(order="F") * pr.dp_per_unit_C(psi) * C
            psib_top = CI * pr.psib_coil + MU0 * CI * pr.dR * pr.dZ * (
                pr.Gsub.T @ Jz[pr.int_idx])
            _Xh: list = []
            _Gh: list = []
            _Fh: list = []
        while itout < ntout and dpsimout > out_tol_eff:
            itout += 1
            if accel_mid:
                psib = psib_top
            else:
                # 自由边界：由当前 ψ 的电流算边界通量（原文 [C] 那一步）
                j1 = pr.jzmid_per_unit_C(psi)
                C = pr.P["Izeta0"] / j1
                Jz = pr.R.ravel(order="F") * pr.dp_per_unit_C(psi) * C
                psib = CI * pr.psib_coil + MU0 * CI * pr.dR * pr.dZ * (
                    pr.Gsub.T @ Jz[pr.int_idx])
            if itCI == 1 and itout <= 2:
                print(f"    [诊断] j1={j1:.4e}  C={C:.4e}  psib∈"
                      f"[{psib.min():.3e},{psib.max():.3e}]  "
                      f"全有限={bool(np.all(np.isfinite(psib)))}  "
                      f"ψ_min={psi.min():.3e}")
            # 内层：固定边界 + 约束内嵌
            r = solve_inner_hybrid(pr, psib, C_base, inner_tol=inner_tol_eff,
                                   inner_max=inner_max, fb_rounds=fb_rounds,
                                   fb_max=fb_max, warm=psi,
                                   verbose=(itCI == 1 and itout <= 1))
            if itCI == 1 and itout <= 2:
                print(f"    [诊断] 内层返回 {r['iters']} 步，res={r['rng']:.3e}，"
                      f"停止原因：{r.get('stall')}")
            psi = r["psi"]
            if r.get("stall") != "收敛":    # 内层没收敛 = 异常，打印现场
                print(f"    [内层早停] CI{itCI} itout={itout} 步数={r['iters']} "
                      f"res={r['rng']:.3e} 停止={r.get('stall')} "
                      f"回退次数={r.get('n_fallback')} "
                      f"ψ_min={psi.min():+.3e} psib∈[{psib.min():.2e},{psib.max():.2e}] "
                      f"C={C:.3e} α={r.get('omega_mean')}")
            inner_iters.append(r["iters"])
            dpsimout = float(np.max(np.abs(psi - psiout)))
            psiout = psi.copy()
            dps_hist.append(dpsimout)
            # ---- **修改版**：第一次拿到实测的"本步响应"后，定下这一层的判据 ----
            # 第一轮内层用的是保守的 inner_tol；这一轮结束后 psi-psi_warm 就是
            # 新线圈电流引起的实际磁通变化（freegsnke 的 d_plasma_psi_step）。
            if step_mode and dpsi_step is None:
                dpsi_step = float(np.max(psi - psi_warm) - np.min(psi - psi_warm))
                if not np.isfinite(dpsi_step) or dpsi_step <= 0.0:
                    dpsi_step = max(dpsimout, 1e-30)
                out_tol_eff = max(float(step_tol) * dpsi_step, out_tol_floor_abs)
                if psi_amp_warm > 0:
                    inner_tol_eff = max(
                        float(step_tol) * dpsi_step / psi_amp_warm,
                        float(step_tol_floor))
            dps_rate = decay_rate(dps_hist, 5)
            outer.append((itCI, itout, dpsimout, r["iters"], r["rng"],
                          float(dps_rate) if np.isfinite(dps_rate) else 0.0))
            # **修改版**：中层（itout）的平台判据（默认关，见 docstring）。
            # 这一层的分辨率地板由内层给（内层按 inner_tol 停），所以判"还降不降"
            # 必须用**窗口趋势**，不能用单点速率——实测 dpsimout 逐轮摆动 ±3 倍，
            # 单点判据会在噪声上误判（会误伤 CI1，紧跟的 CI2 就 stall）。
            if mid_plateau_stop and itout >= 10:
                w_now = float(np.median(dps_hist[-5:]))
                w_pre = float(np.median(dps_hist[-10:-5]))
                trend = w_pre / w_now if w_now > 0 else 1.0
                if trend < 1.01:                    # 两个窗口之间几乎没降
                    diag["mid_plateau"] = (itCI, itout, dpsimout, trend)
                    break
            if verbose:
                print(f"  CI{itCI} itout={itout:2d}  dpsimout={dpsimout:.3e}  "
                      f"内层 {r['iters']:3d} 步  res={r['rng']:.2e}  "
                      f"Jzmid/Izeta0="
                      f"{pr.jzmid(psi, pr.P['Izeta0'] / pr.jzmid_per_unit_C(psi)) / pr.P['Izeta0']:.6f}")
            # ---- **修改版**（第 18 条）：算出 Φ(psib) 并做 Anderson 外推 ----
            # 这一段的计算量与原路径"循环顶部重算 psib"完全相同，只是挪到末尾。
            if accel_mid:
                j1 = pr.jzmid_per_unit_C(psi)
                C = pr.P["Izeta0"] / j1
                Jz = pr.R.ravel(order="F") * pr.dp_per_unit_C(psi) * C
                psib_out = CI * pr.psib_coil + MU0 * CI * pr.dR * pr.dZ * (
                    pr.Gsub.T @ Jz[pr.int_idx])
                if r.get("stall") != "收敛":
                    # 内层在**外推出来的** psib 上停滞：本 CI 轮立刻停止加速、
                    # 清历史、回到普通不动点。否则会连环停滞——实测冷启动
                    # 不加这条时内层步数 1473→2524、墙钟翻三倍。
                    _Xh.clear()
                    _Gh.clear()
                    _Fh.clear()
                    accel_mid = False
                    psib_top = psib_out
                else:
                    _Xh.append(np.array(psib, dtype=float))
                    _Gh.append(psib_out)
                    _Fh.append(psib_out - psib)
                    # 残差突然变差 → 丢掉历史重启（防止被坏的线性组合带跑）
                    if len(_Fh) >= 2:
                        n_new = float(np.linalg.norm(_Fh[-1]))
                        n_old = float(np.linalg.norm(_Fh[-2]))
                        if n_old > 0.0 and n_new > 1.5 * n_old:
                            _Xh[:] = _Xh[-1:]
                            _Gh[:] = _Gh[-1:]
                            _Fh[:] = _Fh[-1:]
                    psib_top = anderson_next(_Xh, _Gh, _Fh, mid_depth, mid_reg,
                                             mid_max_ratio)
                    if len(_Fh) > int(mid_depth) + 1:
                        _Xh.pop(0)
                        _Gh.pop(0)
                        _Fh.pop(0)
        if ci_fixed is not None and not be_loop:
            # **修改版**（第 23 条）：CI 被钉住 → 不做归一化。这里仍把 Be1 算出来
            # 存进诊断历史（它是"物理磁链口径下离压强平衡有多远"的度量），然后收尾。
            ap_ = pr.P["alpha"] / (pr.P["Be"] * pr.P["Rs"] ** 2)
            bp_ = pr.P["sigma"] / (pr.P["Be"] * pr.P["Rs"] ** 2)
            ps_ = psi.reshape(pr.nR, pr.nZ, order="F")
            u_ = bp_ * ps_ + 1.0
            Pm_ = (C * pr.P["Pm0"] * pr.P["betas"]
                   * np.exp(-ap_ * ps_) * u_ ** pr.P["n"])
            Be1 = sqrt(2 * MU0 * float(np.max(Pm_[:, pr.Z0 - 1])))
            CI_hist.append(CI)
            g_hist.append(Be1 - Be)
            if verbose:
                print(f"[CI 钉住] CI={CI:.8f}（不归一化）Be1={Be1:.8f} "
                      f"Be1/Be-1={Be1 / Be - 1.0:+.3e}")
            break

        if be_loop:
            # **修改版**（第 24 条）：B_e 环——与下面的 CI 环完全同构，只是玄量换成
            # 剖面参考场 B_e（`pr.set_params` 会同步改 Pm0=B_e²/2μ0），改完**重跑
            # 中/内层**（源项 ∝ B_e，必须重解）。
            ap_ = pr.P["alpha"] / (pr.P["Be"] * pr.P["Rs"] ** 2)
            bp_ = pr.P["sigma"] / (pr.P["Be"] * pr.P["Rs"] ** 2)
            ps_ = psi.reshape(pr.nR, pr.nZ, order="F")
            u_ = bp_ * ps_ + 1.0
            Pm_ = (C * pr.P["Pm0"] * pr.P["betas"]
                   * np.exp(-ap_ * ps_) * u_ ** pr.P["n"])
            Be1 = sqrt(2 * MU0 * float(np.max(Pm_[:, pr.Z0 - 1])))
            Be_now = float(pr.P["Be"])
            g_now = Be1 - Be_now
            be_hist.append(Be_now)
            bg_hist.append(g_now)
            CI_hist.append(CI)
            g_hist.append(g_now)
            if abs(g_now) / Be_now <= CI_tol:          # 用新鲜值判收敛
                if verbose:
                    print(f"[Be {itCI:3d}] 收敛：Be={Be_now:.8f} Be1={Be1:.8f} "
                          f"（|Be1/Be-1|={abs(g_now) / Be_now:.2e}）")
                break
            Be_new = Be1                      # 首猜：直接取压强等效场
            if be_hist:
                Ba, ga = be_hist[-2] if len(be_hist) > 1 else be_hist[-1], \
                    bg_hist[-2] if len(bg_hist) > 1 else bg_hist[-1]
                if g_now != ga:
                    Be_try = Be_now - g_now * (Be_now - Ba) / (g_now - ga)
                    if np.isfinite(Be_try) and Be_try > 0.0:
                        Be_new = Be_try
            ratio = min(max(Be_new / Be_now, 1 / 1.3), 1.3)   # 同 CI 环的 ±30% 保险
            pr.set_params({"Be": Be_now * ratio})
            Be = float(pr.P["Be"])
            if verbose:
                print(f"[Be {itCI:3d}] Be={Be_now:.8f} -> {Be:.8f} Be1={Be1:.8f} "
                      f"g={g_now:+.3e} ratio={ratio:.5f}")
            continue
        # 外层：CI 归一化（带保护的 secant，同 [F]）
        ap = pr.P["alpha"] / (pr.P["Be"] * pr.P["Rs"] ** 2)
        bp = pr.P["sigma"] / (pr.P["Be"] * pr.P["Rs"] ** 2)
        u = bp * psi.reshape(pr.nR, pr.nZ, order="F") + 1.0
        Pm = (C * pr.P["Pm0"] * pr.P["betas"]
              * np.exp(-ap * psi.reshape(pr.nR, pr.nZ, order="F")) * u ** pr.P["n"])
        Be1 = sqrt(2 * MU0 * float(np.max(Pm[:, pr.Z0 - 1])))
        g_now = Be1 - Be
        CI_new = CI * Be / Be1
        if CI_hist:
            CI_a, g_a = CI_hist[-1], g_hist[-1]
            if g_now != g_a:
                CI_try = CI - g_now * (CI - CI_a) / (g_now - g_a)
                if np.isfinite(CI_try) and CI_try > 0:
                    CI_new = CI_try
        dCI = CI_new / CI
        if not np.isfinite(dCI) or dCI <= 0:
            dCI = Be / Be1
        dCI = min(max(dCI, 1 / 1.3), 1.3)
        if dci_max is not None and dci_max > 0.0:
            # **修改版**：把外层 CI 的一步限成 ±dci_max（TokaMaker 的 nR0_ramp 思路）。
            #
            # ⚠️ 实测是**负面结果**（修改记录第 15 条）：把 −19.6% 那一步拆成
            # ±5% 之后，CI2 首轮的 298 步确实掉到 15 步，但每多一轮 CI 就多付
            # 30 轮 itout（每轮 ~5–8 步），而且摊开之后反而有 8 次内层撞到 300 步
            # 上限、触发原文松弛兜底（4000 轮）——全量 11.1 s → 30.8 s。
            # 所以这个开关**只用于复现那次实验，保持关闭**。
            dCI = min(max(dCI, 1.0 - dci_max), 1.0 + dci_max)
        CI_hist.append(CI)
        g_hist.append(g_now)
        CI = CI * dCI
        # **修改版**：外层的"卡死"判据——按实测速率外推还剩几轮，若 CI 已经
        # 连续 3 轮几乎不动（|dCI-1|<1e-9）却仍没到判据，说明再迭代也是空转，
        # 直接停并说明，而不是一路磨到 itCI_max。
        gabs_hist.append(abs(Be1 / Be - 1.0))
        stuck_hist.append(abs(dCI - 1.0))
        diag["ci_rate"] = decay_rate(gabs_hist, 3)
        if len(stuck_hist) >= 3 and max(stuck_hist[-3:]) < 1e-9:
            diag["ci_stuck"] = (itCI, CI, float(abs(Be1 / Be - 1.0)))
            if verbose:
                print(f"[CI {itCI}] 外层卡死：CI 连续 3 轮不动（|dCI-1|<1e-9），"
                      f"而 |Be1/Be-1|={abs(Be1 / Be - 1.0):.3e} 仍 > {CI_tol:g} → 停")
            break
        if verbose:
            print(f"[CI {itCI}] CI={CI:.8f} Be1={Be1:.8f} g={g_now:+.3e} dCI={dCI:.5f}")
    dt = time.perf_counter() - t0
    print(f"  快速内层：全量 {dt:.1f} s；itCI={itCI}，内层求解 {len(inner_iters)} 次，"
          f"平均 {np.mean(inner_iters):.1f} 步（最多 {max(inner_iters)}）")
    if psi_base is not None:
        d = float(np.max(np.abs(psi - psi_base)) / np.max(np.abs(psi_base)))
        print(f"  与对照解：max|Δψ|/max|ψ_base| = {d:.3e}，"
              f"ψ_min={psi.min():+.4e}（对照 {psi_base.min():+.4e}）")
    F, C, ok = pr.residual(psi, "embedded", "frc", C_base)
    # **修改版**：终态残差算不出来时（F=None）原来会在这一行抛 AttributeError，
    # 把"解坏了"这个信息埋成异常；现在照实打印，并带上分层诊断。
    rng_txt = f"{res_range(psi, F):.2e}" if ok else "算不出（解已落在无效区）"
    print(f"  终态：C={C:.6f}  CI={CI:.6f}  Be1={Be1:.6f}（Be={Be}）"
          f"  res_range={rng_txt}  psi_max={np.nanmax(psi):.4e}"
          f"  psi_min={np.nanmin(psi):+.4e}")
    if diag.get("mid_plateau") or diag.get("ci_stuck"):
        print(f"  分层诊断：中层平台 {diag.get('mid_plateau')}；"
              f"外层卡死 {diag.get('ci_stuck')}")
    out = dict(psi=psi, C=C, CI=CI, Be1=Be1, wall=dt, inner_iters=inner_iters,
               outer=outer, psib=psib, F=F, CI_hist=CI_hist, g_hist=g_hist,
               diag=diag,
               # **修改版**（第 17 条）：时间演化要带走的状态
               dpsi_step=dpsi_step, inner_tol_eff=float(inner_tol_eff),
               out_tol_eff=float(out_tol_eff), step_mode=step_mode,
               secant_state=((CI_hist[-1], g_hist[-1]) if CI_hist else None),
               psi_warm=psi_warm)
    return out
