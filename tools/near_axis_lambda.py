"""近轴正则化实验：把 Λ=ψ/R²（规范 ψ_0=0）用到本项目这套离散上，和现在的 ψ 形式对比。

**实验脚本，不接进程序树、不改任何生产代码。**

背景
----
现在的求解器以 ψ 为未知量，径向是守恒形式

    (R/dR²)[(ψ_{i+1}-ψ_i)/R_{i+1/2} - (ψ_i-ψ_{i-1})/R_{i-1/2}]

第一圈内部点 R=dR 处的系数 1/(R-dR/2)=2/dR 很大。修改记录第 10 条记过：用
sin(aR)sin(bZ) 当检验函数时，二阶算子的误差在 max 范数下不随网格下降。

第一篇论文（Howell & Sovinec 2014）换未知量 Λ=ψ/R²，因为

    Δ*ψ = (1/R)·∂_R(R³∂_RΛ) + R²∂_ZZ Λ
    ⇔   ∂_R(R³∂_RΛ) + ∂_Z(R³∂_ZΛ) = R·Δ*ψ

是**全散度**形式，系数里没有 1/R。再取规范 ψ_0=0（轴上 ψ=0、ψ≈ψ_2(Z)R²），
Λ 在轴上正则，轴上那一行用极限形式 4Λ_RR(0)+Λ_ZZ(0)（因 Λ_RR+3Λ_R/R→4Λ_RR）。

两个检验函数
------------
* regular —— ψ = R²e^{-aR²}(Z²-z_w²)：轴上 ψ~R²，Λ=e^{-aR²}(Z²-z_w²) 正则，
  正是论文那套列式的前提；
* item10  —— ψ = sin(aR)sin(bZ)：轴上 ψ~R，Λ~1/R **不正则**（第 10 条用的就是
  这一类），只有 ψ 形式的数。

在域 R∈[0,0.14]、Z∈[-0.75,0.75]（Fig6 那个算例的域）上做加密研究。

实测结论（2026-09-21，本机，129×257 那一档为例，误差都按 max|ψ| 归一）
--------------------------------------------------------------------
1. **ψ 形式对轴上正则的解本来就是二阶收敛的**：解误差 1.673e-3 → 4.185e-4 →
   1.046e-4 → 2.614e-5（33×65 → 257×513，每档降 ~4 倍），近轴 4 行的误差也
   每档降一个量级。所以修改记录第 10 条那句"近轴带 O(h/R) 项、误差不随网格下降"
   **只对它用的那个检验函数成立**——`sin(aR)sin(bZ)` 在轴上 ψ~R、Λ~1/R，
   根本不是这个域上容许的解（含轴的域上解必须 ψ~ψ_2(Z)R²）。把检验函数换成
   正则的，二阶算子和解都规规矩矩二阶收敛。
2. **Λ 形式确实更准**：同样的 129×257 上解误差 2.122e-5（ψ 形式 1.046e-4，好 ~5 倍）；
   用 65×129 的 Λ 解（8.441e-5）就能压到 ψ 形式 129×257（1.046e-4）的水平，
   也就是**自由度少 4 倍**。Λ 形式的算子误差也二阶收敛（1.907e-3 → 4.773e-4 →
   1.194e-4 → 2.984e-5），而且内部行的截断误差比轴上那一行小 6 个数量级
   （轴上那一行用极限形式 4Λ_RR+Λ_ZZ，误差只落在那一个节点上，而那个节点上 ψ=R²Λ=0）。
3. 代价与前提：多一圈未知量（轴上那一行由 Dirichlet 变成方程）；壁面数据要换算成
   Λ_b=ψ_b/R²（一条乘法）；要求 F(ψ_轴)=0——本项目 F 恒为 0（FRC），自动满足。
4. **但它治不了本项目真正的误差来源**：修改记录第 10 条已经测过，65×129 上换四阶
   与二阶解几乎一样（4.96e-3 vs 4.69e-3），那 0.47% 来自几何/拓扑分辨率（壁面点
   枚举、线圈 Green 耦合、分界面位置），不是内部离散格式。Λ 形式改善的正是后者的
   那一半，所以在细网格上有意义、在项目现在用的粗网格上没有意义。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gseqfrc import mpath  # noqa: E402

mpath.addpath("second_paper/Fig6")
nk = mpath.load("nk_inner")          # 生产代码里那个二阶算子

RW, ZW = 0.14, 0.75
LEVELS = ((33, 65), (65, 129), (129, 257), (257, 513))


def make_grid(nR, nZ):
    rr = np.linspace(0.0, RW, nR)
    zz = np.linspace(-ZW, ZW, nZ)
    R, Z = np.meshgrid(rr, zz, indexing="ij")       # 与 ndgrid(rr,zz) 同序
    return rr, zz, rr[1] - rr[0], zz[1] - zz[0], R, Z


class Regular:
    """ψ = R²e^{-aR²}(Z²-z_w²)：轴上正则，Λ 也正则。"""

    name = "regular"

    def __init__(self, a=200.0):
        self.a = a

    def psi(self, R, Z):
        return R ** 2 * np.exp(-self.a * R ** 2) * (Z ** 2 - ZW ** 2)

    def lam(self, R, Z):
        return np.exp(-self.a * R ** 2) * (Z ** 2 - ZW ** 2)

    def dstar(self, R, Z):
        a = self.a
        G, f = Z ** 2 - ZW ** 2, np.exp(-a * R ** 2)
        return (-8 * a * R ** 2 + 4 * a ** 2 * R ** 4) * f * G + 2 * R ** 2 * f

    def axis_target(self, Z):
        """Δ*ψ/R² 在 R→0 的极限（Λ 形式轴上那一行的右端）。

        对 ψ=R²f(R)G(Z)：Δ*ψ/R² = (-8a+4a²R²)fG + 2f，取 R→0 得 -8aG+2。
        """
        a = self.a
        return -8.0 * a * (Z ** 2 - ZW ** 2) + 2.0


class Item10:
    """ψ = sin(aR)sin(bZ)：轴上 ψ~R，Λ 发散（第 10 条那一类）。"""

    name = "item10"

    def __init__(self, a=100.0, b=10.0):
        self.a, self.b = a, b

    def psi(self, R, Z):
        return np.sin(self.a * R) * np.sin(self.b * Z)

    def lam(self, R, Z):
        return np.full_like(np.asarray(R, dtype=float), np.nan)

    def dstar(self, R, Z):
        a, b = self.a, self.b
        return (-(a ** 2 + b ** 2) * self.psi(R, Z)
                - np.divide(a * np.cos(a * R) * np.sin(b * Z), R,
                            out=np.full_like(R, np.inf), where=R > 0))

    def axis_target(self, Z):
        return np.full_like(Z, np.nan)


# ------------------------------------------------------- 项目现在的 ψ 形式算子
def psi_operator(nR, nZ, dR, dZ, rr):
    """直接调生产代码 `nk_inner.Problem._build_operator_o2`，保证是同一套。"""

    class _Stub:
        pass

    stub = _Stub()
    stub.nR, stub.nZ = nR, nZ
    stub.dR, stub.dR2, stub.dZ2 = dR, dR * dR, dZ * dZ
    stub.rr, stub.N = rr, nR * nZ
    return nk.Problem._build_operator_o2(stub)


# ------------------------------------------------------------- 论文的 Λ 形式
def lambda_operator(nR, nZ, dR, dZ, rr, scaling="physical"):
    """∂_R(R³∂_RΛ) + ∂_Z(R³∂_ZΛ)；轴上那一行用极限形式 4Λ_RR(0)+Λ_ZZ(0)。

    未知量 = 除 Dirichlet（i=nR-1 与 j=0/nZ-1）之外的所有节点，**含轴向 i=0**。
    行与节点的编号都用列优先线性下标 i + j*nR，与项目一致。

    `scaling`：
      * `"physical"`——内部行用物理的 R^3 散度形式（∂_R(R³Λ_R)+R³Λ_ZZ），
        轴上那一行只能写成除以 R² 后的极限形式；两处的行权不一致（相差 R² 量级），
        只用于精度研究（解与行权无关）。
      * `"equation"`——所有行都写成同一个方程 Λ_RR+3Λ_R/R+Λ_ZZ = Δ*ψ/R²
        （内部行把 R^3 形式整体除以 R_i³），行权一致，做迭代性质对比时用这个。
    """
    N = nR * nZ
    dR2, dZ2 = dR * dR, dZ * dZ
    rows, cols, vals = [], [], []

    def is_dirichlet(i, j):
        return i == nR - 1 or j == 0 or j == nZ - 1

    def put(row, col, value):
        rows.append(row)
        cols.append(col)
        vals.append(value)

    for j in range(nZ):
        for i in range(nR):
            row = i + j * nR
            if is_dirichlet(i, j):
                put(row, row, 1.0)
                continue
            r = rr[i]
            if i == 0:                                    # 轴上
                put(row, row, -8.0 / dR2 - 2.0 / dZ2)
                put(row, 1 + j * nR, 8.0 / dR2)
                put(row, 0 + (j - 1) * nR, 1.0 / dZ2)
                put(row, 0 + (j + 1) * nR, 1.0 / dZ2)
                continue
            r3 = r ** 3
            a_p = (r + 0.5 * dR) ** 3                     # 两个面上的 R³
            a_m = (r - 0.5 * dR) ** 3
            # "equation" 把整行除以 R_i³，等价于写成 Λ_RR+3Λ_R/R+Λ_ZZ = Δ*ψ/R²
            wrow = 1.0 if scaling == "physical" else 1.0 / r3
            put(row, i + j * nR, -((a_p + a_m) / dR2 + 2.0 * r3 / dZ2) * wrow)
            put(row, (i + 1) + j * nR, a_p / dR2 * wrow)
            put(row, (i - 1) + j * nR, a_m / dR2 * wrow)
            put(row, i + (j - 1) * nR, r3 / dZ2 * wrow)
            put(row, i + (j + 1) * nR, r3 / dZ2 * wrow)

    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()


def run_case(case, nR, nZ):
    rr, zz, dR, dZ, R, Z = make_grid(nR, nZ)
    psi_e = case.psi(R, Z)
    dstar_e = case.dstar(R, Z)
    psi_f = psi_e.ravel(order="F")
    pnorm = np.abs(psi_e[1:nR - 1, 1:nZ - 1]).max()

    # ---------------- ψ 形式 ----------------
    A = psi_operator(nR, nZ, dR, dZ, rr)
    idr = np.zeros((nR, nZ), dtype=bool)
    idr[1:nR - 1, 1:nZ - 1] = True
    idr_f = idr.ravel(order="F")
    tgt_psi = (-(dZ ** 2) * dstar_e).ravel(order="F")    # 生产算子的符号约定 Aψ=-dZ²Δ*ψ
    Apsi = A @ psi_f
    scale_psi = np.abs(tgt_psi[idr_f]).max()
    op_psi = np.abs(Apsi[idr_f] - tgt_psi[idr_f]).max() / scale_psi
    sign = (Apsi[idr_f] * tgt_psi[idr_f]).sum() / (tgt_psi[idr_f] ** 2).sum()

    bnd_f = ~idr_f
    psi_h = psi_f.copy()
    psi_h[idr_f] = spsolve(A[idr_f][:, idr_f].tocsc(),
                           tgt_psi[idr_f] - A[idr_f][:, bnd_f] @ psi_f[bnd_f])
    psi_h2 = psi_h.reshape(nR, nZ, order="F")
    sol_psi = np.abs(psi_h2 - psi_e)[idr].max() / pnorm
    sol_psi_near = np.abs(psi_h2 - psi_e)[1:5, 1:nZ - 1].max() / pnorm
    row_err = np.abs((Apsi - tgt_psi).reshape(nR, nZ, order="F")) / scale_psi
    first_rows = [row_err[i, 1:nZ - 1].max() for i in range(1, 5)]

    # ---------------- Λ 形式 ----------------
    if np.all(np.isfinite(case.lam(R, Z))):
        B = lambda_operator(nR, nZ, dR, dZ, rr)
        lam_e = case.lam(R, Z).ravel(order="F")
        tgt_lam = np.zeros(nR * nZ)
        rep = np.zeros(nR * nZ, dtype=bool)
        for j in range(1, nZ - 1):
            for i in range(0, nR - 1):
                row = i + j * nR
                rep[row] = True
                tgt_lam[row] = (float(case.axis_target(np.array([Z[i, j]]))[0]) if i == 0
                                else rr[i] * dstar_e[i, j])
        scale_lam = np.abs(tgt_lam[rep]).max()
        B_lam = B @ lam_e
        op_lam = np.abs(B_lam[rep] - tgt_lam[rep]).max() / scale_lam
        row_err_lam = np.abs((B_lam - tgt_lam).reshape(nR, nZ, order="F")) / scale_lam
        lam_h = lam_e.copy()
        lam_h[rep] = spsolve(B[rep][:, rep].tocsc(),
                             tgt_lam[rep] - B[rep][:, ~rep] @ lam_e[~rep])
        psi_from_lam = (R ** 2 * lam_h.reshape(nR, nZ, order="F"))
        sol_lam = np.abs(psi_from_lam - psi_e)[idr].max() / pnorm
        near = np.abs(psi_from_lam - psi_e)[0:5, 1:nZ - 1].max() / pnorm
    else:
        row_err_lam = None
        op_lam = sol_lam = near = float("nan")

    return dict(op_psi=op_psi, sol_psi=sol_psi, sign=sign, row_err=row_err,
                sol_psi_near=sol_psi_near,
                op_lam=op_lam, sol_lam=sol_lam, near_lam=near,
                row_err_lam=row_err_lam)


def main() -> None:
    print("近轴正则化对比：ψ 形式（现状）vs Λ=ψ/R² 形式（第一篇论文）")
    print(f"域 R∈[0,{RW}]、Z∈[-{ZW},{ZW}]；误差都按 max|ψ| 归一，网格四档")
    for case in (Regular(200.0), Regular(60.0), Item10()):
        print(f"\n{'=' * 92}\n检验函数：{case.name}"
              f"{'  (a=%g)' % case.a if hasattr(case, 'a') else ''}\n{'=' * 92}")
        print(f"{'网格':>11} | {'ψ解误差':>10} {'ψ近轴4行':>10} {'ψ算子误差':>11} | "
              f"{'Λ解误差':>10} {'Λ近轴4行':>10} {'Λ算子误差':>11}")
        print("-" * 92)
        for nR, nZ in LEVELS:
            out = run_case(case, nR, nZ)
            print(f"{nR:>5}x{nZ:<5} | {out['sol_psi']:>10.3e} {out['sol_psi_near']:>10.3e} "
                  f"{out['op_psi']:>11.3e} | {out['sol_lam']:>10.3e} "
                  f"{out['near_lam']:>10.3e} {out['op_lam']:>11.3e}")
        print(f"（ψ 形式算子的符号比例 = {out['sign']:+.6f}，应约等于 +1）")

    # 误差住在哪几行：最细网格上逐行看
    print("\n最细网格上 ψ 形式算子误差按行分布（前 6 圈内部点，按全场最大值归一）")
    for case in (Regular(), Item10()):
        out = run_case(case, *LEVELS[-1])
        vals = "  ".join(f"{v:.3e}" for v in out["row_err"][1:7, 1:-1].max(axis=1))
        print(f"  {case.name:>8}: {vals}")
        if out["row_err_lam"] is not None:
            vals = "  ".join(f"{v:.3e}" for v in
                             out["row_err_lam"][0:6, 1:-1].max(axis=1))
            print(f"  {'(Λ)':>8}: {vals}   ← 第 1 个数是轴上那一行")


# ------------------------------------------------------------------ 速度
def speed_probe(nR=129, nZ=257, iters=400):
    """速度只可能从三个地方来：迭代次数、每次迭代的代价、同精度下的自由度。

    这里量前两个（第三个是上面那张表：Λ 在 65×129 上就顶得上 ψ 在 129×257）。
    """
    import time

    from scipy.sparse.linalg import eigsh, splu

    rr, zz, dR, dZ, R, Z = make_grid(nR, nZ)
    A = psi_operator(nR, nZ, dR, dZ, rr)
    B = lambda_operator(nR, nZ, dR, dZ, rr, scaling="equation")
    idr = np.zeros((nR, nZ), dtype=bool)
    idr[1:nR - 1, 1:nZ - 1] = True
    A_int = A[idr.ravel(order="F")][:, idr.ravel(order="F")].tocsc()
    rep = np.zeros(nR * nZ, dtype=bool)
    for j in range(1, nZ - 1):
        rep[0 + j * nR:(nR - 1) + j * nR] = True
    B_int = B[rep][:, rep].tocsc()

    out = {}
    for name, M in (("ψ 形式", A_int), ("Λ 形式", B_int)):
        t0 = time.perf_counter()
        lu = splu(M)
        t_fact = time.perf_counter() - t0
        b = np.random.default_rng(0).standard_normal(M.shape[0])
        t0 = time.perf_counter()
        for _ in range(20):
            lu.solve(b)
        t_solve = (time.perf_counter() - t0) / 20
        # 阻尼 Jacobi 的收敛因子：D^{-1}M 与对称化后的 D^{-1/2} M D^{-1/2} 相似，
        # 所以谱半径精确等于 max|1-w*lambda|，lambda 是后者（对称）的特征值。
        # 整体符号不影响 Jacobi 迭代（M→-M、b→-b 时迭代式一模一样），
        # 但 D^{-1/2}MD^{-1/2} 要求对角线为正，所以先把符号统一成正对角。
        Mn = M if M.diagonal().mean() > 0 else (-M).tocsc()
        d = np.sqrt(Mn.diagonal())
        S = (sp.diags(1.0 / d) @ Mn @ sp.diags(1.0 / d)).tocsc()
        lam_max = float(eigsh(S, k=1, which="LA", tol=1e-8, maxiter=iters)[0][0])
        lam_min = float(eigsh(S, k=1, which="LM", sigma=0.0, tol=1e-8,
                              maxiter=iters)[0][0])
        rho = {w: max(abs(1.0 - w * lam_min), abs(1.0 - w * lam_max))
               for w in (1.0, 0.1)}
        # 按这个收敛因子，残差降 6 个数量级要多少轮
        n_iter = {w: int(np.log(1e-6) / np.log(rho[w])) for w in rho}
        out[name] = dict(n=M.shape[0], fact=t_fact, solve=t_solve, rho=rho,
                         lam=(lam_min, lam_max), n_iter=n_iter, nnz=M.nnz)
    return out


if __name__ == "__main__":
    if "--speed" in sys.argv:
        args = [a for a in sys.argv[1:] if a.isdigit()]
        size = (int(args[0]), int(args[1])) if len(args) >= 2 else (129, 257)
        print(f"速度三问：迭代次数 / 每次迭代代价 / 同精度下的自由度（{size[0]}x{size[1]}）")
        res = speed_probe(*size)
        for name, r in res.items():
            print(f"\n{name}  未知量 {r['n']}  非零 {r['nnz']}")
            print(f"  LU 分解   {r['fact'] * 1e3:7.1f} ms")
            print(f"  一次回代  {r['solve'] * 1e3:7.2f} ms")
            print(f"  D^-1M 特征值范围  [{r['lam'][0]:.3e}, {r['lam'][1]:.3f}]")
            print(f"  阻尼 Jacobi 收敛因子  w=1.00: {r['rho'][1.0]:.6f}  → 残差降 1e-6 要 "
                  f"{r['n_iter'][1.0]:,} 轮；  w=0.10: {r['rho'][0.1]:.6f} → "
                  f"{r['n_iter'][0.1]:,} 轮")
    else:
        main()
