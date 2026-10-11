r"""RMF 电流驱动的内部效率（Hugrass 1984, Aust. J. Phys. 37, 509）——共享内核。

原文第 4 节的三条式子单独放这里，供 `frc_rmf_sustainment.py`（标定 / 维持边界）与
`frc_evolve_design.py`（正向演化的驱动项）共用：

* eq 33：\(\gamma=(\nu/\nu_0)(\omega r/v_{e\theta})\)，\(v_{e\theta}=|J_\zeta|/(n_ee)\)
* eq 34：\(\langle\gamma\rangle=\int\gamma\eta J_\zeta^{2}2\pi r\,\mathrm{d}r
  \big/\int\eta J_\zeta^{2}2\pi r\,\mathrm{d}r\)
* eq 35：\(\mathcal{E}_i=1/\langle\gamma\rangle\le1\)（原文结论：\(O(1)\)、趋近 1）

本仓库的锚定式满足 \(S^{RR}/S_0=\omega/\omega_m=\gamma\)（刚性转子下），所以把
\(\gamma\) 换成原文的 \(\langle\gamma\rangle\) 就得到**理论版**

$$\eta_{\rm cd}^{\rm th}=\frac{\langle\gamma\rangle}{\beta_{\rm ref}^{2}F(x_{\rm ref})}
\qquad(\beta_{\rm ref}=1,\ F(1)=\tfrac12\ \Rightarrow\ \eta_{\rm cd}^{\rm th}=2\langle\gamma\rangle).$$

一个实现要点：eq (34) 的被积函数 \(\gamma\eta J^2\propto|J|r^2\) 在 \(J\to0\) 处是
**有限值**，所以直接算这个乘积、不要先算 \(\gamma=\dots/|J|\) 再乘（那样在零点出 NaN）。
"""

from __future__ import annotations

import numpy as np

E_CHARGE = 1.602176634e-19
M_ELECTRON = 9.1093837015e-31
BETA_REF = 1.0
X_REF = 1.0


def spitzer_resistivity(te_ev: float, ln_lambda: float = 10.0, z: float = 1.0) -> float:
    """Spitzer 电阻率 [Ohm*m]。"""
    return 5.2e-5 * z * ln_lambda / max(float(te_ev), 1e-6) ** 1.5


def electron_collision_time(ne: float, eta: float) -> float:
    r"""电子—离子碰撞时间 \(\tau_e=m_e/(n_ee^2\eta)\) [s]。"""
    return M_ELECTRON / (float(ne) * E_CHARGE ** 2 * float(eta))


def slip_factor(x):
    r"""标准滑移因子 \(F(x)=x/(1+x^2)\)，\(x=\omega\tau_e\)。"""
    x = np.asarray(x, dtype=float)
    return x / (1.0 + x ** 2)


def midplane_slice(field2d, Z):
    """取 \(Z\approx0\) 那一列（沿 \(R\) 的剖面）。"""
    Z = np.asarray(Z, dtype=float)
    i0 = int(np.argmin(np.abs(Z)))
    return np.asarray(field2d, dtype=float)[:, i0], i0


def profile_integrals(R, psi_mid, J_mid):
    r"""分界面以内算 \(\int|J|r^{2}\,\mathrm{d}r\) 与 \(\int J^{2}r\,\mathrm{d}r\)。

    这两个积分就是原文 eq (34) 的分子、分母（\(\eta\)、\(2\pi\) 都约掉）。
    """
    R = np.asarray(R, dtype=float)
    psi_mid = np.asarray(psi_mid, dtype=float)
    J_mid = np.asarray(J_mid, dtype=float)
    inside = np.where(psi_mid < 0.0)[0]
    if inside.size == 0:
        raise RuntimeError("ψ 中找不到分界面（没有 ψ<0 的格点）")
    Rs = float(R[inside.max()])
    m = (R > 0.0) & (R <= Rs)
    Rr, Jr = R[m], J_mid[m]
    iO = int(np.argmin(psi_mid))
    return dict(
        Rs=Rs,
        R_O=float(R[iO]),
        J_peak=float(np.abs(Jr).max()),
        I1=float(np.trapz(np.abs(Jr) * Rr ** 2, Rr)),
        I2=float(np.trapz(Jr ** 2 * Rr, Rr)),
    )


def midplane_integrals(R, Z, psi2d, J2d):
    """从二维场直接取中平面剖面再积分（正向演化里每步都用）。"""
    psi_mid, _ = midplane_slice(psi2d, Z)
    J_mid, _ = midplane_slice(J2d, Z)
    return profile_integrals(R, psi_mid, J_mid)


def theory_eta_cd(ne: float, eta: float, prof: dict, x_ref: float = X_REF,
                  beta_ref: float = BETA_REF, nu_ratio: float = 1.0) -> dict:
    r"""理论版 \(\eta_{\rm cd}\)（Hugrass 1984 内部效率口径）。

    \(\langle\gamma\rangle=(\nu/\nu_0)\,\omega^{*}n_ee\,I_1/I_2\)，其中
    \(\omega^{*}=x_{\rm ref}/\tau_e\)。原文第 518 页的结论是内部效率
    \(\mathcal{E}_i=1/\langle\gamma\rangle\le1\)，所以 \(\langle\gamma\rangle\ge1\)
    是一条自洽性判据（对应一个最低密度 \(n_{e,\min}\)）。
    """
    ne = float(ne)
    eta = float(eta)
    tau_e = electron_collision_time(ne, eta)
    omega = float(x_ref) / tau_e
    gamma = float(nu_ratio) * omega * ne * E_CHARGE * prof["I1"] / prof["I2"]
    F = float(slip_factor(np.array([x_ref]))[0])
    return dict(
        tau_e=tau_e,
        omega=omega,
        gamma=gamma,
        eta_cd=gamma / (float(beta_ref) ** 2 * F),
        E_i=(1.0 / gamma if gamma > 0.0 else float("inf")),
        slip=(1.0 - 1.0 / gamma if gamma > 0.0 else float("nan")),
        ne_min=float(np.sqrt(
            M_ELECTRON * prof["I2"]
            / (float(nu_ratio) * float(x_ref) * E_CHARGE ** 3 * eta * prof["I1"])
        )),
        F=F,
    )


def required_beta(eta_cd: float, x: float) -> float:
    r"""维持所需 \(\beta_\omega=B_\omega/B_e=1/\sqrt{\eta_{\rm cd}F(x)}\)。"""
    F = float(slip_factor(np.array([x]))[0])
    return float(1.0 / np.sqrt(max(float(eta_cd) * F, 1e-300)))
