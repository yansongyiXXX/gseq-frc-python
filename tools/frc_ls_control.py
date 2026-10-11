r"""\(l_s\) 控制的定量关系——用"0D 形成 → 交接 → 二维平衡"这条链。

目标（总目标第 4 条）：给出自由边界下 FRC 长度 \(l_s\) 与**线圈电流、真空室尺寸、
剖面参数**的定量关系。

数据来源（不重复解二维，全部可追溯）：

* **链**：`outputs/frc_formation/handoff.npz`（0D 形成末态）+
  `handoff_check.json`（交接到的二维平衡）→ 给出"在形成链上停在哪"这一维；
* **线圈响应矩阵**：`outputs/design_point/shaping.json`（由
  `tools/equilibrium_shaping.py` 跑出的 \(\partial(z_h,R_s,\Phi,E)/\partial I_j\)
  与三个物理模式，含一次到 \(E=3.6\) 的 Newton 塑形）；
* **尺寸/剖面**：族自相似关系（解析）。

核心关系：

$$l_s=2E\,R_s,\qquad R_s=\sqrt{\Phi/(cB_e)},\qquad
\frac{\delta l_s}{l_s}=\frac{\delta E}{E}
+\frac12\frac{\delta\Phi}{\Phi}-\frac12\frac{\delta B_e}{B_e}$$

用法：`python tools/frc_ls_control.py`（秒级，无二维求解）
产物：`outputs/frc_formation/ls_control.json` / `ls_control.png`
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

OUT = ROOT / "outputs" / "frc_formation"
SHAPING = ROOT / "outputs" / "design_point" / "shaping.json"


def main() -> int:
    if not SHAPING.exists():
        print(f"[ls] 找不到 {SHAPING}，先跑 tools/equilibrium_shaping.py")
        return 1
    sh = json.loads(SHAPING.read_text(encoding="utf-8"))
    nom = sh["nominal"]
    J = sh["jacobian"]
    modes = sh["modes"]
    Il0 = np.asarray(nom["Il"], float)
    ls0 = 2.0 * nom["zh"]
    print(f"[ls] 设计点：l_s={ls0 * 1e2:.2f} cm  R_s={nom['Rs'] * 1e2:.3f} cm  "
          f"E={nom['E']:.3f}  Phi={nom['Phi']:.4e} Wb  I_l={Il0[1] / 1e3:.1f} kA")

    # ---------------- 线圈电流 ----------------
    dzh = np.asarray(J["zh"], float)          # dzh/dI_j  [m/A]
    pct = 0.01 * Il0                          # +1% 安匝
    dls = 2.0 * dzh * pct                     # Delta l_s（每 +1% 单路线圈）
    coil_gain = dls / ls0                     # 相对增益
    amp = 0.01 * Il0[1]
    mode_gain = {name: float(modes[name]["E"] * amp / nom["E"]) for name in modes}

    # ---------------- 尺寸 & 剖面（解析） ----------------
    analytic = dict(
        dln_ls_dln_Rw=1.0,
        dln_ls_dln_zw=1.0,
        dln_ls_dln_Phi=0.5,
        dln_ls_dln_Be=-0.5,
        note="剖面经 c=Phi/(B_e R_s^2) 与 E 影响 l_s："
             "dl_s/l_s = dE/E - 0.5 dC/c + 0.5 dPhi/Phi - 0.5 dBe/Be",
    )

    # ---------------- 链上的点 ----------------
    chain = {}
    hp = OUT / "handoff_check.json"
    if hp.exists():
        hc = json.loads(hp.read_text(encoding="utf-8"))
        sol = hc["solve"]
        ls_hand = 2.0 * sol["elongation"] * sol["Rs_mid_m"]
        chain = dict(
            t_form_s=hc["t_form_s"], Il_handoff_A=hc["Il_handoff_A"],
            Phi_handoff_Wb=sol["Phi_2d_Wb"], Rs_m=sol["Rs_mid_m"],
            E=sol["elongation"], ls_m=ls_hand,
            dln_ls_dln_Il=float(np.log(ls_hand / ls0)
                                / np.log(hc["Il_handoff_A"] / Il0[1])),
            ls_ratio=ls_hand / ls0,
        )

    fin = sh.get("final", {})
    verify = {}
    if fin:
        Ed = fin["y"][1]
        ls_d = 2.0 * Ed * nom["Rs"]
        verify = dict(E=float(Ed), ls_m=float(ls_d), ls_ratio=float(ls_d / ls0),
                      Il=[float(v) for v in fin["Il"]],
                      source="shaping.json newton (target E=3.6)")

    res = dict(
        nominal=dict(ls_m=ls0, Rs_m=nom["Rs"], E=nom["E"], Phi_Wb=nom["Phi"],
                     Il_J=[float(v) for v in Il0]),
        coil_gain_per_1pct=[float(v) for v in coil_gain],
        coil_gain_note="每 +1% 该路线圈电流 -> dl_s/l_s（列序 = 线圈 1/2/3）",
        mode_E_gain_per_1pct=mode_gain,
        analytic=analytic, chain=chain, verify_shaping=verify,
        law="l_s = 2E sqrt(Phi/(c Be)); dl_s/l_s = dE/E + 0.5 dPhi/Phi - 0.5 dBe/Be",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ls_control.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n[ls] 线圈电流增益（每 +1% 单路线圈 -> dl_s/l_s）：")
    for j in range(3):
        print(f"   I{j + 1}: {coil_gain[j] * 1e2:+.3f}%")
    print("[ls] 物理模式（每 +1% 幅度 -> dE/E）：")
    for name, v in mode_gain.items():
        print(f"   {name:>12s}: {v * 1e2:+.3f}%")
    print("[ls] 尺寸 / 剖面（解析）：")
    print("   l_s ~ R_w（自相似，线性）；l_s ~ sqrt(Phi)；l_s ~ 1/sqrt(Be)")
    if chain:
        print(f"[ls] 链上（0D 交接）：I_l {Il0[1] / 1e3:.2f}->"
              f"{chain['Il_handoff_A'] / 1e3:.2f} kA，l_s {ls0 * 1e2:.2f}->"
              f"{chain['ls_m'] * 1e2:.2f} cm（{chain['ls_ratio']:.3f}x）")
        print(f"   dln(l_s)/dln(I_l) = {chain['dln_ls_dln_Il']:+.2f}"
              "（含 E 随停点的变化）")
    if verify:
        print(f"[ls] 验证（线圈塑形到 E={verify['E']:.2f}）："
              f"l_s {ls0 * 1e2:.2f}->{verify['ls_m'] * 1e2:.2f} cm"
              f"（{verify['ls_ratio']:.3f}x）")

    # ---------------- 图 ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from gseqfrc import mstyle
        mstyle.install()
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
        x = np.arange(3)
        ax[0].bar(x, [v * 1e2 for v in coil_gain], color="C0")
        ax[0].set_xticks(x); ax[0].set_xticklabels(["I1 端", "I2 中", "I3 端"])
        ax[0].axhline(0, color="gray", lw=0.8)
        ax[0].set_ylabel("Δl_s/l_s（每 +1%）")
        ax[0].set_title("线圈电流 → l_s 增益")
        labels, vals = list(mode_gain.keys()), [v * 1e2 for v in mode_gain.values()]
        ax[1].bar(np.arange(len(vals)), vals, color="C2")
        ax[1].set_xticks(np.arange(len(vals))); ax[1].set_xticklabels(labels)
        ax[1].axhline(0, color="gray", lw=0.8)
        ax[1].set_ylabel("ΔE/E（每 +1%）")
        ax[1].set_title("物理模式 → 伸长比")
        pts = [("设计点", ls0)]
        if chain:
            pts.append(("0D 交接", chain["ls_m"]))
        if verify:
            pts.append(("塑形 E=3.6", verify["ls_m"]))
        ax[2].bar(np.arange(len(pts)), [p[1] * 1e2 for p in pts], color="C3")
        ax[2].set_xticks(np.arange(len(pts)))
        ax[2].set_xticklabels([p[0] for p in pts])
        ax[2].set_ylabel("l_s (cm)"); ax[2].set_title("三个实测工作点")
        fig.tight_layout()
        fig.savefig(OUT / "ls_control.png", dpi=140)
        plt.close(fig)
        print(f"[ls] 产物：{OUT / 'ls_control.json'} / {OUT / 'ls_control.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[ls] 出图跳过：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
