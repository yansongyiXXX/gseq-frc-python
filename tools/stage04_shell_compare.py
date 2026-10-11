r"""Stage 4 壳环不动点对照：读 `outputs/frc_stage04/nsh*_*.npz`，汇总逐环 eps_l。

用法：

    python tools/stage04_shell_compare.py            # 汇总 + 出图
    python tools/stage04_shell_compare.py --files a.npz b.npz --labels A B

只读产物，不重算任何物理量。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "frc_stage04"

#: 默认对照表：(标签, 文件)。都在 outputs/frc_stage04/ 下。
DEFAULT = [
    ("n=2 关", "nsh2_noloop_history.npz"),
    ("n=2 开", "nsh2_fix_history.npz"),
    ("n=3 关", "nsh3_noloop_history.npz"),
    ("n=3 开", "nsh3_fix_history.npz"),
    ("n=5 开", "nsh5_fix_history.npz"),
]


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", type=str, default="shell_loop_compare.png",
                    help="对照图文件名（写在 outputs/frc_stage04/ 下）")
    args = ap.parse_args()

    if args.files:
        pairs = list(zip(args.labels or [Path(f).stem for f in args.files],
                         args.files))
    else:
        pairs = DEFAULT

    rows, series = [], {}
    for label, name in pairs:
        path = Path(name)
        if not path.is_absolute():
            path = OUT_DIR / path
        if not path.exists():
            print(f"[compare] 跳过（不存在）：{path}")
            continue
        d = load(path)
        t = np.asarray(d["times"], dtype=float)
        eps = np.asarray(d["eps_ring"], dtype=float)
        psi_ring = np.asarray(d["Psi_ring"], dtype=float)
        psi_wall = np.asarray(d["psi_wall"], dtype=float)
        n_sh = int(np.asarray(d["n_shell"]).ravel()[0])
        loop = bool(np.asarray(d["shell_loop"]).ravel()[0])
        coupled = bool(np.asarray(d["coupled"]).ravel()[0]) if "coupled" in d else False
        newton = bool(np.asarray(d["newton"]).ravel()[0]) if "newton" in d else False
        circ = (np.asarray(d["circ_resid"], dtype=float)
                if "circ_resid" in d else None)
        iters = (np.asarray(d["coupled_iter"], dtype=float)
                 if "coupled_iter" in d else None)
        drift = float((psi_wall[-1] - psi_wall[0]) / abs(psi_wall[0]))
        rows.append(dict(label=label, n_sh=n_sh, loop=loop, t=t, eps=eps,
                         psi_ring=psi_ring, psi_wall=psi_wall, drift=drift,
                         coupled=coupled, circ=circ, iters=iters,
                         newton=newton,
                         nodes=np.asarray(d["shell_nodes"]).ravel()))
        series[label] = (t, eps, psi_wall, n_sh, loop, circ)

    if not rows:
        print("[compare] 没有任何产物；先跑 frc_evolve_stage04.py")
        return 1

    print("\n=== 逐环冻结误差 eps_l = |psi_ring(t) - psi_ring(0)| / |psi_ring(0)| ===\n")
    head = (f"{'对照':<9} {'n_shell':>7} {'壳环不动点':>10} "
            f"{'max eps_l(末步)':>16} {'max eps_l(全程)':>16} "
            f"{'中平面 psi_wall 漂移':>20}")
    print(head)
    print("-" * len(head))
    for r in rows:
        e_last = float(np.max(r["eps"][-1])) if r["eps"].size else float("nan")
        e_all = float(np.max(r["eps"])) if r["eps"].size else float("nan")
        print(f"{r['label']:<9} {r['n_sh']:>7d} "
              f"{'开' if r['loop'] else '关':>10} "
              f"{e_last:>16.3e} {e_all:>16.3e} {r['drift']:>+19.2%}")

    print("\n每步逐环 eps_l：")
    for r in rows:
        steps = "  ".join(
            f"t={ti * 1e3:.2f}ms[" + ", ".join(f"{v:.1e}" for v in e) + "]"
            for ti, e in zip(r["t"], r["eps"]))
        print(f"  {r['label']:<9} {steps}")

    if any(r["circ"] is not None and len(r["circ"]) > 1 for r in rows):
        print("\n=== 步内回路<->平衡自洽性（回路方程残差）===\n")
        head2 = (f"{'对照':<9} {'步内自洽':>9} {'每步回路残差':>28} "
                 f"{'迭代次数':>12}")
        print(head2)
        print("-" * len(head2))
        for r in rows:
            if r["circ"] is None or len(r["circ"]) < 2:
                continue
            c = r["circ"][1:]
            it = (r["iters"][1:] if r["iters"] is not None else np.zeros_like(c))
            txt = ", ".join(f"{v:.1e}" for v in c)
            itx = ", ".join(str(int(v)) for v in it)
            mode = ("牛顿" if r["newton"] else
                    ("块迭代" if r["coupled"] else "外推"))
            print(f"{r['label']:<9} {mode:>9} "
                  f"{txt:>28} {itx:>12}")

    if args.no_plot:
        return 0
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sys.path.insert(0, str(ROOT))
        from gseqfrc import mstyle
        mstyle.install()

        fig, ax = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        for label, (t, eps, _pw, _n, loop, _c) in series.items():
            if not eps.size:
                continue
            style = "-o" if loop else "--s"
            ax[0].semilogy(t * 1e3, np.maximum(np.max(eps, axis=1), 1e-18),
                           style, label=label)
        ax[0].set_xlabel("t [ms]")
        ax[0].set_ylabel(r"$\max_l \varepsilon_l$")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3, which="both")

        for label, (t, _eps, pw, _n, loop, _c) in series.items():
            style = "-o" if loop else "--s"
            ax[1].plot(t * 1e3, (pw - pw[0]) / abs(pw[0]) * 100, style,
                       label=label)
        ax[1].set_xlabel("t [ms]")
        ax[1].set_ylabel(r"midplane $\psi_{\rm wall}$ drift [%]")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)
        mstyle.style_figure(fig, fontsize=12.0, grid=mstyle.GRID_STYLE)
        out = OUT_DIR / args.out
        fig.savefig(out, dpi=120)
        print(f"\n[compare] 图：{out}")
    except Exception as exc:  # noqa: BLE001
        print(f"[compare] 画图跳过：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
