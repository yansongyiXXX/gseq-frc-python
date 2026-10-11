r"""展示物 1：原文求解器 vs 加速版的**耗时与残差对照**（一键跑出 JSON + 图）。

跑两遍 Fig6 自由边界求解（同一条线圈表、同一张网格）：

* ``inner="relax"``  —— 原文那条逐点松弛 + 1%/轮 C 控制器（参照路径，逐位复现冻结基准）
* ``inner="hybrid"`` —— 加速版默认路径（约束内嵌 + IMEX + 容差线搜索）

实测记录墙钟、GS 方程幅值相对残差 ``gs_res``、松弛映射固定点缺陷 ``defect``、``C``/``CI``，
写成 ``outputs/showcase/solver_compare.json`` 与 ``solver_compare.png``。

图上另有两个**记录值**参照点（不在本机跑，取自 ``修改记录.md``）：

* 原文 MATLAB ``GSEQ_coil_MRR.m``：8.3 h，而且自己没收敛到底（``dpsimout`` 只到 2.1e-5）；
* 加速版 MATLAB：665 s（与本 Python 版差 1e-13 相对量级，只来自求和次序）。

用法（仓库根目录）::

    python tools/showcase_solver_compare.py            # relax + hybrid，约 8 分钟
    python tools/showcase_solver_compare.py --only hybrid
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

try:  # 控制台编码兜底（cp936 下打印 U+2212 之类会炸）
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

from gseqfrc import mpath  # noqa: E402

OUT_DIR = ROOT / "outputs" / "showcase"

#: 来自 `修改记录.md` / `accel/README.md` 的记录值（不在本机重跑，作图时单独标注）
RECORDED = {
    "matlab_original": {
        "label": "原文 MATLAB GSEQ_coil_MRR.m",
        "wall_s": 8.3 * 3600.0,
        "note": "8.3 h；而且自己没收敛到底（dpsimout 只到 2.1e-5）",
    },
    "matlab_accel": {
        "label": "加速版 MATLAB（同一套向量化/矩阵乘）",
        "wall_s": 665.0,
        "note": "与本 Python 版差 1e-13 相对量级（只来自求和次序）",
    },
}


def measure(gcm, inner: str, **kwargs) -> dict:
    """跑一遍 ``solve``，把墙钟与残差诊断量记下来。"""
    print(f"\n[compare] 开始：inner={inner!r}"
          + (f" {kwargs}" if kwargs else ""), flush=True)
    t0 = time.perf_counter()
    out = gcm.solve(gcm.SRC_DIR, inner=inner, **kwargs)
    wall = time.perf_counter() - t0
    iters = out.get("nk_inner_iters")
    rec = {
        "inner": inner,
        "wall_s": wall,
        "gs_res": float(out["gs_res"]),
        "defect": float(out["defect"]),
        "C": float(out["C"]),
        "CI": float(out["CI"]),
        "n_calls": (int(len(iters)) if iters is not None else None),
        "n_steps": (int(sum(int(v) for v in iters)) if iters is not None else None),
        "omega": kwargs.get("inner_tol") if inner == "hybrid" else None,
    }
    print(f"[compare] inner={inner!r}：墙钟 {wall:.1f} s，gs_res={rec['gs_res']:.3e}，"
          f"defect={rec['defect']:.3e}，C={rec['C']:.9f}，CI={rec['CI']:.9f}",
          flush=True)
    return rec


def plot(records: list[dict], path: Path) -> None:
    """两联图：左 = 耗时（对数轴），右 = 终态残差。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # 中文标签：Windows 上优先用雅黑/黑体，退化到 DejaVu（那时中文会变方框）
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun",
                                       "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.6), constrained_layout=True)
    # 给底部留一条注脚的位置（免得注脚压到数据点上）
    fig.get_layout_engine().set(rect=(0.0, 0.07, 1.0, 0.93))

    # ---- (a) 耗时 -----------------------------------------------------------
    labels = [RECORDED["matlab_original"]["label"], RECORDED["matlab_accel"]["label"]]
    times = [RECORDED["matlab_original"]["wall_s"], RECORDED["matlab_accel"]["wall_s"]]
    kinds = ["recorded", "recorded"]
    colors = ["#b0b0b0", "#b0b0b0"]
    for r in records:
        if r["inner"] == "relax":
            labels.append("本 Python 版 · 原文逐点松弛（参照路径）")
            colors.append("#d62728")
        else:
            labels.append("本 Python 版 · 加速版默认路径")
            colors.append("#1f77b4")
        times.append(r["wall_s"])
        kinds.append("measured")

    y = np.arange(len(labels))[::-1]
    ax[0].barh(y, times, color=colors, edgecolor="black", linewidth=0.6, height=0.62)
    for yi, (lab, t, k) in zip(y, zip(labels, times, kinds)):
        txt = f"{t:,.1f} s" if t < 3600 else f"{t / 3600:.1f} h"
        if k == "recorded":
            txt += "（记录值）"
        ax[0].text(t * 1.25, yi, txt, va="center", fontsize=9.5)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(labels, fontsize=9.5)
    ax[0].set_xscale("log")
    ax[0].set_xlim(1.0, 3.0e5)
    ax[0].set_xlabel("求解墙钟 [s]（对数轴）")
    ax[0].set_title("(a) 同一个算例的求解耗时", fontsize=11)
    ax[0].grid(axis="x", alpha=0.3)
    ratio = RECORDED["matlab_original"]["wall_s"] / min(r["wall_s"] for r in records)
    ax[0].text(0.98, 0.04,
               f"原文 → 加速版默认路径：快约 {ratio:,.0f} 倍",
               transform=ax[0].transAxes, ha="right", fontsize=10,
               bbox=dict(boxstyle="round", fc="lightyellow", ec="gray"))

    # ---- (b) 终态残差 -------------------------------------------------------
    from matplotlib.lines import Line2D

    ax[1].axhline(1e-5, ls="--", lw=1.1, color="green", zorder=1)
    ax[1].axhline(4.75e-6, ls=":", lw=1.4, color="gray", zorder=1)
    for i, r in enumerate(records):
        color = "#d62728" if r["inner"] == "relax" else "#1f77b4"
        # 圆 = 方程残差 gs_res；方 = 松弛映射固定点缺陷 defect（两种量分开画）
        ax[1].scatter([i - 0.18], [r["gs_res"]], marker="o", s=140, color=color,
                      edgecolor="black", linewidth=0.9, zorder=4)
        ax[1].scatter([i + 0.18], [r["defect"]], marker="s", s=120, color=color,
                      edgecolor="black", linewidth=0.9, zorder=4)
        ax[1].annotate(f"{r['gs_res']:.3e}", (i - 0.18, r["gs_res"]),
                       textcoords="offset points", xytext=(8, 7), fontsize=9,
                       color=color)
        ax[1].annotate(f"{r['defect']:.3e}", (i + 0.18, r["defect"]),
                       textcoords="offset points", xytext=(8, -14), fontsize=9,
                       color=color)
    handles = [
        Line2D([], [], marker="o", ls="none", markerfacecolor="0.35",
               markeredgecolor="black", markersize=9, label="方程残差 gs_res"),
        Line2D([], [], marker="s", ls="none", markerfacecolor="0.35",
               markeredgecolor="black", markersize=8, label="固定点缺陷 defect"),
        Line2D([], [], ls="--", color="green", label="内层判据 1e-5（配方口径）"),
        Line2D([], [], ls=":", color="gray", label="原文路径方程残差 4.75e-6（记录值互印）"),
    ]
    ax[1].legend(handles=handles, loc="upper center", fontsize=9, ncol=2,
                 framealpha=0.92)
    ax[1].set_yscale("log")
    ax[1].set_xticks(range(len(records)))
    ax[1].set_xticklabels(
        [("原文逐点松弛\n（参照路径）" if r["inner"] == "relax" else "加速版默认路径")
         for r in records], fontsize=9.5)
    ax[1].set_xlim(-0.6, len(records) - 0.4)
    lo = min(min(r["gs_res"], r["defect"]) for r in records)
    hi = max(max(r["gs_res"], r["defect"]) for r in records)
    ax[1].set_ylim(lo / 1.9, hi * 4.2)
    ax[1].set_ylabel("终态残差（无量纲）")
    ax[1].set_title("(b) 终态残差诊断（圆 = 方程残差，方 = 固定点缺陷）", fontsize=11)
    ax[1].grid(alpha=0.3)

    fig.suptitle("Fig6 自由边界求解器：原文 vs 加速版（同一机器、同一算例）", fontsize=12.5)
    # 注脚放在图外底部，绝不压数据点
    fig.text(0.5, 0.015,
             "原文那句「dpsim ≤ 1e-9」只是每步 ψ 的变化量；方程残差实测仍有 4.75e-6"
             "——判收敛要看残差。",
             ha="center", va="bottom", fontsize=9.5,
             bbox=dict(boxstyle="round", fc="#fff3cd", ec="gray", alpha=0.9))
    fig.savefig(path, dpi=140)
    print(f"[compare] 图：{path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=("hybrid", "relax"), default=None,
                    help="只跑其中一条（默认两条都跑）")
    ap.add_argument("--replot", action="store_true",
                    help="不重算，直接用已有 solver_compare.json 重画（改图用）")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.replot:
        payload = json.loads((OUT_DIR / "solver_compare.json").read_text(
            encoding="utf-8"))
        plot(payload["measured"], OUT_DIR / "solver_compare.png")
        return 0

    mpath.addpath("second_paper/Fig6")
    gcm = mpath.load("GSEQ_coil_MRR")

    wanted = [args.only] if args.only else ["relax", "hybrid"]
    records: list[dict] = []
    for inner in wanted:
        if inner == "relax":
            records.append(measure(gcm, "relax"))
        else:
            records.append(measure(gcm, "hybrid", inner_tol=1e-5))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"recorded": RECORDED, "measured": records,
               "machine": {"python": sys.version.split()[0],
                           "numpy": __import__("numpy").__version__,
                           "scipy": __import__("scipy").__version__}}
    (OUT_DIR / "solver_compare.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[compare] 数据：{OUT_DIR / 'solver_compare.json'}")
    plot(records, OUT_DIR / "solver_compare.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
