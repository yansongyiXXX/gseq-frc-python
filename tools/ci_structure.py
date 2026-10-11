"""诊断：三层嵌套里，贵的那些内层调用出现在哪一层、被什么触发。

`full_case` 已经把每轮 itout 的记录放在返回值的 `outer` 里：
`(itCI, itout, dpsimout, 内层步数, 内层残差)`。这里把它摊开看。

用法：`python tools/ci_structure.py [out_tol]`
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from gseqfrc import mpath  # noqa: E402

mpath.addpath("second_paper/Fig6")
gcm = mpath.load("GSEQ_coil_MRR")
nk = gcm._load_nk_inner()

# 逐次记录内层调用走的路径（IMEX / 原文松弛兜底）与兜底轮数
_paths: list[str] = []
_relax: list[int] = []
_stall: list[str] = []
_hybrid = nk.solve_inner_hybrid


def _counting_hybrid(*args, **kwargs):
    out = _hybrid(*args, **kwargs)
    _paths.append(out.get("path", "IMEX"))
    _relax.append(out.get("relax_rounds", 0))
    _stall.append(out.get("stall", ""))
    return out


nk.solve_inner_hybrid = _counting_hybrid

# 兜底那条逐点松弛的"每 500 轮残差轨迹"：它到底第几轮进平台
_relax_traj: list[list[tuple[int, float]]] = []
_cur_traj: list[tuple[int, float]] = []
_relax_rounds_fn = nk._relax_rounds


def _counting_relax(pr, psi, psib, mode, shape, C_target, lam, rounds, **kw):
    out = _relax_rounds_fn(pr, psi, psib, mode, shape, C_target, lam, rounds, **kw)
    F, _C, ok = pr.residual(out, mode, shape, C_target, lam)
    _cur_traj.append((rounds, nk.res_range(out, F) if ok else float("inf")))
    return out


nk._relax_rounds = _counting_relax

_real_hybrid2 = nk.solve_inner_hybrid


def _hybrid_with_traj(*args, **kwargs):
    global _cur_traj
    _cur_traj = []
    out = _real_hybrid2(*args, **kwargs)
    if _cur_traj:
        _relax_traj.append(list(_cur_traj))
    return out


nk.solve_inner_hybrid = _hybrid_with_traj

OUT_TOL = float(sys.argv[1]) if len(sys.argv) > 1 else 2e-9

s = gcm._setup(ROOT / "GSEQ-FRC-open" / "second_paper" / "Fig6")
pr = nk.Problem(s, gcm._coil_green(s)[3], gcm._grz_jz(s)[0])

t0 = time.perf_counter()
fast = nk.full_case(pr, 1.0, psi_base=None, ntout=30, CI_tol=5e-4, itCI_max=200,
                    verbose=False, inner_tol=1e-5, out_tol=OUT_TOL)
dt = time.perf_counter() - t0

outer = [(int(a), int(b), float(c), int(d), float(e)) for a, b, c, d, e, _f in fast["outer"]]
steps = np.array([r[3] for r in outer])
print(f"out_tol={OUT_TOL:g}  全量 {dt:.2f} s  CI 轮 {max(r[0] for r in outer)}  "
      f"内层调用 {len(outer)}  步数合计 {steps.sum()}  终态 C={fast['C']:.9f} "
      f"CI={fast['CI']:.9f}")
print(f"\n{'CI':>3}{'itout':>6}{'dpsimout':>12}{'内层步数':>9}{'残差':>11}")
print("-" * 44)
for itCI, itout, dps, it, rng in outer:
    flag = "  <==" if it > 30 else ""
    if itout <= 6 or it > 30 or itout == 30:
        print(f"{itCI:>3}{itout:>6}{dps:>12.3e}{it:>9}{rng:>11.2e}{flag}")

print("\n每轮 CI 的步数与最贵的一次调用：")
for ci in sorted({r[0] for r in outer}):
    sub = [r for r in outer if r[0] == ci]
    s_ = np.array([r[3] for r in sub])
    print(f"  CI{ci}: 轮数 {len(sub):2d}  步数 {s_.sum():5d}  "
          f"最贵 {s_.max():3d} 步（第 {int(np.argmax(s_)) + 1} 轮 itout）  "
          f"首轮 {s_[0]:3d}  其余平均 {(s_.sum() - s_[0]) / max(len(s_) - 1, 1):.1f}")

print("\n逐次调用：步数 > 20 或走了兜底的")
print(f"{'#':>4}{'CI':>4}{'itout':>6}{'步数':>6}{'dpsimout':>12}{'路径':>16}"
      f"{'兜底轮':>8}  停止原因")
for i, (itCI, itout, dps, it, rng) in enumerate(outer):
    path = _paths[i] if i < len(_paths) else "?"
    if it > 20 or path != "IMEX":
        print(f"{i:>4}{itCI:>4}{itout:>6}{it:>6}{dps:>12.3e}{path:>16}"
              f"{_relax[i]:>8}  {_stall[i]}")
print(f"\n兜底调用 {sum(1 for p in _paths if p != 'IMEX')} 次，"
      f"兜底轮数合计 {sum(_relax):,}")

if _relax_traj:
    t = _relax_traj[0]
    print("\n第一条兜底轨迹（每 500 轮一次判据）——平台在哪：")
    print(f"{'累计轮数':>10}{'残差':>13}{'本轮降幅':>12}")
    prev = None
    for k, (rounds, r) in enumerate(t[:20], 1):
        drop = "" if prev is None else f"{prev / r:>12.3f}"
        print(f"{k * rounds:>10}{r:>13.3e}{drop}")
        prev = r
    if len(t) > 20:
        print(f"  … 共 {len(t)} 段，末段残差 {t[-1][1]:.3e}")
