"""验证"分层停止判据"：每层按**实测速率外推 + 剩余预算**停，而不是写死的轮数上限。

对照三组：
  ① 默认配置——必须逐位不变（新判据在默认下都不该触发）；
  ② 中层判据放到 1e-6（原来 47 s 的那条）；
  ③ CI 限步 ±0.05（原来 30.8 s 的那条）。

用法：`python tools/stopping_layers.py`
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

_paths: list[str] = []
_relax: list[int] = []
_stop: list[str] = []
_hybrid = nk.solve_inner_hybrid


def _counting(*args, **kwargs):
    out = _hybrid(*args, **kwargs)
    _paths.append(out.get("path", "IMEX"))
    _relax.append(out.get("relax_rounds", 0))
    _stop.append(str(out.get("relax_stop", "")))
    return out


nk.solve_inner_hybrid = _counting

FIG6 = ROOT / "GSEQ-FRC-open" / "second_paper" / "Fig6"
s = gcm._setup(FIG6)
psib_coil = gcm._coil_green(s)[3]
G2 = gcm._grz_jz(s)[0]

BASE = dict(ntout=30, CI_tol=5e-4, itCI_max=200, verbose=False, inner_tol=1e-5)


def run(label, out_tol=2e-9, dci_max=None):
    _paths.clear(); _relax.clear(); _stop.clear()
    pr = nk.Problem(s, psib_coil, G2)
    t0 = time.perf_counter()
    fast = nk.full_case(pr, 1.0, psi_base=None, out_tol=out_tol, dci_max=dci_max,
                        **BASE)
    wall = time.perf_counter() - t0
    steps = [int(r[3]) for r in fast["outer"]]
    res = float((fast["F"].max() - fast["F"].min())
                / (fast["psi"].max() - fast["psi"].min()))
    print(f"\n=== {label}")
    print(f"  墙钟 {wall:6.2f} s  CI 轮 {len(fast['CI_hist']) + 1}  内层调用 "
          f"{len(steps)}  步数 {sum(steps)}  最贵 {max(steps)}")
    print(f"  终态 C={float(fast['C']):.12f}  CI={float(fast['CI']):.12f}  res={res:.2e}")
    print(f"  兜底：{sum(1 for p in _paths if p != 'IMEX')} 次 / "
          f"{sum(_relax):,} 轮；外推判死触发 "
          f"{sum(1 for t in _stop if t)} 次")
    d = fast.get("diag", {})
    print(f"  分层诊断：中层平台 {d.get('mid_plateau')}；外层卡死 {d.get('ci_stuck')}；"
          f"外层速率 {d.get('ci_rate')}")
    return fast, wall


base_fast, base_wall = run("① 默认配置（应与原来逐位一致）")
run("② 中层判据 1e-6（原来 47 s）", out_tol=1e-6)
run("③ CI 限步 ±0.05（原来 30.8 s）", dci_max=0.05)

print("\n默认配置复核：")
print(f"  C  = {float(base_fast['C'])!r}  基准 1.000315231002813  "
      f"{'✓' if abs(float(base_fast['C']) - 1.000315231002813) < 1e-15 else '✗'}")
print(f"  CI = {float(base_fast['CI'])!r}  基准 0.7898532668537745  "
      f"{'✓' if abs(float(base_fast['CI']) - 0.7898532668537745) < 1e-15 else '✗'}")
