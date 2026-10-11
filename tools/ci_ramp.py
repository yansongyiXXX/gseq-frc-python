"""实验：把外层 CI 的单步变化限幅（TokaMaker `nR0_ramp` 的思路）对冷启动有什么影响。

对照量：全量墙钟、内层调用次数、内层步数合计、最贵的一次调用、终态 C/CI/残差。
默认（`dci_max=None`）就是现在那条路径，应当逐位复现 C=1.000315231、CI=0.789853267。

用法：`python tools/ci_ramp.py`
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

# 数一下内层每次调用走的路径（IMEX / 原文松弛兜底）与兜底轮数
_paths: list[str] = []
_relax: list[int] = []
_hybrid = nk.solve_inner_hybrid


def _counting_hybrid(*args, **kwargs):
    out = _hybrid(*args, **kwargs)
    _paths.append(out.get("path", "IMEX"))
    _relax.append(out.get("relax_rounds", 0))
    return out


nk.solve_inner_hybrid = _counting_hybrid

FIG6 = ROOT / "GSEQ-FRC-open" / "second_paper" / "Fig6"
s = gcm._setup(FIG6)
psib_coil = gcm._coil_green(s)[3]
G2 = gcm._grz_jz(s)[0]

BASE = dict(ntout=30, CI_tol=5e-4, itCI_max=200, verbose=False, inner_tol=1e-5,
            out_tol=2e-9)


def run(dci_max):
    _paths.clear()
    _relax.clear()
    pr = nk.Problem(s, psib_coil, G2)
    t0 = time.perf_counter()
    fast = nk.full_case(pr, 1.0, psi_base=None, dci_max=dci_max, **BASE)
    wall = time.perf_counter() - t0
    steps = [int(r[3]) for r in fast["outer"]]
    ci_hist = [float(v) for v in fast["CI_hist"]]
    res = float((fast["F"].max() - fast["F"].min()) / (fast["psi"].max() - fast["psi"].min()))
    return dict(dci_max=dci_max, wall=wall, nci=len(ci_hist) + 1, ncalls=len(steps),
                total=sum(steps), worst=max(steps), ci=float(fast["CI"]),
                C=float(fast["C"]), res=res,
                nfb=sum(1 for p in _paths if p != "IMEX"), relax=sum(_relax),
                first_steps=[steps[i] for i in range(len(steps))
                             if i == 0 or fast["outer"][i][1] == 1][:8],
                ci_path=[1.0] + ci_hist[:8])


print(f"{'dci_max':>9}{'墙钟 s':>9}{'CI 轮':>7}{'内层调用':>9}{'步数合计':>9}"
      f"{'最贵':>6}{'兜底次':>7}{'兜底轮':>8}{'C':>13}{'CI':>12}{'res':>10}")
print("-" * 100)
rows = []
for dm in (None, 0.20, 0.10, 0.05, 0.02):
    r = run(dm)
    rows.append(r)
    tag = "关（现状）" if dm is None else f"{dm:g}"
    print(f"{tag:>9}{r['wall']:>9.2f}{r['nci']:>7}{r['ncalls']:>9}{r['total']:>9}"
          f"{r['worst']:>6}{r['nfb']:>7}{r['relax']:>8}{r['C']:>13.9f}"
          f"{r['ci']:>12.9f}{r['res']:>10.2e}")

print("\nCI 的爬升路径（前几步）：")
for r in rows:
    tag = "关（现状）" if r["dci_max"] is None else f"±{r['dci_max']:g}"
    print(f"  {tag:>9}: " + " → ".join(f"{v:.4f}" for v in r["ci_path"]))
print("\n每轮 CI 的首轮 itout 步数：")
for r in rows:
    tag = "关（现状）" if r["dci_max"] is None else f"±{r['dci_max']:g}"
    print(f"  {tag:>9}: " + " ".join(f"{v:4d}" for v in r["first_steps"]))
