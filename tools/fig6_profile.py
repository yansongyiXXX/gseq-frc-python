"""自由边界求解器（Fig6 那条）的耗时剖面：时间到底花在哪一层、哪种操作上。

**诊断脚本，不改程序树。** 做法是把 `nk_inner` 里几个热点方法包一层计时/计数，
然后跑一次完整的冷启动 `full_case`，把结果按"层"和"操作"两种口径汇总。

用法：`python tools/fig6_profile.py`
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from gseqfrc import mio, mpath  # noqa: E402

mpath.addpath("second_paper/Fig6")
gcm = mpath.load("GSEQ_coil_MRR")
nk = gcm._load_nk_inner()

T: dict[str, float] = defaultdict(float)
N: dict[str, int] = defaultdict(int)


def wrap(cls, name, label=None, accumulate=None):
    """给类方法套一层计时/计数。`accumulate` 为真时按调用累计额外量（如步数）。"""
    original = getattr(cls, name)
    label = label or f"{cls.__name__}.{name}"

    def wrapper(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = original(self, *args, **kwargs)
        T[label] += time.perf_counter() - t0
        N[label] += 1
        return out

    setattr(cls, name, wrapper)


class _LuProxy:
    """包住 SuperLU，统计稀疏回代的次数与总时间。"""

    def __init__(self, lu):
        self._lu = lu

    def solve(self, b):
        t0 = time.perf_counter()
        out = self._lu.solve(b)
        T["SuperLU.solve（稀疏回代）"] += time.perf_counter() - t0
        N["SuperLU.solve（稀疏回代）"] += 1
        return out

    def __getattr__(self, item):
        return getattr(self._lu, item)


_real_lu_full = nk.Problem.lu_full

#: `--fast-lu`：换 SuperLU 的排序（正对称模式 + MMD）。隔离测量里回代快 2.5 倍，
#: 但端到端反而慢 4.7 倍——本脚本就是为了看清这件事。
_FAST_LU = "--fast-lu" in sys.argv


def _lu_full(self):
    if _FAST_LU:
        from scipy.sparse.linalg import splu
        if getattr(self, "_luF", None) is None:
            self._luF = splu(self.A, permc_spec="MMD_AT_PLUS_A",
                             diag_pivot_thresh=0.0,
                             options=dict(SymmetricMode=True))
        return _LuProxy(self._luF)
    return _LuProxy(_real_lu_full(self))


nk.Problem.lu_full = _lu_full
wrap(nk.Problem, "dp_per_unit_C")
wrap(nk.Problem, "jzmid_per_unit_C")
wrap(nk.Problem, "jzmid")
wrap(nk.Problem, "residual")
wrap(nk.Problem, "rhs_full")
wrap(nk.Problem, "amplitude")
wrap(nk.Problem, "psib_from_psi")
wrap(nk.Problem, "__init__", label="Problem.__init__（建算子）")
wrap(nk.Problem, "lu_A")

_real_hybrid = nk.solve_inner_hybrid
_steps: list[int] = []
_fb: list[int] = []
_paths: list[str] = []
_relax_rounds: list[int] = []


def _hybrid(*args, **kwargs):
    t0 = time.perf_counter()
    out = _real_hybrid(*args, **kwargs)
    T["solve_inner_hybrid（内层一次调用）"] += time.perf_counter() - t0
    N["solve_inner_hybrid（内层一次调用）"] += 1
    _steps.append(out.get("iters", 0))
    _fb.append(out.get("n_fallback", 0))
    _paths.append(out.get("path", "IMEX"))
    _relax_rounds.append(out.get("relax_rounds", 0))
    N["内层迭代步数（合计）"] += out.get("iters", 0)
    return out


nk.solve_inner_hybrid = _hybrid

_real_full = nk.full_case


def _full(*args, **kwargs):
    t0 = time.perf_counter()
    out = _real_full(*args, **kwargs)
    T["full_case（整条自由边界求解）"] += time.perf_counter() - t0
    return out


nk.full_case = _full

# ---------------------------------------------------------------- 跑
s = gcm._setup(Path(ROOT) / "GSEQ-FRC-open" / "second_paper" / "Fig6")
psiw_coil, Br_coil, Bz_coil, psib_coil = gcm._coil_green(s)
G2, int_lin = gcm._grz_jz(s)

t_setup = time.perf_counter()
pr = nk.Problem(s, psib_coil, G2)
pr.lu_full()
t_setup = time.perf_counter() - t_setup

_args = [a for a in sys.argv[1:] if not a.startswith("-")]
OUT_TOL = float(_args[0]) if _args else 2e-9
fast = nk.full_case(pr, 1.0, psi_base=None, ntout=30, CI_tol=5e-4,
                    itCI_max=200, verbose=False, inner_tol=1e-5, out_tol=OUT_TOL)

# ---------------------------------------------------------------- 汇总
total = T["full_case（整条自由边界求解）"]
print(f"\n自由边界求解（冷启动）= {total:.2f} s；另有一次性准备 {t_setup:.2f} s"
      f"（含 _coil_green/_grz_jz 共约 1.2 s）")
print(f"out_tol = {OUT_TOL:g}")
print(f"外层 CI 轮数 = {fast['outer'][-1][0]}；内层调用 {len(fast['inner_iters'])} 次；"
      f"内层步数合计 {sum(fast['inner_iters'])}；"
      f"itout 每轮实际跑了 "
      f"{len(fast['inner_iters']) // max(fast['outer'][-1][0], 1)} 次（上限 30）")

print("\n按操作（时间是累计；'占求解'按整条求解归一；注意嵌套，不能相加）")
print(f"{'操作':<32}{'次数':>8}{'累计 s':>10}{'占求解':>9}{'单次 ms':>10}")
print("-" * 72)
for key in sorted(T, key=lambda k: -T[k]):
    if key.startswith("full_case"):
        continue
    per = T[key] / max(N[key], 1) * 1e3
    print(f"{key:<32}{N[key]:>8}{T[key]:>10.2f}{T[key] / total:>9.1%}{per:>10.3f}")

inner = T["solve_inner_hybrid（内层一次调用）"]
print(f"\n内层合计 {inner:.2f} s（占求解 {inner / total:.1%}），"
      f"平均每次 {sum(_steps) / max(len(_steps), 1):.1f} 步")
print(f"其中稀疏回代 {T['SuperLU.solve（稀疏回代）']:.2f} s"
      f"（{N['SuperLU.solve（稀疏回代）']} 次，平均 "
      f"{T['SuperLU.solve（稀疏回代）'] / max(N['SuperLU.solve（稀疏回代）'], 1) * 1e3:.2f} ms）")
print(f"     源项 dp_per_unit_C {T['Problem.dp_per_unit_C']:.2f} s"
      f"（{N['Problem.dp_per_unit_C']} 次）")
print(f"     残差 residual {T['Problem.residual']:.2f} s"
      f"（{N['Problem.residual']} 次）")
print(f"     右端 rhs_full {T['Problem.rhs_full']:.2f} s"
      f"（{N['Problem.rhs_full']} 次）")

steps = np.array(_steps)
order = np.argsort(steps)[::-1]
print(f"\n内层调用里步数最多的 5 次：{steps[order[:5]].tolist()}"
      f"（合计 {steps[order[:5]].sum()} 步 = 全部 {steps.sum()} 步的 "
      f"{steps[order[:5]].sum() / steps.sum():.0%}）")
print(f"触发回退（relax 兜底）的调用：{sum(1 for v in _fb if v)} 次；"
      f"回退轮数合计 {sum(_fb)}")
from collections import Counter  # noqa: E402

print(f"内层走的路径：{dict(Counter(_paths))}；"
      f"其中原文松弛兜底跑的总轮数 {sum(_relax_rounds):,}")
print(f"步数分布：≤12 步 {int((steps <= 12).sum())} 次，"
      f"13–50 步 {int(((steps > 12) & (steps <= 50)).sum())} 次，"
      f">50 步 {int((steps > 50).sum())} 次")
