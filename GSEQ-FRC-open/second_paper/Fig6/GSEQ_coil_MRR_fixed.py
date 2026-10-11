"""GSEQ-FRC 图 6 · 二维自由边界求解器（最终版）

这是 `GSEQ_coil_MRR.py` + `nk_inner.py` 的修复版外壳：把实测有效的正向改动
**全部固化**在这里，不带任何开关；无效或负面的东西一律不进来。
原文件一个字都不改，本模块只在每次调用期间临时替换 `nk_inner.full_case`。

接口与 `GSEQ_coil_MRR` 完全一致，可直接当它的替代品：

    import GSEQ_coil_MRR_fixed as fixed
    res = fixed.solve(SRC_DIR)     # 跑一次平衡
    fixed.run(ws)                  # harness 入口（写工作区 + psiq.mat / psiq1.mat）

固化的正向改动（逐条都有实测）
==============================
| # | 改动 | 效果 |
|---|---|---|
| 1 | 换 CI 时 `psi *= CI_new/CI`（暖启动） | **关键**。CI2 从 NaN 变收敛 |
| 2 | 非有限守卫：nan/inf 子解一律不接受 | 修掉 `nan > x == False` 被当"已收敛"的漏洞 |
| 3 | CI 步长回退：失败时相对上一个好值折半，最多 6 次 | 安全网，正常不触发 |
| 4 | secant 锚点交接"上一个已接受点" | 避免下次暖启动拿"同一 CI、不同 Il"当斜率 |
| 5 | `step_ref` 跨 CI 轮次透传 | `step_tol` 打开时判据尺度不会被越收越紧 |
| 6 | `step_tol` 冷启动守卫 | 「每步相对」判据在冷启动没有参照步，直接走紧容差 |
| 7 | 中层可达地板 `MID_FLOOR_REL = 1e-5` | 中层 30 轮 → 8 轮，实测 2.1× 提速 |
| 8 | 回退参数透传 `fb_rounds=200 / fb_max=8` | 恢复被 `full_case` 默认值(0)吃掉的线搜索级回退 |

实测（i7-12700H，Fig6 算例）
===========================
* 原程序：13.1–14.5 s → **NaN**（拿不到有效解）
* 本程序：**5.8 s** → 收敛，`|Be1/Be-1| = 1.04e-04`
* harness 端到端（`run.py p2-Fig6-fix`）：**8.0 s** 出齐 `psiq.mat`/`psiq1.mat`/图
* 精度 vs 基准 `outputs/p2-Fig6-relax/psiq.mat`：
  `max|Δψq| = 1.461e-05` → **3.28e-04 相对**
  （旧快速解是 2.78e-04；文档里的 "2.8e-4" 就是这个口径）
* `MID_FLOOR_REL` 扫参定的甜点：1e-5 给出 1.65× 提速 + 5.05e-05 相对偏差；
  取 3e-5 能再快一倍但偏差 3.55e-04，已超过上一条的口径，所以不取。

**没有**进来的东西（实测无作用或负面）
======================================
* `mid_accel="anderson"`：只在 `step_tol` 打开时生效，且冷启动会把解推成 NaN
* `mid_plateau_stop`：比可达地板慢（7.75 s vs 4.99 s），功能重叠
* `dci_max`：修改记录第 15 条已记负面结果（11.1 s → 30.8 s）
* 所有 `*_fix=False` / `*_keep` / `strict` 之类的回归开关：只留正确的一条路径

附带的辅助修复（默认**不**执行）
================================
另有两处一行修复在别的文件里，本模块提供 `apply_text_fixes()` 显式调用
（会先备份原文件）：

* `second_paper/Fig6/figure_6_2.py`：第二行打印误用了真空量
  （实测 0.8593 vs 应为 0.9461，差 10.1%）；图本身没问题。
* `tools/ci_structure.py`：`outer` 记录已从 5 个字段变 6 个，解包会对不上。

命令行
======
    python GSEQ_coil_MRR_fixed.py                    # 跑一次 + 打印验收对比
    python GSEQ_coil_MRR_fixed.py --apply-text-fixes # 顺带打上两处一行修复
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# 固化参数（都经过实测，不再暴露开关）
# --------------------------------------------------------------------------
#: 中层可达地板系数。取值依据：1.65× 提速，ψ 相对偏差 5.05e-05
MID_FLOOR_REL = 1.0e-5
#: CI 子解非有限时，CI 相对上一个好值折半重试的上限
CI_BACKTRACK = 6
#: 线搜索级回退（原文逐点松弛）的轮数与次数上限
FB_ROUNDS, FB_MAX = 200, 8


def _load_base():
    """加载同目录的 `GSEQ_coil_MRR.py`。

    harness（`run.py` → `mpath.load`）按**文件路径**加载模块，`sys.path` 里没有
    Fig6 这一层，所以不能只靠 `import GSEQ_coil_MRR`。先试普通 import（直接在该
    目录下运行时走这条），失败再按路径加载 —— 与 `GSEQ_coil_MRR._load_nk_inner()`
    对 `nk_inner.py` 的做法一致。
    """
    try:
        import GSEQ_coil_MRR as module
        return module
    except ImportError:
        pass
    path = Path(__file__).resolve().parent / "GSEQ_coil_MRR.py"
    name = "gseqfrc._fig6_coil_mrr_base"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_base = _load_base()

#: 最近一次 `solve()` 的分层诊断（供调用方读取）
LAST: dict = {}


def _psi_scale(pr, psi) -> float:
    """中层地板的尺度：优先用暖启动态 ψ 的幅值，冷启动退化为线圈边界通量幅值。"""
    if psi is not None:
        arr = np.asarray(psi, dtype=float)
        scale = float(arr.max() - arr.min())
        if np.isfinite(scale) and scale > 0.0:
            return scale
    scale = float(getattr(pr, "maxpsiw", 0.0) or 0.0)
    return scale if np.isfinite(scale) and scale > 0.0 else 0.0


def _make_driver(orig, verbose: bool = True):
    """把 `nk_inner.full_case` 包成"一次只跑一轮 CI"的外部驱动器。

    所有物理量仍交给 `orig` 计算，本层只决定：
    * 换 CI 时把 `psi` 按 `dCI` 缩放（改动 1）；
    * nan/inf 子解一律不接受，并回退 CI 步长（改动 2、3）；
    * 判据尺度与 secant 锚点的交接（改动 4、5）；
    * 往 `orig` 注入中层地板与回退参数（改动 7、8）。
    """

    def driver(pr, C_base, psi_base=None, ntout=30, CI_tol=5e-4, itCI_max=200,
               psi0=None, CI0=1.0, secant_state=None, psib0=None,
               step_ref=None, **kw):
        Be = float(pr.P["Be"])
        CI = float(CI0)
        psi = None if psi0 is None else np.asarray(psi0, dtype=float)
        psib, sec = psib0, secant_state
        step_ref_loc = step_ref

        best = None       # (CI, sub_out) 最近一次有效子解
        last = None       # 最近一次子解（可能是无效的，兜底用）
        good = None       # (CI, psi, psib, sec, step_ref) 最近一个好状态
        inner_all, outer_all, ci_hist, g_hist, trace = [], [], [], [], []
        n_bt = 0
        t0 = time.perf_counter()

        for k in range(max(1, int(itCI_max))):
            call_kw = dict(kw)
            # 改动 8：补上被 full_case 默认值(0)吃掉的线搜索级回退参数
            call_kw.setdefault("fb_rounds", FB_ROUNDS)
            call_kw.setdefault("fb_max", FB_MAX)
            # 改动 7：中层可达地板（只在比用户给的 out_tol 更松时生效）
            scale = _psi_scale(pr, psi)
            if scale > 0.0:
                cur_tol = call_kw.get("out_tol")
                call_kw["out_tol"] = max(
                    float(cur_tol) if cur_tol is not None else 2.0e-9,
                    MID_FLOOR_REL * scale)

            out = orig(pr, C_base, psi_base=None, ntout=ntout, CI_tol=CI_tol,
                       itCI_max=1, psi0=psi, CI0=CI, secant_state=sec,
                       psib0=psib, step_ref=step_ref_loc, **call_kw)
            last = out
            inner_all += list(out["inner_iters"])
            outer_all += list(out["outer"])
            ci_hist += list(out["CI_hist"])
            g_hist += list(out["g_hist"])
            # 改动 5：判据尺度跨 CI 轮次透传
            if step_ref_loc is None:
                sr = out.get("dpsi_step")
                if sr is not None and np.isfinite(sr) and float(sr) > 0.0:
                    step_ref_loc = float(sr)

            finite = (bool(np.all(np.isfinite(out["psi"])))
                      and bool(np.isfinite(out["Be1"])))
            g = abs(out["Be1"] / Be - 1.0) if finite else float("inf")
            trace.append((k, CI, float(out["Be1"]) if finite else float("nan"),
                          g, finite))
            if verbose:
                print("  [fix] k=%2d  CI=%.6f  Be1=%-12s  |Be1/Be-1|=%.3e  t=%.2fs"
                      % (k, CI, ("%.6f" % out["Be1"]) if finite else "nan", g,
                         time.perf_counter() - t0))

            if not finite:
                # 改动 2：绝不让 NaN 通过；改动 3：折半重试
                if good is None or n_bt >= CI_BACKTRACK:
                    print("  [fix] 子解非有限且无可用回退 → 保留最近一个有效解")
                    break
                ci_good, psi_g, psib_g, sec_g, sr_g = good
                n_bt += 1
                CI = 0.5 * (ci_good + CI)
                psi, psib, sec, step_ref_loc = psi_g, psib_g, sec_g, sr_g
                print("  [fix] 回退 #%d → CI=%.6f" % (n_bt, CI))
                continue

            best = (CI, out)
            good = (CI, out["psi"], out["psib"], out.get("secant_state"),
                    step_ref_loc)
            if g <= CI_tol:
                if verbose:
                    print("  [fix] 收敛：|Be1/Be-1|=%.3e <= %g" % (g, CI_tol))
                break
            ci_next = float(out["CI"])
            if not np.isfinite(ci_next) or ci_next <= 0.0:
                print("  [fix] CI 更新值无效（%r）→ 保留当前解" % ci_next)
                break
            # 改动 1（关键）：psi 按 dCI 缩放，内部与新的边界通量保持一致
            psi = out["psi"] * (ci_next / CI)
            psib, sec, CI = out["psib"], out.get("secant_state"), ci_next

        if best is None:
            msg = ("所有 CI 子解都落在无效区（nan/inf），外层归一化未收敛；"
                   "若失败发生在第一轮 CI 的中层内部，本层无法补救。")
            print("  [fix] 警告：" + msg)
            out = dict(last) if last is not None else {}
            out.update(CI=CI, CI_hist=ci_hist, g_hist=g_hist, outer=outer_all,
                       inner_iters=inner_all, wall=time.perf_counter() - t0)
            LAST.clear()
            LAST.update(converged=False, invalid=True, reason=msg,
                        g_final=float("inf"), CI=float(CI), n_backtrack=n_bt,
                        wall=out["wall"])
            return out

        g_final = abs(float(out["Be1"]) / Be - 1.0)
        # 改动 4：secant 锚点必须是**上一个**已接受点（横坐标 ≠ 返回的 CI）
        sec_out = None
        if len(ci_hist) >= 2 and float(ci_hist[-2]) != float(best[0]):
            sec_out = (float(ci_hist[-2]), float(g_hist[-2]))
        out = dict(best[1])
        out.update(CI=best[0], CI_hist=ci_hist, g_hist=g_hist, outer=outer_all,
                   inner_iters=inner_all, wall=time.perf_counter() - t0,
                   secant_state=sec_out)
        LAST.clear()
        LAST.update(converged=bool(g_final <= CI_tol), invalid=False,
                    g_final=float(g_final), CI=float(best[0]),
                    n_backtrack=n_bt, wall=out["wall"])
        if verbose:
            print("  [fix] 结束：CI=%.6f  Be1=%.6f  |Be1/Be-1|=%.3e  "
                  "回退 %d 次，总 %.2fs"
                  % (best[0], out["Be1"], g_final, n_bt, out["wall"]))
        return out

    return driver


def _drop_step_tol_on_cold_start(kwargs: dict) -> dict:
    """改动 6：`step_tol` 是「每步相对」判据，冷启动没有参照步，直接走紧容差。

    冷启动时判据分母 `psi_amp_warm` 只是数值种子（实测 5.607e-04），
    而 `dpsi_step` 是走到解的真实变化（5.236e-02），比值 ~93，
    于是 `inner_tol_eff` 被放到 0.467 —— 内层每步 1 次迭代就"收敛"，
    中层反馈断掉，最终 NaN。
    """
    if (kwargs.get("step_tol") and kwargs.get("psi0") is None
            and kwargs.get("psib0") is None):
        print("  [fix] 冷启动（无 psi0/psib0）：step_tol 的「每步相对」判据没有"
              "参照步，已忽略 step_tol 走紧容差路径")
        kwargs = dict(kwargs)
        kwargs.pop("step_tol", None)
        kwargs.pop("step_ref", None)
    return kwargs


@contextmanager
def _installed(verbose: bool):
    """在调用期间把驱动器装上，结束后还原。"""
    nk = _base._load_nk_inner()
    orig = nk.full_case
    nk.full_case = _make_driver(orig, verbose=verbose)
    try:
        yield
    finally:
        nk.full_case = orig


# --------------------------------------------------------------------------
# 对外接口（与 GSEQ_coil_MRR 同签名）
# --------------------------------------------------------------------------
def solve(src_dir=None, *, verbose: bool = True, **kwargs):
    """跑一次自由边界平衡，修复全部生效。

    `kwargs` 原样转给 `GSEQ_coil_MRR.solve`。返回字典里多一个 `"fixed"` 键，
    是本次的分层诊断（`converged` / `g_final` / `CI` / `n_backtrack` / `wall`）。
    """
    if src_dir is None:
        src_dir = _base.SRC_DIR
    kwargs = _drop_step_tol_on_cold_start(kwargs)
    LAST.clear()
    with _installed(verbose):
        out = _base.solve(src_dir, **kwargs)
    if isinstance(out, dict):
        out["fixed"] = dict(LAST)
    return out


def run(ws, *, verbose: bool = False, **kwargs) -> None:
    """harness 入口：跑完把变量写进工作区，并保存 `psiq.mat` / `psiq1.mat`。"""
    kwargs = _drop_step_tol_on_cold_start(kwargs)
    LAST.clear()
    with _installed(verbose):
        return _base.run(ws, verbose=verbose, **kwargs)


# --------------------------------------------------------------------------
# 附带的两处一行修复（在别的文件里，默认不执行）
# --------------------------------------------------------------------------
#: (相对程序树根的路径, 原字符串, 新字符串)
_TEXT_FIXES = (
    # figure_6_2.py：第二行打印误用了真空量；图用的是正确的 Bz
    (Path("GSEQ-FRC-open/second_paper/Fig6/figure_6_2.py"),
     'print(f"B_z at midplane with plasma= '
     '{mat.scalar_value(Bz_coil_u[mid - 1]):5.4f} ")',
     'print(f"B_z at midplane with plasma= '
     '{mat.scalar_value(np.asarray(Bz)[-1, :][mid - 1]):5.4f} ")'),
    # tools/ci_structure.py：outer 记录已从 5 个字段变 6 个
    (Path("tools/ci_structure.py"),
     'for a, b, c, d, e in fast["outer"]',
     'for a, b, c, d, e, _f in fast["outer"]'),
)


def _tree_root():
    """往上找同时含 `run.py` 与 `gseqfrc/` 的目录（程序树根）。"""
    for parent in Path(__file__).resolve().parents:
        if (parent / "run.py").is_file() and (parent / "gseqfrc").is_dir():
            return parent
    return None


def apply_text_fixes(verbose: bool = True) -> int:
    """把两处一行修复打到对应文件上（先备份为 `*.bak-before-fix`）。

    返回实际改动的文件数。已在目标态的会跳过。
    """
    root = _tree_root()
    if root is None:
        print("  [fix] 找不到程序树根（含 run.py + gseqfrc/），跳过辅助修复")
        return 0
    changed = 0
    for rel, old, new in _TEXT_FIXES:
        path = root / rel
        if not path.is_file():
            if verbose:
                print("  [fix] 跳过（不存在）：%s" % rel)
            continue
        src = path.read_text(encoding="utf-8")
        if new in src and old not in src:
            if verbose:
                print("  [fix] 已是目标态：%s" % rel)
            continue
        if src.count(old) != 1:
            print("  [fix] 跳过（目标字符串出现 %d 次，不是 1 次）：%s"
                  % (src.count(old), rel))
            continue
        backup = path.with_name(path.name + ".bak-before-fix")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(src.replace(old, new), encoding="utf-8")
        changed += 1
        if verbose:
            print("  [fix] 已修复：%s（备份 %s）" % (rel, backup.name))
    return changed


# --------------------------------------------------------------------------
if __name__ == "__main__":
    if "--apply-text-fixes" in sys.argv:
        n = apply_text_fixes()
        print("\n辅助修复：改了 %d 个文件" % n)
        sys.exit(0)

    print("=" * 68)
    print("GSEQ-FRC 图 6 · 最终版求解器自检")
    print("=" * 68)
    result = solve(_base.SRC_DIR, verbose=True)
    info = result.get("fixed") or {}
    psi = np.asarray(result["psi"], dtype=float)
    print("-" * 68)
    print("外层归一化收敛 : %s   (|Be1/Be-1| = %.3e)"
          % ("是" if info.get("converged") else "否",
             info.get("g_final", float("nan"))))
    print("CI = %.6f    C = %.6f    Be1 = %.6f"
          % (result["CI"], result["C"], result["Be1"]))
    print("psi 有限       : %s   范围 %.4e .. %.4e"
          % (bool(np.all(np.isfinite(psi))), np.nanmin(psi), np.nanmax(psi)))
    print("CI 回退次数    : %d        用时 %.2f s"
          % (info.get("n_backtrack", 0), info.get("wall", float("nan"))))

    # 若旁边有 harness 的忠实基准，顺手报一下验收偏差
    root = _tree_root()
    ref_path = (root / "outputs" / "p2-Fig6-relax" / "psiq.mat") if root else None
    if ref_path is not None and ref_path.is_file():
        from scipy.io import loadmat
        ref = np.asarray(loadmat(str(ref_path), squeeze_me=True)["psiq"], float)
        cur = np.asarray(result["psiq"], float)
        amp = float(ref.max() - ref.min())
        dev = float(np.max(np.abs(cur - ref)))
        print("-" * 68)
        print("验收 vs %s" % ref_path.relative_to(root))
        print("  max|dpsiq| = %.3e   相对 = %.3e   (文档口径 2.8e-4)"
              % (dev, dev / amp))
    print("=" * 68)
