"""CI2 / 中层空转的修复包。

对应 `GSEQ_coil_MRR.py` / `nk_inner.py` 的第 19–21 条修改。**这两个文件一个字都没动**，
本文件只在调用期间临时替换 `nk_inner.full_case`，结束后立即还原。

包含三项彼此独立的修复，都能单独关掉：

| 编号 | 问题 | 开关（默认） |
|---|---|---|
| 19 | CI2 把已收敛的 CI1 解毁成 NaN | `ci_fix=True` |
| 20 | `step_tol` 冷启动时判据分母用了数值种子 | `ci_step_mode_cold="disable"` |
| 21 | 中层 7 轮后纯空转 + 回退被默认参数关掉 | `ci_mid_floor_rel=1e-5`、`ci_fb_rounds=200` |

用法
----
    import GSEQ_CI2_fix as fix
    res = fix.solve(SRC_DIR)          # 与 GSEQ_coil_MRR.solve 同签名
    fix.run(ws)                       # 与 GSEQ_coil_MRR.run 同签名（会写 .mat）
    fix.solve(SRC_DIR, ci_fix=False)  # 逐位回到原程序，便于对拍

命令行直接跑默认算例：`python GSEQ_CI2_fix.py`

------------------------------------------------------------------------
第 19 条：CI2 的 NaN
------------------------------------------------------------------------
`nk_inner.full_case` 最外层用 `Be1 = sqrt(2*mu0*max P_mid)`、`g(CI) = Be1 - Be`
的带保护 secant 调 `CI` 使 `Be1 -> Be`。冷启动实测（`inner="hybrid"`）：

* **CI1**（CI=1.0）2.7–3.8 s 收敛，`Be1/Be = 1.2437`；
* **CI2** 把 CI 一步拉到 0.804（−19.6%），中层首个内层解就在
  "线搜索找不到既降残差、又守住电流下界的步" 处停滞，随后残差算不出来；
* 中层退出条件是 `while itout < ntout and dpsimout > out_tol_eff`，而
  **`nan > x` 恒为 False**，于是 NaN 状态被当成"已收敛"退出；外层拿到
  `Be1 = nan` 后再退出一次，最终返回一个全 NaN 的解。

修法：
1. **`psi` 按 `dCI` 缩放做暖启动（关键）**。约束内嵌把 `Jzmid` 钉在 `Izeta0`，
   而边界通量里线圈项与等离子体项都显式乘了 `CI`
   （`psib = CI*psib_coil + mu0*CI*dR*dZ*(Gsub.T@Jz)`），所以换 CI 时 `psi`
   近似正比于 CI。原来直接拿旧 `psi` 配新 CI，内部与边界差一个 19.6% 的跳变，
   线搜索自然找不到可接受步；改成 `psi *= CI_new/CI` 后 CI2 平滑收敛。
2. **非有限守卫**：子解只要出现 nan/inf 一律不接受，回退到上一个好状态。
3. **CI 步长回退**：子解失败时把 CI 相对上一个好值折半重试（`ci_backtrack` 次）。
4. **`step_ref` 跨 CI 轮次透传**：`step_tol` 打开时判据尺度由本时间步的磁通变化
   决定；不传的话每一轮 CI 都会从自己的暖启动点重新估一个更小的尺度，
   判据越收越紧、中层必然跑满 `ntout`。
5. **secant 交接**：驱动器返回的 `CI` 是"与 ψ 对应的已接受值"，若把配套的
   `(CI, g)` 原样当作 `secant_state` 交给下一次暖启动，下一次就会拿
   "同一个 CI、不同 Il"的两点当斜率。现在改为交接**上一个已接受点**。

物理公式零改动：剖面、边界通量、电流归一化、收敛判据全部沿用 `nk_inner` 原件，
本文件只接管"CI 该不该接受、失败了怎么退"。

------------------------------------------------------------------------
第 20 条：step_mode 的冷启动发散
------------------------------------------------------------------------
`step_tol` 打开时（`step_mode`）判据来自

    inner_tol_eff = max(step_tol * dpsi_step / psi_amp_warm, step_tol_floor)

其中 `psi_amp_warm` 取的是**暖启动态** ψ 的幅值。冷启动时那只是一个数值种子：

    psi_amp_warm = 5.607e-04      # 种子 ψ 的幅值
    dpsi_step    = 5.236e-02      # 走到解的真实变化
    inner_tol_eff = max(5e-3 * 93.4, 1e-5) = 0.467

内层判据被放到 **0.467**，于是 itout=2..5 每次都**1 步就宣布收敛**
（实测 rng = 3.7e-3 / 2.6e-3 / 1.6e-2 / 1.8e-2，全都 < 0.467），中层
"边界通量不动点"的反馈随之断掉（正是 `nk_inner` 模块开头警告过的情形），
状态漂移到 itout=6 时 IMEX 步爆掉（残差 3.7e3 → 1.86e16，缩步重试 45 次后
放弃），回退到原文松弛也救不回来 → NaN。

**语义上** `step_tol` 是"每步相对"判据，需要一个"上一步"作参照；冷启动没有
上一步，这个判据不成立。`GSEQ_coil_MRR.evolve()` 其实已经这么处理了
（首步 `prev is None` 时不传 `step_tol`），本模块只是把同样的守卫补到直接
调用 `solve()` 的路径上。要保留原行为传 `ci_step_mode_cold="keep"`。

实测（`step_tol=5e-3`，i7-12700H）：

| 场景 | step_mode | 结果 | 时间 |
|---|---|---|---|
| 冷启动 + step_tol（原程序） | True | **NaN** | 3.2 s |
| 冷启动 + step_tol（本模块，守卫拦下） | False | `Be1/Be = 1.05e-05` | 9.7 s |
| 暖启动 + step_tol（`evolve` 的用法） | True | `Be1/Be = 1.43e-04` | 2.9 s |
| 暖启动 + step_tol + 带 secant（非默认） | True | `Be1/Be = 1.69e-04` | 22.6 s |

------------------------------------------------------------------------
第 21 条：中层空转 + 回退被关掉
------------------------------------------------------------------------
### (a) 中层 7 轮后纯空转

默认路径（`step_mode=False`）里 `out_tol_eff = out_tol = 2e-9`。插桩实测
`dpsimout` 轨迹：

    itout  1..7 :  5.2e-2 → 8.7e-7      ← 真降，约 10 倍/轮
    itout  8..30:  在 3e-8 ~ 7e-7 之间乱跳，不再下降   ← 已经是噪声
    判据 out_tol_eff = 2.0e-9           ← 比可达地板低 50~100 倍

**30 轮里只有 7 轮在干活**，后 23 轮在追内层求解器的交付噪声。代码里其实
有"中层可达地板"（`mid_floor_rel`），但它被 step_mode 挡住了：

    out_tol_floor_abs = (float(mid_floor_rel) * psi_amp_warm) if step_mode else 0.0

本模块在默认路径上补同样的地板：`out_tol_eff = max(out_tol, ci_mid_floor_rel × ψ幅值)`，
尺度取暖启动态 ψ 的幅值（冷启动第一轮退化为 `pr.maxpsiw`）。
扫参实测（基准为不加地板的紧容差解，ψ 幅值 4.456e-02）：

| `ci_mid_floor_rel` | 时间 | `|Be1/Be-1|` | `max|Δψ|` | 相对 | 每轮 CI 的内层调用 |
|---|---|---|---|---|---|
| 关（`None`） | 10.83 s | 1.05e-05 | 0 | 0 | [30,30,30,30] |
| 3e-5（= 原 `mid_floor_rel`） | 5.40 s | 9.48e-05 | 1.58e-05 | 3.55e-04 | [7,9,5,3] |
| **1e-5（默认）** | **6.55 s** | 1.04e-04 | 2.25e-06 | **5.05e-05** | [8,13,8,6] |
| 3e-6 | 10.08 s | 5.71e-06 | 2.52e-07 | 5.66e-06 | [13,21,30,30] |

默认取 **1e-5**：1.65 倍提速，ψ 相对偏差 5.05e-05，比本代码已接受的
快速内层偏差（文档记 2.8e-4）还小 5 倍多。取 3e-5 能快一倍但偏差
3.55e-04 已超过那条线，所以没选。要完全回到原行为传 `ci_mid_floor_rel=None`。

### (b) 线搜索级的回退保护被默认参数关掉

**先澄清一处容易说错的地方**：内层有**两层**兜底，只有一层是坏的。

| 层 | 位置 | 触发条件 | 现状 |
|---|---|---|---|
| ① 线搜索级回退 | `solve_imex` 的 `fallback_rounds`/`fallback_max` | 线搜索连续减半 + 缩步重试都失败 | **被关掉** |
| ② 外层松弛兜底 | `solve_inner_hybrid` 的 `relax_chunk=500`/`relax_max=20000` | IMEX 未收敛就整段接管 | **一直在跑** |

`nk_inner.py` 里 `full_case(..., fb_rounds=0, fb_max=0, ...)`，而调用方
`_solve_fast_inner`（`GSEQ_coil_MRR.py:500` 起）**没有传**这两个参数，于是取到
`full_case` 的 **0**，覆盖掉了 `solve_inner_hybrid` 自己声明的默认
`fb_rounds=200, fb_max=8`（`nk_inner.py:625`），最终传进 `solve_imex` 的
`fallback_rounds` 就是 0 —— **第 ① 层从来没启用过**（日志里的 `回退次数=0`
指的就是它，`n_fallback`）。

第 ② 层不受影响：用你自己的 `tools/ci_structure.py` 跑 CI2 那次失败，
输出里 `路径=原文松弛兜底、兜底轮=3000 / 2000 / 500`，说明它确实在跑。

本模块在驱动器里把 `fb_rounds`/`fb_max` 补上（`ci_fb_rounds=200`、`ci_fb_max=8`）。
注入后实测停止原因会从"线搜索找不到…"变成"**回退后仍落在无效区**"
（后者是 `solve_imex` 里第 ① 层分支才有的措辞），即第 ① 层代码确实被调到了。
正常收敛时它不触发，所以没有额外开销。

------------------------------------------------------------------------
精度验证：修复后的解算得对不对
------------------------------------------------------------------------
**验收基准是 `outputs/p2-Fig6-relax/psiq.mat`**（harness 流程 `p2-Fig6-relax`，
`run.py:148` 的标题写明"内层用原文逐点松弛……逐位复现冻结基准"）。
三份解在同一张 129×257 网格上，`ψ` 幅值 4.4588e-02：

| 对比 | `max|Δψq|` | 相对 | `|ΔCI|` |
|---|---|---|---|
| 旧快速解（`outputs/p2-Fig6`，09-22 18:48）vs relax 基准 | 1.239e-05 | **2.782e-04** | 6.10e-05 |
| **本次修复版 vs relax 基准** | 1.461e-05 | **3.280e-04** | 1.09e-05 |
| **本次修复版 vs 旧快速解** | 2.241e-06 | **5.030e-05** | 5.00e-05 |

三点结论：

1. 文档里那个 **2.8e-4 就是"旧快速 vs relax 基准"**（实测 2.782e-04），不是与
   任何冻结 `.mat` 的差 —— 口径现在对上了。
2. **修复版与旧快速解只差 5.03e-05**，即本模块把快速路径恢复到了它原先
   （09-22 18:48）正常工作时的那份解。
3. 修复版与忠实基准差 3.28e-04，比旧快速解（2.78e-04）略大但同量级 ——
   仍在本代码已接受的快速内层代价范围内。扫 `CI_tol`（5e-4 → 1e-5）该偏差不变，
   说明它不是收敛容差的残差。

**一条值得追的时间线**：`outputs/p2-Fig6/` 是 **09-22 18:48** 跑出来的有限解，
而 Fig6 的求解器代码在 **19:04 之后**跑同一条流程就会得到 NaN。也就是说
**快速路径在 18:48 还是好的，中间某次改动把它弄坏了**（无 git，无法回溯那 16 分钟）。
本模块把这一路恢复了。

**操作性提示**：
* 树里 Fig6 目录下的 `psiq.mat` / `psiq1.mat`（09-14 同一批）是**更早版本**的产物，
  不是验收基准；验收看 `outputs/`。它们不参与验收，只是绘图脚本的输入。
* 这两个文件会被 `run()` 直接覆盖，而 `figure_6_1.py` 读的正是
  `psiq1.mat` —— 不重跑求解器就出图，画的是目录里那份旧解。
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np


def _load_base():
    """加载同目录的 `GSEQ_coil_MRR.py`。

    harness（`run.py` → `mpath.load`）是按**文件路径**加载模块的，`sys.path`
    里并没有 Fig6 那一层，所以不能只靠 `import GSEQ_coil_MRR`。这里先试普通
    import（直接在该目录下跑时走这条），失败再按路径加载 —— 与
    `GSEQ_coil_MRR._load_nk_inner()` 对 `nk_inner.py` 的做法一致。
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

__all__ = ["solve", "run", "make_ci_driver", "CI_FIX_DEFAULTS"]

#: 默认参数
CI_FIX_DEFAULTS = dict(
    rescale=True,        # 换 CI 时按 dCI 缩放 psi（第 19 条关键修复）
    backtrack=6,         # 子解非有限时 CI 折半重试次数
    mid_floor_rel=1.0e-5,  # 中层可达地板（第 21a 条），None = 关闭
    fb_rounds=200,       # 回退到原文松弛的轮数（第 21b 条），None = 不注入
    fb_max=8,
)


def _psi_scale(pr, psi_arr) -> float:
    """中层地板的尺度：优先用暖启动态 ψ 的幅值，冷启动退化为线圈边界通量幅值。"""
    if psi_arr is not None:
        a = np.asarray(psi_arr, dtype=float)
        s = float(a.max() - a.min())
        if np.isfinite(s) and s > 0.0:
            return s
    s = float(getattr(pr, "maxpsiw", 0.0) or 0.0)
    return s if np.isfinite(s) and s > 0.0 else 0.0


def make_ci_driver(orig, *, rescale: bool = True, backtrack: int = 6,
                   verbose: bool = True, report: dict | None = None,
                   strict: bool = False, mid_floor_rel: float | None = 1.0e-5,
                   fb_rounds: int | None = 200, fb_max: int | None = 8):
    """把 `nk_inner.full_case` 包成"一次只跑一轮 CI"的外部驱动器。

    参数
    ----
    orig          : 原始的 `nk_inner.full_case`
    rescale       : 换 CI 时把 `psi` 按 `CI_new/CI` 缩放（第 19 条关键修复）
    backtrack     : 子解非有限时 CI 折半重试的最大次数（0 = 关闭）
    verbose       : 打印每一轮 CI 的 `CI / Be1 / |Be1/Be-1| / 耗时`
    report        : 可选字典；驱动器把最终收敛诊断写进去。必须这样回传——
                    `GSEQ_coil_MRR._solve_fast_inner` 会重建返回字典。
    strict        : 连一轮有效子解都拿不到时，`True` 抛异常；`False` 照原样
                    返回并置 `report["invalid"]=True`（默认，不改变原有契约）。
    mid_floor_rel : 中层可达地板系数（第 21a 条），`None` = 不干预
    fb_rounds/fb_max : 注入给 `solve_inner_hybrid` 的回退轮数（第 21b 条）

    返回一个与 `full_case` 同签名的函数。所有物理量都转交给 `orig` 计算，
    本函数只决定 CI 的接受与回退、以及几个判据参数。
    """

    def driver(pr, C_base, psi_base=None, ntout=30, CI_tol=5e-4, itCI_max=200,
               psi0=None, CI0=1.0, secant_state=None, psib0=None,
               step_ref=None, **kw):
        Be = float(pr.P["Be"])
        CI = float(CI0)
        psi = None if psi0 is None else np.asarray(psi0, dtype=float)
        psib, sec = psib0, secant_state
        step_ref_loc = step_ref

        best = None      # (CI, sub_out) 最近一次有效子解
        last_out = None  # 最近一次子解（可能是无效的，兜底用）
        good = None      # (CI, psi, psib, sec, step_ref) 最近一个好状态
        inner_all, outer_all, ci_hist, g_hist, trace = [], [], [], [], []
        n_sub = n_bt = 0
        t0 = time.perf_counter()

        for k in range(max(1, int(itCI_max))):
            n_sub += 1
            call_kw = dict(kw)
            # ---- 第 21b 条：补上被 full_case 默认值吃掉的回退参数 ----
            if fb_rounds is not None:
                call_kw.setdefault("fb_rounds", int(fb_rounds))
            if fb_max is not None:
                call_kw.setdefault("fb_max", int(fb_max))
            # ---- 第 21a 条：默认路径也套用"中层可达地板" ----
            if mid_floor_rel is not None and float(mid_floor_rel) > 0.0:
                scale = _psi_scale(pr, psi)
                if scale > 0.0:
                    floor = float(mid_floor_rel) * scale
                    cur_tol = call_kw.get("out_tol")
                    call_kw["out_tol"] = max(
                        float(cur_tol) if cur_tol is not None else 2.0e-9, floor)

            out = orig(pr, C_base, psi_base=None, ntout=ntout, CI_tol=CI_tol,
                       itCI_max=1, psi0=psi, CI0=CI, secant_state=sec,
                       psib0=psib, step_ref=step_ref_loc, **call_kw)
            last_out = out
            inner_all += list(out["inner_iters"])
            outer_all += list(out["outer"])
            ci_hist += list(out["CI_hist"])
            g_hist += list(out["g_hist"])
            # 判据尺度跨 CI 轮次透传（step_tol 打开时才有意义）
            if step_ref_loc is None:
                sr = out.get("dpsi_step")
                if sr is not None and np.isfinite(sr) and float(sr) > 0.0:
                    step_ref_loc = float(sr)

            finite = (bool(np.all(np.isfinite(out["psi"])))
                      and bool(np.isfinite(out["Be1"])))
            g = abs(out["Be1"] / Be - 1.0) if finite else float("inf")
            trace.append((k, CI, float(out["Be1"]) if finite else float("nan"),
                          g, bool(finite)))
            if verbose:
                print("  [CI-fix] k=%2d  CI=%.6f  Be1=%-12s  |Be1/Be-1|=%.3e  "
                      "ok=%d  t=%.2fs"
                      % (k, CI, ("%.6f" % out["Be1"]) if finite else "nan",
                         g, finite, time.perf_counter() - t0))

            if finite:
                best = (CI, out)
                good = (CI, out["psi"], out["psib"], out.get("secant_state"),
                        step_ref_loc)
                if g <= CI_tol:
                    if verbose:
                        print("  [CI-fix] 收敛：|Be1/Be-1|=%.3e <= %g"
                              % (g, CI_tol))
                    break
                CI_next = float(out["CI"])
                if not np.isfinite(CI_next) or CI_next <= 0.0:
                    if verbose:
                        print("  [CI-fix] CI 更新值无效（%r）→ 保留当前解" % CI_next)
                    break
                # 第 19 条关键修复：psi 按 dCI 缩放，内部与新的边界通量一致
                psi = out["psi"] * (CI_next / CI) if rescale else out["psi"]
                psib, sec, CI = out["psib"], out.get("secant_state"), CI_next
            else:
                if good is None or backtrack <= 0 or n_bt >= backtrack:
                    if verbose:
                        print("  [CI-fix] 子解非有限且无可用回退 → 保留最近一个有效解")
                    break
                CI_good, psi_good, psib_good, sec_good, sr_good = good
                n_bt += 1
                CI = 0.5 * (CI_good + CI)
                psi, psib, sec, step_ref_loc = (psi_good, psib_good, sec_good,
                                                sr_good)
                if verbose:
                    print("  [CI-fix] 回退 #%d → CI=%.6f" % (n_bt, CI))

        if best is None:
            msg = ("所有 CI 子解都落在无效区（nan/inf）；外层归一化未收敛。"
                   "注意：若失败发生在第一轮 CI 的**中层**内部，这一层驱动器"
                   "无法补救。")
            print("  [CI-fix] 警告：" + msg)
            if strict:
                raise RuntimeError("CI-fix：" + msg)
            out = dict(last_out) if last_out is not None else {}
            out.update(CI=CI, CI_hist=ci_hist, g_hist=g_hist, outer=outer_all,
                       inner_iters=inner_all, wall=time.perf_counter() - t0)
            if report is not None:
                report.clear()
                report.update(converged=False, invalid=True, reason=msg,
                              g_final=float("inf"), final_CI=float(CI),
                              n_sub=n_sub, n_backtrack=n_bt, trace=trace,
                              wall=out["wall"])
            return out

        g_final = abs(float(out["Be1"]) / Be - 1.0)
        # secant 交接：横坐标必须与返回的 CI 不同，否则下一步会拿
        # "同一个 CI、不同 Il"的两点当斜率，把 CI 甩飞（实测跑满 200 轮）。
        sec_out = None
        if len(ci_hist) >= 2 and float(ci_hist[-2]) != float(best[0]):
            sec_out = (float(ci_hist[-2]), float(g_hist[-2]))
        out = dict(best[1])
        out.update(
            CI=best[0], CI_hist=ci_hist, g_hist=g_hist, outer=outer_all,
            inner_iters=inner_all, wall=time.perf_counter() - t0,
            secant_state=sec_out,
            ci_fix=dict(converged=bool(g_final <= CI_tol), invalid=False,
                        g_final=float(g_final), final_CI=float(best[0]),
                        n_sub=n_sub, n_backtrack=n_bt, trace=trace),
        )
        if report is not None:
            report.clear()
            report.update(out["ci_fix"])
            report["wall"] = out["wall"]
        if verbose:
            print("  [CI-fix] 结束：CI=%.6f  Be1=%.6f  |Be1/Be-1|=%.3e  "
                  "子解 %d 次，回退 %d 次，总 %.2fs"
                  % (best[0], out["Be1"], g_final, n_sub, n_bt, out["wall"]))
        return out

    return driver


def _apply_step_mode_cold_guard(kwargs, mode):
    """第 20 条：冷启动时 `step_tol` 的"每步相对"判据没有参照步，忽略之。"""
    if (mode == "disable" and kwargs.get("step_tol")
            and kwargs.get("psi0") is None and kwargs.get("psib0") is None):
        print("  [CI-fix] 冷启动（无 psi0/psib0）：step_tol 的「每步相对」判据"
              "没有参照步，已忽略 step_tol 走紧容差路径；要保留请传 "
              "ci_step_mode_cold='keep'")
        kwargs = dict(kwargs)
        kwargs.pop("step_tol", None)
        kwargs.pop("step_ref", None)
    return kwargs


def solve(src_dir=None, *, ci_fix: bool = True, ci_rescale: bool = True,
          ci_backtrack: int = 6, ci_verbose: bool = True,
          ci_strict: bool = False, ci_step_mode_cold: str = "disable",
          ci_mid_floor_rel: float | None = 1.0e-5,
          ci_fb_rounds: int | None = 200, ci_fb_max: int | None = 8, **kwargs):
    """`GSEQ_coil_MRR.solve` 的同签名版本，默认启用第 19–21 条修复。

    `ci_fix=False` 时直接转交原程序（逐位一致，便于对拍）。其余 `kwargs`
    原样传给 `GSEQ_coil_MRR.solve`。
    """
    if src_dir is None:
        src_dir = _base.SRC_DIR
    kwargs = _apply_step_mode_cold_guard(kwargs, ci_step_mode_cold)
    nk = _base._load_nk_inner()
    if not ci_fix:
        return _base.solve(src_dir, **kwargs)
    orig = nk.full_case
    report: dict = {}
    nk.full_case = make_ci_driver(orig, rescale=ci_rescale,
                                  backtrack=ci_backtrack, verbose=ci_verbose,
                                  report=report, strict=ci_strict,
                                  mid_floor_rel=ci_mid_floor_rel,
                                  fb_rounds=ci_fb_rounds, fb_max=ci_fb_max)
    try:
        out = _base.solve(src_dir, **kwargs)
    finally:
        nk.full_case = orig
    if isinstance(out, dict):
        out["ci_fix"] = report
    return out


def run(ws, *, verbose: bool = False, ci_fix: bool = True,
        ci_rescale: bool = True, ci_backtrack: int = 6,
        ci_verbose: bool = True, ci_step_mode_cold: str = "disable",
        ci_mid_floor_rel: float | None = 1.0e-5,
        ci_fb_rounds: int | None = 200, ci_fb_max: int | None = 8, **kwargs) -> None:
    """`GSEQ_coil_MRR.run` 的同签名版本（会写 `psiq.mat` / `psiq1.mat`）。"""
    nk = _base._load_nk_inner()
    if not ci_fix:
        return _base.run(ws, verbose=verbose, **kwargs)
    orig = nk.full_case
    nk.full_case = make_ci_driver(orig, rescale=ci_rescale,
                                  backtrack=ci_backtrack, verbose=ci_verbose,
                                  mid_floor_rel=ci_mid_floor_rel,
                                  fb_rounds=ci_fb_rounds, fb_max=ci_fb_max)
    try:
        return _base.run(ws, verbose=verbose, **kwargs)
    finally:
        nk.full_case = orig


if __name__ == "__main__":
    import sys

    no_fix = "--no-fix" in sys.argv
    res = solve(_base.SRC_DIR, ci_fix=not no_fix, ci_verbose=True)
    info = res.get("ci_fix") or {}
    psi = np.asarray(res["psi"], dtype=float)
    print("\n%s" % ("=" * 62))
    print("外层归一化收敛 : %s   (|Be1/Be-1| = %.3e)"
          % ("是" if info.get("converged") else "否",
             info.get("g_final", float("nan"))))
    print("CI = %.6f    C = %.6f    Be1 = %.6f" % (res["CI"], res["C"], res["Be1"]))
    print("psi 有限       : %s   范围 %.4e .. %.4e"
          % (bool(np.all(np.isfinite(psi))), np.nanmin(psi), np.nanmax(psi)))
    print("子解/回退次数  : %d / %d" % (info.get("n_sub", 0),
                                        info.get("n_backtrack", 0)))
    print("=" * 62)
