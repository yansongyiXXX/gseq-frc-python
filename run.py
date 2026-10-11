"""跑流程：按原文的会话顺序，把 `GSEQ-FRC-open/` 里的脚本跑起来。

原文里每个图/附录都是一段"会话"：先 `addpath` 几个目录，然后按顺序跑几个脚本，
脚本之间靠基本工作区传数据。移植版把每段会话照搬到这里（加路径的顺序、脚本的顺序、
哪一个用上一步留下的工作区，都与原文一致）。

用法（在本目录下）：

    python run.py                     # 跑全部流程
    python run.py --list              # 看有哪些流程
    python run.py p2-Fig6 p2-Fig4     # 只跑其中几个

流程名带 `p1-` / `p2-` 前缀，对应两篇论文；两篇里都有 "Fig4/Fig5/…"，
不加前缀在 Windows 上会撞成同一个产物目录。

产物（脚本 `save` 出的 `.mat`、`print` 出的图）落在 `outputs/<流程名>/` 下。
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gseqfrc import mglobals  # noqa: E402
from gseqfrc import mio  # noqa: E402
from gseqfrc import mpath  # noqa: E402
from gseqfrc import mplot  # noqa: E402
from gseqfrc.workspace import Workspace  # noqa: E402

FIRST = "first_paper"
SECOND = "second_paper"


# --------------------------------------------------------------- 流程的写法
def script(directory: str, name: str, *, chain: bool = False,
           extra: tuple[str, ...] = (), stops: bool = False, **kwargs) -> dict:
    """一个脚本步骤：把目录加进搜索路径，然后调用同名的 `run(ws)`。

    `chain=True` 表示"接着上一步的工作区跑"（原文的会话就是这个意思）；
    否则新开一个干净工作区。`extra` 是这一步还要额外加进路径的目录
    （对应原文算例登记表里的 `extraPath`）。`stops=True` 表示**原文自己**会在这里
    中止（脚本调了仓库里不存在的函数），移植版照抄，不当作失败。

    目录写法与 `mpath.addpath` 一致：相对于程序树 `GSEQ-FRC-open/`。
    """
    return {"dirs": [directory, *extra], "script": name, "chain": chain,
            "stops": stops, "kwargs": kwargs}


# --------------------------------------------------------------- 流程登记表
FLOWS: dict[str, dict] = {
    "p1-1d": {
        "title": "第一篇 · 一维平衡解（RR / MRR1 / MRR2 / 2PE×2 / 3PE / valid1d）",
        "steps": [
            script(f"{FIRST}/figure1_2/RR", "solve_RR"),
            script(f"{FIRST}/figure1_2/MRR1", "solve_MRR1"),
            script(f"{FIRST}/figure1_2/MRR2", "solve_MRR2"),
            script(f"{FIRST}/figure1_2/2PE", "solve_2PE08"),
            script(f"{FIRST}/figure1_2/2PE", "solve_2PE09"),
            script(f"{FIRST}/figure1_2/3PE", "solve_3pe"),
            script(f"{FIRST}/figure1_2/valid1d", "valid1d"),
        ],
    },
    "p1-fig1-2": {
        "title": "第一篇 · 图 1/图 2（一维剖面成图）",
        "steps": [
            script(f"{FIRST}/figure1_2", "MRR_figure1"),
            script(f"{FIRST}/figure1_2", "MRR_figure2"),
        ],
    },
    "p1-fig3": {"title": "第一篇 · 图 3（二维 MRR 平衡剖面）",
             "steps": [script(f"{FIRST}/figure3", "figure3")]},
    "p1-fig4": {"title": "第一篇 · 图 4", "steps": [script(f"{FIRST}/figure4", "figure4")]},
    "p1-fig5": {"title": "第一篇 · 图 5", "steps": [script(f"{FIRST}/figure5", "figure5")]},
    "p1-fig6": {"title": "第一篇 · 图 6", "steps": [script(f"{FIRST}/figure6", "figure6")]},
    "p1-fig7": {"title": "第一篇 · 图 7（零维参数扫描）",
             "steps": [script(f"{FIRST}/figure7", "figure7")]},
    "p1-fig8": {
        "title": "第一篇 · 图 8（GSEQ 二维求解：gen_psib → gseq → 成图）",
        "steps": [
            script(f"{FIRST}/figure8", "gen_psib"),
            script(f"{FIRST}/figure8", "gseq", chain=True, stops=True),
            script(f"{FIRST}/figure8", "figure_8_1"),
            script(f"{FIRST}/figure8", "plt_psi_r", chain=True),
            script(f"{FIRST}/figure8", "figure_8_2", chain=True),
        ],
    },
    "p1-figB1": {"title": "第一篇 · 附录 B1（含壁面的一维验证）",
              "steps": [script(f"{FIRST}/figureB1", "valid1d")]},
    "p2-appendix-A": {
        "title": "第二篇 · 附录 A（诊断图链：figure_A_1 → A_2 → A_3）",
        "steps": [
            script(f"{SECOND}/appendix_A", "figure_A_1"),
            script(f"{SECOND}/appendix_A", "figure_A_2", chain=True),
            script(f"{SECOND}/appendix_A", "figure_A_3", chain=True),
        ],
    },
    "p2-appendix-B-GSEQ": {
        "title": "第二篇 · 附录 B（GSEQ 版：gen_psib → gseq_ver_2 → Figure_B）",
        "steps": [
            script(f"{SECOND}/appendix_B/GSEQ", "gen_psib"),
            script(f"{SECOND}/appendix_B/GSEQ", "gseq_ver_2", chain=True),
            script(f"{SECOND}/appendix_B/appendix_B", "Figure_B"),
        ],
    },
    "p2-appendix-B-grass-ft": {
        "title": "第二篇 · 附录 B（grass_ft 版：gen_psib → gseq_grass_ft → 成图）",
        "steps": [
            script(f"{SECOND}/appendix_B/grass_ft", "gen_psib"),
            script(f"{SECOND}/appendix_B/grass_ft", "gseq_grass_ft", chain=True),
            script(f"{SECOND}/appendix_B/grass_ft", "contour_psi"),
            script(f"{SECOND}/appendix_B/grass_ft", "plot_grass_ft_f90"),
        ],
    },
    "p2-Fig4": {
        "title": "第二篇 · 图 4（二维定边界：solve_MRR → gseq_frc_fixed → 成图）",
        "steps": [
            script(f"{SECOND}/Fig4", "solve_MRR"),
            script(f"{SECOND}/Fig4", "gseq_frc_fixed",
                   extra=(f"{SECOND}/Fig4/gui", f"{FIRST}/figure8")),
            script(f"{SECOND}/Fig4", "figure_4_1", chain=True),
            script(f"{SECOND}/Fig4", "figure_4_2", chain=True),
        ],
    },
    "p2-Fig5": {
        "title": "第二篇 · 图 5（二维定边界 FIX：GSEQ_FRC_fixed_FIX → 成图）",
        "steps": [
            script(f"{SECOND}/Fig5", "GSEQ_FRC_fixed_FIX",
                   extra=(f"{FIRST}/figure8",)),
            script(f"{SECOND}/Fig5", "fix5_1", chain=True),
            script(f"{SECOND}/Fig5", "fix5_2", chain=True),
        ],
    },
    "p2-Fig6": {
        "title": "第二篇 · 图 6（二维自由边界：快速内层 → 成图）",
        "steps": [
            script(f"{SECOND}/Fig6", "GSEQ_coil_MRR"),
            script(f"{SECOND}/Fig6", "figure_6_1", chain=True),
            script(f"{SECOND}/Fig6", "figure_6_2", chain=True),
        ],
    },
    "p2-Fig6-relax": {
        "title": "第二篇 · 图 6（同上，但内层用原文逐点松弛：约 7 分钟，逐位复现冻结基准）",
        "steps": [
            script(f"{SECOND}/Fig6", "GSEQ_coil_MRR", inner="relax"),
            script(f"{SECOND}/Fig6", "figure_6_1", chain=True),
            script(f"{SECOND}/Fig6", "figure_6_2", chain=True),
        ],
    },
    "p2-Fig6-fix": {
        "title": "第二篇 · 图 6（最终修复版 GSEQ_coil_MRR_fixed：无需任何开关）",
        "steps": [
            # `GSEQ_coil_MRR_fixed` 把正向改动全部固化，**不接受任何 ci_* 开关**，
            # 所以这里不传参数。它里面有意排除的几项及实测理由：
            #   * mid_accel="anderson"：只在 step_mode 下生效，冷启动会把解推到 NaN
            #     （见 nk_inner 第 18 条注释）；本流程是冷启动，不适用。
            #   * mid_plateau_stop=True：实测比"可达地板"慢（7.75 s vs 4.99 s），
            #     且功能重叠。
            #   * dci_max：修改记录第 15 条已记为负面结果（11.1 s → 30.8 s）。
            script(f"{SECOND}/Fig6", "GSEQ_coil_MRR_fixed"),
            script(f"{SECOND}/Fig6", "figure_6_1", chain=True),
            script(f"{SECOND}/Fig6", "figure_6_2", chain=True),
        ],
    },
    "p2-Fig7": {
        "title": "第二篇 · 图 7（固定/自由边界两套的 ψ 与 Bz 剖面）",
        "steps": [
            script(f"{SECOND}/Fig7/Bz_fixed_072", "plt_psi_p"),
            script(f"{SECOND}/Fig7/Bz_fixed_072", "plot_Bz_radial", chain=True),
            script(f"{SECOND}/Fig7/Bz_free_072", "plt_psi_p"),
            script(f"{SECOND}/Fig7/Bz_free_072", "plot_Bz_radial", chain=True),
        ],
    },
    "p2-Fig8-9": {
        "title": "第二篇 · 图 8/9（参数扫描成图）",
        "steps": [
            script(f"{SECOND}/Fig8-9", "Figure_8"),
            script(f"{SECOND}/Fig8-9", "Figure_9"),
        ],
    },
}


# ------------------------------------------------------------------ 跑流程
def new_session(out_dir: Path) -> Workspace:
    """开一段新会话：清空搜索路径 / 全局量 / 产出目录（等价于新开一个 MATLAB）。"""
    mpath.reset()
    mglobals.reset()
    mplot.close_all()
    mio.reset()
    mio.set_output_dir(out_dir)
    return Workspace()


def _save_new_figures(out_dir: Path, stem: str, before: list) -> int:
    """把这一步新画出来的图窗存成 `stem.png`、`stem_fig2.png`…，返回存了几张。

    原文这些脚本在 MATLAB 里是"弹图窗给人看"，不落盘；移植版用无窗口后端跑，
    所以这里替它们把图存下来，读者才能对着论文的 `.eps`/`.png` 逐张比。
    `before` 是这一步开始前已经开着的图窗，只存这之后新出现的那些。
    """
    saved = 0
    fresh = [fig for fig in mplot.open_figures()
             if not any(fig is old for old in before)]
    for index, fig in enumerate(fresh, 1):
        name = stem if index == 1 else f"{stem}_fig{index}"
        try:
            fig.savefig(out_dir / f"{name}.png", dpi=100)
        except Exception:  # noqa: BLE001 - 存图失败不该让整条流程失败
            continue
        saved += 1
    return saved


def run_flow(name: str, spec: dict, verbose: bool = False) -> tuple[bool, float]:
    out_dir = ROOT / "outputs" / name
    print(f"\n=== {name}：{spec['title']}")
    print(f"    产物目录：{out_dir}")
    started = time.perf_counter()
    ws = Workspace()
    session_open = False
    ok = True
    try:
        for step_index, st in enumerate(spec["steps"], 1):
            if not st.get("chain") or not session_open:
                ws = new_session(out_dir)
                session_open = True
            before = mplot.open_figures()
            label = f"{st['script']} @ {st['dirs'][0]}"
            print(f"  - {label} ...", flush=True)
            step_started = time.perf_counter()
            for directory in st["dirs"]:
                mpath.addpath(directory)
            module = mpath.load(st["script"])
            # 脚本类移植件一律暴露 `run(ws)`（函数类移植件才用与文件名同名的
            # 入口，那种由 `mpath.call` 调用，不在这里）
            entry = getattr(module, "run", None) or getattr(module, st["script"])
            try:
                entry(ws, **st["kwargs"])
            except Exception as exc:  # noqa: BLE001
                if not st["stops"]:
                    raise
                print(f"    （原文到这里自行中止：{type(exc).__name__}: {exc}）",
                      flush=True)
            step_stem = f"{step_index:02d}_{st['script']}"
            saved = _save_new_figures(out_dir, step_stem, before)
            if saved:
                print(f"    （存了 {saved} 张图）", flush=True)
            print(f"    {time.perf_counter() - step_started:.1f} s", flush=True)
    except Exception:  # noqa: BLE001
        ok = False
        print(f"  !! 这一步失败：\n{traceback.format_exc()}")
    elapsed = time.perf_counter() - started
    print(f"  => {'完成' if ok else '失败'}，用时 {elapsed:.1f} s")
    return ok, elapsed


def main(argv: list[str]) -> int:
    if "--list" in argv or "-l" in argv:
        width = max(len(name) for name in FLOWS)
        for name, spec in FLOWS.items():
            print(f"{name:<{width}}  {spec['title']}")
        return 0

    names = [a for a in argv if not a.startswith("-")]
    unknown = [n for n in names if n not in FLOWS]
    if unknown:
        print(f"没有这些流程：{', '.join(unknown)}；用 --list 看清单")
        return 2
    selected = names or list(FLOWS)

    (ROOT / "outputs").mkdir(exist_ok=True)
    results = []
    started = time.perf_counter()
    for name in selected:
        results.append((name, *run_flow(name, FLOWS[name], verbose="-v" in argv)))

    print("\n=== 汇总 ===")
    failed = 0
    for name, ok, elapsed in results:
        print(f"  {name:<22} {'完成' if ok else '失败'}  {elapsed:8.1f} s")
        failed += 0 if ok else 1
    print(f"\n{len(results) - failed}/{len(results)} 个流程跑通；"
          f"总用时 {time.perf_counter() - started:.1f} s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
