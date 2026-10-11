r"""展示物 2：17 条流程的**用时与产物齐备情况**表（从一次完整运行的日志 + 产物目录生成）。

输入：

* ``run_all_timing.log`` —— ``python run.py`` 的完整输出（含末尾的「汇总」块）；
* ``outputs/<流程名>/`` —— 本次运行产出的产物；
* ``outputs_backup_<日期>/`` —— 上一次完整运行的产物快照（用来判"齐备"）。

输出：

* ``outputs/showcase/flows.md`` —— 可直接贴进汇报材料的 Markdown 表；
* ``outputs/showcase/flows.csv`` —— 同样内容的机器可读版。

用法（仓库根目录，先跑完 ``python run.py``）::

    python tools/showcase_flow_table.py
    python tools/showcase_flow_table.py --log run_all_timing.log --ref outputs_backup_20260927
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

_LINE = re.compile(r"^\s{2}(\S+)\s+(完成|失败)\s+([\d.]+) s\s*$")
_TOTAL = re.compile(r"(\d+)/(\d+) 个流程跑通；总用时\s+([\d.]+) s")


def parse_log(log: Path) -> tuple[list[dict], dict]:
    """从 ``run.py`` 的日志里取出每个流程的用时与总用时（只认末尾「汇总」块）。"""
    text = log.read_text(encoding="utf-8", errors="replace")
    tail = text.split("=== 汇总 ===")[-1]
    rows: list[dict] = []
    for line in tail.splitlines():
        m = _LINE.match(line)
        if m:
            name, status, secs = m.groups()
            rows.append({"flow": name, "ok": status == "完成", "wall_s": float(secs)})
    total = {"n_ok": None, "n_all": None, "total_s": None}
    m = _TOTAL.search(tail)
    if m:
        total = {"n_ok": int(m.group(1)), "n_all": int(m.group(2)),
                 "total_s": float(m.group(3))}
    return rows, total


def dir_files(path: Path) -> dict[str, int]:
    """目录里的文件名 → 字节数（目录不存在时返回空）。"""
    if not path.is_dir():
        return {}
    return {p.name: p.stat().st_size for p in path.iterdir() if p.is_file()}


def describe(flow: str, ref_root: Path | None) -> dict:
    """一个流程的产物情况：文件数、总大小、与上一次运行相比的缺失/新增。"""
    now = dir_files(ROOT / "outputs" / flow)
    info = {"n_files": len(now), "size_mb": sum(now.values()) / 1e6,
            "missing": [], "extra": []}
    if ref_root is not None:
        ref = dir_files(ref_root / flow)
        if ref:
            info["missing"] = sorted(set(ref) - set(now))
            info["extra"] = sorted(set(now) - set(ref))
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", default="run_all_timing.log")
    ap.add_argument("--ref", default=None,
                    help="上一次运行的产物目录（默认自动找 outputs_backup_*）")
    ap.add_argument("--note", action="append", default=[],
                    help="追加到表尾的脚注（可重复给多次）")
    args = ap.parse_args()

    log = ROOT / args.log
    if not log.is_file():
        print(f"找不到日志 {log}；先在仓库根目录跑 `python run.py` 再来。")
        return 2
    ref_root = None
    if args.ref:
        ref_root = ROOT / args.ref
    else:
        cands = sorted(p for p in ROOT.glob("outputs_backup_*") if p.is_dir())
        ref_root = cands[-1] if cands else None

    rows, total = parse_log(log)
    for r in rows:
        r.update(describe(r["flow"], ref_root))

    out_dir = ROOT / "outputs" / "showcase"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Markdown ----
    lines = [
        f"# {len(rows)} 条流程：用时与产物齐备情况",
        "",
        f"- 日志：`{log.name}`；对照快照：`{ref_root.name if ref_root else '（无）'}`",
        (f"- 结果：{total['n_ok']}/{total['n_all']} 条跑通，"
         f"总用时 {total['total_s'] / 60:.1f} 分钟" if total["total_s"] else ""),
        "",
        "| 流程 | 结果 | 用时 [s] | 产物文件数 | 产物大小 [MB] | 与上次相比 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        diff = "—"
        if ref_root is not None:
            bits = []
            if r["missing"]:
                bits.append(f"缺 {len(r['missing'])}")
            if r["extra"]:
                bits.append(f"多 {len(r['extra'])}")
            diff = "、".join(bits) if bits else "一致"
        lines.append(
            f"| `{r['flow']}` | {'跑通' if r['ok'] else '**失败**'} | {r['wall_s']:.1f} "
            f"| {r['n_files']} | {r['size_mb']:.2f} | {diff} |")
    if total["total_s"]:
        lines.append(f"| **合计** | {total['n_ok']}/{total['n_all']} | "
                     f"**{total['total_s']:.1f}** | "
                     f"{sum(r['n_files'] for r in rows)} | "
                     f"{sum(r['size_mb'] for r in rows):.1f} | — |")
    for i, note in enumerate(args.note, 1):
        lines += ["", f"注 {i}：{note}"]
    (out_dir / "flows.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- CSV ----
    with (out_dir / "flows.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["flow", "ok", "wall_s", "n_files",
                                           "size_mb", "missing", "extra"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "missing": ";".join(r["missing"]),
                        "extra": ";".join(r["extra"])})
    (out_dir / "flows.json").write_text(
        json.dumps({"total": total, "flows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    print("\n".join(lines))
    print(f"\n[flows] 产物：{out_dir / 'flows.md'}、{out_dir / 'flows.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
