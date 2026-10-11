"""valid1d.m（figure1_2 副本）的移植——**薄壳**，实现见 `gseqfrc/shared/valid1d.py`。

两份 `valid1d.m` 只差绘图风格（figureB1 那份用 tight_subplot + LaTeX、字号 30、
末尾的打印块被注释掉），2026-09-17 已合并、差异提成 `style` 参数。
**本副本取 `style="figure1_2"`**（默认）。
"""

from functools import partial

from gseqfrc.shared.valid1d import run as _run

run = partial(_run, style="figure1_2")
