"""valid1d.m（figureB1 副本）的移植——**薄壳**，实现见 `gseqfrc/shared/valid1d.py`。

两份只差绘图风格，2026-09-17 已合并、差异提成 `style` 参数。
**本副本取 `style="figureB1"`**（tight_subplot + LaTeX、不打印图片）。
"""

from functools import partial

from gseqfrc.shared.valid1d import run as _run

run = partial(_run, style="figureB1")
