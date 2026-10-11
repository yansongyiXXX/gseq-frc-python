"""fmrr0.m（本目录那一份副本）的移植——**薄壳**，实现在 `gseqfrc/shared/fmrr0.py`。

四份 `fmrr0.m`（Fig4/Fig5/Fig6/Fig7-Bz_free_072）逐字节相同，2026-09-17 已合并成
一份（见 PORTING.md §8 与 `gseqfrc/shared/__init__.py`）。原位置保留这个薄壳，
是为了让 `mpath` 按"路径上有没有这个名字"解析的行为与合并前完全一致：
MATLAB 那边这里确实有一份 `fmrr0.m`。

数值行为与合并前逐位相同（合并的只是文件，不是算法）。
"""

from gseqfrc.shared.fmrr0 import fmrr0  # noqa: F401
