"""fun_ellipEK.m（本目录那一份副本）的移植——**薄壳**，实现见 `gseqfrc/shared/fun_ellipEK.py`。

三份 `fun_ellipEK.m`（Fig4/gui、Fig5/gui、Fig6）逐字节相同，2026-09-17 已合并成
一份（见 PORTING.md §8 与 `gseqfrc/shared/__init__.py`）。原位置保留这个薄壳，
是为了让 `mpath` 按"路径上有没有这个名字"解析的行为与合并前完全一致。

调用方：`fun_Bfield.py`（线圈磁场）、`u_ellipEK*`/`u_grzjz`/`u_psib_coil` 探针。
数值行为与合并前逐位相同（合并的只是文件，不是算法）。
"""

from gseqfrc.shared.fun_ellipEK import fun_ellipEK  # noqa: F401
