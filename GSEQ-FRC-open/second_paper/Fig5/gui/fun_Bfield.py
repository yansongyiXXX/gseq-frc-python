"""fun_Bfield.m（Fig5/gui 副本）的移植——**薄壳**，实现见 `gseqfrc/shared/fun_Bfield.py`。

两份副本只差 `psi` 累加里那个 `/2`，2026-09-17 已合并、差异提成 `psi_div` 参数。
**本副本取 1.0（默认）**，即 Fig5 原本就没有那个 `/2` 的写法。

`u_Bfield_fig5` 这个冻结基准钉住的是这一份；`GSEQ_coil_MRR.m` 用的也是这一份的
归一化。
"""

from gseqfrc.shared.fun_Bfield import fun_Bfield  # noqa: F401
