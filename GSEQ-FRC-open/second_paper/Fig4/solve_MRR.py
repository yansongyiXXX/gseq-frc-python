"""solve_MRR.m（Fig4 副本）的移植——**薄壳**，实现见 `gseqfrc/shared/solve_MRR.py`。

三份 `solve_MRR.m` 的求解器骨架逐字相同，差异只有参数块与绘图/打印，2026-09-17
已合并、差异登记在 `PROFILES` 里（见 PORTING.md §8 的规则 2）。
**本副本取 `profile="Fig4"`**（betas=0.6、`ls=0.4*ls0`、psig=−0.054、x0=2.4、实线）。

它写出的 `MRR1d.mat` 是 Fig4 二维定边界算例（`g_fixed_fig4`）的输入。
"""

from functools import partial

from gseqfrc.shared.solve_MRR import run as _run

run = partial(_run, profile="Fig4")
