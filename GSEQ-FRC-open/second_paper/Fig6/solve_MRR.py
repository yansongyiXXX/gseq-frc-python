"""solve_MRR.m（Fig6 副本）的移植——**薄壳**，实现见 `gseqfrc/shared/solve_MRR.py`。

三份的差异已合并成 `PROFILES` 里的参数。
**本副本取 `profile="Fig6"`**（betas=0.6、Be=1、Rs=0.075、Rw=0.14、`ls=0.40*ls0`、
psig=−0.045、x0=2.4，另外四条曲线画成 `k--`、横轴标签写 `r (cm)`）。
"""

from functools import partial

from gseqfrc.shared.solve_MRR import run as _run

run = partial(_run, profile="Fig6")
