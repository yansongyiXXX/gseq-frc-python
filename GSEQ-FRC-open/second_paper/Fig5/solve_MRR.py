"""solve_MRR.m（Fig5 副本）的移植——**薄壳**，实现见 `gseqfrc/shared/solve_MRR.py`。

三份的差异已合并成 `PROFILES` 里的参数（含本副本特有的 `icoil` 开关）。
**本副本取 `profile="Fig5"`**（`icoil=1`：betas=0.7、Be=0.5、Rs=0.17、Rw=0.4、
`ls=0.5*ls0`、psig=−0.045、x0=3.7）；它写出的 `MRR1d.mat` 里多一个 `icoil`。
"""

from functools import partial

from gseqfrc.shared.solve_MRR import run as _run

run = partial(_run, profile="Fig5")
