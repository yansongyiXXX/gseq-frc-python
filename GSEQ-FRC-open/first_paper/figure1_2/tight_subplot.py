"""tight_subplot.m（本目录那一份副本）——**薄壳**，实现见 `mplot.tight_subplot`。

全仓 9 份 `tight_subplot.m` 逐字节相同，早已折进 `gseqfrc/mplot.py`；这里补一层
薄壳，是为了让按名字调用它的脚本（`figure3/4/5/6/7.m`、`valid1d.m`、`Figure_8/9.m`
等）经 `mpath.resolve("tight_subplot")` 能找到文件——MATLAB 那边这里确实有一份。
"""

from gseqfrc.mplot import tight_subplot  # noqa: F401
