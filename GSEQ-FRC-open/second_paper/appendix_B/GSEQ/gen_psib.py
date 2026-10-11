"""gen_psib.m（appendix_B/GSEQ 副本）的移植——**薄壳**，实现见
`gseqfrc/shared/gen_psib_appendix_B.py`。

appendix_B 下的两份 `gen_psib.m` 只差一行注释、冻结基准也完全相同，2026-09-17
已合并成一份实现（见 PORTING.md §8）；原位置保留薄壳，让 `mpath` 按路径解析的
行为与合并前一致。`first_paper/figure8/gen_psib.m` 是另一个程序，没有并进来。
"""

from gseqfrc.shared.gen_psib_appendix_B import run  # noqa: F401
