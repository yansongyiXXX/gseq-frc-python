"""GSEQ-FRC 的 Python 逐行移植（独立程序树）。

本目录是"运行垫片"：把 MATLAB 的语义补出来，让 `GSEQ-FRC-open/` 里的 `.py`
能像原文的 `.m` 一样跑。移植规则是**一个 `.m` 对应一个 `.py`**，语句顺序与运算
顺序照抄（原文的笔误、死代码、写死的容差都一并照抄，不"顺手修正"）。

    GSEQ-FRC-open/       程序树：与原文仓库逐目录对应，数据文件就在脚本旁边
    gseqfrc/             本包：MATLAB 语义垫片 + addpath 模拟 + 写 .mat 的工具
    fastpy/              加速版求解器（替换原文那条慢的自由边界解法）
    run.py               入口：按论文的流程把脚本按顺序跑起来
"""

from pathlib import Path

__all__ = ["repo_root", "__version__"]

__version__ = "0.1.0"


def repo_root() -> Path:
    """返回本目录（同时包含 `GSEQ-FRC-open/` 程序树与 `gseqfrc/` 垫片的那一级）。"""
    return Path(__file__).resolve().parent.parent
