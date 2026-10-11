"""MATLAB 的文件读写：`save` 与 `load`。

* `save('rr_datq.mat', ...)` 真的写出一个 `.mat`（`scipy.io` 的 v5 格式），
  并顺手加载回来记一份内容快照；
* `load('MRR1d.mat')` 先找产出目录，再沿搜索路径找——**随论文提供的数据文件
  就放在脚本旁边**，所以相对文件名照原文那样写就能读到最后移植版自己写出的那
  一份（例如先跑 `solve_MRR` 产出 `MRR1d.mat`，再跑求解器读它）。

产出目录默认是每跑一次新建一个临时目录；`run.py` 用 `set_output_dir()` 把它固定到
`outputs/<流程名>/`，脚本 `save` 出的 `.mat`、`print` 出的图都落在那里。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


class ProducedMats:
    """一个算例写出的所有 `.mat` 文件。"""

    def __init__(self) -> None:
        self._directory: Path | None = None
        self._contents: dict[str, dict] = {}

    def reset(self) -> None:
        self._directory = None
        self._contents.clear()

    @property
    def directory(self) -> Path:
        if self._directory is None:
            self._directory = Path(tempfile.mkdtemp(prefix="gseqfrc_out_"))
        return self._directory

    def save(self, filename: str, **variables) -> Path:
        """等价于 MATLAB 的 `save(filename,'a','b',...)`。"""
        path = self.directory / filename
        savemat(path, dict(variables))
        # 照基准的做法：文件写完之后再加载回来记快照
        self._contents[filename] = {
            name: value for name, value in loadmat(path).items()
            if not name.startswith("__")
        }
        return path

    def save_struct(self, filename: str, variables: dict) -> Path:
        """`save(filename)` 把当前工作区全部写出去的等价物。"""
        return self.save(filename, **variables)

    def contents(self) -> dict[str, dict]:
        return {name: dict(values) for name, values in self._contents.items()}

    def filenames(self) -> list[str]:
        return list(self._contents)


#: 当前算例的产出快照（和 mglobals 一样是每算例一份的全局状态）
PRODUCED = ProducedMats()


def reset() -> ProducedMats:
    PRODUCED.reset()
    return PRODUCED


def set_output_dir(path) -> Path:
    """把产出目录固定到指定位置（默认是每跑一次就新建一个临时目录）。

    `save()` 写出的 `.mat`、`print()`/`imwrite()` 画出的图都落在这里。
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    PRODUCED._directory = directory
    return directory


def save(filename: str, **variables) -> Path:
    return PRODUCED.save(filename, **variables)


def load(filename: str) -> dict:
    """等价于 MATLAB 的 `load(filename)`：返回文件里的变量。

    MATLAB 的 `load` 会先在当前目录找，再沿搜索路径找。二维求解器那些脚本正是
    靠这条读**随论文提供**的基线数据（`psib_FIX.mat`、`MRR1d.mat` 等），
    它们不在算例的产出目录里。所以这里也按"先产出目录、再搜索路径"的顺序找。
    """
    from . import mpath

    candidates = [PRODUCED.directory / filename]
    for directory in mpath.DEFAULT_PATH.dirs:
        candidates.append(directory / filename)
        original = mpath.DEFAULT_PATH.original_dir(directory)
        if original is not None:
            candidates.append(original / filename)
    path = next((c for c in candidates if c.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"load: 找不到 {filename}（找过：{', '.join(str(c) for c in candidates)}）"
        )
    return {
        name: value for name, value in loadmat(path).items()
        if not name.startswith("__")
    }


def key_for(filename: str) -> str:
    """把文件名规范化成与基准快照同样的键（对应 MATLAB 的 `regexprep(nb,'\\W','_')`）。

    基准里的键形如 `f02_rr_datq`，前缀是产出文件序号，不可复现也无意义，
    所以两边都只用"去掉扩展名、非单词字符换成下划线"的部分做配对。
    """
    stem = Path(filename).stem
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in stem)


def strip_matlab_keys(raw: dict) -> dict:
    return {k: v for k, v in raw.items() if not k.startswith("__")}
