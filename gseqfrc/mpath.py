"""MATLAB 搜索路径（addpath）的模拟。

原程序不用 `import`，它靠"把目录加进搜索路径，然后按函数名调用"。同名函数散落在
多个目录里，谁被调用完全取决于路径顺序——基准文档里专门警告过这件事（两个仓库副本
共存时 `load('psi.mat')` 会解析到不同的文件）。所以移植版把同样的机制显式做出来：

    mpath.addpath("second_paper/Fig4/gui")
    K, E = mpath.call("fun_ellipEK", k2, 1)

`resolve()` 严格按加入顺序找第一个 `名字.py`。如果已加入的目录里都没有，才会去整棵
程序树里搜；搜到一个就发警告使用（提醒调用方该显式 addpath），搜到多个就报错，
因为那意味着路径顺序真的会影响结果，不能猜。

程序树的根见 `PROGRAM_ROOT`：它就是与原文仓库 `GSEQ-FRC-open/` **逐目录对应**的
那棵树（一个 `.m` 对应一个 `.py`），随论文提供的数据文件（`.mat`/`.dat`）也放在
同一批目录里——跟原仓库一样，脚本和数据是邻居，所以 `analyze('psiq.mat')` 这类
相对路径直接就能找到。

`addpath` 的路径因此都相对于这棵树来写（相当于 MATLAB 里先 `cd GSEQ-FRC-open`）。
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

PROGRAM_ROOT = Path(__file__).resolve().parent.parent / "GSEQ-FRC-open"


class MatlabPath:
    """一份 `addpath` 状态。"""

    def __init__(self, root: Path | str = PROGRAM_ROOT) -> None:
        self.root = Path(root)
        self._dirs: list[Path] = []
        self._modules: dict[Path, object] = {}
        self._fallback: dict[str, Path | None] = {}

    # ------------------------------------------------------------- 路径管理
    def addpath(self, *relative_dirs: str) -> "MatlabPath":
        for rel in relative_dirs:
            path = (self.root / rel).resolve()
            if not path.is_dir():
                raise FileNotFoundError(f"addpath 的目录不存在：{path}")
            if path not in self._dirs:
                self._dirs.append(path)
        return self

    def rmpath(self, *relative_dirs: str) -> "MatlabPath":
        for rel in relative_dirs:
            path = (self.root / rel).resolve()
            if path in self._dirs:
                self._dirs.remove(path)
        return self

    def reset(self) -> "MatlabPath":
        """清空路径，等价于每个算例各自开一个新 MATLAB 进程。"""
        self._dirs.clear()
        self._modules.clear()
        self._fallback.clear()
        return self

    @property
    def dirs(self) -> list[Path]:
        return list(self._dirs)

    def original_dir(self, directory: Path) -> Path | None:
        """数据文件所在的那个目录。

        这份独立程序树里，随论文提供的**数据文件**（`psib_FIX.mat`、`*.dat` 这些）
        与脚本放在**同一个目录**里，所以这里返回目录自身；不在程序树里的目录返回
        `None`（调用方会跳过）。
        """
        try:
            relative = Path(directory).relative_to(self.root)
        except ValueError:
            return None
        return self.root / relative

    # ------------------------------------------------------------- 函数解析
    def resolve(self, name: str) -> Path:
        """按 MATLAB 的 `which` 规则定位 `名字.py`。"""
        for directory in self._dirs:
            candidate = directory / f"{name}.py"
            if candidate.is_file():
                return candidate

        # 兜底搜索整棵树很贵（每次约 0.5 s，而一个算例里可能有几十万次调用），
        # 所以结果必须缓存。缓存里存 None 表示"确实一个副本都没有"。
        if name not in self._fallback:
            found = sorted(self.root.rglob(f"{name}.py"))
            if len(found) > 1:
                listed = "\n  ".join(str(h.relative_to(self.root)) for h in found)
                raise NameError(
                    f"{name}.py 存在 {len(found)} 个副本，且都不在已加入的路径里，"
                    f"无法决定用哪一个：\n  {listed}"
                )
            self._fallback[name] = found[0] if found else None
            if found:
                warnings.warn(
                    f"{name}: 已加入的路径里没有它，用了全树搜索找到的唯一副本 "
                    f"{found[0].relative_to(self.root)}；建议显式 addpath。",
                    RuntimeWarning,
                    stacklevel=2,
                )
        hit = self._fallback[name]
        if hit is None:
            joined = (
                "、".join(str(d.relative_to(self.root)) for d in self._dirs) or "（空）"
            )
            raise NameError(f"未找到 {name}.py；已加入的路径：{joined}")
        return hit

    def load(self, name: str):
        """按文件路径加载并缓存模块。同一份文件只加载一次。"""
        path = self.resolve(name)
        module = self._modules.get(path)
        if module is None:
            module_name = f"gseqfrc._program.{name}_{abs(hash(str(path))) & 0xFFFFFF:06x}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"无法加载 {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            self._modules[path] = module
        return module

    def call(self, name: str, *args, **kwargs):
        """按名字调用，等价于 MATLAB 里的 `名字(...)`。"""
        return getattr(self.load(name), name)(*args, **kwargs)


# 全局默认路径：MATLAB 的搜索路径也是全局状态，这里保持一致。
DEFAULT_PATH = MatlabPath()


def addpath(*relative_dirs: str) -> MatlabPath:
    return DEFAULT_PATH.addpath(*relative_dirs)


def rmpath(*relative_dirs: str) -> MatlabPath:
    return DEFAULT_PATH.rmpath(*relative_dirs)


def reset() -> MatlabPath:
    return DEFAULT_PATH.reset()


def resolve(name: str) -> Path:
    return DEFAULT_PATH.resolve(name)


def load(name: str):
    """按名字加载移植文件（相当于 MATLAB 在路径上找到那个 `.m`）。"""
    return DEFAULT_PATH.load(name)


def call(name: str, *args, **kwargs):
    return DEFAULT_PATH.call(name, *args, **kwargs)
