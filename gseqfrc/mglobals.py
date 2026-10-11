"""MATLAB 全局工作区（`global` 语句）的模拟。

原程序里一维剖面与平衡求解器之间靠全局变量传参：

    global betas ls n tol alpha0 ls0 psim0 psig

`solve_MRR.m` 之类先给这些全局赋值，再调用 `fmrr0(x)`，后者从全局里取参数。
MATLAB 的全局工作区是整个进程共享的，所以移植版也用一个共享对象 `G`：

    G.declare("betas", "ls", "n", "tol")   % 相当于 global betas ls n tol
    G.betas = 0.6                          % 相当于 betas = 0.6
    alpha = ((1/G.ls)/np.sqrt(1-G.betas)+G.n*sigma)

每跑一个算例之前由 `run.py` 调 `reset()` 清空，等价于给每个算例
开一个干净的 MATLAB 进程。

按 MATLAB 的规则，`global x` 会把尚未赋值的 `x` 建成空矩阵 `[]`，所以未赋值的
全局读出来是 `EMPTY`，而不是报错。
"""

from __future__ import annotations

import numpy as np

#: MATLAB 的空矩阵 `[]`（0×0 double）
EMPTY = np.zeros((0, 0))


class Globals:
    """一份全局工作区。"""

    def __init__(self) -> None:
        object.__setattr__(self, "_values", {})
        object.__setattr__(self, "_declared", set())

    def declare(self, *names: str) -> "Globals":
        """等价于 MATLAB 的 `global a b c`。"""
        for name in names:
            self._declared.add(name)
            self._values.setdefault(name, EMPTY)
        return self

    def clear(self) -> None:
        self._values.clear()
        self._declared.clear()

    def as_dict(self) -> dict:
        return dict(self._values)

    def __getattr__(self, name: str):
        values = object.__getattribute__(self, "_values")
        if name in values:
            return values[name]
        raise NameError(
            f"{name} 不在全局工作区里（原程序里没有对它做 global 声明）"
        )

    def __setattr__(self, name: str, value) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        # MATLAB 里给一个名字赋值，它就成为全局工作区的一部分
        values = object.__getattribute__(self, "_values")
        values[name] = value
        object.__getattribute__(self, "_declared").add(name)

    def __repr__(self) -> str:
        return f"Globals({', '.join(sorted(self._values))})"


#: 全局工作区（MATLAB 的全局区也是每进程一份）
G = Globals()


def reset() -> Globals:
    G.clear()
    return G
