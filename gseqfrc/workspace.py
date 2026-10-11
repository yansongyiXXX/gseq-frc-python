"""MATLAB 基本工作区的模拟。

原程序靠基本工作区在脚本之间传数据：`clear all` 之后跑一个脚本，它定义的每个变量
都留在工作区里，下一个脚本直接拿来用。冻结工具抓的就是这个工作区。

移植时脚本对应 `run(ws)`，凡是原脚本写在基本工作区里的变量，都写到 `ws` 上
（同名、同形状），下游脚本和跑完后的检查都能直接拿到。
"""

from __future__ import annotations

import re


class Workspace:
    """一个可以 `ws.foo = ...` 的普通命名空间。"""

    def __init__(self, **items: object) -> None:
        self.__dict__.update(items)

    def __contains__(self, name: str) -> bool:
        return name in self.__dict__

    def __getitem__(self, name: str):
        return self.__dict__[name]

    def __setitem__(self, name: str, value: object) -> None:
        self.__dict__[name] = value

    def keys(self) -> list[str]:
        return [k for k in self.__dict__ if not k.startswith("_")]

    def as_dict(self) -> dict[str, object]:
        return {k: self.__dict__[k] for k in self.keys()}

    def clear(self) -> None:
        self.__dict__.clear()

    def __repr__(self) -> str:
        return f"Workspace({', '.join(sorted(self.keys()))})"


def eval_workspace_name(ws: Workspace, expression: str):
    """求值 `evalin('base', expr)` 这类字符串表达式里的单个变量名。"""
    name = expression.strip()
    if not re.fullmatch(r"[A-Za-z_]\w*", name):
        raise ValueError(f"不是合法的变量名：{expression!r}")
    return ws[name]
