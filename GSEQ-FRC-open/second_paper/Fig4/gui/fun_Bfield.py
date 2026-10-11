"""`fun_Bfield.m`（Fig4/gui 与 Fig5/gui 两份合并后的实现，见 `gseqfrc/shared/`）。

**修改版**：这里去掉了原文 Fig4 那份多出来的 `/2`。

两份 `.m` 只差 `psi` 累加里那个除数（`Br`、`Bz` 逐字相同）：Fig4 那份是
`...*((2-k2)*EK2-2*EE2)/2`，Fig5 那份没有 `/2`。同一个函数算出来的 \(\psi\)
与 \(B_r,B_z\) 必须用同一套归一化才自洽——这个 `/2` 与 `fun_Bfield.m` 自己的
`Br`/`Bz` 矛盾，也与 `GSEQ_coil_MRR.m` 里同一根灯丝的公式差一倍，所以按笔误处理，
统一到 Fig5 那份（也就是 `psi_div=1.0`，本薄壳直接用共享实现的默认值）。

影响：`Fig4/gui` 里 `frc_coil_gui`、`plt_wall` 用到的线圈真空磁通 \(\psi_{coil}\)
变成原来的 2 倍（与 Fig5 的算法一致）；这一项只用于画壁面上的线圈磁通分布，
不参与任何平衡求解。改动记录见 `修改记录.md`。
"""

from gseqfrc.shared.fun_Bfield import fun_Bfield  # noqa: F401
