# GSEQ-FRC 的 Python 版程序树（**修改版**）

这是 GSEQ-FRC 两篇论文（Ma 等，Nucl. Fusion 2021）那套 MATLAB 程序的 Python 移植，
整棵树独立自足：**不依赖任何 MATLAB 文件、不依赖原来的工作目录**，只装
numpy / scipy / matplotlib 就能跑完整套流程。

与 `GSEQ-FRC-Python`（严格移植版）的区别：

1. **加速版自由边界求解器就地顶替了原程序**：`second_paper/Fig6/GSEQ_coil_MRR.py`
   就是加速版（原来那份 8.3 小时、自己还没收敛的 `GSEQ_coil_MRR.m` 不再逐行保留，
   也没有单独的 `fastpy/` 目录）。
2. **修掉了查得出来的缺陷**：线圈 Green 函数里 `Br` 漏掉的 \(1/(2\pi)\)、Fig4 副本
   `fun_Bfield.m` 里自相矛盾的 `/2`、`Figure_8.m` 的变量名笔误、原文缺失的
   `solve_lp`/`solve_h`、被老网格约定卡死的 `plt_psi_r` 下标，以及 `gseq` 右端项
   的向量化（纯性能）。

每一条改动的位置、依据、影响面都写在 **[`修改记录.md`](修改记录.md)** 里；
源码里凡是改过的地方都标了「**修改版**」。没改的地方（会动论文数值的那些）
也在那份记录里单列说明。

除上述之外，移植规矩不变：**一个 `.m` 对应一个 `.py`**，目录名、语句顺序、
运算顺序照抄，原文写死的容差、死代码都保留。

## 目录

```
GSEQ-FRC-Python/
  GSEQ-FRC-open/      程序树：与原文仓库逐目录对应
    first_paper/       第一篇论文：figure1_2（RR/MRR1/MRR2/2PE/3PE/valid1d）、figure3–8、figureB1
    second_paper/      第二篇论文：appendix_A、appendix_B（GSEQ / grass_ft）、Fig4–Fig7、Fig8-9
                        （随论文提供的数据 .mat / .dat 就在各自目录里，与脚本为邻）
      Fig6/nk_inner.py 图 6 的**快速内层**（默认；`inner="relax"` 回到原文那条，见下文）
  gseqfrc/            运行垫片：MATLAB 语义（1 基索引、all()、size…）、addpath 模拟、save/load、绘图
  run.py              入口：按论文的会话把脚本按顺序跑起来
  修改记录.md          这份修改版改了哪些、为什么、影响了什么
  outputs/            run.py 的产物（每个流程一个子目录）
```

## 怎么跑

```powershell
C:\Users\dell\miniforge3\envs\freegsnke\python.exe run.py --list          # 有哪些流程
C:\Users\dell\miniforge3\envs\freegsnke\python.exe run.py p2-Fig6        # 跑某一个流程
C:\Users\dell\miniforge3\envs\freegsnke\python.exe run.py                 # 跑全部流程
```

每个流程是一段"会话"：先 `addpath` 几个目录，再按顺序跑脚本（脚本之间靠基本工作区
传数据，跟 MATLAB 里一样）。跑完的 `.mat` 和图片落在 `outputs/<流程名>/`：

* 脚本自己 `save`/`print` 出来的东西，按原文的名字落盘；
* 每一步跑完，`run.py` 还会把**这一步新画出来的图窗**存成 `NN_脚本名.png`
  （原文这些脚本在 MATLAB 里只是弹图窗给人看、不落盘；移植版用无窗口后端跑，
  所以替它们存一份，方便对着论文里的 `.eps`/`.png` 逐张比）。

流程名带 `p1-` / `p2-` 前缀（两篇论文里都有 Fig4/Fig5…，不加前缀在 Windows 上
会撞成同一个产物目录）：

| 流程 | 内容 |
| --- | --- |
| `p1-1d` | 一维平衡解：`solve_RR`、`solve_MRR1`、`solve_MRR2`、`solve_2PE08`、`solve_2PE09`、`solve_3pe`、`valid1d` |
| `p1-fig1-2` | 图 1/图 2 成图（`MRR_figure1`、`MRR_figure2`） |
| `p1-fig3` … `p1-fig7` | 第一篇的图 3–7（含 `figure7` 的零维参数扫描） |
| `p1-fig8` | 第一篇图 8：`gen_psib` → `gseq` → `figure_8_1` → `plt_psi_r` → `figure_8_2` |
| `p1-figB1` | 附录 B1 的一维验证 |
| `p2-appendix-A` | 第二篇附录 A 的诊断图链（`figure_A_1 → A_2 → A_3`） |
| `p2-appendix-B-GSEQ` | 附录 B 的 GSEQ 版：`gen_psib` → `gseq_ver_2` → `Figure_B` |
| `p2-appendix-B-grass-ft` | 附录 B 的 grass_ft 版：`gen_psib` → `gseq_grass_ft` → `contour_psi` / `plot_grass_ft_f90` |
| `p2-Fig4` | 图 4：`solve_MRR` → `gseq_frc_fixed`（二维定边界）→ `figure_4_1` → `figure_4_2` |
| `p2-Fig5` | 图 5：`GSEQ_FRC_fixed_FIX` → `fix5_1` → `fix5_2` |
| `p2-Fig6` | 图 6：`GSEQ_coil_MRR`（加速版自由边界求解器，**默认快速内层**）→ `figure_6_1` → `figure_6_2`（约 16 s） |
| `p2-Fig6-relax` | 图 6 同上，但内层用原文逐点松弛（`inner="relax"`）：约 7 分钟，逐位复现冻结基准 |
| `p2-Fig7` | 图 7：固定/自由边界两套的 ψ 与 Bz 剖面 |
| `p2-Fig8-9` | 图 8/9 的参数扫描成图 |

想单跑树里某一个脚本，照原文的办法自己加路径即可：

```python
import sys; sys.path.insert(0, ".")
from gseqfrc import mpath
from gseqfrc.workspace import Workspace
mpath.addpath("GSEQ-FRC-open/second_paper/Fig6")
mpath.load("figure_6_1").run(Workspace())
```

## 二维自由边界求解器（图 6，加速版）

原文那份解法要跑 8.3 小时，而且自己没收敛到底，所以这一格直接放**加速版**：
`second_paper/Fig6/GSEQ_coil_MRR.py`。线圈 Green 函数与 GrzJz 张量向量化、
边界通量走矩阵乘、内层循环用预分配缓冲、最外层 CI 归一化换成带保护的 secant。
同配置下 MATLAB 加速版 665 s、这份 Python 版 **413 s**，两者数值差在 \(10^{-13}\)
相对量级（只来自求和次序），逐点算式一字未改；其中 \(B_r\) 漏掉的 \(1/(2\pi)\)
按 `修改记录.md` 第 1 条补上了（只影响这个诊断量）。

它和其他求解脚本用法一样（`mpath.load("GSEQ_coil_MRR").run(ws)`），`p2-Fig6`
这条流程跑的就是它，结果写成 `psiq.mat` / `psiq1.mat` 供 `figure_6_1`、
`figure_6_2` 接着画图。

### 内层有两套，默认是快的那个（`inner=` 开关）

同一格里的**内层求解器**有两个选择——外层（边界通量不动点 + CI 归一化 secant）完全一样：

| `inner=` | 是什么 | 全量耗时 | 与冻结基准 |
| --- | --- | --- | --- |
| `"hybrid"`（**默认**） | `nk_inner.py` 里的 FreeGSNKE 式配方：约束内嵌 + IMEX + 容差线搜索，卡住回退到 `relax` | **10.1 s**（流程 13.2 s） | 差 0.028%（停在外层容差带里的另一个点） |
| `"relax"` | 原文那条逐点松弛 + 1%/轮 C 控制器 | 约 415 s（本机 420–580 s） | 逐位复现 |

`hybrid` 的内层判据用的是这套配方自己的口径 `inner_tol=1e-5`（freegsnke
`forward_solve` 的原始判据，见 `nk_inner.TOL_RANGE`）。取更严的 1e-6 会让
CI2 那几步永远达不到、每次都跑满 20000 轮松弛兜底——那是全量 ~50 s 的来源；
1e-5 下兜底基本不触发，解只移动 7e-6。这些实验（包括失败的强松弛与四个
"换方向"变体）都记在 `修改记录.md` 第 7 条里。

另外向 freegsnke 借了两处（`修改记录.md` 第 8 条）：内层的**缩步重试**
（线搜索找不到可接受的步时，按残差比缩步再走，而不是立刻交给松弛兜底）、
以及**初值兜底**（起手残差算不出来就把初值放大 25% 重试，再不行退回默认初始化）。
效果是**换初值不再崩**：×1.3、×4.0、甚至全零初值都能收敛到同一个解。

**每次跑完都会打印两个残差诊断**（`修改记录.md` 第 9 条）：

| 名字 | 含义 | 典型值 |
| --- | --- | --- |
| `gs_res` | 离散 G-S 方程残差 \\(F=A\\psi-\\mathrm{rhs}\\) 的幅值相对值 | `relax` 路径 4.75e−6；`hybrid` 8.38e−6 |
| `defect` | 松弛映射的固定点缺陷（把终态 ψ 代进无阻尼一轮更新后差多少） | 2.1e−6 上下 |

这不是装饰：`relax` 路径的停判据是 `dpsim=1e−9`，而它的**方程残差仍有 4.75e−6**
（与加速线记录里"冻结基准 4.7e−6"对得上）——`dpsim` 小不等于方程解好了。
要按残差停，用 `solve(..., inner="relax", inner_stop="residual", res_tol=1e−6)`；
默认保留 `"dpsim"` 是为了能逐位复现冻结基准。
（这个残差模式也不做矩阵乘：它用恒等式 \\(F=-2\\Delta\\psi/(1-w)\\) 直接从每轮的 Δψ 得到，
比原来每轮一次稀疏矩阵乘快一倍——见 `修改记录.md` 第 12 条。）

还有一个**离散阶数**开关 `gs_order`（`修改记录.md` 第 10 条）：默认 2（原文那套
3 点离散，逐位复现基准）；`gs_order=4` 是四阶 5 点算子（与 freegsnke 的
`gs_operator_order=4` 同一思路）。实测结论：**四阶能更准、但不能更快**——
在 129×257 上它与二阶解差 1.69e−04（这是二阶离散误差的量级），代价是 29.4 s vs 14.9 s；
而"粗一档网格 + 四阶"省不了时间，因为粗网格的 0.47% 偏差主要来自几何/拓扑分辨率，
不是内部离散阶数。

最后一件**试过但没有采纳**的事（`修改记录.md` 第 14 条）：把最外层的标量耦合
\((C,CI)\) 写成小方程、用线性化灵敏度走牛顿步。线性化本身是验过的（三块雅可比
逐块与无穷小差分吻合；把真差分的 dψ/dCI 代回灵敏度公式，与前向差分差 0.03%），
但外层一步要跨 20%，这个尺度上线性化只有 ~30% 精度，于是外层反而多两轮
（6 轮 vs 割线 4 轮、内层步数多 17%），在默认 `out_tol=2e−9` 下还会把 CI 推进
内层发散的区间。所以**默认仍是割线**，那条路径没并进主树，实验脚本留在
`tools/exp_ci_newton_b.py`（`sec|newton` 两档可复现对比表）。

```powershell
python run.py p2-Fig6          # 默认：走 hybrid（快）
python run.py p2-Fig6-relax    # 复现冻结基准那条：走原文松弛（慢）
```

也可以直接在代码里选：

```python
mpath.load("GSEQ_coil_MRR").run(ws)               # 默认 hybrid
mpath.load("GSEQ_coil_MRR").run(ws, inner="relax")  # 原文那条
```

`"relax"` 那条留着是有意的：它逐位复现冻结基准，是这棵树用来对照的参照物，
跑一次约 7 分钟。两套的差别、独立复核结论与耗时记录见 `修改记录.md` 第 7 条和
`accel/README.md`（原始实验记录）。**两者的产物不要混用**：`hybrid` 快约 27 倍
（流程 13.2 s 对 427.5 s），
但解在 \(10^{-4}\) 量级上不同（ψ 0.028%、C 0.057%），各写各的目录。

## 原文里本来就跑不通的地方

* ~~`first_paper/figure8/gseq.m` 缺 `solve_lp`/`solve_h`~~ —— **修改版已补上**
  （见 `修改记录.md` 第 4 条），现在这条流程能按作者的迭代结构跑到底。
* `second_paper/appendix_B/grass_ft/plot_data.m`：读 `data/%06d_pse.DAT`，随仓库提供的
  数据里没有这些 PIC 输出文件，MATLAB 同样停在第一句 `importdata` 上。所以没进流程表。
* `second_paper/{Fig4,Fig5}/gui/` 里的 `plt_wall*`、`frc_coil_gui` 是**交互式 GUI**的
  一部分：`plt_wall*` 必须接在 GUI 会话后面跑（单跑会停在缺变量上），因此也没进流程表。
  它们需要的 `fun_Bfield` / `fun_ellipEK` 都在同一目录里。

## 数据与来源

`GSEQ-FRC-open/` 里的 `.mat` / `.dat` 就是随论文提供的输入数据（`MRR1d.mat`、
`psib_paper.mat`、`grass_ft/data/*.dat` 等），按原目录放在脚本旁边；脚本用相对文件名
读它们，读法与 MATLAB 的"先当前目录、再搜索路径"一致（见 `gseqfrc/mio.py`、
`gseqfrc/matlab_compat.py` 的 `data_path`）。

代码注释里偶尔会出现原文 `.m` 的文件名（`gseq-main/.../x.m`）、移植期的记录编号
（`PORTING.md §…`）或加速版实验的名字（`accel/...`）——那些都是**出处标注**，
对应的文件不在本目录里，也不影响运行。

## 修改版的实测

17 个流程全跑通，总用时 25.5 分钟（同一台机器、Python 3.10 / numpy 1.26 /
scipy 1.15）。每个流程跑完都在 `outputs/<流程名>/` 留下产物（`.mat` + 图片），
可以对照原文的 `.eps`/`.png` 看。整棵树同样做过文件访问审计：跑流程时
**没有读过本文件夹以外的任何工程文件**。

（同样是 17 条流程，总时间比严格移植版多 4 分钟——多出来的全部是 `p1-fig8`：
严格版在第 2 轮就"按预期报错"结束（2 s），修改版把它跑满 70000 轮（170 s）。）

| 流程 | 用时 | 流程 | 用时 |
| --- | --- | --- | --- |
| `p1-1d` | 7.4 s | `p2-appendix-A` | 1.1 s |
| `p1-fig1-2` | 0.5 s | `p2-appendix-B-GSEQ` | 63.3 s |
| `p1-fig3` / `fig4` / `fig5` / `fig6` | 各 0.3 s | `p2-appendix-B-grass-ft` | 66.9 s |
| `p1-fig7` | 30.3 s | `p2-Fig4` | 439.0 s |
| `p1-fig8` | **170.5 s**（跑满 70000 轮） | `p2-Fig5` | 167.3 s |
| `p1-figB1` | 0.1 s | `p2-Fig6`（默认快速内层） | **13.2 s** |
| `p2-Fig7` | 2.1 s | `p2-Fig8-9` | 1.0 s |

改成默认快速内层、并把内层判据按配方自己的口径定在 1e-5 之后，`p2-Fig6` 这条
从 581.9 s 降到 **13.2 s**（其中求解本身 10.1 s）——整轮 17 条流程的最贵一项现在是
`p2-Fig4`（430–440 s）。
想看原文那条逐点松弛（427.5 s，逐位复现冻结基准）就跑 `p2-Fig6-relax`；
两条路径的逐变量差异见 `修改记录.md` 第 7 条。

（对照：原文那份解法在同一台机器上要 8.3 小时；`Fig4` / `Fig5` 是二维定边界求解器
跑满原文设定的迭代轮数 `nt=100000` / `40000`，所以也在几百秒量级。）

`outputs/` 只是跑出来的产物，随时可以删掉重跑。

## 当前进度
新的开始，尝试使用git管理项目版本，并把之前冗余的东西删除。