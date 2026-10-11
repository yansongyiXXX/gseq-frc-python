"""MATLAB 数值例程的 Python 替代：`fsolve`、`integral`。

这两个函数在原程序里承担的角色不同：

* `fsolve(F1, psig, optimset('TolFun',tol,'Tolx',tol))` 解一维剖面方程；
* `integral(fpsi, 0, psim, 'RelTol',tol, 'AbsTol',tol)` 沿磁通求积分。

**两者都是容差受限的迭代算法**：MATLAB 只保证做到用户给的容差。所以这些核函数
冻结下来的数值本身就是"容差内的结果"，不是精确值。实测（`u_fmrr0`，tol=1e-10）：
用一个独立的高精度方法算出的真值与冻结值相差 \(1.8\times10^{-10}\)——正好是原程序
自己要求的量级。**任何实现都不可能比这更接近冻结值，MATLAB 自己换个容差也一样。**

但这不等于"随便实现一个都行"：**迭代路径**（每一步走到哪）会决定多根问题落在哪个
根上。`s_solveMRR_f6` 就是这个教训——起初自己写了一个"看起来对"的信赖域，
结果 `solve_MRR` 沿 r 逐点解出的 `psi(r)` 整条剖面都不同。所以这两个函数现在
**按 MATLAB 的源码逐行移植**，常数、判据、步长公式都照搬：

* `fsolve` → `optim/private/trustnleqn.m` + `optim/private/dogleg.m`
  （+ `finitedifferences` 的默认差分步长，实测反推）；见下文的 `fsolve`。
* `integral` → `matlab/toolbox/matlab/funfun/private/integralCalc.m`
  （Shampine 的 quadva）+ `integralParseArgs.m` 的默认参数
  （Gauss7Kronrod15、AbsTol 1e-10、RelTol 1e-6、InitialIntervalCount 10）。

`integralCalc` 最关键的一点是**区间 [A,B] 先做三次变换到 [-1,1]**：

$$x(t)=\tfrac14(B-A)t(3-t^2)+\tfrac12(A+B),\qquad w(t)=\tfrac34(B-A)(1-t^2)$$

端点附近 \(x-A=O(\varepsilon^2)\) 而 \(w=O(\varepsilon)\)，所以 \(f\sim
(x-A)^{-1/2}\) 这种平方根奇点被**自动抵消**成有限值——`fmrr0` 里被积函数的奇点
恰好就是这一类。这比自己另做一套奇点检测/代换既准又快（而且不用猜什么时候失效）。
"""

from __future__ import annotations

import numpy as np

_EPS = float(np.finfo(float).eps)

# --------------------------------------------------------------- Gauss-Kronrod

#: `integralParseArgs.m` 的默认规则 `Gauss7Kronrod15`（Kronrod 15 点 +
#: 嵌入的 Gauss 7 点）。`_GK_EW = HighWeights - LowWeights` 是误差估计用的权。
_GK_NODES = np.array([
    -0.9914553711208126, -0.9491079123427585, -0.8648644233597691,
    -0.7415311855993944, -0.5860872354676911, -0.4058451513773972,
    -0.2077849550078985, 0.0, 0.2077849550078985, 0.4058451513773972,
    0.5860872354676911, 0.7415311855993944, 0.8648644233597691,
    0.9491079123427585, 0.9914553711208126,
])
_GK_HIGH = np.array([
    0.02293532201052922, 0.06309209262997855, 0.1047900103222502,
    0.1406532597155259, 0.1690047266392679, 0.1903505780647854,
    0.2044329400752989, 0.2094821410847278, 0.2044329400752989,
    0.1903505780647854, 0.1690047266392679, 0.1406532597155259,
    0.1047900103222502, 0.06309209262997855, 0.02293532201052922,
])
_GK_LOW = np.array([
    0.0, 0.1294849661688697, 0.0, 0.2797053914892767, 0.0,
    0.3818300505051189, 0.0, 0.4179591836734694, 0.0,
    0.3818300505051189, 0.0, 0.2797053914892767, 0.0,
    0.1294849661688697, 0.0,
])
_GK_EW = _GK_HIGH - _GK_LOW

#: `integralCalc.m` 的固定参数与 `integralParseArgs.m` 的默认参数。
_DEFAULT_MAXINTERVALCOUNT = 16384          # 2^14
_INITIAL_INTERVAL_COUNT = 10


def integral(f, a, b, rel_tol: float = 1e-6, abs_tol: float = 1e-10,
             max_interval_count: int | None = None):
    """MATLAB `integral(f,a,b,'RelTol',rel_tol,'AbsTol',abs_tol)` 的等价物。

    这是 `integralCalc.m` 的逐行移植（Shampine 的 quadva 自适应求积）：

    1. 有限区间先做**三次变换** \(x(t)=\\frac14(B-A)t(3-t^2)+\\frac12(A+B)\)、
       \(w(t)=\\frac34(B-A)(1-t^2)\)，把区间搬到 \\([-1,1]\\)。变换在端点附近
       把 \\((x-A)^{-1/2}\\) 型奇点自动抹平（`fmrr0` 的被积函数正是这一类）；
    2. 区间先切成 `InitialIntervalCount=10` 段，每段用 Gauss7Kronrod15 求积；
       哪一段的误差估计超过 `(2*tol/pathlen)*|halfh|` 就继续二分，其余累加进
       `q_ok`（`tol = max(AbsTol, RelTol*|q|)`，与 MATLAB 完全一致）；
    3. 误差界 `errbnd = |err_ok| + Σ|errsubs|`，满足 `errbnd <= tol` 就停；
       区间数超过 `MAXINTERVALCOUNT` 或节点间距小到机器精度就按 MATLAB 的
       做法停下并返回当前值。

    `b < a` 时先交换端点再取负号；`a == b` 按 MATLAB 的 `midpArea` 返回
    \\((b-a)f((a+b)/2)\\)。
    """
    a = _scalar(a)
    b = _scalar(b)
    rel_tol = _scalar(rel_tol)
    abs_tol = _scalar(abs_tol)
    max_interval_count = (
        _DEFAULT_MAXINTERVALCOUNT if max_interval_count is None
        else int(max_interval_count)
    )

    reversedir = b < a
    A, B = (b, a) if reversedir else (a, b)

    if A == B:
        q = _midp_area(f, A, B)
    elif np.isfinite(A) and np.isfinite(B):
        q, _ = _vadapt(f, np.array([-1.0, 1.0]),
                       _AtoB_inv_transform(A, B),
                       abs_tol, rel_tol, max_interval_count)
    elif np.isfinite(A) and np.isinf(B):
        q, _ = _vadapt(f, np.array([0.0, 1.0]),
                       _AToInf_inv_transform(A),
                       abs_tol, rel_tol, max_interval_count)
    elif np.isinf(A) and np.isfinite(B):
        q, _ = _vadapt(f, np.array([-1.0, 0.0]),
                       _minusInfToB_inv_transform(B),
                       abs_tol, rel_tol, max_interval_count)
    else:
        q, _ = _vadapt(f, np.array([-1.0, 1.0]),
                       _minusInfToInf_inv_transform(),
                       abs_tol, rel_tol, max_interval_count)

    return -q if reversedir else q


def _vadapt(f, tinterval, transform, abs_tol, rel_tol, max_interval_count):
    """`integralCalc/vadapt` + `iterateScalarValued`。"""
    tinterval, pathlen = _split_interval(tinterval, _INITIAL_INTERVAL_COUNT)
    if pathlen == 0:
        q = _midp_area(f, tinterval[0], tinterval[-1])
        return q, q
    max_interval_count = max(max_interval_count, 2 * (tinterval.size - 1))
    return _iterate_scalar_valued(f, tinterval, pathlen, transform, abs_tol,
                                  rel_tol, max_interval_count)


def _iterate_scalar_valued(f, tinterval, pathlen, transform, abs_tol, rel_tol,
                           max_interval_count):
    """`integralCalc/iterateScalarValued`：主自适应循环。"""
    subs = np.vstack([tinterval[:-1], tinterval[1:]])
    q_ok = 0.0
    err_ok = 0.0
    first_eval = True
    q = 0.0
    while True:
        midpt = 0.5 * (subs[0] + subs[1])
        halfh = 0.5 * (subs[1] - subs[0])
        x = (np.outer(_GK_NODES, halfh) + midpt).reshape(-1)
        t, w = transform(x)
        if first_eval:
            fx = _integrand_values(f, t) * w
            # finalInputChecks：除非"纯绝对误差控制"，RelTol 至少要 100*eps
            if not (abs_tol > 0 and rel_tol == 0):
                rel_tol = max(rel_tol, 100 * _EPS)
            first_eval = False
        else:
            if _check_spacing(t):
                break
            fx = _integrand_values(f, t) * w
        fx = fx.reshape(_GK_NODES.size, -1)
        qsubs = (_GK_HIGH @ fx) * halfh
        errsubs = (_GK_EW @ fx) * halfh
        q = np.sum(qsubs) + q_ok
        tol = max(abs_tol, rel_tol * abs(q))
        accurate = np.abs(errsubs) <= (2 * tol / pathlen) * np.abs(halfh)
        err_ok = err_ok + np.sum(errsubs[accurate])
        errsubs = errsubs[~accurate]
        errbnd = abs(err_ok) + np.sum(np.abs(errsubs))
        if not (np.isfinite(abs(q)) and np.isfinite(errbnd)):
            # MATLAB 这里只是 warning（NonFiniteValue），然后返回当前值
            break
        if errbnd <= tol:
            break
        subs = subs[:, ~accurate]
        q_ok = q_ok + np.sum(qsubs[accurate])
        midpt = midpt[~accurate]
        if subs.shape[1] == 0:
            break
        if 2 * subs.shape[1] > max_interval_count:
            # MATLAB：MaxIntervalCountReached，warning 后返回当前值
            break
        subs = np.vstack([np.concatenate([subs[0], midpt]),
                          np.concatenate([midpt, subs[1]])])
    return q, errbnd


def _check_spacing(t) -> bool:
    """`integralCalc/checkSpacing`：相邻节点间距小到机器精度就该停手了。

    MATLAB 在这一点上会打出"最小步长已快达到 x = ..."的警告——原程序跑
    `solve_MRR` 时确实会打（`fmrr0` 的积分在 `psim` 处有奇点），但结果照常返回。
    """
    ax = np.abs(t)
    limit = 100 * _EPS * np.maximum(ax[:-1], ax[1:])
    return bool(np.any(np.abs(np.diff(t)) <= limit))


def _integrand_values(f, t) -> np.ndarray:
    """按 MATLAB 的方式对整段节点数组求值（用户函数要能接受数组）。"""
    try:
        value = np.asarray(f(t))
    except Exception:  # noqa: BLE001 - 只接受标量的被积函数走下面的逐个求值
        return np.array([_num(f(ti)) for ti in t])
    if value.size == 1:
        value = np.full(t.shape, value.reshape(-1)[0])
    elif value.size == t.size:
        value = value.reshape(t.shape)
    else:
        return np.array([_num(f(ti)) for ti in t])
    if value.dtype.kind in "iub":
        value = value.astype(float)
    return value


def _midp_area(f, a, b):
    """`integralCalc/midpArea`：`a == b`（或端点非有限）时的返回值。"""
    x = 0.5 * (a + b)
    if np.isfinite(a) and np.isfinite(b) and not np.isfinite(x):
        x = a / 2 + b / 2
    return (b - a) * _num(f(x))


def _AtoB_inv_transform(A: float, B: float):
    """`integralCalc/AtoBInvTransform`：[A,B] → [-1,1] 的三次变换。"""

    def transform(t):
        x = 0.25 * (B - A) * t * (3 - t * t) + 0.5 * (B + A)
        w = 0.75 * (B - A) * (1 - t * t)
        # 端点附近改用等价但更稳的表达式
        tshift = np.abs(t) - 1.0
        xoffset = 0.25 * (B - A) * tshift ** 2 * (tshift + 3.0)
        bd = 0.25
        x = np.where(t < -bd, A + xoffset, x)
        x = np.where(t > bd, B - xoffset, x)
        wboundaries = -0.75 * (B - A) * tshift * (tshift + 2.0)
        w = np.where(np.abs(t) > bd, wboundaries, w)
        return x, w

    return transform


def _AToInf_inv_transform(A: float):
    """`integralCalc/AToInfInvTransform`：[A,Inf) → [0,1]。"""

    def transform(t):
        tt = t / (1 - t)
        return A + tt ** 2, 2 * tt / (1 - t) ** 2

    return transform


def _minusInfToB_inv_transform(B: float):
    """`integralCalc/minusInfToBInvTransform`：(-Inf,B] → [-1,0]。"""

    def transform(t):
        tt = t / (1 + t)
        return B - tt ** 2, -2 * tt / (1 + t) ** 2

    return transform


def _minusInfToInf_inv_transform():
    """`integralCalc/minusInfToInfInvTransform`：(-Inf,Inf) → [-1,1]。"""

    def transform(t):
        return t / (1 - t ** 2), (1 + t ** 2) / (1 - t ** 2) ** 2

    return transform


def _split_interval(x, minsubs):
    """`integralCalc/split`：把区间切到"每段不超过总长的 1/minsubs"。"""
    absdx = np.abs(np.diff(x))
    pathlen = x[-1] - x[0]
    if minsubs > 1 and pathlen > 0:
        udelta = minsubs / pathlen
        nnew = np.ceil(absdx * udelta).astype(int) - 1
        for j in np.nonzero(nnew > 0)[0][::-1]:
            nnj = int(nnew[j])
            newpts = x[j] + np.arange(1, nnj + 1) / (nnj + 1) * (x[j + 1] - x[j])
            x = np.concatenate([x[:j + 1], newpts, x[j + 1:]])
    keep = np.concatenate(([True], np.diff(x) != 0))
    x = x[keep]
    if x.size == 1:
        x = np.array([x[0], x[0]])
    return x, pathlen


# --------------------------------------------------------------------- 求根


#: MATLAB R2023a 的 `optimset` 结构体字段（顺序即快照里的顺序）。
#: 调用时没给的字段留空矩阵——原程序里被保存下来的就是这个结构。
_OPTIMSET_FIELDS = (
    "Display", "MaxFunEvals", "MaxIter", "TolFun", "TolX", "FunValCheck",
    "OutputFcn", "PlotFcns", "ActiveConstrTol", "Algorithm",
    "AlwaysHonorConstraints", "DerivativeCheck", "Diagnostics",
    "DiffMaxChange", "DiffMinChange", "FinDiffRelStep", "FinDiffType",
    "GoalsExactAchieve", "GradConstr", "GradObj", "HessFcn", "Hessian",
    "HessMult", "HessPattern", "HessUpdate", "InitBarrierParam",
    "InitTrustRegionRadius", "Jacobian", "JacobMult", "JacobPattern",
    "LargeScale", "MaxNodes", "MaxPCGIter", "MaxProjCGIter", "MaxSQPIter",
    "MaxTime", "MeritFunction", "MinAbsMax", "NoStopIfFlatInfeas",
    "ObjectiveLimit", "PhaseOneTotalScaling", "Preconditioner",
    "PrecondBandWidth", "RelLineSrchBnd", "RelLineSrchBndDuration",
    "ScaleProblem", "SubproblemAlgorithm", "TolCon", "TolConSQP",
    "TolGradCon", "TolPCG", "TolProjCG", "TolProjCGAbs", "TypicalX",
    "UseParallel",
)

_OPTIMSET_CANONICAL = {name.lower(): name for name in _OPTIMSET_FIELDS}


def optimset(**options):
    """MATLAB 的 `optimset('TolFun',tol,'Tolx',tol,'Display','off')`。

    返回一个 55 字段的选项结构体：给过的字段有值，其余是空矩阵。
    选项名大小写不敏感，**而且允许缩写**——原程序写的是 `'Tolx'` 和
    `'Maxfuneval'`，MATLAB 把它们分别归到 `TolX` 与 `MaxFunEvals`
    （快照里能直接看到 `MaxFunEvals = 10000`）。这里按同样的规则归一化：
    先精确匹配，再按无歧义前缀匹配。

    这个结构体本身不参与求根计算，但脚本会把它留在工作区、快照也会记下来，
    所以必须原样复现。
    """
    from . import matlab_compat as mat

    fields = {name: np.zeros((0, 0)) for name in _OPTIMSET_FIELDS}
    for key, value in options.items():
        canonical = _canonical_option(str(key))
        # 数值选项在 MATLAB 里是 1×1 double；字符选项保持文本
        if not isinstance(value, str):
            value = mat.scalar(value)
        fields[canonical] = value
    return mat.struct(**fields)


def _canonical_option(key: str) -> str:
    """把选项名归一化成 MATLAB 的字段名（大小写不敏感，允许无歧义缩写）。"""
    lowered = key.lower()
    exact = _OPTIMSET_CANONICAL.get(lowered)
    if exact is not None:
        return exact
    matches = [name for name in _OPTIMSET_FIELDS if name.lower().startswith(lowered)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"optimset: 未知选项 {key}")
    raise KeyError(f"optimset: 选项名 {key} 有歧义，可能是 {', '.join(matches)}")


def options_value(options, name: str):
    """从 `optimset` 结构体里取一个选项；空矩阵返回 None。"""
    if options is None:
        return None
    names = getattr(np.asarray(options).dtype, "names", None) or ()
    if name not in names:
        return None
    value = np.asarray(options)[name][0, 0]
    if np.size(value) == 0:
        return None
    return value


def fsolve(f, x0, options=None, tol_fun: float = 1e-6, tol_x: float = 1e-6,
           max_iter: int = 200) -> np.ndarray:
    """标量版 `fsolve`：解 `f(x)=0`，返回 1×1 矩阵（与 MATLAB 的形状一致）。

    原程序只在标量方程上用 `fsolve`（`fmrr0/1/2`、`f3pe`、`solve_*`），
    所以这里只实现标量情形，但**算法按 MATLAB 的源码逐条搬**：

    * 主循环 = `optim/private/trustnleqn.m`（MATLAB `fsolve` 的默认算法
      `trust-region-dogleg`）；
    * 子问题 = `optim/private/dogleg.m`（Cauchy 点 + 高斯–牛顿步 + dogleg 折线
      与信赖域边界的交点）；
    * 有限差分雅可比 = `shared/optimlib/finitedifferences` 的默认设置
      （前向差分、相对步长 \(\sqrt{\varepsilon}\)、`TypicalX` 全 1）；
    * 停机判据 = `trustnleqn.m` 里的 `testStop`，用**一阶最优性**
      \(\|J^\top F\|_\infty<TolFun\) 而不是 \(|F|<TolFun\)。

    这些常数和判据都是原程序行为的组成部分，不是"随便选一个"：`Delta` 从 1 起、
    接受阈值 \(\eta_1=0.05\)、放大 \(\eta_2=0.9\) 配 2.5 倍、失败缩到 0.25 倍
    但**留在原地重试**（不是放弃）、新点算出 Inf/NaN 时按"未定义点"处理并把半径
    缩 20 倍——少任何一条，`s_solveMRR_f6` 这类多根问题就会走到另一个根上。

    有限差分步长公式不是从源码里读到的（`finitedifferences` 是 `.p` 文件），而是
    用 `tools/fsolve_step_probe.m` 在 MATLAB 里实测反推出来的：
    \(h=\sqrt{\varepsilon}\,\mathrm{sign}(x)\max(|x|,1)\)（\(x=0\) 取正号）。
    """
    from . import matlab_compat as mat

    # 原程序有两种调用方式：给 TolFun/TolX，或者给一个 optimset 结构体。
    # 结构体里没给的字段在 MATLAB 里取 fsolve.m 的 defaultopt。
    if options is not None:
        value = options_value(options, "TolFun")
        if value is not None:
            tol_fun = value
        value = options_value(options, "TolX")
        if value is not None:
            tol_x = value
        value = options_value(options, "MaxIter")
        if value is not None and max_iter is None:
            max_iter = value
        max_fun_evals = options_value(options, "MaxFunEvals")
        tol_fun_value = options_value(options, "TolFunValue")
    else:
        max_fun_evals = None
        tol_fun_value = None

    x = _root_scalar(x0)
    tol_fun = _scalar(tol_fun)
    tol_x = _scalar(tol_x)
    max_iter = 400 if max_iter is None else _scalar(max_iter)
    max_fun_evals = 100.0 if max_fun_evals is None else _scalar(max_fun_evals)
    # `prepareOptionsForSolver.m`：用户给的是结构体、里面有 TolFun 而没有
    # TolFunValue 时，MATLAB 把 **TolFunValue 设成 TolFun**。这一条影响停机判据
    # （ExitFlag=3 的阈值是 max(TolFunValue^2,eps)），`s_solveMRR_f4` 的内层求根
    # 恰好卡在这上面：按 1e-6 算会提前停下，按 1e-10 算才会像 MATLAB 那样继续迭代。
    tol_fun_value = tol_fun if tol_fun_value is None else _scalar(tol_fun_value)

    fx = _finite_or_nan(f, x)
    if not np.isfinite(fx):
        # trustnleqn 的第一条：初值处函数值非有限，直接报错
        raise ValueError(f"fsolve: 初值处函数值非有限：f({x}) = {fx}")

    x, fx, _ = _trustnleqn_scalar(f, x, fx, tol_fun, tol_x, tol_fun_value,
                                  max_fun_evals, max_iter)
    return mat.scalar(x)


#: `fsolve.m` 里 `defaultopt` 的数值项（原程序没给的选项取其默认值）。
_FSOLVE_DEFAULTS = {
    "MaxIter": 400.0,
    "TolFun": 1e-6,
    "TolFunValue": 1e-6,
    "TolX": 1e-6,
    "TypicalX": 1.0,
    "DiffMinChange": 0.0,
    "DiffMaxChange": np.inf,
}

#: 诊断开关：设成一个 list 时，`_trustnleqn_scalar` 会把每次迭代的
#: `(iter, numFevals, 2*obj, normd, normgradinf, Delta)` 记进去。
#: 与 MATLAB 的 `fsolve(...,'Display','iter')` 表格逐行对齐时用得上
#: （见 `tools/fsolve_probe.m`）。
DEBUG_TRACE: list | None = None


def _trustnleqn_scalar(f, x: float, fx, tol_fun: float, tol_x: float,
                       tol_fun_value: float, max_fun_evals: float,
                       max_iter: float):
    """`optim/private/trustnleqn.m` 的标量移植。返回 `(x, fx, exitflag)`。"""
    # Initialize some trust region parameters.
    Delta = 1e0
    DeltaMax = 1e10
    eta1 = 0.05
    eta2 = 0.9
    alpha1 = 2.5
    alpha2 = 0.25

    it = 0
    num_evals = 1          # 初值那次求值算在 fsolve.m 头上
    step_accept = True
    normd = 0.0
    objold = 1e0
    obj = 0.5 * _sq(fx)

    # 初值处的前向差分雅可比（MATLAB 在第一次 testStop 之前就算好了）
    jac, eval_ok = _fd_jacobian_scalar(f, x, fx)
    num_evals += 1
    if not eval_ok:
        # trustnleqn：初值处差分雅可比无定义 → 报错
        raise ValueError("fsolve: 初值处的有限差分雅可比无定义")
    grad = _conj(jac) * fx
    normgradinf = abs(grad)

    if DEBUG_TRACE is not None:
        DEBUG_TRACE.append((it, num_evals, 2 * obj, None, normgradinf, Delta))

    done, exitflag = _trustnleqn_stop(
        normgradinf, tol_fun, tol_fun_value, tol_x, step_accept, it, eval_ok,
        max_iter, num_evals, max_fun_evals, Delta, normd, obj, objold, 0.0, x)

    while not done:
        it += 1

        # Compute step, d, using dogleg approach.
        d, quad_obj, normd, normdscal = _dogleg_scalar(fx, jac, grad, Delta)
        pred = -quad_obj

        x_trial = x + d
        f_trial = _finite_or_nan(f, x_trial)
        num_evals += 1
        obj_trial = 0.5 * _sq(f_trial)
        ared = obj - obj_trial
        # MATLAB：if pred <= 0, ratio = 0; else ratio = ared/pred;
        ratio = 0.0 if float(np.real(pred)) <= 0.0 else ared / pred

        # faultTolStruct：新点没算出有限的目标函数值时按"未定义点"处理
        well_defined = bool(np.isfinite(obj_trial))

        if float(np.real(ratio)) > eta1 and well_defined:
            # 接受这一步
            x, fx, objold, obj = x_trial, f_trial, obj, obj_trial
            jac, eval_ok = _fd_jacobian_scalar(f, x, fx)
            num_evals += 1
            grad = _conj(jac) * fx
            normgradinf = abs(grad)
            step_accept = True
        else:
            # 拒绝这一步：**不重算雅可比**，下一轮换个半径再试
            step_accept = False

        if DEBUG_TRACE is not None:
            DEBUG_TRACE.append((it, num_evals, 2 * obj, normd, normgradinf, Delta))

        # Update trust region radius.
        if not well_defined:
            Delta = min(normdscal / 20.0, Delta / 20.0)
        elif float(np.real(ratio)) < eta1:
            Delta = alpha2 * normdscal
        elif float(np.real(ratio)) >= eta2:
            Delta = max(Delta, alpha1 * normdscal)
        Delta = min(Delta, DeltaMax)

        done, exitflag = _trustnleqn_stop(
            normgradinf, tol_fun, tol_fun_value, tol_x, step_accept, it,
            eval_ok, max_iter, num_evals, max_fun_evals, Delta, normd, obj,
            objold, d, x)

    return x, fx, exitflag


def _trustnleqn_stop(normgradinf, tol_fun, tol_fun_value, tol_x, step_accept,
                     it, eval_ok, max_iter, num_evals, max_fun_evals, Delta,
                     normd, obj, objold, d, x):
    """`trustnleqn.m` 里的 `testStop`。判据的顺序也要一致。"""
    if not eval_ok:
        return True, 2
    if step_accept and normgradinf < tol_fun:
        return True, 1
    if it > 1 and abs(d) / (abs(x) + 1.0) < max(tol_x ** 2, _EPS):
        return True, 2
    if (it > 1 and step_accept and normd < 0.9 * Delta
            and abs(objold - obj)
            < max(tol_fun_value ** 2, _EPS) * (1.0 + abs(objold))):
        return True, 3
    if Delta < 2 * _EPS:
        return True, -3
    if it >= max_iter:
        return True, 0
    if num_evals >= max_fun_evals:
        return True, 0
    return False, 0


def _dogleg_scalar(fx, jac, grad, Delta, scal: float = 1.0):
    """`optim/private/dogleg.m` 的标量移植（`scalMat` 在标量下恒为 1）。

    返回 `(step, quadObj, normStep, normStepScal)`：`step` 是**未缩放**的步长，
    `normStepScal` 是缩放空间里的步长模（`updateDelta` 用的是它）。
    """
    # Compute scaled gradient and other scaled terms.
    gradscal = grad / scal
    gradscal2 = gradscal / scal
    normgradscal = abs(gradscal)

    if normgradscal >= _EPS:
        # First compute the Cauchy step (in scaled space).
        d_cauchy = -(Delta / normgradscal) * gradscal
        jacvec = jac * gradscal2
        denom = Delta * (jacvec * jacvec)
        with np.errstate(all="ignore"):
            tauterm = normgradscal ** 3 / denom
        # MATLAB 的 min(1,NaN) 给 1（NaN 不算"更小"），numpy 会传播 NaN
        tau_c = tauterm if float(np.real(tauterm)) < 1.0 else 1.0
        d_cauchy = tau_c * d_cauchy

        # Compute quadratic objective at Cauchy point.
        jacvec = jac * (d_cauchy / scal)
        obj_cauchy = gradscal * d_cauchy + 0.5 * (jacvec * jacvec)
        # 同样地，min(NaN,Delta) 在 MATLAB 里给 Delta
        normdcauchy = abs(d_cauchy)
        normd_cauchy = normdcauchy if normdcauchy < Delta else Delta
    else:
        # Set Cauchy step to zero step and continue.
        obj_cauchy = 0.0
        normd_cauchy = 0.0
        d_cauchy = 0.0

    if Delta - normd_cauchy < _EPS:
        # Take the Cauchy step if it is at the boundary of the trust region.
        step, quad_obj = d_cauchy, obj_cauchy
    else:
        # Compute the Gauss-Newton step (in scaled space).
        with np.errstate(all="ignore"):
            d_newton = -(fx / jac) * scal
        if not np.isfinite(abs(d_newton)):
            # Take the Cauchy step if the Gauss-Newton step gives bad values.
            step, quad_obj = d_cauchy, obj_cauchy
        else:
            normd_newt = abs(d_newton)
            if normd_newt <= Delta:
                # Use the Newton direction as the trial step
                step = d_newton
            else:
                # Find the intersect point along dogleg path.
                Delta2 = Delta ** 2
                normdcauchy2 = min(abs(d_cauchy) ** 2, Delta2)
                normdnewt2 = normd_newt ** 2
                dcdn = _conj(d_cauchy) * d_newton
                dcdndist2 = normdcauchy2 + normdnewt2 - 2 * dcdn
                dcdndist2 = float(np.real(dcdndist2))
                if dcdndist2 < 0.0:      # MATLAB 的 max(...,0)
                    dcdndist2 = 0.0
                if dcdndist2 == 0.0:
                    tau_i = 0.0
                else:
                    # Stable method for solving 1-D quadratic
                    a = 0.5 * dcdndist2
                    b = dcdn - normdcauchy2
                    c = 0.5 * (normdcauchy2 - Delta2)
                    q = -0.5 * (b + np.sign(b) * np.sqrt(b ** 2 - 4 * a * c))
                    tau_i = c / q if float(np.real(b)) > 0.0 else q / a
                    if not np.isfinite(abs(tau_i)):
                        tau_i = 0.0      # Take Cauchy step
                step = d_cauchy + tau_i * (d_newton - d_cauchy)
            # Compute quadratic objective at trial point.
            jacvec = jac * (step / scal)
            quad_obj = gradscal * step + 0.5 * (jacvec * jacvec)
            # Compare Cauchy step and trial step (Newton or intersection)
            if float(np.real(obj_cauchy)) < float(np.real(quad_obj)):
                step, quad_obj = d_cauchy, obj_cauchy

    # The step computed was the scaled step.  Unscale it.
    norm_step_scal = abs(step)
    step = step / scal
    norm_step = abs(step)
    return step, quad_obj, norm_step, norm_step_scal


def _fd_jacobian_scalar(f, x: float, fx, rel_step: float | None = None,
                        typical_x: float = 1.0, diff_min_change: float = 0.0,
                        diff_max_change: float = np.inf):
    """前向差分雅可比的标量版，返回 `(jac, eval_ok)`。

    步长公式在 MATLAB 的 `finitedifferences`（`.p` 文件）里，读不到源码，
    由 `tools/fsolve_step_probe.m` 实测反推：对 `x0 = ±2.4, ±0.5, 0, 12, 1e6`
    逐个测出差分点，得到的规律是

    $$h=\sqrt{\varepsilon}\,\mathrm{sign}(x)\max(|x|,\text{TypicalX})$$

    （\(x=0\) 取正号；`TypicalX` 默认 1）。注意步长带 \(x\) 的符号：对负的
    \(x\)，差分点比 \(x\) 更负。`DiffMinChange/DiffMaxChange` 默认 0/Inf，
    对这里的用法不起作用，但保留同样的截断规则。

    差分点上函数值非有限时（MATLAB 的 `chkFunEval`），`eval_ok` 置假，
    由调用方按 `trustnleqn` 的规则终止。
    """
    if rel_step is None:
        rel_step = np.sqrt(_EPS)
    h = rel_step * (1.0 if x >= 0.0 else -1.0) * max(abs(x), typical_x)
    h = min(max(abs(h), diff_min_change), diff_max_change)
    h = h if x >= 0.0 else -h
    value = _finite_or_nan(f, x + h)
    if not np.isfinite(abs(value)):
        return np.nan, False
    return (value - fx) / h, True


def _sq(value) -> float:
    """`Fvec'*Fvec` 的标量版：复数按 \(|F|^2\)（MATLAB 的 `'` 是共轭转置）。"""
    value = complex(value)
    return float((value.conjugate() * value).real)


def _conj(value):
    """MATLAB 的 `'` 对实数没有影响，对复数取共轭。"""
    return np.conj(value) if np.iscomplexobj(value) else value


def _root_scalar(value) -> float:
    """取初值里的那个数（`x0` 在 MATLAB 里是 1×1 矩阵）。"""
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"fsolve: 目前只支持标量方程，x0 形状是 {array.shape}")
    return float(array.reshape(-1)[0])


def _finite_or_nan(f, x: float):
    """求值并把非有限结果原样返回（不抛异常），供迭代判断用。

    与 `_scalar` 的区别是**保留复数**：原程序的 `f3pe`/`fmrr2` 在实数轴上某些点会
    走进复数域，MATLAB 的 `fsolve` 允许函数值取复数（`sqrt(-1)` 给 `1i`）。
    """
    with np.errstate(all="ignore"):
        return _num(f(x))


def _num(value):
    """把标量返回值压成 Python 的 `float` 或 `complex`（保住复数）。"""
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"期望标量，得到形状 {array.shape}")
    scalar = array.reshape(-1)[0]
    if np.iscomplexobj(scalar):
        return complex(scalar)
    return float(scalar)


def _scalar(value) -> float:
    """把被积/被求根函数的返回值压成 Python 浮点。"""
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"期望标量，得到形状 {array.shape}")
    return float(array.reshape(-1)[0])
