r"""平移控制与压缩控制的极限扫描。

把两个控制旋钮分别往一个方向推到"二维平衡解不出来"为止，并把失效区间二分细化：

* **压缩**：只动中心线圈（`frc_ls_target.py` 的主控旋钮），目标 \(l_s\) 从设计点
  往下走，看 \(l_s\) 能压到多小、需要的中心线圈电流是多少、伸长比 \(E\) 会不会
  先掉到 1 以下。
* **平移**：只动差模（两端线圈反向），目标 \(z_c\) 往 \(\pm\) 两个方向走，
  看能推多远、需要的差模电流是多少、长度 \(l_s\) 被带着漂多少。

每一步都用上一步的解暖启动（这是能把一套扫描跑完的关键），命中用割线法。
失败（求解器报"起手状态无解"或结果非有限）就记为该方向的**极限**，再在
最后一个好点与第一个坏点之间二分 3 次，把极限收窄到大约 1/8 步长。

用法：

    python tools/frc_control_limits.py                 # 默认扫描
    python tools/frc_control_limits.py --ls-step 1 --zc-step 1   # 更细

产物：`outputs/frc_formation/control_limits.{json,png}`
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.chdir(ROOT)
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

import design_point as dp  # noqa: E402

OUT = ROOT / "outputs" / "frc_formation"
SHAPING = ROOT / "outputs" / "design_point" / "shaping.json"

DIFF = np.array([1.0, 0.0, -1.0])          # 差模方向：两端反向
COIL_L_H = 314.1624953042869e-6
COIL_TURNS = 20.0
CTRL_VOLT = 450.0
ZC_QUANT = 0.75 / 128.0 / 2.0              # z_c 网格量化步长 [m]
LS_QUANT = 2.0 * ZC_QUANT                  # l_s 网格量化步长 [m]（≈0.586 cm）


def measure(psi, diag, s) -> dict:
    """与 `frc_zc_control.py` / `frc_ls_target.py` 同口径的观测量。"""
    nR, nZ = s["nR"], s["nZ"]
    zz = np.asarray(s["zz"], float)
    p2 = np.asarray(psi, float).reshape(nR, nZ, order="F")
    out = dict(Phi=float(diag["Phi_private_Wb"]), Rs=float(diag["Rs_mid"]))
    cols = np.nonzero((p2 < 0.0).any(axis=0))[0]
    if cols.size:
        zlo, zhi = float(zz[cols].min()), float(zz[cols].max())
        out["zc"] = 0.5 * (zlo + zhi)
        out["zh"] = 0.5 * (zhi - zlo)
        out["E"] = out["zh"] / out["Rs"]
        out["ls"] = 2.0 * out["zh"]
    else:
        out["zc"] = out["zh"] = out["E"] = out["ls"] = float("nan")
    return out


class Sweeper:
    """一次建模型，之后所有点都从上一点的解暖启动。"""

    def __init__(self):
        self.s, self.model, self.il_scale, _ = dp.build_design_model()
        self.s0 = dp.gcm._setup(dp.gcm.SRC_DIR)
        self.il_base = np.asarray(self.s["Il"], float)
        self.psi0, self.ci0 = dp.warm_start_guess(self.il_scale)
        # 基线解一次，供两条扫描分别作起点
        self.base = None
        m, psi, ci = self.solve(self.il_base, iz_scale=True,
                                psi=self.psi0, ci=self.ci0)
        if m is None:
            raise SystemExit("[limits] 基线平衡都解不出来，先检查设计点产物")
        self.base = dict(Il=self.il_base.copy(), m=m, psi=psi, ci=ci)

    # ------------------------------------------------------------------
    def solve(self, il, *, iz_scale=False, psi=None, ci=None):
        """解一次二维平衡；失败返回 (None, None, None)。"""
        il = np.asarray(il, float)
        if iz_scale:
            iz = float(self.s["Izeta0"]) * float(np.mean(il / self.il_base))
        else:
            iz = float(self.s["Izeta0"])
        t0 = time.perf_counter()
        try:
            out = dp.gcm.solve(dp.gcm.SRC_DIR, model=self.model, Il=il,
                               params=dict(Izeta0=iz),
                               psi0=self.psi0 if psi is None else psi,
                               CI0=self.ci0 if ci is None else ci)
        except Exception as exc:  # noqa: BLE001
            print(f"      求解异常：{type(exc).__name__}: {exc}")
            return None, None, None
        idx = np.asarray(out["psiq"], float)
        nci = float(out["CI"])
        m = measure(idx, out["diag"], self.s0)
        m["C"] = float(out["C"])
        m["wall_s"] = time.perf_counter() - t0
        if not (np.isfinite(m["ls"]) and np.isfinite(m["zc"])):
            print("      解非有限（反场区消失）")
            return None, idx, nci
        return m, idx, nci

    # ------------------------------------------------------------------
    def hit(self, kind, target, il_guess, J, psi, ci, *, tol, J0,
            max_iter=8, step_lim=0.08, allow_oblate=False):
        """割线法命中目标。

        返回 (观测量 or None, psi, ci, il, J, status)；
        status = ``ok`` / ``no_converge`` / ``no_solution``。
        """
        il = np.asarray(il_guess, float).copy()
        # 上一步的割线斜率跑飞了就退回标称灵敏度（否则会一直限步、磨不出结果）
        if not (0.25 * abs(J0) <= abs(J) <= 4.0 * abs(J0)):
            J = J0
        prev = None
        for it in range(1, max_iter + 1):
            m, psi, ci = self.solve(il, iz_scale=(kind == "ls"), psi=psi, ci=ci)
            if m is None:
                return None, psi, ci, il, J, "no_solution"
            if kind == "ls" and np.isfinite(m["E"]) and m["E"] < 1.0:
                # 伸长比 < 1 意味着"比球还扁"，不是 FRC 了
                print(f"      E={m['E']:.3f} < 1：超出 FRC 的几何极限")
                return (None if not allow_oblate else m), psi, ci, il, J, (
                    "E_lt_1" if not allow_oblate else "ok")
            x = il[1] if kind == "ls" else 0.5 * (il[0] - il[2])
            val = m["ls"] if kind == "ls" else m["zc"]
            err = target - val
            print(f"      it{it}: {'l_s' if kind == 'ls' else 'z_c'}="
                  f"{val * 1e2:+8.3f} cm  err={err * 1e2:+.3f} cm")
            if abs(err) <= tol:          # tol 一律按"绝对误差"给（米）
                return m, psi, ci, il, J, "ok"
            if prev is not None:
                x0, v0 = prev
                if x != x0:
                    jn = (val - v0) / (x - x0)
                    if (np.isfinite(jn) and jn != 0
                            and np.sign(jn) == np.sign(J0)
                            and 0.25 * abs(J0) <= abs(jn) <= 4.0 * abs(J0)):
                        J = jn
            prev = (x, val)
            if not np.isfinite(J) or J == 0:
                return None, psi, ci, il, J, "no_converge"
            d = float(err / J)
            if kind == "ls":
                d = float(np.clip(d, -step_lim * self.il_base[1],
                                  step_lim * self.il_base[1]))
                il = il.copy()
                il[1] = il[1] + d
            else:
                d = float(np.clip(d, -step_lim * self.il_base[0],
                                  step_lim * self.il_base[0]))
                il = self.il_base + (x + d) * DIFF
        return None, psi, ci, il, J, "no_converge"


def record(kind, target, m, il, base_il, wall, base_ls=None):
    if kind == "ls":
        cur = il[1]
        d_cur = float(cur - base_il[1])
        return dict(target=float(target), ok=True, achieved=float(m["ls"]),
                    d_Il_A=d_cur, Il_A=[float(v) for v in il],
                    E=float(m["E"]), Rs=float(m["Rs"]), Phi=float(m["Phi"]),
                    zc=float(m["zc"]), ls=float(m["ls"]),
                    a_A=float(0.5 * (il[0] - il[2])),
                    d_ls=float(m["ls"] - base_ls) if base_ls else float("nan"),
                    t_ramp_s=float(abs(d_cur / COIL_TURNS) * COIL_L_H / CTRL_VOLT),
                    wall_s=float(wall), reason="")
    a = float(0.5 * (il[0] - il[2]))
    return dict(target=float(target), ok=True, achieved=float(m["zc"]),
                a_A=a, Il_A=[float(v) for v in il],
                ls=float(m["ls"]), Rs=float(m["Rs"]), Phi=float(m["Phi"]),
                zc=float(m["zc"]), d_Il_A=float("nan"),
                d_ls=float(m["ls"] - base_ls) if base_ls else float("nan"),
                t_ramp_s=float(abs(a) / COIL_TURNS * COIL_L_H / CTRL_VOLT),
                wall_s=float(wall), reason="")


def bad_record(kind, target, reason=""):
    return dict(target=float(target), ok=False, achieved=float("nan"),
                a_A=float("nan"), d_Il_A=float("nan"), E=float("nan"),
                Rs=float("nan"), Phi=float("nan"), ls=float("nan"),
                zc=float("nan"), d_ls=float("nan"), t_ramp_s=float("nan"),
                wall_s=float("nan"), reason=reason)


def sweep(sweeper, kind, targets, *, J0, tol, base_ls, label, on_row=None,
          rows=None, allow_oblate=False):
    """按 targets 顺序推进；返回 (记录表, 第一个坏点 or None, 末态雅可比)。"""
    il = sweeper.base["Il"].copy()
    psi = np.asarray(sweeper.base["psi"], float)
    ci = float(sweeper.base["ci"])
    J = float(J0)
    a = 0.0
    prev_t = None
    rows = [] if rows is None else rows
    first_bad = None
    for tgt in targets:
        t0 = time.perf_counter()
        if kind == "zc":
            a_guess = (tgt / J if prev_t is None
                       else a + (tgt - prev_t) / J)
            a_guess = float(np.clip(a_guess, -0.35 * sweeper.il_base[0],
                                    0.35 * sweeper.il_base[0]))
            guess = sweeper.base["Il"] + a_guess * DIFF
        else:
            guess = il.copy()
        print(f"[{label}] 目标 {tgt * 1e2:+.2f} cm")
        m, psi2, ci2, il2, J2, status = sweeper.hit(
            kind, tgt, guess, J, psi, ci, tol=tol, J0=J0,
            allow_oblate=allow_oblate)
        J = J2
        wall = time.perf_counter() - t0
        if status != "ok":
            rows.append(bad_record(kind, tgt, status))
            if on_row:
                on_row()
            first_bad = tgt
            break
        psi, ci, il = psi2, ci2, il2
        if kind == "zc":
            a = 0.5 * (il[0] - il[2])
        prev_t = tgt
        rows.append(record(kind, tgt, m, il, sweeper.base["Il"], wall, base_ls))
        if on_row:
            on_row()
    return rows, first_bad, J


def refine(sweeper, kind, good, bad, *, J0, tol, base_ls, label, n=3,
           on_row=None, rows=None, allow_oblate=False):
    """在 good/bad 之间二分，逼近极限。"""
    il = sweeper.base["Il"].copy()
    psi = np.asarray(sweeper.base["psi"], float)
    ci = float(sweeper.base["ci"])
    J = float(J0)
    rows = [] if rows is None else rows
    lo, hi = good, bad
    for k in range(n):
        mid = 0.5 * (lo + hi)
        if kind == "zc":
            guess = sweeper.base["Il"] + (mid / J if J else 0.0) * DIFF
        else:
            guess = il.copy()
        print(f"[{label}] 二分{k + 1}/{n}：目标 {mid * 1e2:+.3f} cm")
        m, psi2, ci2, il2, J2, status = sweeper.hit(
            kind, mid, guess, J, psi, ci, tol=tol, J0=J0,
            allow_oblate=allow_oblate)
        J = J2
        if status != "ok":
            rows.append(bad_record(kind, mid, status))
            if on_row:
                on_row()
            hi = mid
        else:
            psi, ci, il = psi2, ci2, il2
            rows.append(record(kind, mid, m, il, sweeper.base["Il"], 0.0,
                               base_ls))
            if on_row:
                on_row()
            lo = mid
    return rows, lo, hi


def plot_limits(res, out_png):
    """四格汇总：压缩电流 / 压缩形状 / 平移电流 / 平移长度漂移。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mstyle_ok = True
    try:
        from gseqfrc import mstyle

        mstyle.install()
    except Exception:  # noqa: BLE001
        mstyle_ok = False

    comp = res["compression"]["rows"]
    # 多轮扫描会留下同一个目标点的旧记录：按目标去重，保留最后一次
    seen = {}
    for r in res["translation"]["rows"]:
        seen[round(r["target"], 6)] = r
    trans = [seen[k] for k in sorted(seen)]
    base_I2_kA = float(res["baseline"]["Il_kA"][1])

    fig, ax = plt.subplots(2, 2, figsize=(13.0, 8.6), constrained_layout=True)

    def split(rows):
        xs = np.asarray([r["target"] * 1e2 for r in rows], float)
        ok = np.asarray([bool(r["ok"]) for r in rows], bool)
        return xs, ok

    def mark_bad(axis, rows, xs, ok, ymin, ymax):
        for x, good, r in zip(xs, ok, rows):
            if not good:
                axis.axvline(x, color=mstyle.RED if mstyle_ok else "r",
                             lw=1.2, ls=":")
                axis.annotate(f"{r['reason']}\n解不出", (x, 0.5 * (ymin + ymax)),
                              color=mstyle.RED if mstyle_ok else "r",
                              fontsize=9, ha="right", va="center", rotation=90)

    # (a) 压缩：中心线圈电流
    x, ok = split(comp)
    i2 = np.asarray([base_I2_kA + r["d_Il_A"] / 1e3 if r["ok"] else np.nan
                     for r in comp])
    ax[0, 0].plot(x[ok], i2[ok], "o-", color=mstyle.BLUE if mstyle_ok else "b",
                  lw=2.0, label="需要的电流")
    ax[0, 0].axhline(base_I2_kA, color="0.6", lw=1.0, ls="--",
                     label=f"设计点 {base_I2_kA:.1f} kA")
    ax[0, 0].set_xlabel(r"$l_s$ [cm]")
    ax[0, 0].set_ylabel(r"$I_2$ [kA]")
    ax[0, 0].set_title("压缩：需要的中心线圈电流")
    ax[0, 0].legend(fontsize=9, frameon=False)
    lim = res["compression"].get("limit")
    if lim:
        ax[0, 0].axvline(lim["fail_at"] * 1e2, color=mstyle.RED if mstyle_ok
                         else "r", lw=1.2, ls=":")
        ax[0, 0].annotate(f"极限 ≈ {lim['ok_to'] * 1e2:.1f} cm",
                          (lim["ok_to"] * 1e2, np.nanmin(i2[ok])),
                          fontsize=9, ha="right", va="bottom",
                          color=mstyle.RED if mstyle_ok else "r")
    ax[0, 0].invert_xaxis()

    # (b) 压缩：伸长比 E 与 Rs
    e = np.asarray([r["E"] if r["ok"] else np.nan for r in comp])
    rs = np.asarray([r["Rs"] * 1e2 if r["ok"] else np.nan for r in comp])
    ax01 = ax[0, 1]
    ax01.plot(x[ok], e[ok], "o-", color=mstyle.GREEN if mstyle_ok else "g",
              lw=2.0)
    ax01.axhline(1.0, color="0.5", lw=1.2, ls=":",
                 label="E=1（球）")
    ax01.set_xlabel(r"$l_s$ [cm]")
    ax01.set_ylabel(r"$E$", color=mstyle.GREEN if mstyle_ok else "g")
    ax01.set_title("压缩：形状先到哪一步失效")
    ax02 = ax01.twinx()
    ax02.plot(x[ok], rs[ok], "s--", color="0.45", lw=1.4, ms=4,
              label=r"$R_s$ [cm]")
    ax02.set_ylabel(r"$R_s$ [cm]")
    ax01.legend(fontsize=9, frameon=False, loc="upper left")
    ax02.legend(fontsize=9, frameon=False, loc="lower right")
    if lim:
        ax01.axvline(lim["fail_at"] * 1e2, color=mstyle.RED if mstyle_ok
                     else "r", lw=1.2, ls=":")
    ax01.invert_xaxis()

    # (c) 平移：差模电流
    x, ok = split(trans)
    a = np.asarray([r["a_A"] / 1e3 if r["ok"] else np.nan for r in trans])
    order = np.argsort(x)
    ax[1, 0].plot(x[order][ok[order]], a[order][ok[order]], "o-",
                  color=mstyle.MAGENTA if mstyle_ok else "m", lw=2.0)
    ax[1, 0].axhline(0.0, color="0.7", lw=1.0)
    ax[1, 0].set_xlabel(r"$z_c$ [cm]")
    ax[1, 0].set_ylabel(r"$a$ [kA]")
    ax[1, 0].set_title("平移：需要的差模电流")
    for tag, l in res["translation"]["limits"].items():
        ax[1, 0].axvline(l["fail_at"] * 1e2, color=mstyle.RED if mstyle_ok
                         else "r", lw=1.2, ls=":")

    # (d) 平移：长度漂移
    d = np.asarray([r["d_ls"] * 1e2 if r["ok"] else np.nan for r in trans])
    ax[1, 1].plot(x[order][ok[order]], d[order][ok[order]], "o-",
                  color=mstyle.BLUE if mstyle_ok else "b", lw=2.0)
    ax[1, 1].axhline(0.0, color="0.7", lw=1.0)
    ax[1, 1].set_xlabel(r"$z_c$ [cm]")
    ax[1, 1].set_ylabel(r"$\Delta l_s$ [cm]")
    ax[1, 1].set_title("平移：被带着漂的长度（二阶耦合）")
    for tag, l in res["translation"]["limits"].items():
        ax[1, 1].axvline(l["fail_at"] * 1e2, color=mstyle.RED if mstyle_ok
                         else "r", lw=1.2, ls=":")

    for a_ in ax.ravel():
        a_.grid(alpha=0.3)
    if mstyle_ok:
        mstyle.style_figure(fig, fontsize=12.0)
        for a_ in ax.ravel():
            a_.grid(alpha=0.3)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ls-step", type=float, default=2.0, help="压缩步长 [cm]")
    ap.add_argument("--ls-min", type=float, default=26.0, help="压缩扫描下限 [cm]")
    ap.add_argument("--ls-fine-from", type=float, default=None,
                    help="l_s 低于这个值（cm）以后改用 --ls-fine-step 细步")
    ap.add_argument("--ls-fine-step", type=float, default=0.5,
                    help="深压缩段的细步长 [cm]")
    ap.add_argument("--allow-oblate", action="store_true",
                    help="不把伸长比 E<1 判为失效（用来分清数值极限与几何极限）")
    ap.add_argument("--zc-max", type=float, default=9.0, help="平移扫描上限 [cm]")
    ap.add_argument("--zc-step", type=float, default=1.5, help="平移步长 [cm]")
    ap.add_argument("--tol-ls", type=float, default=1.0,
                    help="压缩收敛容差，单位是 l_s 的网格格数（≈0.586 cm/格）")
    ap.add_argument("--tol-zc", type=float, default=0.6 * ZC_QUANT)
    ap.add_argument("--only", choices=("both", "compression", "translation"),
                    default="both", help="只跑其中一段")
    ap.add_argument("--replot", action="store_true",
                    help="不重跑扫描，只读 control_limits.json 重新出图")
    args = ap.parse_args()

    if args.replot:
        res = json.loads((OUT / "control_limits.json").read_text(encoding="utf-8"))
        png = plot_limits(res, OUT / "control_limits.png")
        print(f"[limits] 产物：{png}")
        return 0

    sw = Sweeper()
    sh = json.loads(SHAPING.read_text(encoding="utf-8"))
    J_ls = 2.0 * float(np.asarray(sh["jacobian"]["zh"], float)[1])
    J_zc = float(np.asarray(sh["jacobian"]["zc"], float) @ DIFF)
    base_ls = float(sw.base["m"]["ls"])
    base_il = sw.base["Il"]
    print(f"[limits] 基线：l_s={base_ls * 1e2:.2f} cm、"
          f"z_c={sw.base['m']['zc'] * 1e2:+.3f} cm、"
          f"I_l={np.asarray(base_il) / 1e3} kA")
    print(f"[limits] 灵敏度：dl_s/dI2={J_ls:.3e} m/A、"
          f"dz_c/da={J_zc:.3e} m/A、z_c 量化={ZC_QUANT * 1e3:.3f} mm")

    t_start = time.perf_counter()
    tol_ls_abs = args.tol_ls * LS_QUANT
    OUT.mkdir(parents=True, exist_ok=True)
    # 与已有结果合并（只跑一段时不擦掉另一段）
    if (OUT / "control_limits.json").is_file():
        res = json.loads((OUT / "control_limits.json").read_text(encoding="utf-8"))
    else:
        res = {}
    res["baseline"] = dict(ls_cm=base_ls * 1e2,
                           zc_cm=sw.base["m"]["zc"] * 1e2,
                           Il_kA=[float(v / 1e3) for v in base_il])
    res["gains"] = dict(dls_dI2_m_per_A=J_ls, dzc_da_m_per_A=J_zc,
                        zc_quant_m=ZC_QUANT)

    def save():
        res["wall_s"] = time.perf_counter() - t_start
        (OUT / "control_limits.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 压缩 ----
    ref_ls = None
    if args.only in ("both", "compression"):
        top = round(base_ls * 1e2) - 1
        if args.ls_fine_from is None:
            ls_cm = np.arange(top, args.ls_min - 1e-9, -args.ls_step)
        else:
            coarse = np.arange(top, args.ls_fine_from - 1e-9, -args.ls_step)
            fine = np.arange(args.ls_fine_from - args.ls_fine_step,
                             args.ls_min - 1e-9, -args.ls_fine_step)
            ls_cm = np.concatenate([coarse, fine])
        ls_targets = ls_cm * 1e-2
        rows_ls = []
        res["compression"] = dict(rows=rows_ls, limit=None)
        _, bad_ls, _ = sweep(sw, "ls", ls_targets, J0=abs(J_ls),
                             tol=tol_ls_abs, base_ls=base_ls, label="压缩",
                             on_row=save, rows=rows_ls,
                             allow_oblate=args.allow_oblate)
        if bad_ls is not None and rows_ls:
            good = [r["target"] for r in rows_ls if r["ok"]][-1]
            _, lo, hi = refine(sw, "ls", good, bad_ls, J0=abs(J_ls),
                               tol=tol_ls_abs, base_ls=base_ls,
                               label="压缩", on_row=save, rows=rows_ls,
                               allow_oblate=args.allow_oblate)
            ref_ls = dict(ok_to=float(lo), fail_at=float(hi))
            res["compression"]["limit"] = ref_ls
        rows_ls.sort(key=lambda r: r["target"])
        save()

    # ---- 平移（正反向各一轮，都从基线重新出发）----
    zc_lim = {}
    if args.only in ("both", "translation"):
        prev_rows = list(res.get("translation", {}).get("rows", []))
        zc_lim = dict(res.get("translation", {}).get("limits", {}))
        zc_rows = []
        res["translation"] = dict(rows=zc_rows, limits=zc_lim)
        for sign, tag in ((+1.0, "正向 +z"), (-1.0, "反向 -z")):
            targets = sign * np.arange(args.zc_step, args.zc_max + 1e-9,
                                       args.zc_step) * 1e-2
            _, bad, _ = sweep(sw, "zc", targets, J0=J_zc, tol=args.tol_zc,
                              base_ls=base_ls, label=f"平移{tag}",
                              on_row=save, rows=zc_rows)
            if bad is not None:
                good = [r["target"] for r in zc_rows if r["ok"]][-1]
                _, lo, hi = refine(sw, "zc", good, bad, J0=J_zc,
                                   tol=args.tol_zc, base_ls=base_ls,
                                   label=f"平移{tag}", on_row=save,
                                   rows=zc_rows)
                zc_lim[tag] = dict(ok_to=float(lo), fail_at=float(hi))
            res["translation"]["limits"] = zc_lim
            have = {round(r["target"], 8) for r in zc_rows}
            zc_rows.extend(r for r in prev_rows
                           if round(r["target"], 8) not in have)
            zc_rows.sort(key=lambda r: r["target"])
            save()

    wall = time.perf_counter() - t_start
    save()
    print(f"\n[limits] 总用时 {wall:.0f} s；产物：{OUT / 'control_limits.json'}")
    try:
        png = plot_limits(res, OUT / "control_limits.png")
        print(f"[limits] 产物：{png}")
    except Exception as exc:  # noqa: BLE001
        print(f"[limits] 出图跳过：{exc}")
    if res.get("compression", {}).get("limit"):
        ref_ls = res["compression"]["limit"]
        print(f"[limits] 压缩极限：能到 {ref_ls['ok_to'] * 1e2:.2f} cm，"
              f"在 {ref_ls['fail_at'] * 1e2:.2f} cm 处解不出来")
    for tag, lim in res.get("translation", {}).get("limits", {}).items():
        print(f"[limits] 平移{tag}极限：能到 {lim['ok_to'] * 1e2:+.2f} cm，"
              f"在 {lim['fail_at'] * 1e2:+.2f} cm 处解不出来")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
