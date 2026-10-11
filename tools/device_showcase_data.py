r"""Extract compact, faithful data for the design-point device showcase.

Reads ``outputs/design_point/forward.npz`` and writes
``outputs/design_point/showcase_data.json`` containing:

* per-frame scalars (t, Phi, Rs, S0, drive, Bz0, beta, elongation),
* the normalised separatrix / nested-flux-surface profiles,
* the machine geometry (vessel + coils).

The profiles are near self-similar along the decay, so one normalised family
plus per-frame (Rs, z_s) reproduces the 2-D shapes at ~1 % level.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import contourpy

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "design_point" / "forward.npz"
OUT = ROOT / "outputs" / "design_point" / "showcase_data.json"

LEVELS = (0.012, 0.20, 0.40, 0.65, 0.90)   # fractions of the O-point psi
N_PTS = 40


def _main_loop(rr, zz, field, level):
    """Longest contour line of ``field`` at ``level``, as (z, r) points."""
    gen = contourpy.contour_generator(x=zz, y=rr, z=field)
    lines = gen.lines(level)
    if not lines:
        return None
    best = max(lines, key=lambda a: len(a))
    return np.asarray(best, dtype=float)          # columns: z, r


def _loop(poly, zm, rm):
    """Resample a closed poloidal contour to N_PTS points, normalised by the
    outer scale ``(zm, rm)``, starting at the +z tip.

    The result is the real flux-surface cross-section (a spindle-torus loop),
    so revolving it reproduces the separatrix and the nested surfaces.
    """
    z, r = poly[:, 0], poly[:, 1]
    if np.hypot(z[0] - z[-1], r[0] - r[-1]) > 1e-9:
        z = np.append(z, z[0])
        r = np.append(r, r[0])
    if zm <= 0 or rm <= 0:
        return None
    k0 = int(np.argmax(z))                                  # start at the tip
    z, r = np.roll(z, -k0), np.roll(r, -k0)
    zn, rn = z / zm, r / rm
    seg = np.hypot(np.diff(zn), np.diff(rn))
    s = np.concatenate(([0.0], np.cumsum(seg)))
    tgt = np.linspace(0.0, s[-1], N_PTS, endpoint=False)
    out = np.column_stack((np.interp(tgt, s, zn), np.interp(tgt, s, rn)))
    return out


def main() -> None:
    d = np.load(SRC)
    rr, zz = d["rr"], d["zz"]
    psi = d["psi_all"]
    n = len(d["t"])

    frames = []
    outer_profiles = []
    for k in range(n):
        phi = float(d["Phi"][k])
        rs = float(d["Rs"][k])
        if not np.isfinite(rs) or psi[k].min() >= 0:
            continue
        p = psi[k]
        p_o = float(p.min())
        outer = _main_loop(rr, zz, p, LEVELS[0] * p_o)
        if outer is None:
            continue
        zm, rm = float(np.max(np.abs(outer[:, 0]))), float(np.max(outer[:, 1]))
        prof = []
        for lv in LEVELS:
            line = outer if lv == LEVELS[0] else _main_loop(rr, zz, p, lv * p_o)
            prof.append(None if line is None else _loop(line, zm, rm))
        outer_profiles.append(prof)
        frames.append(dict(
            t=float(d["t"][k]), Phi=phi, Rs=rs,
            S0=float(d["S0"][k]), drive=float(d["drive"][k]),
            Bz0=float(d["Bz0"][k]), beta=float(d["beta"][k]),
            W=float(d["W"][k]), Izeta0=float(d["Izeta0"][k]),
            zm=zm, rm=rm,
        ))

    # sanity: how self-similar is the normalised separatrix?
    ref = outer_profiles[0][0]
    dev = []
    for prof in outer_profiles[1:]:
        cur = prof[0]
        dev.append(float(np.max(np.hypot(cur[:, 0] - ref[:, 0], cur[:, 1] - ref[:, 1]))))

    # keep the median frame as the canonical shape family
    mid = len(outer_profiles) // 2
    family = [[[round(float(x), 3), round(float(y), 3)] for x, y in prof]
              for prof in outer_profiles[mid]]

    def col(key, nd=6):
        return [round(float(fr[key]), nd) for fr in frames]

    payload = dict(
        shapeDev=round(float(np.max(dev)) if dev else 0.0, 3),
        levels=list(LEVELS),
        contours=family,
        geom=dict(
            Rw=float(d["Rw"]), zw=float(d["zw"]), Be=float(d["Be"]),
            rl=d["rl"].tolist(), zl=d["zl"].tolist(),
            hl=d["hl"].tolist(), wl=d["wl"].tolist(),
        ),
        meta=json.loads(str(d["meta"])),
        f=dict(
            t=[round(float(fr["t"]) * 1e3, 5) for fr in frames],
            Phi=col("Phi", 10),
            Rs=col("Rs", 5),
            S0=col("S0", 4),
            drive=col("drive", 4),
            zm=col("zm", 5),
            beta=col("beta", 4),
            Bz0=col("Bz0", 5),
        ),
    )
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"[showcase] frames={len(frames)} wrote {OUT}")
    print(f"[showcase] normalised-shape max deviation = {payload['shapeDev']:.3f}")
    for f in frames[::4]:
        print(f"  t={f['t']*1e3:6.3f} ms  Phi={f['Phi']:.4e}  Rs={f['Rs']*1e2:5.2f} cm "
              f" zm={f['zm']*1e2:5.2f} cm  S0={f['S0']:.3f}  drive={f['drive']:.3f}")


if __name__ == "__main__":
    main()
