r"""把 `frc_discharge_run.py` 的放电结果做成可交互可视化（真 ψ 等值线）。

读 `outputs/frc_formation/discharge_psi.npz`（每步的 ψ 场）+ `discharge.json`
（标量时间线），生成一个自包含的 HTML 片段：装置剖面按真实比例画线圈，磁面用
**真 ψ 的等值线**（`d3.contours` 在浏览器里对降采样网格做 marching squares）。

用法：`python tools/make_discharge_viz.py [输出路径]`
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

SRC = ROOT / "outputs" / "frc_formation"
DEFAULT_OUT = Path(
    r"C:/Users/dell/.codex/visualizations/2026/10/07/"
    r"01a11467-efab-7880-9a55-15289028d126/discharge-timeline.html")

#: ψ 存成整数：psi_scaled = round(psi * PSI_SCALE)
PSI_SCALE = 1e7
#: 内部闭合磁面（相对当帧 ψ_min）与外部开场线（相对当帧 ψ_max）
LEVELS_IN = (0.86, 0.66, 0.46, 0.26)
LEVELS_OUT = (0.05, 0.12, 0.22, 0.38)


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    d = np.load(SRC / "discharge_psi.npz")
    psi, rr, zz, tt = d["psi"], d["rr"], d["zz"], d["t"]
    sc = json.loads((SRC / "discharge.json").read_text(encoding="utf-8"))
    tr = sc["track"]

    # 注意：npz 里的 t 是**秒**；这里统一换成毫秒（之前漏了 ×1e3，
    # 结果所有时刻都被舍成 0，泳道曲线全塌到左边缘）
    T = [round(float(v) * 1e3, 4) for v in tt]
    ZC = [round(float(r["zc"]) * 1e2, 3) for r in tr]
    ZS = [round(float(r["zc_star"]) * 1e2, 3) for r in tr]
    LS = [round(float(r["ls"]) * 1e2, 3) for r in tr]
    LSS = [round(float(r["ls_star"]) * 1e2, 3) for r in tr]
    A = [round(float(r["a"]) / 1e3, 3) for r in tr]
    I2 = [round(float(r["i2"]) / 1e3, 2) for r in tr]
    PHI = [round(float(r["Phi"]) * 1e3, 5) for r in tr]
    PH = {"初态": "初态", "平移": "平移", "压缩": "压缩", "平顶": "平顶"}
    PHASE = [PH.get(r["phase"], r["phase"]) for r in tr]

    # 只保留互不相同的 ψ 帧（平顶段是重复的），并给出 t -> 帧 的映射
    keep = [0]
    for i in range(1, len(psi)):
        if not np.allclose(psi[i], psi[keep[-1]], rtol=1e-9, atol=1e-12):
            keep.append(i)
    # d3.contours 要的是「y 行、x 最快」的行主序：这里存成 [iz][ir]，
    # 即 (NZ, NR) 的 C 序 —— 所以先转置再 ravel（直接 ravel 会是转置的场）。
    frames = [np.rint(psi[i] * PSI_SCALE).astype(int).T.ravel(order="C").tolist()
              for i in keep]
    fidx = [sum(1 for k in keep if k <= i) - 1 for i in range(len(psi))]
    print(f"[viz] ψ 帧 {len(keep)}/{len(psi)}，网格 {psi.shape[1]}x{psi.shape[2]}")

    tpl = TEMPLATE
    strip = strip_png(T, ZC, ZS, LS, LSS, A, I2, T_END=0.6, I2_BASE=56.889)
    for key, val in (
        ("__T__", json.dumps(T)), ("__ZC__", json.dumps(ZC)),
        ("__ZS__", json.dumps(ZS)), ("__LS__", json.dumps(LS)),
        ("__LSS__", json.dumps(LSS)), ("__A__", json.dumps(A)),
        ("__I2__", json.dumps(I2)), ("__PHI__", json.dumps(PHI)),
        ("__PHASE__", json.dumps(PHASE, ensure_ascii=False)),
        ("__RR__", json.dumps([round(float(v), 6) for v in rr])),
        ("__ZZ__", json.dumps([round(float(v), 6) for v in zz])),
        ("__NR__", str(int(psi.shape[1]))), ("__NZ__", str(int(psi.shape[2]))),
        ("__FIDX__", json.dumps(fidx)),
        ("__PSI__", json.dumps(frames, separators=(",", ":"))),
        ("__LIN__", json.dumps(list(LEVELS_IN))),
        ("__LOUT__", json.dumps(list(LEVELS_OUT))),
        ("__SCALE__", str(int(PSI_SCALE))),
        ("__STRIPSVG__", strip),
    ):
        tpl = tpl.replace(key, val)
    if "__" in tpl.replace("__", "", 0)[:0]:  # no-op，保持结构清晰
        pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tpl, encoding="utf-8")
    print(f"[viz] 写出 {out_path}（{out_path.stat().st_size / 1024:.0f} KB）")
    return 0


def strip_png(T, ZC, ZS, LS, LSS, A, I2, *, T_END, I2_BASE):
    """三个泳道用 matplotlib 离线渲染成 PNG（内嵌 data URI）。

    浏览器端不再做任何坐标计算——这一版就是被浏览器里的尺度坑掉的。
    图区在画布上的横向范围固定为 [PL, PR]，游标按这个比例定位。
    """
    import base64
    import io as _io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:                       # 复用程序树的中文字体设置，否则标签会变成方框
        from gseqfrc import mstyle
        mstyle.install()
    except Exception:          # noqa: BLE001
        pass

    fig = plt.figure(figsize=(7.2, 2.2), dpi=140)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.97, bottom=0.17, hspace=0.6)
    axes = [fig.add_subplot(3, 1, i + 1) for i in range(3)]
    ink, grid = "#8a929b", "#c9ced4"
    axes[0].plot(T, ZC, color="#3b6fb6", lw=2)
    axes[0].plot(T, ZS, color=ink, lw=1.2, ls=(0, (4, 3)))
    axes[1].plot(T, LS, color="#2f8f6b", lw=2)
    axes[1].plot(T, LSS, color=ink, lw=1.2, ls=(0, (4, 3)))
    axes[2].plot(T, A, color="#7a5bbf", lw=2)
    axes[2].plot(T, [v - I2_BASE for v in I2], color="#d0822c", lw=1.6)
    for ax, ttl in zip(axes, ("位置 z_c (cm)", "长度 l_s (cm)", "控制电流 (kA)")):
        ax.set_xlim(0, T_END)
        ax.set_ylabel(ttl, fontsize=7, color=ink)
        ax.tick_params(labelsize=6, colors=ink, length=2)
        ax.grid(axis="x", color="#e3e7ea", lw=0.5)
        for sp in ax.spines.values():
            sp.set_color(grid)
            sp.set_linewidth(0.6)
        for xb in (0.1, 0.2):
            ax.axvline(xb, color=grid, lw=0.6, ls=":")
    axes[0].set_xticklabels([])
    axes[1].set_xticklabels([])
    axes[2].set_xlabel("时间 t (ms)", fontsize=7, color=ink)
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def strip_svg(T, ZC, ZS, LS, LSS, A, I2, *, T_END, T_TRANS, T_COMP, I2_BASE):
    """把三个泳道（位置 / 长度 / 电流）在 Python 里算成静态 SVG。

    路径坐标全部在这里算好写死，浏览器端不做任何尺度变换——上一版就是栽在
    浏览器里那套 d3 尺度上（同一次渲染里 rect 用对了、path 却差 1000 倍）。
    """
    W, H = 720, 220
    L, R, TOP, BOT = 56, 14, 16, 26
    iw, ih = W - L - R, H - TOP - BOT
    laneH = ih / 3.0

    def X(v):
        return L + iw * (v / T_END)

    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" height="{H}" role="img" '
         f'aria-label="时间轴上的位置、长度与两个控制量">']
    bands = ((0.0, T_TRANS, 0.09), (T_TRANS, T_TRANS + T_COMP, 0.18),
             (T_TRANS + T_COMP, T_END, 0.09))
    for a, b, op in bands:
        o.append(f'<rect class="frcd-phase" x="{X(a):.1f}" y="{TOP}" '
                 f'width="{X(b) - X(a):.1f}" height="{ih:.1f}" opacity="{op}"/>')
    specs = (('位置 z_c (cm)', (('frcd-s1', ZC, None, 2.0),
                                ('frcd-fg', ZS, '4 3', 1.3))),
             ('长度 l_s (cm)', (('frcd-s3', LS, None, 2.0),
                                ('frcd-fg', LSS, '4 3', 1.3))),
             ('控制电流 (kA)', (('frcd-s5', A, None, 2.0),
                                ('frcd-s2', [v - I2_BASE for v in I2], None, 1.3))))
    for li, (title, series) in enumerate(specs):
        y0 = li * laneH
        vals = [v for _, arr, _, _ in series for v in arr]
        lo, hi = min(vals), max(vals)
        pad = max((hi - lo) * 0.18, 0.4)
        d0, d1 = lo - pad, hi + pad
        r0, r1 = y0 + laneH - 6, y0 + 12

        def Y(v, r0=r0, r1=r1, d0=d0, d1=d1):
            return r0 + (v - d0) / (d1 - d0) * (r1 - r0)

        for cls, arr, dash, sw in series:
            d = ' '.join(('M' if i == 0 else 'L') +
                         f'{X(T[i]):.1f},{Y(v):.1f}' for i, v in enumerate(arr))
            da = f' stroke-dasharray="{dash}"' if dash else ''
            o.append(f'<path class="{cls}" fill="none" stroke-width="{sw}"{da} d="{d}"/>')
        gy = TOP + y0 + laneH
        o.append(f'<line class="frcd-guide" x1="{L}" x2="{L + iw}" '
                 f'y1="{gy:.1f}" y2="{gy:.1f}"/>')
        o.append(f'<text class="frcd-sub" x="6" y="{TOP + y0 + 11:.1f}">{title}</text>')
    ya = TOP + ih
    o.append('<g class="frcd-axis">')
    o.append(f'<line x1="{L}" x2="{L + iw}" y1="{ya}" y2="{ya}"/>')
    for k in range(7):
        v = k / 10.0
        xv = X(v)
        o.append(f'<line x1="{xv:.1f}" x2="{xv:.1f}" y1="{ya}" y2="{ya + 5}"/>')
        o.append(f'<text x="{xv:.1f}" y="{ya + 17}" text-anchor="middle">{v:.1f}</text>')
    o.append('</g>')
    o.append(f'<line id="frcd-cursor" class="frcd-cursor" x1="{L}" x2="{L}" '
             f'y1="{TOP}" y2="{TOP + ih}"/>')
    o.append(f'<text class="frcd-lbl" x="{L + iw / 2:.1f}" y="{H - 6}" '
             f'text-anchor="middle">时间 t (ms)</text>')
    o.append('</svg>')
    return '\n'.join(o)


TEMPLATE = r"""<div id="frcd-viz" class="frcd-viz">
  <style>
    #frcd-viz h3 { margin: 0 0 .35rem; }
    #frcd-viz svg { display: block; max-width: 100%; }
    #frcd-viz .frcd-lbl { fill: var(--foreground, #111418); font-size: 12px; }
    #frcd-viz .frcd-sub { fill: var(--muted-foreground, #5b6b7b); font-size: 11px; }
    #frcd-viz .frcd-struct { fill: none; stroke: var(--border, #9aa0a6); stroke-width: 1.2; }
    #frcd-viz .frcd-coil { fill: var(--foreground, #111418); fill-opacity: .5; stroke: none; }
    #frcd-viz .frcd-flux { fill: none; stroke: var(--viz-series-1, #3b6fb6); stroke-width: 1; stroke-opacity: .75; }
    #frcd-viz .frcd-sep { fill: none; stroke: var(--viz-series-1, #3b6fb6); stroke-width: 2; }
    #frcd-viz .frcd-ext { fill: none; stroke: var(--viz-series-2, #d0822c); stroke-width: 1; stroke-opacity: .6; }
    #frcd-viz .frcd-op { fill: var(--viz-series-1, #3b6fb6); }
    #frcd-viz .frcd-cursor { stroke: var(--foreground, #111418); stroke-width: 1.2; stroke-dasharray: 3 3; }
    #frcd-viz .frcd-guide { stroke: var(--border, #9aa0a6); stroke-width: .7; }
    #frcd-viz .frcd-phase { fill: var(--muted, rgba(128,128,128,.18)); }
    #frcd-viz .frcd-axis text { fill: var(--foreground, #111418); font-size: 11px; }
    #frcd-viz .frcd-axis line, #frcd-viz .frcd-axis .domain { stroke: var(--border, #9aa0a6); }
    #frcd-viz .frcd-s1 { stroke: var(--viz-series-1, #3b6fb6); }
    #frcd-viz .frcd-s2 { stroke: var(--viz-series-2, #d0822c); }
    #frcd-viz .frcd-s3 { stroke: var(--viz-series-3, #2f8f6b); }
    #frcd-viz .frcd-s5 { stroke: var(--viz-series-5, #7a5bbf); }
    #frcd-viz .frcd-fg { stroke: var(--foreground, #111418); }
  </style>
  <h3>一次完整放电：平移 → 压缩 → 平顶（0.60 ms）</h3>
  <p id="frcd-status" class="text-small" aria-live="polite"></p>
  <p class="text-small" style="display:flex;flex-wrap:wrap;gap:1rem;margin:.15rem 0 .35rem">
    <span><span aria-hidden="true" style="display:inline-block;width:14px;height:2px;background:var(--viz-series-1, #3b6fb6);vertical-align:middle;margin-right:6px"></span>分离面 / 闭合磁面（ψ&lt;0）</span>
    <span><span aria-hidden="true" style="display:inline-block;width:14px;height:2px;background:var(--viz-series-2, #d0822c);vertical-align:middle;margin-right:6px"></span>开场线（ψ&gt;0）</span>
    <span><span aria-hidden="true" style="display:inline-block;width:14px;height:8px;background:var(--foreground, #111418);opacity:.5;vertical-align:middle;margin-right:6px"></span>线圈截面</span>
  </p>
  <div id="frcd-device"></div>
  <div id="frcd-strip" style="position:relative">
    <img src="__STRIPSVG__" alt="位置、长度与控制电流随时间的变化" style="width:100%;display:block">
    <div id="frcd-cursor" style="position:absolute;top:1%;height:74%;width:0;left:7.5%;border-left:1px dashed var(--foreground, #111418)"></div>
  </div>
  <div class="viz-controls">
    <button id="frcd-play" class="btn" type="button" aria-pressed="false">播放</button>
    <label class="form-label" for="frcd-time">时间</label>
    <input id="frcd-time" class="form-range" type="range" min="0" max="1" step="0.0005" value="0">
    <span id="frcd-tval" class="text-small tabular-nums">0.000 ms</span>
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(function () {
  var root = document.getElementById('frcd-viz');
  if (!root || typeof d3 === 'undefined') return;
  var devHost = root.querySelector('#frcd-device');
  var stripHost = root.querySelector('#frcd-strip');
  var status = root.querySelector('#frcd-status');
  var playBtn = root.querySelector('#frcd-play');
  var timeInp = root.querySelector('#frcd-time');
  var tval = root.querySelector('#frcd-tval');

  var T = __T__, ZC = __ZC__, ZS = __ZS__, LS = __LS__, LSS = __LSS__;
  var A = __A__, I2 = __I2__, PHI = __PHI__, PHASE = __PHASE__;
  var RR = __RR__, ZZ = __ZZ__, NR = __NR__, NZ = __NZ__;
  var FIDX = __FIDX__, PSI = __PSI__;
  var LEVELS_IN = __LIN__, LEVELS_OUT = __LOUT__, PSI_SCALE = __SCALE__;
  var T_END = 0.6, T_TRANS = 0.1, T_COMP = 0.1, I2_BASE = 56.889;
  var DR = RR[1] - RR[0], DZ = ZZ[1] - ZZ[0];

  var ZW = 0.75, RW = 0.14;
  var COIL_R = 0.15, COIL_H = 0.006;
  var R_IN = COIL_R - COIL_H / 2, R_OUT = COIL_R + COIL_H / 2;
  var COILS = [[-0.7, 0.1], [0.0, 1.2], [0.7, 0.1]];

  function interp(arr, t) {
    if (t <= T[0]) return arr[0];
    if (t >= T[T.length - 1]) return arr[arr.length - 1];
    var i = d3.bisectLeft(T, t);
    var w = (t - T[i - 1]) / (T[i] - T[i - 1]);
    return arr[i - 1] + (arr[i] - arr[i - 1]) * w;
  }

  var H_DEV = 196, H_STRIP = 220;
  var MD = { top: 20, right: 14, bottom: 26, left: 46 };
  var MS = { top: 16, right: 14, bottom: 26, left: 56 };
  var xz = d3.scaleLinear().domain([-0.82, 0.82]);
  var xr = d3.scaleLinear().domain([-0.175, 0.175]);
  var devW = 640, fluxG = null, sepG = null, extG = null, opEl = null;

  function buildDevice(w) {
    devW = w;
    var iw = w - MD.left - MD.right, ih = H_DEV - MD.top - MD.bottom;
    xz.range([0, iw]); xr.range([ih, 0]);
    devHost.innerHTML = '';
    var svg = d3.select(devHost).append('svg')
      .attr('viewBox', '0 0 ' + w + ' ' + H_DEV).attr('width', w).attr('height', H_DEV)
      .attr('role', 'img')
      .attr('aria-label', '装置剖面（真实比例）与真实 ψ 场的磁面；随放电时间演化');
    var g = svg.append('g').attr('transform', 'translate(' + MD.left + ',' + MD.top + ')');
    g.append('rect').attr('class', 'frcd-struct')
      .attr('x', xz(-ZW)).attr('y', Math.min(xr(-RW), xr(RW)))
      .attr('width', xz(ZW) - xz(-ZW)).attr('height', Math.abs(xr(-RW) - xr(RW)));
    g.append('line').attr('x1', 0).attr('x2', iw).attr('y1', xr(0)).attr('y2', xr(0))
      .attr('stroke', 'var(--border)').attr('stroke-width', 1);
    COILS.forEach(function (c) {
      var z0 = c[0] - c[1] / 2, z1 = c[0] + c[1] / 2;
      [1, -1].forEach(function (s) {
        g.append('rect').attr('class', 'frcd-coil')
          .attr('x', xz(z0)).attr('y', Math.min(xr(s * R_IN), xr(s * R_OUT)))
          .attr('width', xz(z1) - xz(z0))
          .attr('height', Math.abs(xr(s * R_OUT) - xr(s * R_IN)));
      });
    });
    extG = g.append('g'); sepG = g.append('g'); fluxG = g.append('g');
    opEl = g.append('circle').attr('class', 'frcd-op').attr('r', 3);
    [['端线圈', -0.7], ['中心线圈', 0], ['端线圈', 0.7]].forEach(function (c) {
      svg.append('text').attr('class', 'frcd-sub').attr('x', MD.left + xz(c[1]))
        .attr('y', 14).attr('text-anchor', 'middle').text(c[0]);
    });
    g.append('g').attr('class', 'frcd-axis')
      .attr('transform', 'translate(0,' + ih + ')')
      .call(d3.axisBottom(xz).ticks(5));
    svg.append('text').attr('class', 'frcd-lbl').attr('x', MD.left + iw / 2)
      .attr('y', H_DEV - 6).attr('text-anchor', 'middle').text('轴向位置 z (m)');
    svg.append('text').attr('class', 'frcd-lbl').attr('transform', 'rotate(-90)')
      .attr('x', -(MD.top + ih / 2)).attr('y', 12).attr('text-anchor', 'middle')
      .text('r (m)');
  }

  function pathOf(coords, sgn) {
    var d = '';
    for (var p = 0; p < coords.length; p++) {
      var rings = coords[p];
      for (var q = 0; q < rings.length; q++) {
        var ring = rings[q];
        for (var j = 0; j < ring.length; j++) {
          var z = ZZ[0] + ring[j][1] * DZ, r = RR[0] + ring[j][0] * DR;
          d += (j ? 'L' : 'M') + xz(z).toFixed(1) + ' ' + xr(sgn * r).toFixed(1);
        }
        d += 'Z';
      }
    }
    return d;
  }

  var laneX = null, cursor = null, cursorEl = null;

  function buildStrip(w) {
    cursorEl = root.querySelector('#frcd-cursor');   // 泳道图是内嵌 PNG
    return;                                          // 不再动态重画
    var iw = w - MS.left - MS.right, ih = H_STRIP - MS.top - MS.bottom;
    var laneH = ih / 3;
    var x = d3.scaleLinear().domain([0, T_END]).range([0, iw]);
    stripHost.innerHTML = '';
    var svg = d3.select(stripHost).append('svg')
      .attr('viewBox', '0 0 ' + w + ' ' + H_STRIP).attr('width', w).attr('height', H_STRIP)
      .attr('role', 'img').attr('aria-label', '时间轴上的位置、长度与两个控制量');
    var g = svg.append('g').attr('transform', 'translate(' + MS.left + ',' + MS.top + ')');
    [[0, T_TRANS], [T_TRANS, T_TRANS + T_COMP], [T_TRANS + T_COMP, T_END]].forEach(function (p, i) {
      g.append('rect').attr('class', 'frcd-phase')
        .attr('x', x(p[0])).attr('y', 0).attr('width', x(p[1]) - x(p[0]))
        .attr('height', ih).attr('opacity', i === 1 ? .18 : .09);
    });
    function val(k, i) {
      return k === 'a' ? A[i] : k === 'i2' ? I2[i] - I2_BASE
        : k === 'zc' ? ZC[i] : k === 'zs' ? ZS[i] : k === 'ls' ? LS[i] : LSS[i];
    }
    [{ keys: ['zc', 'zs'], cls: ['frcd-s1', 'frcd-fg'],
       dashes: [null, '4 3'], title: '位置 z_c (cm)' },
     { keys: ['ls', 'lss'], cls: ['frcd-s3', 'frcd-fg'],
       dashes: [null, '4 3'], title: '长度 l_s (cm)' },
     { keys: ['a', 'i2'], cls: ['frcd-s5', 'frcd-s2'],
       dashes: [null, null], title: '控制电流 (kA)' }].forEach(function (sp, li) {
      var y0 = li * laneH, vals = [];
      sp.keys.forEach(function (k) { T.forEach(function (_, i) { vals.push(val(k, i)); }); });
      var ex = d3.extent(vals);
      var pad = Math.max((ex[1] - ex[0]) * 0.18, 0.4);
      var y = d3.scaleLinear().domain([ex[0] - pad, ex[1] + pad]).range([y0 + laneH - 6, y0 + 12]);
      var gl = g.append('g');
      sp.keys.forEach(function (k, si) {
        var d = '';
        for (var j = 0; j < T.length; j++) {
          var X = iw * (T[j] / T_END);      // 不依赖闭包里的 x 尺度
          d += (j ? 'L' : 'M') + X.toFixed(1) + ' ' + y(val(k, j)).toFixed(1);
        }
        gl.append('path').attr('class', sp.cls[si]).attr('fill', 'none')
          .attr('stroke-width', si === 0 ? 2 : 1.3)
          .attr('stroke-dasharray', sp.dashes[si])
          .attr('d', d);
      });
      gl.append('line').attr('class', 'frcd-guide')
        .attr('x1', 0).attr('x2', iw).attr('y1', y0 + laneH).attr('y2', y0 + laneH);
      svg.append('text').attr('class', 'frcd-sub').attr('x', 6)
        .attr('y', MS.top + y0 + 11).text(sp.title);
    });
    g.append('g').attr('class', 'frcd-axis')
      .attr('transform', 'translate(0,' + ih + ')')
      .call(d3.axisBottom(x).ticks(6));
    svg.append('text').attr('class', 'frcd-lbl').attr('x', MS.left + iw / 2)
      .attr('y', H_STRIP - 6).attr('text-anchor', 'middle').text('时间 t (ms)');
    cursor = g.append('line').attr('class', 'frcd-cursor').attr('y1', 0).attr('y2', ih)
      .attr('x1', 0).attr('x2', 0);
    laneX = function (v) { return iw * (v / T_END); };
  }

  var lastFrame = -1;
  function drawFields(i) {
    if (i === lastFrame) return;
    lastFrame = i;
    var psi = PSI[FIDX[i]];
    // 阈值必须和数据同量纲：psi 存的是整数（ψ×PSI_SCALE），这里不要再除
    var pmin = d3.min(psi), pmax = d3.max(psi);
    var levels = LEVELS_IN.map(function (f) { return pmin * f; })
      .concat([0])
      .concat(LEVELS_OUT.map(function (f) { return pmax * f; }));
    var cs = d3.contours().size([NR, NZ]).thresholds(levels)(psi);
    var dFlux = '', dSep = '', dExt = '';
    cs.forEach(function (multi, k) {
      var lv = levels[k];
      var dps = pathOf(multi.coordinates, 1) + pathOf(multi.coordinates, -1);
      if (lv > 0) { dExt += dps; }
      else if (lv === 0) { dSep += dps; }
      else { dFlux += dps; }
    });
    fluxG.selectAll('path.frcd-flux').data([0]).join('path')
      .attr('class', 'frcd-flux').attr('d', dFlux);
    sepG.selectAll('path.frcd-sep').data([0]).join('path')
      .attr('class', 'frcd-sep').attr('d', dSep);
    extG.selectAll('path.frcd-ext').data([0]).join('path')
      .attr('class', 'frcd-ext').attr('d', dExt);
  }

  function draw(t) {
    var i = Math.min(Math.max(d3.bisectLeft(T, t + 1e-9), 0), T.length - 1);
    drawFields(i);
    var zc = interp(ZC, t) / 100;
    opEl.attr('cx', xz(zc)).attr('cy', xr(0));
    if (cursorEl) {
      cursorEl.style.left = (7.5 + 92.0 * (t / T_END)).toFixed(2) + '%';
    }
    tval.textContent = t.toFixed(3) + ' ms';
    status.textContent = '阶段：' + PHASE[i]
      + '　位置 z_c = ' + interp(ZC, t).toFixed(2) + ' cm（目标 ' + interp(ZS, t).toFixed(2) + '）'
      + '　长度 l_s = ' + interp(LS, t).toFixed(2) + ' cm（目标 ' + interp(LSS, t).toFixed(2) + '）'
      + '　差模 a = ' + interp(A, t).toFixed(2) + ' kA'
      + '　中心线圈 I₂ = ' + interp(I2, t).toFixed(2) + ' kA'
      + '　Φ/Φ₀ = ' + (interp(PHI, t) / PHI[0]).toFixed(3);
  }

  function render() {
    var w = Math.max(320, (devHost.clientWidth || 640));
    buildDevice(w); buildStrip(w); lastFrame = -1;
    draw(+timeInp.value * T_END);
  }

  var raf = null, t0 = null;
  function loop(ts) {
    if (t0 === null) t0 = ts;
    var frac = Math.min((ts - t0) / 8000, 1);
    timeInp.value = frac;
    draw(frac * T_END);
    if (frac < 1) { raf = requestAnimationFrame(loop); }
    else { playBtn.textContent = '播放'; playBtn.setAttribute('aria-pressed', 'false'); raf = null; }
  }
  playBtn.addEventListener('click', function () {
    if (raf) {
      cancelAnimationFrame(raf); raf = null; t0 = null;
      playBtn.textContent = '播放'; playBtn.setAttribute('aria-pressed', 'false'); return;
    }
    t0 = null; playBtn.textContent = '暂停'; playBtn.setAttribute('aria-pressed', 'true');
    raf = requestAnimationFrame(loop);
  });
  timeInp.addEventListener('input', function () {
    if (raf) {
      cancelAnimationFrame(raf); raf = null; t0 = null;
      playBtn.textContent = '播放'; playBtn.setAttribute('aria-pressed', 'false');
    }
    draw(+timeInp.value * T_END);
  });
  if (window.ResizeObserver) { new ResizeObserver(render).observe(devHost); }
  else { window.addEventListener('resize', render); }
  render();
})();
</script>
"""


if __name__ == "__main__":
    raise SystemExit(main())
