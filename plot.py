#!/usr/bin/env python3
"""Figure generator for the GRPO length-penalty study.

Figures (written to --outdir, default figures/):
  fig_thinking_roll.png   rolling mean thinking_tokens per arm over steps
  fig_correct_roll.png    rolling correct rate per arm over steps
  fig_difficulty.png      thinking length by prompt difficulty tier
  fig_penalty_shape.png   actual penalty curves from qwen_normal.py
  fig_eval_bars.png       greedy GSM8K eval bars: accuracy + correct-per-1000-tokens

Style: near-black background, left-aligned title + muted subtitle, recessive
axes, faint raw per-step traces under EMA-smoothed lines, direct end labels
(staggered when they overlap) instead of legend boxes, Inter + JetBrains Mono.
Colors: baseline light grey, frozen base dark grey, deepseek lime #b2f827,
composer pink #f974c3.

Use --demo to run on synthetic data when real runs/ data is missing.
"""
import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG = "#0b0b0c"
FG = "#f2f2f2"
MUTED = "#8c8c8c"
GRID = "#242426"
SPINE = "#3a3a3d"
WHITE = "#d9d9d9"   # trained baseline: neutral reference, not a brand hue
FROZEN = "#6e6e72"  # frozen base model
LIME = "#b2f827"
PINK = "#f974c3"

ARMS = ["qwen17-base", "qwen17-ds", "qwen17-composer"]
ARM_LABEL = {
    "qwen17-base": "baseline",
    "qwen17-ds": "deepseek",
    "qwen17-composer": "composer",
}
ARM_COLOR = {
    "qwen17-base": WHITE,
    "qwen17-ds": LIME,
    "qwen17-composer": PINK,
}

SANS = ["Inter", "DejaVu Sans"]
MONO = ["JetBrainsMono NF", "DejaVu Sans Mono"]

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "font.family": SANS,
    "font.size": 12,
    "text.color": FG,
    "axes.edgecolor": SPINE,
    "axes.labelcolor": MUTED,
    "axes.labelsize": 12,
    "axes.labelpad": 10,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 1.0,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.major.pad": 8,
    "ytick.major.pad": 8,
    "lines.solid_capstyle": "round",
    "savefig.dpi": 200,
})


def new_fig(ncols=1, w=11, h=6.4, **kw):
    fig, ax = plt.subplots(1, ncols, figsize=(w, h), **kw)
    return fig, ax


def _mono_ticks(ax):
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(MONO)


def _style(fig, ax, title=None):
    """Recessive axes; optional small per-panel title, left-aligned."""
    if title:
        ax.set_title(title, loc="left", fontsize=13, fontweight="semibold",
                     color=FG, pad=12)
    _mono_ticks(ax)


def _header(fig, title, subtitle=None, gap=0.0):
    """Figure title + muted subtitle, left-aligned; spacing fixed in inches."""
    h = fig.get_figheight()
    fig.text(0.03, 1 - 0.28 / h, title, fontsize=19, fontweight="semibold",
             color=FG, va="top", ha="left")
    if subtitle:
        fig.text(0.03, 1 - 0.72 / h, subtitle, fontsize=11.5, color=MUTED,
                 va="top", ha="left")
    fig.subplots_adjust(top=1 - (1.3 + gap) / h)


def _legend(fig, items, y=0.905):
    """Compact frameless key, top-right: [(label, color), ...]."""
    x = 0.97
    for label, color in reversed(items):
        t = fig.text(x, y, label, fontsize=11, color=FG, va="top", ha="right")
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(fig.transFigure.inverted())
        fig.add_artist(matplotlib.lines.Line2D(
            [bb.x0 - 0.026, bb.x0 - 0.008], [bb.y0 + bb.height / 2] * 2,
            color=color, lw=2.6, solid_capstyle="round",
            transform=fig.transFigure))
        x = bb.x0 - 0.045


def _end_label(ax, x, y, color, text, ly=None, name=None):
    """Dot at the true endpoint (x, y); label at ly (staggered) if given."""
    ax.plot([x], [y], "o", color=color, markersize=6.5,
            markeredgecolor=BG, markeredgewidth=1.5, zorder=6, clip_on=False)
    if ly is None:
        ly = y
    ax.annotate(text, (x, ly), xytext=(10, 0), textcoords="offset points",
                color=FG, fontsize=10.5, fontfamily=MONO, va="center",
                zorder=6, annotation_clip=False)
    if name:
        ax.annotate(name, (x, ly), xytext=(10 + 7.2 * len(text) + 6, 0),
                    textcoords="offset points", color=color, fontsize=10.5,
                    va="center", zorder=6, annotation_clip=False)


def _ref_line(ax, y, color, text):
    """Dashed horizontal reference line labelled just outside its right end."""
    ax.axhline(y, color=color, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate(text, (1, y), xycoords=("axes fraction", "data"),
                xytext=(6, 0), textcoords="offset points", ha="left",
                va="center", color=color, fontsize=10, annotation_clip=False)


def staggered(ys, y0, y1, frac=0.045):
    """Return label y positions: same order-preserving values pushed apart by
    at least frac * (y1 - y0), bottom-up."""
    rng = y1 - y0 if y1 > y0 else 1.0
    min_sep = frac * rng
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = [0.0] * len(ys)
    prev = None
    for i in order:
        v = ys[i]
        if prev is not None and v - prev < min_sep:
            v = prev + min_sep
        out[i] = v
        prev = v
    return out


def smooth(x, w=11):
    """Centered moving average; NaN where the window would run off the ends,
    so a few edge steps can't drag the line."""
    x = np.asarray(x, dtype=float)
    y = np.full_like(x, np.nan)
    y[w // 2: len(x) - w // 2] = np.convolve(x, np.ones(w) / w, "valid")
    return y


# ---------------------------------------------------------------- data

def load_reward_samples(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def steps_means(rows, key):
    """Group every 4 rows into one training step; mean of key per step."""
    vals = [float(r[key]) for r in rows]
    n = len(vals) // 4
    arr = np.array(vals[: n * 4]).reshape(n, 4)
    return arr.mean(axis=1)


def rolling(x, w=21):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    k = np.ones(w) / w
    y = np.convolve(x, k, mode="same")
    y[: w // 2] = np.nan
    y[-(w // 2):] = np.nan
    return y


def load_summary(path):
    """eval summary.json, or analyze.py's analysis.json (reference = base)."""
    with open(path) as f:
        data = json.load(f)
    if "comparison" in data:
        comp = data["comparison"]
        arms = {data["reference"]: next(iter(comp.values()))["reference"]}
        arms.update({k: v["arm"] for k, v in comp.items()})
        return {"arms": arms}
    return data


# ---------------------------------------------------------------- demo

def make_demo_data(seed=7):
    rng = np.random.default_rng(seed)
    logs = {}
    for arm in ARMS:
        rows = []
        for step in range(100):
            for _ in range(4):
                if arm == "qwen17-base":
                    base_think = 900 + step * 4
                    corr = 0.42 + 0.15 * (1 - math.exp(-step / 45))
                elif arm == "qwen17-ds":
                    base_think = 700 + step * 1.5
                    corr = 0.45 + 0.20 * (1 - math.exp(-step / 35))
                else:
                    base_think = 620 + step * 0.5
                    corr = 0.46 + 0.22 * (1 - math.exp(-step / 30))
                tt = int(max(80, rng.normal(base_think, 140)))
                ok = bool(rng.random() < corr + rng.normal(0, 0.02))
                comp = tt + int(rng.normal(120, 30))
                pen = -min(0.5, 0.5 * tt / 512) if arm != "qwen17-base" else 0.0
                rows.append({
                    "correct": ok,
                    "truncated": comp >= 4096,
                    "thinking_tokens": tt,
                    "completion_tokens": comp,
                    "penalty": pen,
                    "reward": (1.0 if ok else 0.0) + pen,
                })
        logs[arm] = rows

    summary = {"arms": {
        "base":          {"accuracy": 0.418, "total_generated_tokens": 52100, "median_thinking_all": 870.0, "correct_per_1000_tokens": 8.02, "truncated": 3, "finished": 1316},
        "trained-base":  {"accuracy": 0.552, "total_generated_tokens": 90300, "median_thinking_all": 1180.0, "correct_per_1000_tokens": 6.11, "truncated": 9, "finished": 1310},
        "deepseek":      {"accuracy": 0.634, "total_generated_tokens": 64100, "median_thinking_all": 760.0, "correct_per_1000_tokens": 9.89, "truncated": 1, "finished": 1318},
        "composer":      {"accuracy": 0.661, "total_generated_tokens": 59700, "median_thinking_all": 690.0, "correct_per_1000_tokens": 11.07, "truncated": 0, "finished": 1319},
    }}
    return logs, summary


# ---------------------------------------------------------------- figs

def _roll_fig(logs, key, title, subtitle, ylabel, outpath, ylim=None,
              fmt="{:.0f}"):
    fig, ax = new_fig()
    ends = []
    for arm in ARMS:
        m = steps_means(logs[arm], key)
        xs = np.arange(len(m))
        sm = smooth(m)
        c = ARM_COLOR[arm]
        ax.plot(xs, m, color=c, lw=0.8, alpha=0.13, zorder=2)
        ax.plot(xs, sm, color=c, lw=2.4, zorder=3)
        last = np.flatnonzero(~np.isnan(sm))[-1]
        ends.append((arm, xs[last], sm[last]))
    if ylim:
        ax.set_ylim(*ylim)
    else:  # scale to the smoothed lines; raw spikes run off the top
        hi = max(np.nanmax(l.get_ydata()) for l in ax.lines[1::2])
        ax.set_ylim(0, hi * 1.3)
    ax.set_xlim(0, len(xs) - 1)
    y0, y1 = ax.get_ylim()
    lys = staggered([e[2] for e in ends], y0, y1, frac=0.05)
    for (arm, x, last), ly in zip(ends, lys):
        _end_label(ax, x, last, ARM_COLOR[arm], fmt.format(last), ly=ly,
                   name=ARM_LABEL[arm])
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    _style(fig, ax)
    _header(fig, title, subtitle)
    fig.subplots_adjust(left=0.08, right=0.83, bottom=0.11)
    fig.savefig(outpath)
    plt.close(fig)


def fig_thinking_roll(logs, outdir):
    _roll_fig(logs, "thinking_tokens", "Thinking length over training",
              "Mean thinking tokens per step (faint), 11-step moving average (bold)",
              "thinking tokens", os.path.join(outdir, "fig_thinking_roll.png"),
              fmt="{:.0f}")


def fig_correct_roll(logs, outdir):
    _roll_fig(logs, "correct", "Correct rate over training",
              "Share of the 4 completions per step that are correct, 11-step moving average",
              "correct rate", os.path.join(outdir, "fig_correct_roll.png"),
              ylim=(0, 1), fmt="{:.2f}")


# ---- penalty formulas, copied from qwen_normal.py ---------------------

def deepseek_penalty(length, norm=512, cap=0.5, strength=0.5, effort=1.0):
    return -min(cap, strength * effort * length / norm)


def composer_raw(length, k, q):
    if k <= 0 or q < 1:
        raise ValueError("composer k must be > 0 and q must be >= 1")
    if q == 1:
        return math.log1p(k * length) / k
    return math.expm1((1 - q) * math.log1p(k * length)) / (k * (1 - q))


def composer_penalty(length, norm=512, cap=0.5, strength=0.5, effort=1.0,
                     k=1 / 512, q=1.5):
    anchor = min(cap, strength * effort)
    return -anchor * composer_raw(length, k, q) / composer_raw(norm, k, q)


def fig_penalty_shape(outdir, T=512):
    L = np.linspace(0, 4096, 600)
    ds = np.array([deepseek_penalty(l) for l in L])
    co = np.array([composer_penalty(l) for l in L])

    fig, ax = new_fig()
    ax.fill_between(L, ds, co, where=L > T, color=PINK, alpha=0.07, lw=0)
    ax.plot(L, ds, color=LIME, lw=2.4, zorder=3)
    ax.plot(L, co, color=PINK, lw=2.4, zorder=3)
    ax.axvline(T, color=SPINE, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.plot([T], [-0.5], "o", color=FG, markersize=7,
            markeredgecolor=BG, markeredgewidth=1.5, zorder=5)
    ax.annotate(f"both cost -0.50 at the anchor, T = {T}", (T, -0.5),
                xytext=(14, 22), textcoords="offset points", color=FG,
                fontsize=10.5)
    ax.annotate("linear: flat after the cap", (2300, -0.5), xytext=(0, 8),
                textcoords="offset points", ha="center", color=MUTED,
                fontsize=10.5)
    ax.annotate("concave: keeps charging, slope falls", (2300, -1.0),
                xytext=(0, -28), textcoords="offset points", ha="center",
                color=MUTED, fontsize=10.5)
    _end_label(ax, L[-1], ds[-1], LIME, f"{ds[-1]:.2f}", name="deepseek")
    _end_label(ax, L[-1], co[-1], PINK, f"{co[-1]:.2f}", name="composer")
    ax.set_xlim(0, 4096)
    ax.set_ylim(-1.25, 0.05)
    ax.set_xticks([0, 512, 1024, 2048, 3072, 4096])
    ax.set_xlabel("thinking length (tokens)")
    ax.set_ylabel("reward penalty")
    _style(fig, ax)
    _header(fig, "Length penalty shapes",
            "Capped linear (DeepSeek) vs concave q-log (Composer), "
            "matched at the anchor")
    fig.subplots_adjust(left=0.08, right=0.83, bottom=0.11)
    fig.savefig(os.path.join(outdir, "fig_penalty_shape.png"))
    plt.close(fig)


EVAL_ROWS = [("base", "frozen base", FROZEN), ("trained-base", "baseline", WHITE),
             ("deepseek", "deepseek", LIME), ("composer", "composer", PINK)]


def _hbars(ax, labels, vals, colors, fmt, xmax):
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, height=0.56, color=colors, zorder=3)
    for yi, v in zip(y, vals):
        ax.annotate(fmt.format(v), (v, yi), xytext=(6, 0),
                    textcoords="offset points", va="center", color=FG,
                    fontsize=10.5, fontfamily=MONO)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=FG, fontsize=11)
    ax.set_xlim(0, xmax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.spines["bottom"].set_visible(False)


def fig_eval_bars(summary, outdir):
    arms = summary.get("arms", summary)
    rows = [r for r in EVAL_ROWS if r[0] in arms]
    labels = [r[1] for r in rows]
    colors = [r[2] for r in rows]
    acc = [100 * float(arms[r[0]]["accuracy"]) for r in rows]
    eff = [float(arms[r[0]]["correct_per_1000_tokens"]) for r in rows]

    fig, (a1, a2) = new_fig(2, h=4.8, gridspec_kw={"wspace": 0.22})
    _hbars(a1, labels, acc, colors, "{:.1f}%", 85)
    _hbars(a2, labels, eff, colors, "{:.2f}", max(eff) * 1.25)
    a2.set_yticklabels([])
    _style(fig, a1, "accuracy")
    _style(fig, a2, "correct answers per 1,000 tokens")
    for lbl in a1.get_yticklabels():
        lbl.set_fontfamily(SANS)
    _header(fig, "Greedy GSM8K evaluation",
            "128 held-out questions, temperature 0", gap=0.35)
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.08)
    fig.savefig(os.path.join(outdir, "fig_eval_bars.png"))
    plt.close(fig)


def load_pass4(path):
    """Rows ordered per question, 4 rows each. Return (pass@4, per-sample acc)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    n = len(rows) // 4
    corr = np.array([bool(r["correct"]) for r in rows[: n * 4]]).reshape(n, 4)
    return corr.any(axis=1).mean(), corr.mean()


PASS4_NAMES = ["base", "trained-base", "deepseek", "composer"]
PASS4_TITLE = {"gsm8k": "GSM8K", "math500": "MATH-500"}


def fig_pass4_bars(datadir, name, outdir):
    rows = [r for r in EVAL_ROWS if r[0] in PASS4_NAMES]
    p4, acc = [], []
    for key, _, _ in rows:
        p, a = load_pass4(os.path.join(datadir, f"{key}.jsonl"))
        p4.append(100 * p)
        acc.append(100 * a)

    fig, ax = new_fig(h=4.8)
    y = np.arange(len(rows))[::-1]
    for yi, (_, label, c), a, p in zip(y, rows, acc, p4):
        ax.plot([a, p], [yi, yi], color=c, lw=2.4, alpha=0.45, zorder=2)
        ax.plot([a], [yi], "o", ms=8, color=BG, markeredgecolor=c,
                markeredgewidth=2, zorder=3)
        ax.plot([p], [yi], "o", ms=8, color=c, markeredgecolor=BG,
                markeredgewidth=1.5, zorder=3)
        ax.annotate(f"{a:.1f}", (a, yi), xytext=(-10, 0),
                    textcoords="offset points", ha="right", va="center",
                    color=MUTED, fontsize=10.5, fontfamily=MONO)
        ax.annotate(f"{p:.1f}", (p, yi), xytext=(10, 0),
                    textcoords="offset points", ha="left", va="center",
                    color=FG, fontsize=10.5, fontfamily=MONO)
    ax.set_yticks(y)
    ax.set_yticklabels([r[1] for r in rows], color=FG, fontsize=11)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(max(0, 10 * math.floor(min(acc) / 10) - 10),
                min(100, 10 * math.ceil(max(p4) * 1.15 / 10)))
    ax.set_xlabel("percent of questions")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.spines["bottom"].set_visible(False)
    _style(fig, ax)
    for lbl in ax.get_yticklabels():
        lbl.set_fontfamily(SANS)
    title = PASS4_TITLE.get(name, name)
    _header(fig, f"{title}: per-sample accuracy → pass@4",
            "64 questions, 4 samples each at temperature 0.8.  "
            "○ per-sample accuracy   ● pass@4 (any of 4 correct)",
            gap=0.1)
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.14)
    fig.savefig(os.path.join(outdir, f"fig_pass4_{name}.png"))
    plt.close(fig)


def make_demo_pass4():
    rng = np.random.default_rng(11)
    out = {}
    for bench, rates in (("gsm8k", (0.42, 0.55, 0.63, 0.66)),
                         ("math500", (0.30, 0.44, 0.52, 0.55))):
        d = {}
        for nm, p in zip(PASS4_NAMES, rates):
            rows = []
            for _ in range(64):
                for s in range(4):
                    rows.append({"correct": bool(rng.random() < p), "sample": s})
            d[nm] = rows
        out[bench] = d
    return out


def write_demo_pass4(tmpdir, demo):
    import tempfile
    for bench, d in demo.items():
        bd = os.path.join(tmpdir, f"{bench}-pass4")
        os.makedirs(bd, exist_ok=True)
        for nm, rows in d.items():
            with open(os.path.join(bd, f"{nm}.jsonl"), "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")


# ---- prompt difficulty -------------------------------------------------

# ordinal ramp, one hue (blue), brighter = harder on the dark surface
TIERS = [("easy", "#184f95"), ("medium", "#3987e5"), ("hard", "#cde2fb")]


def prompt_tiers(logs):
    """Tier per step from the pooled correct rate of all arms' completions
    (each step is one prompt, shared by every arm)."""
    pc = np.mean([steps_means(logs[a], "correct") for a in ARMS], axis=0)
    return np.where(pc >= 0.75, 0, np.where(pc >= 0.4, 1, 2)), pc


def fig_difficulty(logs, outdir):
    tier, _ = prompt_tiers(logs)
    think = {a: np.array([float(r["thinking_tokens"]) for r in logs[a]])
             .reshape(-1, 4) for a in ARMS}
    trunc = {a: np.array([bool(r["truncated"]) for r in logs[a]]).reshape(-1, 4)
             for a in ARMS}
    pooled = np.mean([think[a].mean(1) for a in ARMS], axis=0)
    n = len(tier)

    fig, (a1, a2) = new_fig(2, w=12, gridspec_kw={"wspace": 0.3,
                                                   "width_ratios": [1.5, 1]})
    # left: every training step, coloured by how hard its prompt was
    a1.bar(np.arange(n), pooled, width=0.72,
           color=[TIERS[t][1] for t in tier], zorder=3)
    a1.axvline(n / 2 - 0.5, color=SPINE, lw=1.0, ls=(0, (4, 3)), zorder=1)
    for t, (name, _) in enumerate(TIERS):
        early, late = (tier[: n // 2] == t).sum(), (tier[n // 2:] == t).sum()
        y = -4 - 17 * t
        a1.annotate("■", (0.0, 1.0), xycoords="axes fraction",
                    xytext=(4, y), textcoords="offset points", va="top",
                    color=TIERS[t][1], fontsize=11)
        a1.annotate(f"{name}: {early} in steps 0–49, {late} in 50–99",
                    (0.0, 1.0), xycoords="axes fraction", xytext=(20, y),
                    textcoords="offset points", va="top", color=FG,
                    fontsize=10)
    a1.set_xlim(-1, n)
    a1.set_ylim(0, pooled.max() * 1.25)
    a1.set_xlabel("training step (one prompt per step)")
    a1.set_ylabel("thinking tokens, mean of all 12 completions")
    _style(fig, a1, "each step's prompt, by difficulty")

    # right: median thinking per tier, one dot per arm
    ys = np.arange(len(TIERS))[::-1]
    for t, (name, c) in enumerate(TIERS):
        m = tier == t
        meds = [np.median(think[a][m]) for a in ARMS]
        for k, (a, v) in enumerate(zip(ARMS, meds)):
            yy = ys[t] + 0.22 - 0.22 * k
            a2.plot([0, v], [yy, yy], color=ARM_COLOR[a], lw=1.2, alpha=0.35,
                    zorder=2)
            a2.plot([v], [yy], "o", ms=8, color=ARM_COLOR[a],
                    markeredgecolor=BG, markeredgewidth=1.5, zorder=3)
            a2.annotate(f"{v:.0f}", (v, yy), xytext=(9, 0),
                        textcoords="offset points", va="center", color=FG,
                        fontsize=10, fontfamily=MONO)
        tr = np.mean([trunc[a][m].mean() for a in ARMS])
        a2.annotate(f"n={m.sum()}  ·  {100 * tr:.0f}% truncated",
                    (0, ys[t]), xycoords=("axes fraction", "data"),
                    xytext=(0, -30), textcoords="offset points", color=MUTED,
                    fontsize=9.5, va="top")
    a2.set_yticks(ys)
    a2.set_yticklabels([name for name, _ in TIERS], color=FG, fontsize=11)
    a2.set_ylim(-0.9, len(TIERS) - 0.55)
    a2.set_xlim(0, 1500)
    a2.grid(axis="y", visible=False)
    a2.grid(axis="x", visible=True)
    a2.spines["bottom"].set_visible(False)
    a2.set_xlabel("median thinking tokens")
    _style(fig, a2, "per arm, within each tier")
    for lbl in a2.get_yticklabels():
        lbl.set_fontfamily(SANS)

    _header(fig, "Thinking length tracks prompt difficulty",
            "Tier = share of the prompt's 12 completions (3 arms × 4) answered correctly:"
            " easy ≥ 75%, medium 40–75%, hard < 40%", gap=0.35)
    _legend(fig, [(ARM_LABEL[a], ARM_COLOR[a]) for a in ARMS],
            y=1 - 0.4 / fig.get_figheight())
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.12)
    fig.savefig(os.path.join(outdir, "fig_difficulty.png"))
    plt.close(fig)


# ---- strength ablation helpers (used by fig_sweep / fig_strength_*) ---

ABL = "runs/analysis-local"
STRENGTHS = [0.125, 0.5, 0.75]
DS_ARMS = ["ds-weak", "deepseek", "ds-strong"]
CO_ARMS = ["co-weak", "composer", "co-strong"]


def load_greedy_all(path=f"{ABL}/gsm8k-greedy-all/analysis.json"):
    return load_summary(path)["arms"]


def strength_panel(ax, ds, co, base, frozen, fmt, ylabel):
    """deepseek and composer vs strength, baseline/frozen as reference lines."""
    _ref_line(ax, base, WHITE, "baseline")
    _ref_line(ax, frozen, FROZEN, "frozen base")
    for vals, c in ((ds, LIME), (co, PINK)):
        ax.plot(STRENGTHS, vals, color=c, lw=2.4, zorder=3)
        ax.plot(STRENGTHS, vals, "o", color=c, ms=8, markeredgecolor=BG,
                markeredgewidth=1.5, zorder=4)
    for x, d, k in zip(STRENGTHS, ds, co):
        pairs = [(d, k)] if abs(d - k) < 1e-9 else [(d, k), (k, d)]
        for v, other in pairs:
            up = v >= other
            ax.annotate(fmt.format(v), (x, v), xytext=(0, 10 if up else -10),
                        textcoords="offset points", ha="center",
                        va="bottom" if up else "top", color=FG, fontsize=10,
                        fontfamily=MONO, zorder=5)
    ax.set_xticks(STRENGTHS)
    ax.set_xticklabels(["0.125", "0.5", "0.75"])
    ax.set_xlim(0.05, 0.82)
    ax.set_xlabel("penalty strength")
    ax.set_ylabel(ylabel)
    ax.margins(y=0.18)


# ---------------------------------------------------------------- main

RUN = "runs/logs/length-qwen17-run-38c53964/remote-runs/runs/arms"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", nargs=3, metavar=("BASE", "DS", "COMPOSER"),
                   default=[f"{RUN}/{a}/reward_samples.jsonl" for a in ARMS],
                   help="reward_samples.jsonl paths, one per arm")
    p.add_argument("--eval", default="runs/analysis-local/gsm8k-greedy/analysis.json",
                   help="eval summary.json or analysis.json path")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--pass4-gsm8k", default="runs/analysis-local/gsm8k-pass4",
                   help="dir with {base,trained-base,deepseek,composer}.jsonl")
    p.add_argument("--pass4-math500", default="runs/analysis-local/math500-pass4",
                   help="dir with {base,trained-base,deepseek,composer}.jsonl")
    p.add_argument("--demo", action="store_true", help="use synthetic data")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    have_logs = all(os.path.exists(x) for x in args.logs)
    have_eval = os.path.exists(args.eval)

    if args.demo or not (have_logs and have_eval):
        if not args.demo:
            print("[plot] real data missing -> falling back to synthetic demo data", file=sys.stderr)
        logs, summary = make_demo_data()
    else:
        logs = {arm: load_reward_samples(path) for arm, path in zip(ARMS, args.logs)}
        summary = load_summary(args.eval)

    fig_thinking_roll(logs, args.outdir)
    fig_correct_roll(logs, args.outdir)
    fig_difficulty(logs, args.outdir)
    fig_penalty_shape(args.outdir)
    fig_eval_bars(summary, args.outdir)

    import tempfile
    pass4_dirs = {"gsm8k": args.pass4_gsm8k, "math500": args.pass4_math500}
    if args.demo or not all(
            all(os.path.exists(os.path.join(d, f"{n}.jsonl")) for n in PASS4_NAMES)
            for d in pass4_dirs.values()):
        if not args.demo:
            print("[plot] pass@4 data missing -> using synthetic demo data", file=sys.stderr)
        tmp = tempfile.mkdtemp(prefix="plot-pass4-")
        write_demo_pass4(tmp, make_demo_pass4())
        pass4_dirs = {"gsm8k": os.path.join(tmp, "gsm8k-pass4"),
                      "math500": os.path.join(tmp, "math500-pass4")}
    fig_pass4_bars(pass4_dirs["gsm8k"], "gsm8k", args.outdir)
    fig_pass4_bars(pass4_dirs["math500"], "math500", args.outdir)
    print(f"[plot] wrote 7 figures to {args.outdir}/")


if __name__ == "__main__":
    main()
