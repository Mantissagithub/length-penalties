#!/usr/bin/env python3
"""Figure generator for the GRPO length-penalty study.

Figures (written to --outdir, default figures/):
  fig_thinking_roll.png   rolling mean thinking_tokens per arm over steps
  fig_correct_roll.png    rolling correct rate per arm over steps
  fig_penalty_shape.png   actual penalty curves from qwen_normal.py
  fig_eval_bars.png       greedy GSM8K eval bars: accuracy + correct-per-1000-tokens

Style (matches figures/theme.png): black background, bold white lowercase
title/labels, dashed subtle grid, faint raw traces + bold smoothed lines with
end dots and monospace numeric end labels (staggered when they overlap),
rounded legend box.
Colors: baseline white, deepseek lime #b2f827, composer pink #f974c3.

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

BG = "#000000"
FG = "#ffffff"
WHITE = "#ffffff"
LIME = "#b2f827"
PINK = "#f974c3"
GRID = "#444444"
SPINE = "#555555"
LEGEND_BG = "#0d0d0d"
LEGEND_EDGE = "#666666"

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

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.edgecolor": SPINE,
    "axes.labelcolor": FG,
    "axes.labelweight": "bold",
    "axes.labelsize": 15,
    "text.color": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "font.weight": "bold",
    "font.size": 14,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.9,
    "grid.alpha": 0.55,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "font.family": "DejaVu Sans",
})


def _style(fig, ax, title):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_title(title.lower(), fontsize=26, fontweight="bold", color=FG, pad=16)
    ax.grid(True, linestyle=(0, (4, 4)))
    ax.tick_params(length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("monospace")
        lbl.set_fontsize(12)


def _legend(ax, **kwargs):
    leg = ax.legend(frameon=True, fontsize=13, labelcolor=FG,
                    facecolor=LEGEND_BG, edgecolor=LEGEND_EDGE,
                    fancybox=True, framealpha=1.0, borderpad=0.9,
                    labelspacing=0.7, **kwargs)
    leg.get_frame().set_boxstyle("round", pad=0.4, rounding_size=0.25)
    leg.get_frame().set_linewidth(1.2)
    for line in leg.get_lines():
        line.set_linewidth(4.5)
    return leg


def _end_label(ax, x, y, color, text, ly=None):
    """Dot at the true endpoint (x, y); label at ly (staggered) if given."""
    ax.plot([x], [y], "o", color=color, markersize=9,
            markeredgecolor=BG, markeredgewidth=1.0, zorder=5)
    if ly is None:
        ly = y
    elif abs(ly - y) > 1e-12:
        ax.plot([x, x], [y, ly], color=color, lw=1.0, alpha=0.5, zorder=4)
    ax.annotate(text, (x, ly), xytext=(10, 0), textcoords="offset points",
                color=color, fontsize=14, fontfamily="monospace",
                fontweight="bold", va="center", zorder=5)


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


def new_fig():
    fig, ax = plt.subplots(figsize=(12.07, 9.05), dpi=160)  # ~1930x1450
    return fig, ax


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
    with open(path) as f:
        return json.load(f)


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

def _roll_fig(logs, key, title, ylabel, outpath, ylim=None, fmt="{:.0f}"):
    fig, ax = new_fig()
    n = None
    ends = []
    for arm in ARMS:
        vals = [float(r[key]) for r in logs[arm]]
        n = len(vals) // 4
        arr = np.array(vals[: n * 4]).reshape(n, 4)
        lo = rolling(np.percentile(arr, 25, axis=1), w=11)
        hi = rolling(np.percentile(arr, 75, axis=1), w=11)
        m = arr.mean(axis=1)
        sm = rolling(m)
        ax.fill_between(np.arange(n), lo, hi, color=ARM_COLOR[arm],
                        alpha=0.12, linewidth=0, zorder=2)
        ax.plot(sm, color=ARM_COLOR[arm], lw=3.6, label=ARM_LABEL[arm],
                solid_capstyle="round", zorder=3)
        ends.append((arm, sm[~np.isnan(sm)][-1]))
    if ylim:
        ax.set_ylim(*ylim)
    y0, y1 = ax.get_ylim()
    lys = staggered([e[1] for e in ends], y0, y1)
    for (arm, last), ly in zip(ends, lys):
        _end_label(ax, n - 1 - 4, last, ARM_COLOR[arm], fmt.format(last), ly=ly)
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, n - 1)
    ax.margins(x=0.02)
    if ylim:
        ax.set_ylim(*ylim)
    _legend(ax, loc="lower right")
    _style(fig, ax, title)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def fig_thinking_roll(logs, outdir):
    _roll_fig(logs, "thinking_tokens", "thinking length over training",
              "thinking tokens", os.path.join(outdir, "fig_thinking_roll.png"),
              fmt="{:.0f}")


def fig_correct_roll(logs, outdir):
    _roll_fig(logs, "correct", "correct rate over training",
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
    ax.plot(L, ds, color=LIME, lw=3.6, label="deepseek (linear, capped)",
            solid_capstyle="round")
    ax.plot(L, co, color=PINK, lw=3.6, label="composer (concave)",
            solid_capstyle="round")
    ax.axvline(T, color=FG, lw=1.2, ls=(0, (4, 4)), alpha=0.5)
    ax.plot([T], [-0.5], "o", color=FG, markersize=9,
            markeredgecolor=BG, markeredgewidth=1.0, zorder=5)
    ax.text(T + 90, -0.44, f"anchor t = {T}", color=FG, fontsize=13,
            fontfamily="monospace", fontweight="bold", va="center", zorder=5)
    _end_label(ax, L[-1], ds[-1], LIME, f"{ds[-1]:.2f}")
    _end_label(ax, L[-1], co[-1], PINK, f"{co[-1]:.2f}")
    ax.set_xlim(0, 4096)
    ax.set_ylim(min(co) * 1.12, 0.05)
    ax.set_xlabel("generated length (tokens)")
    ax.set_ylabel("penalty")
    _legend(ax, loc="upper right")
    _style(fig, ax, "length penalty shapes")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_penalty_shape.png"))
    plt.close(fig)


def fig_eval_bars(summary, outdir):
    arms = summary.get("arms", summary)
    names = list(arms.keys())
    acc = [100 * float(arms[n]["accuracy"]) for n in names]
    eff = [float(arms[n]["correct_per_1000_tokens"]) for n in names]
    colors = [WHITE, "#bdbdbd", LIME, PINK][: len(names)]

    fig, axes = plt.subplots(1, 2, figsize=(12.07, 9.05), dpi=160)
    for ax, vals, title, unit, fstr in (
        (axes[0], acc, "gsm8k accuracy", "%", "{:.1f}"),
        (axes[1], eff, "correct per 1000 tokens", "", "{:.2f}"),
    ):
        bars = ax.bar(names, vals, color=colors, width=0.6, edgecolor="none")
        ax.set_ylim(0, max(vals) * 1.22)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * 0.03,
                    fstr.format(v) + unit, ha="center", va="bottom",
                    color=FG, fontsize=14, fontfamily="monospace",
                    fontweight="bold")
        _style(fig, ax, title)
        ax.grid(axis="x", visible=False)
    fig.suptitle("greedy gsm8k evaluation", fontsize=26, fontweight="bold",
                 color=FG, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
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
PASS4_COLORS = {"base": WHITE, "trained-base": "#bdbdbd",
                "deepseek": LIME, "composer": PINK}


def fig_pass4_bars(datadir, name, outdir):
    names = PASS4_NAMES
    p4, acc = [], []
    for n_ in names:
        p, a = load_pass4(os.path.join(datadir, f"{n_}.jsonl"))
        p4.append(100 * p)
        acc.append(100 * a)

    fig, ax = new_fig()
    x = np.arange(len(names))
    w = 0.34
    for i, nm in enumerate(names):
        c = PASS4_COLORS[nm]
        b1 = ax.bar(x[i] - w / 2 - 0.02, p4[i], w, color=c, edgecolor="none")[0]
        b2 = ax.bar(x[i] + w / 2 + 0.02, acc[i], w, color=c, edgecolor="none",
                    alpha=0.45, hatch="///")[0]
        for b, v in ((b1, p4[i]), (b2, acc[i])):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                    f"{v:.1f}%", ha="center", va="bottom", color=FG,
                    fontsize=13, fontfamily="monospace", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    top = max(max(p4), max(acc))
    ax.set_ylim(0, min(100, top * 1.25))
    ax.set_ylabel("percent of questions")
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color="#dddddd", label="pass@4 (any of 4 correct)"),
               mpatches.Patch(color="#dddddd", alpha=0.45, hatch="///",
                              label="per-sample accuracy")]
    leg = fig.legend(handles=handles, frameon=True, fontsize=13, labelcolor=FG,
                     facecolor=LEGEND_BG, edgecolor=LEGEND_EDGE, fancybox=True,
                     framealpha=1.0, borderpad=0.9, labelspacing=0.7,
                     loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.005))
    leg.get_frame().set_boxstyle("round", pad=0.4, rounding_size=0.25)
    leg.get_frame().set_linewidth(1.2)
    _style(fig, ax, f"{name} pass@4 vs per-sample accuracy")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
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


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", nargs=3, metavar=("BASE", "DS", "COMPOSER"),
                   default=[f"runs/arms/{a}/reward_samples.jsonl" for a in ARMS],
                   help="reward_samples.jsonl paths, one per arm")
    p.add_argument("--eval", default="runs/eval/gsm8k/summary.json",
                   help="eval summary.json path")
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
    print(f"[plot] wrote 6 figures to {args.outdir}/")


if __name__ == "__main__":
    main()
