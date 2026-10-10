import matplotlib
matplotlib.use("Agg")
import numpy as np
import plot

LOGDIR = "runs/logs/length-qwen17-ablation-fd2b4145/remote-runs/runs/arms"
# per method: strengths 0.125 / 0.5 / 0.75 as a dim -> bright ramp of its hue
PANELS = [
    ("deepseek", [("qwen17-ds-weak", "0.125", "#5e8a10"),
                  ("qwen17-ds", "0.5", plot.LIME),
                  ("qwen17-ds-strong", "0.75", "#e6ffb3")]),
    ("composer", [("qwen17-co-weak", "0.125", "#a3477e"),
                  ("qwen17-composer", "0.5", plot.PINK),
                  ("qwen17-co-strong", "0.75", "#ffd1ec")]),
]


def series(arm, key):
    rows = plot.load_reward_samples(f"{LOGDIR}/{arm}/reward_samples.jsonl")
    return plot.smooth(plot.steps_means(rows, key))


def draw(key, title, ylabel, out, fmt, ylim=None):
    fig, axes = plot.new_fig(2, sharey=True, gridspec_kw={"wspace": 0.36})
    base = series("qwen17-base", key)
    for ax, (method, arms) in zip(axes, PANELS):
        ax.set_xmargin(0)
        lines = [("baseline", plot.WHITE, base)]
        lines += [(s, c, series(a, key)) for a, s, c in arms]
        for label, c, y in lines:
            dashed = label == "baseline"
            ax.plot(np.arange(len(y)), y, color=c, lw=1.4 if dashed else 2.2,
                    ls=(0, (4, 3)) if dashed else "-", zorder=2 if dashed else 3)
        if ylim:
            ax.set_ylim(*ylim)
        y0, y1 = ax.get_ylim()
        last = np.flatnonzero(~np.isnan(base))[-1]
        lys = plot.staggered([l[2][last] for l in lines], y0, y1, frac=0.055)
        for (label, c, y), ly in zip(lines, lys):
            plot._end_label(ax, last, y[last], c, fmt.format(y[last]), ly=ly,
                            name=label)
        ax.set_xlabel("training step")
        plot._style(fig, ax, method)
    axes[0].set_ylabel(ylabel)
    plot._header(fig, title, "11-step moving average; one line per penalty strength, baseline dashed",
                 gap=0.3)
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.12)
    fig.savefig(out)
    print("wrote", out)


draw("thinking_tokens", "Thinking length over training, strength ablation",
     "thinking tokens", "figures/fig_strength_thinking_roll.png", "{:.0f}")
draw("correct", "Correct rate over training, strength ablation", "correct rate",
     "figures/fig_strength_correct_roll.png", "{:.2f}", ylim=(0, 1))
