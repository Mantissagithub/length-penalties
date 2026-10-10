import os

import matplotlib
matplotlib.use("Agg")
import plot

# pass@1: greedy accuracy by strength
arms = plot.load_greedy_all()
acc = lambda a: 100 * arms[a]["accuracy"]
fig, ax = plot.new_fig(w=8, h=6.4)
plot.strength_panel(ax, [acc(a) for a in plot.DS_ARMS], [acc(a) for a in plot.CO_ARMS],
                    acc("trained-base"), acc("base"), "{:.1f}", "accuracy (%)")
plot._style(fig, ax)
plot._header(fig, "GSM8K pass@1 by strength", "Greedy decoding, 128 questions",
             gap=0.2)
plot._legend(fig, [("deepseek", plot.LIME), ("composer", plot.PINK)], y=0.86)
fig.subplots_adjust(left=0.11, right=0.84, bottom=0.12)
fig.savefig("figures/fig_strength_pass1.png")

# pass@4 by strength, gsm8k and math-500
fig, axes = plot.new_fig(2, gridspec_kw={"wspace": 0.42})
for ax, bench, title in zip(axes, ("gsm8k", "math500"), ("GSM8K", "MATH-500")):
    p4 = lambda a: 100 * plot.load_pass4(
        os.path.join(plot.ABL, f"{bench}-pass4-all", f"{a}.jsonl"))[0]
    plot.strength_panel(ax, [p4(a) for a in plot.DS_ARMS], [p4(a) for a in plot.CO_ARMS],
                        p4("trained-base"), p4("base"), "{:.1f}", "pass@4 (%)")
    plot._style(fig, ax, title)
plot._header(fig, "pass@4 by strength",
             "64 questions, 4 samples each at temperature 0.8", gap=0.35)
plot._legend(fig, [("deepseek", plot.LIME), ("composer", plot.PINK)])
fig.subplots_adjust(left=0.07, right=0.9, bottom=0.12)
fig.savefig("figures/fig_strength_pass4.png")
print("written")
