import matplotlib
matplotlib.use("Agg")
import plot

arms = plot.load_greedy_all()
acc = lambda a: 100 * arms[a]["accuracy"]
tok = lambda a: arms[a]["total_generated_tokens"] / 1000

fig, (a1, a2) = plot.new_fig(2, gridspec_kw={"wspace": 0.42})
plot.strength_panel(a1, [acc(a) for a in plot.DS_ARMS], [acc(a) for a in plot.CO_ARMS],
                    acc("trained-base"), acc("base"), "{:.1f}", "accuracy (%)")
plot.strength_panel(a2, [tok(a) for a in plot.DS_ARMS], [tok(a) for a in plot.CO_ARMS],
                    tok("trained-base"), tok("base"), "{:.1f}k", "total tokens (k)")
plot._style(fig, a1, "greedy accuracy")
plot._style(fig, a2, "tokens generated")
plot._header(fig, "Penalty strength sweep",
             "Greedy GSM8K, 128 questions.  Higher accuracy and fewer tokens are better.",
             gap=0.35)
plot._legend(fig, [("deepseek", plot.LIME), ("composer", plot.PINK)])
fig.subplots_adjust(left=0.07, right=0.9, bottom=0.12)
fig.savefig("figures/fig_strength_sweep.png")
print("written")
