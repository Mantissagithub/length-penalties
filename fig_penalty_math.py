"""Penalty shapes as plain functions of x = L / L_norm (length in units of the anchor).

Capped linear, as in DeepSeek-V4.1-Flash (sec. 5.1.4, eq. 9-10):
    f_b(x) = min(C, k(b) x),  k(b) = k0 exp(-(b - b_min) / tau)
The report gives no values for k0 and tau; the ones below are illustrative.
With k0 = C = 1 the lowest effort b_min bends exactly at the anchor x = 1,
which is the setup trained in this repo (cap = strength = 0.5, fixed effort).
Concave (Composer) is matched to cost 1 at the anchor. Multiply by 0.5 for reward.
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import plot

Q = 1.5
C, K0, TAU, B_MIN = 1.0, 1.0, 25.0, 25
# effort level -> line colour, brightest = lowest effort = strongest penalty
EFFORTS = [(25, plot.LIME), (50, "#8fc71f"), (75, "#6d9a17"), (100, "#53760f")]

x = np.linspace(0, 8, 800)
g = lambda v: (1 - (1 + v) ** (1 - Q)) / (Q - 1)
concave = g(x) / g(1)
k = lambda b: K0 * np.exp(-(b - B_MIN) / TAU)

fig, ax = plot.new_fig()
ax.set_xmargin(0)
H_INF = 1 / (1 - 2 ** (1 - Q))
ax.axhline(H_INF, color=plot.PINK, lw=1.0, ls=(0, (4, 3)), alpha=0.45, zorder=1)
ax.axvline(1, color=plot.SPINE, lw=1.0, ls=(0, (4, 3)), zorder=1)

for b, c in reversed(EFFORTS):
    ax.plot(x, np.minimum(C, k(b) * x), color=c, lw=2.4, zorder=3)
    knee = C / k(b)
    if knee <= x[-1]:
        ax.plot([knee], [C], "o", color=c, ms=7, markeredgecolor=plot.BG,
                markeredgewidth=1.5, zorder=5)
        ax.annotate(f"b = {b}", (knee, C), xytext=(0, 10),
                    textcoords="offset points", ha="center", color=c,
                    fontsize=10.5)
    else:
        ax.annotate(f"b = {b}", (x[-1], k(b) * x[-1]), xytext=(-4, 10),
                    textcoords="offset points", ha="right", color=c,
                    fontsize=10.5)
ax.plot(x, concave, color=plot.PINK, lw=2.4, zorder=4)

ax.annotate(r"$f_b(x) = \min\left(C,\ k(b)\,x\right),\quad"
            r"k(b) = k_0\, e^{-(b - b_{\min})/\tau}$",
            (1.25, 3.0), ha="left", va="center", color=plot.LIME, fontsize=13.5)
ax.annotate("bend at $x = C/k(b)$: higher effort $b$ bends later",
            (1.25, 3.0), xytext=(0, -24), textcoords="offset points", ha="left",
            va="center", color=plot.MUTED, fontsize=10.5)
ax.annotate(r"$h(x) = \dfrac{1-(1+x)^{1-q}}{1-2^{1-q}},\ \ q = 1.5$",
            (5.2, concave[np.searchsorted(x, 5.2)]), xytext=(0, 18),
            textcoords="offset points", ha="center", color=plot.PINK,
            fontsize=14)
ax.annotate(r"limit as $x \to \infty$:  $\dfrac{1}{1-2^{1-q}} \approx 3.41$",
            (8, H_INF), xytext=(-4, 6), textcoords="offset points", ha="right",
            va="bottom", color=plot.MUTED, fontsize=10.5)
ax.annotate("anchor $x = 1$", (1, 3.8), xytext=(6, -6),
            textcoords="offset points", va="top", color=plot.MUTED, fontsize=10.5)

ax.set_xlim(0, 8)
ax.set_ylim(0, 3.8)
ax.set_xticks(range(9))
ax.set_xlabel(r"$x = L\,/\,L_{\mathrm{norm}}$   (thinking length in units of the anchor)")
ax.set_ylabel("length cost")
plot._style(fig, ax)
plot._header(fig, "Capped linear across effort levels vs concave",
             f"DeepSeek-V4.1-Flash penalty with illustrative k₀ = {K0:g}, "
             f"τ = {TAU:g}, C = {C:g} (not given in the report). "
             "b = 25 matches this repo's runs.")
fig.subplots_adjust(left=0.08, right=0.96, bottom=0.12)
fig.savefig("figures/fig_penalty_math.png")
print("wrote figures/fig_penalty_math.png")
