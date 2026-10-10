import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plot

STRENGTHS=[0.125,0.5,0.75]
ds_acc=[63.3,63.3,64.8]; co_acc=[64.1,68.0,60.2]
ds_tok=[162.2,153.4,160.5]; co_tok=[157.1,157.8,168.4]
BASE_ACC=64.1; BASE_TOK=161.1; FROZEN_ACC=58.6; FROZEN_TOK=168.9

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,6.75),dpi=160)
for ax,title,ylabel,ds,co,base,frozen,fmt in [
    (ax1,"greedy accuracy","accuracy (%)",ds_acc,co_acc,BASE_ACC,FROZEN_ACC,"{:.1f}%"),
    (ax2,"tokens generated","total tokens (k)",ds_tok,co_tok,BASE_TOK,FROZEN_TOK,"{:.1f}k")]:
    ax.plot(STRENGTHS,ds,color=plot.LIME,lw=3.5,marker="o",ms=9,label="deepseek",zorder=3)
    ax.plot(STRENGTHS,co,color=plot.PINK,lw=3.5,marker="o",ms=9,label="composer",zorder=3)
    ax.axhline(base,color="#bdbdbd",lw=2,ls=(0,(4,4)),alpha=0.9)
    ax.axhline(frozen,color="#666666",lw=2,ls=(0,(4,4)),alpha=0.9)
    ax.text(0.4 if ylabel.startswith("acc") else 0.55,base,"baseline",color="#bdbdbd",fontsize=12,va="bottom" if ylabel.startswith("acc") else "top",fontfamily="monospace")
    ax.text(0.35 if ylabel.startswith("tok") else 0.128,frozen,"frozen",color="#666666",fontsize=12,va="top" if ylabel.startswith("acc") else "bottom",fontfamily="monospace")
    for x,y,c in zip(STRENGTHS,ds,[plot.LIME]*3): ax.annotate(fmt.format(y),(x,y),textcoords="offset points",xytext=(0,11 if ylabel.startswith("tok") else -20),color=c,fontsize=11,ha="center",fontfamily="monospace")
    for x,y,c in zip(STRENGTHS,co,[plot.PINK]*3): ax.annotate(fmt.format(y),(x,y),textcoords="offset points",xytext=(0,11 if not ylabel.startswith("tok") else -20),color=c,fontsize=11,ha="center",fontfamily="monospace")
    plot._style(fig,ax,title); ax.set_ylabel(ylabel); ax.set_xlabel("penalty strength")
    ax.set_xticks(STRENGTHS); ax.set_xticklabels(["0.125","0.5","0.75"])
    ax.margins(x=0.12)
plot._legend(ax2,loc="upper left")
fig.subplots_adjust(top=0.85)
fig.savefig("figures/fig_strength_sweep.png",bbox_inches="tight")
print("written")
