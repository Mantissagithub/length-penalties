import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plot

# strength order: weak, 0.5, strong; colors: deepseek limes, composer pinks
SHADE={"weak":0.55,"0.5":1.0,"strong":0.75}
def shade(hexc,f):
    hexc=hexc.lstrip('#'); r,g,b=[int(hexc[i:i+2],16) for i in (0,2,4)]
    return '#%02x%02x%02x'%(int(r*f),int(g*f),int(b*f))

DS=[("ds-weak",79.7,20.3,"weak"),("deepseek",90.6,20.3,"0.5"),("ds-strong",84.4,20.3,"strong")]
CO=[("co-weak",82.8,20.3,"weak"),("composer",87.5,25.0,"0.5"),("co-strong",81.3,20.3,"strong")]
DS_ACC=[63.3,63.3,64.8]; CO_ACC=[64.1,68.0,60.2]

def grouped(ax,ds_vals,co_vals,title,ylabel,fmt):
    x=np.arange(3); w=0.36
    for i,(v,st) in enumerate(ds_vals): ax.bar(x[i]-w/2,v,w,color=shade("#b2f827",SHADE[st]),zorder=3)
    for i,(v,st) in enumerate(co_vals): ax.bar(x[i]+w/2,v,w,color=shade("#f974c3",SHADE[st]),zorder=3)
    for i,(v,st) in enumerate(ds_vals): ax.annotate(fmt.format(v),(x[i]-w/2,v),textcoords="offset points",xytext=(0,6),ha="center",fontsize=10,color=plot.FG,fontfamily="monospace")
    for i,(v,st) in enumerate(co_vals): ax.annotate(fmt.format(v),(x[i]+w/2,v),textcoords="offset points",xytext=(0,6),ha="center",fontsize=10,color=plot.FG,fontfamily="monospace")
    ax.set_xticks(x); ax.set_xticklabels(["0.125","0.5","0.75"])
    plot._style(fig,ax,title); ax.set_ylabel(ylabel); ax.set_xlabel("penalty strength")
    ax.margins(y=0.15)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=plot.LIME,label="deepseek"),Patch(color=plot.PINK,label="composer")],loc="upper left",frameon=True,facecolor=plot.LEGEND_BG,edgecolor=plot.LEGEND_EDGE,labelcolor=plot.FG,fontsize=13)

fig,(a1,a2)=plt.subplots(1,2,figsize=(12,6.75),dpi=160)
grouped(a1,[(79.7,"weak"),(90.6,"0.5"),(84.4,"strong")],[(82.8,"weak"),(87.5,"0.5"),(81.3,"strong")],"gsm8k","pass@4 (%)","{:.1f}%")
grouped(a2,[(20.3,"weak"),(20.3,"0.5"),(20.3,"strong")],[(20.3,"weak"),(25.0,"0.5"),(20.3,"strong")],"math-500","pass@4 (%)","{:.1f}%")
fig.subplots_adjust(top=0.85)
fig.savefig("figures/fig_strength_pass4.png",bbox_inches="tight")

fig2,ax=plt.subplots(figsize=(12,6.75),dpi=160)
grouped(ax,[(63.3,"weak"),(63.3,"0.5"),(64.8,"strong")],[(64.1,"weak"),(68.0,"0.5"),(60.2,"strong")],"gsm8k pass@1 (greedy) accuracy","accuracy (%)","{:.1f}%")
ax.axhline(64.1,color="#bdbdbd",lw=2,ls=(0,(4,4))); ax.text(2.13,64.4,"baseline",color="#bdbdbd",fontsize=12,fontfamily="monospace")
ax.axhline(58.6,color="#666666",lw=2,ls=(0,(4,4))); ax.text(2.13,58.9,"frozen",color="#666666",fontsize=12,fontfamily="monospace")
fig2.subplots_adjust(top=0.85)
fig2.savefig("figures/fig_strength_pass1.png",bbox_inches="tight")
print("written")
