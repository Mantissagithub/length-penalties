import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plot

LOGDIR="runs/logs/length-qwen17-ablation-fd2b4145/remote-runs/runs/arms"
ARMS=[("qwen17-base","baseline","#ffffff"),("qwen17-ds-weak","deepseek 0.125","#7fb81d"),
      ("qwen17-ds-strong","deepseek 0.75","#d7ff7a"),("qwen17-co-weak","composer 0.125","#c14a92"),
      ("qwen17-co-strong","composer 0.75","#ffb3de")]
BASELOG="runs/logs/length-qwen17-run-38c53964/remote-runs/runs/arms/qwen17-base/reward_samples.jsonl"

def draw(key,title,ylabel,out,fmt):
    fig,ax=plot.new_fig()
    ends=[]
    for arm,label,color in ARMS:
        path=BASELOG if arm=="qwen17-base" else f"{LOGDIR}/{arm}/reward_samples.jsonl"
        rows=plot.load_reward_samples(path); steps=plot.steps_means(rows,key)
        sm=plot.rolling(np.array(steps,dtype=float),11)
        xs=np.arange(len(steps))
        lo=np.array([np.percentile([float(r[key]) for r in rows[i*4:(i+1)*4]],25) for i in range(len(steps))])
        hi=np.array([np.percentile([float(r[key]) for r in rows[i*4:(i+1)*4]],75) for i in range(len(steps))])
        ax.fill_between(xs,plot.rolling(lo,11),plot.rolling(hi,11),color=color,alpha=0.10,lw=0)
        ax.plot(xs,sm,color=color,lw=3.2,label=label,zorder=3)
        ends.append((sm[-1],color,label))
    for i,(y,color,label) in enumerate(plot.staggered([e[0] for e in ends],ax.get_ylim()[0],ax.get_ylim()[1]) if False else []): pass
    ys=[e[0] for e in ends]; ylo,yhi=min(ys),max(ys)
    labs=plot.staggered(ys,ax.get_ylim()[0],ax.get_ylim()[1]) if hasattr(plot,"staggered") else None
    plot._style(fig,ax,title); ax.set_ylabel(ylabel); ax.set_xlabel("training step")
    plot._legend(ax,loc="lower right")
    fig.savefig(out,bbox_inches="tight"); print("wrote",out)

draw("thinking_tokens","thinking length over training (strength ablation)","thinking tokens","figures/fig_strength_thinking_roll.png","{:.0f}")
draw("correct","correct rate over training (strength ablation)","correct rate","figures/fig_strength_correct_roll.png","{:.2f}")
