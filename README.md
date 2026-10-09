# thinking-token length penalties: capped-linear vs concave

GRPO comparison of two length-penalty shapes on `Qwen/Qwen3-1.7B` (GSM8K):
DeepSeek's **capped linear** penalty vs Composer's **concave** penalty.
One seed, 100 steps per arm — blog-scale, not paper-grade.

## setup

Three matched arms (same data order, seed, 4 completions per prompt,
4096-token cap, LoRA rank 16). Reward for a correct, finished answer:

- **baseline** (`qwen_normal.py`): $+2$
- **deepseek** (`qwen_length.py --penalty deepseek`): $2 - \min(0.5,\ 0.5\,L/512)$
- **composer** (`qwen_length.py --penalty composer`): $2 - 0.5\,g(L)/g(512)$,
  $g(L)=\frac{1-(1+L/512)^{1-q}}{(q-1)/512}$, $q=1.5$

Both cost $-0.5$ at the anchor $T=512$. Past the cap the linear penalty is
flat; the concave one keeps charging ($\approx -1.14$ at 4096, still below
the $+2$ reward). Wrong or unfinished answers get $0$. Training: TRL 1.15.0
GRPO + colocated vLLM on one A100 40 GB, ~9 s/step, ~15 min per arm.

## results

held-out gsm8k greedy (128 questions, temperature 0):

| arm | accuracy | total tokens | correct per 1000 tokens |
| --- | --- | --- | --- |
| frozen base | 54.7% | 167k | 0.42 |
| baseline | 64.1% | 160k | 0.51 |
| deepseek | 65.6% | 171k | 0.49 |
| **composer** | **65.6%** | **153k** | **0.55** |

pass@4 (64 questions, temperature 0.8):

| arm | gsm8k pass@4 | per-sample | math-500 pass@4 | per-sample |
| --- | --- | --- | --- | --- |
| frozen base | 82.8% | 54.3% | 17.2% | 7.8% |
| baseline | 79.7% | 62.9% | 23.4% | 8.6% |
| deepseek | 90.6% | 64.5% | 20.3% | 7.8% |
| composer | 87.5% | 63.3% | **25.0%** | 9.0% |

Both penalties matched baseline accuracy; only the concave one used fewer
tokens ($-10\%$ vs trained baseline; deepseek used more). Deltas between
penalty arms are a few questions wide at this sample size — see paired
bootstrap CIs in `runs/analysis-local/`.

## reproducing

```bash
# train (each arm ~15 min on one A100 40 GB)
accelerate launch qwen_normal.py --model qwen-small --use-vllm   --dataset gsm8k --steps 100 --max-completion-length 4096   --norm 512 --composer-k 0.001953125 --seed 42 --output runs/arms/base
# penalty arms: qwen_length.py plus --penalty deepseek --strength 0.5 --cap 0.5
#               (or --penalty composer), same other flags

python evaluate.py --model qwen-small   --adapter deepseek=runs/arms/ds --adapter composer=runs/arms/co   --dataset gsm8k --samples 128 --output runs/eval/gsm8k
python analyze.py --eval-dir runs/eval/gsm8k --arm deepseek --arm composer   --output runs/analysis.json
python plot.py --logs runs/arms/*/reward_samples.jsonl --outdir figures
```

Adapters on HF: `Pradheep1647/grpo-think-len-baseline`,
`Pradheep1647/grpo-think-len-capped-linear`,
`Pradheep1647/grpo-think-len-concave`. Verified run archive (logs, configs,
rewards, checkpoints, evals) is in `runs/logs/length-qwen17-run-38c53964/`;
`runs/` is gitignored.
