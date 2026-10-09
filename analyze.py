import argparse
import json
import math
import random
import statistics
from pathlib import Path


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def quantile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    rank = (len(values) - 1) * fraction
    lo = math.floor(rank)
    return values[lo] + (values[math.ceil(rank)] - values[lo]) * (rank - lo)


def metrics(rows):
    n = len(rows)
    correct = [r for r in rows if r["correct"]]
    tokens = sum(r["total_tokens"] for r in rows)
    return {"n": n, "accuracy": sum(bool(r["correct"]) for r in rows) / n,
            "finished_rate": sum(bool(r["finished"]) for r in rows) / n,
            "truncated_rate": sum(bool(r["truncated"]) for r in rows) / n,
            "mean_total_tokens": tokens / n,
            "total_generated_tokens": tokens,
            "correct_per_1000_tokens": 1000 * len(correct) / tokens if tokens else None,
            "median_thinking_all": quantile([r["thinking_tokens"] for r in rows], .5),
            "p90_thinking_all": quantile([r["thinking_tokens"] for r in rows], .9),
            "median_thinking_correct": quantile([r["thinking_tokens"] for r in correct], .5),
            "p90_thinking_correct": quantile([r["thinking_tokens"] for r in correct], .9)}


DELTA_KEYS = ("accuracy", "finished_rate", "truncated_rate", "mean_total_tokens",
              "correct_per_1000_tokens", "median_thinking_all", "p90_thinking_all",
              "median_thinking_correct", "p90_thinking_correct")


def paired_comparison(reference, arm, draws=2000, seed=42):
    if len(reference) != len(arm) or not reference:
        raise ValueError("paired arms must have equal nonzero row counts")
    for i, (a, b) in enumerate(zip(reference, arm)):
        if (a["prompt"], a["target"]) != (b["prompt"], b["target"]):
            raise ValueError(f"row {i} has a different prompt or target; cannot pair")
    n = len(reference)
    rng = random.Random(seed)
    base = metrics(reference)
    candidate = metrics(arm)
    observed = {key: (candidate[key] - base[key] if candidate[key] is not None
                       and base[key] is not None else None) for key in DELTA_KEYS}
    samples = {key: [] for key in DELTA_KEYS}
    for _ in range(draws):
        indices = [rng.randrange(n) for _ in range(n)]
        left = metrics([reference[i] for i in indices])
        right = metrics([arm[i] for i in indices])
        for key in DELTA_KEYS:
            if left[key] is not None and right[key] is not None:
                samples[key].append(right[key] - left[key])
    return {"reference": base, "arm": candidate, "delta_arm_minus_reference": observed,
            "delta_95pct_paired_bootstrap_ci": {
                key: [quantile(vals, .025), quantile(vals, .975)] for key, vals in samples.items()},
            "bootstrap_draws": draws, "bootstrap_seed": seed}


def length_curve(rows, edges):
    # bins describe each arm's observed outputs, not paired effects
    bins = [[] for _ in range(len(edges) + 1)]
    for row in rows:
        index = next((i for i, edge in enumerate(edges) if row["thinking_tokens"] < edge), len(edges))
        bins[index].append(row)
    labels = [f"<{edges[0]}"] + [f"[{a},{b})" for a, b in zip(edges, edges[1:])] + [f">={edges[-1]}"]
    return [{"thinking_tokens": label, "n": len(bucket),
             "accuracy": sum(bool(r["correct"]) for r in bucket) / len(bucket) if bucket else None,
             "truncated_rate": sum(bool(r["truncated"]) for r in bucket) / len(bucket) if bucket else None,
             "finished_rate": sum(bool(r["finished"]) for r in bucket) / len(bucket) if bucket else None}
            for label, bucket in zip(labels, bins)]


def training_diagnostics(rows, block_size):
    def summarize(chunk):
        n = len(chunk)
        return {"n": n, "correct_rate": sum(bool(r["correct"]) for r in chunk) / n,
                "truncated_rate": sum(bool(r["truncated"]) for r in chunk) / n,
                "finished_rate": sum(bool(r["finished_thinking"]) for r in chunk) / n,
                "mean_thinking_tokens": statistics.mean(r["thinking_tokens"] for r in chunk),
                "mean_completion_tokens": statistics.mean(r["completion_tokens"] for r in chunk),
                "mean_penalty": statistics.mean(r["penalty"] for r in chunk),
                "mean_reward": statistics.mean(r["reward"] for r in chunk),
                "nonzero_penalty_rate": sum(r["penalty"] != 0 for r in chunk) / n}
    if not rows:
        raise ValueError("empty training reward log")
    return {"overall": summarize(rows), "blocks": [
        {"row_start": i, "row_end_exclusive": min(i + block_size, len(rows)),
         **summarize(rows[i:i + block_size])} for i in range(0, len(rows), block_size)],
        "note": "Training samples are on-policy and unpaired; blocks are log order, not optimizer steps."}


def main():
    parser = argparse.ArgumentParser(description="analyze paired held-out evaluations and training rewards")
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--reference", default="base")
    parser.add_argument("--arm", action="append", required=True, help="Name of an evaluation JSONL arm; repeat")
    parser.add_argument("--train", action="append", default=[], metavar="NAME=DIR",
                        help="Optional training directory containing reward_samples.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-size", type=int, default=100)
    parser.add_argument("--length-edges", default="256,512,1024,2048,4096,8192")
    args = parser.parse_args()
    if args.bootstrap < 1 or args.block_size < 1:
        parser.error("bootstrap and block-size must be positive")
    try:
        edges = [int(x) for x in args.length_edges.split(",")]
        if not edges or edges != sorted(set(edges)) or edges[0] <= 0:
            parser.error("length-edges must be increasing positive integers")
        names = [args.reference, *args.arm]
        if len(set(names)) != len(names) or any(not n or Path(n).name != n or n in (".", "..") for n in names):
            parser.error("arm names must be unique plain file names")
        records = {name: read_jsonl(args.eval_dir / f"{name}.jsonl") for name in names}
        output = {"evaluation": str(args.eval_dir), "reference": args.reference,
                  "comparison": {name: paired_comparison(records[args.reference], records[name],
                                       args.bootstrap, args.seed) for name in args.arm},
                  "length_curves": {name: length_curve(rows, edges) for name, rows in records.items()},
                  "training_diagnostics": {},
                  "interpretation": ["The CI resamples matched test questions; it does not capture training-seed uncertainty.",
                                     "Correct-per-1000-token is an aggregate ratio with paired resampling, not fixed-budget accuracy.",
                                     "Length bins condition on each arm's output, so they are descriptive, not causal.",
                                     "Truncation is measured at the common generation cap; censored lengths must not be interpreted as short reasoning."]}
        for spec in args.train:
            name, sep, directory = spec.partition("=")
            if not sep or not name or name in output["training_diagnostics"]:
                parser.error(f"invalid or duplicate --train NAME=DIR: {spec}")
            path = Path(directory)
            result = training_diagnostics(read_jsonl(path / "reward_samples.jsonl"), args.block_size)
            config = path / "experiment.json"
            if config.exists():
                result["experiment"] = json.loads(config.read_text())
            output["training_diagnostics"][name] = result
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print(f"Saved {args.output}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
