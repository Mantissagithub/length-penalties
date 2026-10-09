# evaluate the base model and adapters on the same held-out questions
import argparse
import json
import statistics
from pathlib import Path

from qwen_normal import MODELS, final_answer_ids, load_data, thinking_span


def median(values):
    return statistics.median(values) if values else None


def percentile90(values):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))]


def main():
    p = argparse.ArgumentParser(description="evaluate the base model and adapters on held-out questions")
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--output", required=True, help="Directory for per-arm records and summary")
    p.add_argument("--adapter", action="append", default=[], metavar="NAME=PATH",
                   help="Repeat for each named PEFT adapter to compare; frozen base is automatic")
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--dataset", choices=("gsm8k", "math-integer", "math-integer-hard", "math-integer-medium", "math500"), default="gsm8k")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-completion-length", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--n", type=int, default=1, help="Completions per question; enables pass@n")
    p.add_argument("--skip-base", action="store_true", help="Skip the frozen base arm (already evaluated)")
    args = p.parse_args()
    adapters = []
    names = {"base"}
    for item in args.adapter:
        name, sep, path = item.partition("=")
        if not sep or not name or name in names or "/" in name or not Path(path).is_dir():
            p.error(f"adapter must be a unique NAME=existing-path (not 'base'): {item}")
        adapters.append((name, str(Path(path).resolve())))
        names.add(name)
    if args.samples <= 0 or args.max_completion_length <= 0:
        p.error("samples and max-completion-length must be positive")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model])
    data = list(load_data("test", args.samples, args.seed, tokenizer, args.model, args.dataset))
    prompts = [row["prompt"] for row in data]
    max_prompt_tokens = max(len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts)
    engine = LLM(model=MODELS[args.model], max_model_len=args.max_completion_length + max_prompt_tokens,
                 gpu_memory_utilization=0.8, max_num_seqs=64,
                 enable_lora=bool(adapters))
    sampling = SamplingParams(n=args.n, temperature=args.temperature, top_p=1.0,
                              max_tokens=args.max_completion_length, seed=args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"model": args.model, "model_id": MODELS[args.model], "samples": len(data),
               "n_per_prompt": args.n, "split": args.dataset + "/test", "seed": args.seed,
               "temperature": args.temperature, "max_completion_length": args.max_completion_length,
               "arms": {}}
    arm_list = adapters if args.skip_base else [("base", None), *adapters]
    for index, (name, adapter) in enumerate(arm_list):
        request = LoRARequest(name, index + 1, adapter) if adapter else None
        results = engine.generate(prompts, sampling, lora_request=request)
        records = []
        for row, result in zip(data, results):
            for sample_index, generated in enumerate(result.outputs):
                ids = list(generated.token_ids)
                length, final_start = thinking_span(ids, tokenizer, args.model, row["prompt"])
                parsed = final_answer_ids(ids, tokenizer, args.model, row["prompt"])
                records.append({"target": row["target"], "answer": parsed,
                                "correct": parsed == row["target"], "finished": final_start is not None,
                                "thinking_tokens": length, "total_tokens": len(ids),
                                "truncated": len(ids) >= args.max_completion_length,
                                "sample": sample_index, "prompt": row["prompt"],
                                "completion": tokenizer.decode(ids, skip_special_tokens=False)})
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        good = [r for r in records if r["correct"]]
        total = sum(r["total_tokens"] for r in records)
        per_question = {}
        for r in records:
            per_question.setdefault(r["prompt"], []).append(r["correct"])
        pass_at_n = sum(any(flags) for flags in per_question.values()) / len(per_question)
        summary["arms"][name] = {"adapter": adapter, "pass_at_n": pass_at_n, "correct": len(good),
            "accuracy": len(good) / len(records), "finished": sum(r["finished"] for r in records),
            "truncated": sum(r["truncated"] for r in records),
            "median_thinking_all": median([r["thinking_tokens"] for r in records]),
            "p90_thinking_all": percentile90([r["thinking_tokens"] for r in records]),
            "median_thinking_correct": median([r["thinking_tokens"] for r in good]),
            "p90_thinking_correct": percentile90([r["thinking_tokens"] for r in good]),
            "total_generated_tokens": total,
            "correct_per_1000_tokens": 1000 * len(good) / total if total else 0}
        print(name, json.dumps(summary["arms"][name]), flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
