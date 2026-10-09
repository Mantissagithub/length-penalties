# sample a frozen model to choose a thinking-length anchor
import argparse
import json
import statistics
from pathlib import Path

from qwen_normal import MODELS, final_answer_ids, load_data, thinking_span


def main():
    p = argparse.ArgumentParser(description="sample a frozen model for a length anchor")
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--output", required=True, help="Path for JSONL records; summary uses .summary.json")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--dataset", choices=("gsm8k", "math-integer", "math-integer-hard", "math-integer-medium", "math500"), default="gsm8k")
    p.add_argument("--split", choices=("train", "test"), default="train")
    p.add_argument("--adapter", help="Trained PEFT adapter; leave unset for frozen base")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-completion-length", type=int, default=4096)
    p.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    p.add_argument("--batch-size", type=int, default=4, help="Transformers fallback only")
    args = p.parse_args()
    if min(args.samples, args.max_completion_length, args.batch_size) <= 0:
        p.error("samples, max-completion-length, and batch-size must be positive")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model])
    data = load_data(args.split, args.samples, args.seed, tokenizer, args.model, args.dataset)
    rows = list(data)
    prompts = [row["prompt"] for row in rows]
    max_prompt_tokens = max(len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts)
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        engine = LLM(model=MODELS[args.model], max_model_len=args.max_completion_length + max_prompt_tokens,
                     gpu_memory_utilization=0.8, max_num_seqs=8, enforce_eager=True,
                     enable_lora=bool(args.adapter))
        lora_request = None
        if args.adapter:
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest("comparison-adapter", 1, args.adapter)
        result = engine.generate(prompts, SamplingParams(n=1, temperature=0.8, top_p=1.0,
                               max_tokens=args.max_completion_length, seed=args.seed),
                               lora_request=lora_request)
        completions = [list(item.outputs[0].token_ids) for item in result]
    else:
        import torch
        from transformers import AutoConfig
        import transformers
        config = AutoConfig.from_pretrained(MODELS[args.model])
        architecture = getattr(transformers, config.architectures[0])
        engine = architecture.from_pretrained(MODELS[args.model], dtype="auto", device_map="auto")
        if args.adapter:
            from peft import PeftModel
            engine = PeftModel.from_pretrained(engine, args.adapter)
        engine.eval()
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        completions = []
        for i in range(0, len(prompts), args.batch_size):
            batch = tokenizer(prompts[i:i + args.batch_size], padding=True, return_tensors="pt")
            batch = {key: value.to(engine.device) for key, value in batch.items()}
            with torch.inference_mode():
                outputs = engine.generate(**batch, max_new_tokens=args.max_completion_length,
                                          do_sample=True, temperature=0.8, top_p=1.0,
                                          pad_token_id=tokenizer.pad_token_id)
            width = batch["input_ids"].shape[1]
            completions.extend([output[width:].tolist() for output in outputs])

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    correct_lengths = []
    records = []
    for row, ids in zip(rows, completions):
        length, final_start = thinking_span(ids, tokenizer, args.model, row["prompt"])
        answer = final_answer_ids(ids, tokenizer, args.model, row["prompt"])
        correct = answer == row["target"]
        if correct:
            correct_lengths.append(length)
        records.append({"prompt": row["prompt"], "target": row["target"],
                        "completion": tokenizer.decode(ids, skip_special_tokens=False),
                        "answer": answer, "correct": correct, "thinking_tokens": length,
                        "total_tokens": len(ids), "finished_thinking": final_start is not None})
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records))
    summary = {**vars(args), "model_id": MODELS[args.model], "count": len(records),
               "correct": len(correct_lengths), "finished": sum(r["finished_thinking"] for r in records),
               "truncated": sum(r["total_tokens"] >= args.max_completion_length for r in records),
               "median_correct_thinking": statistics.median(correct_lengths) if correct_lengths else None,
               "median_all_thinking": statistics.median(r["thinking_tokens"] for r in records)}
    path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not correct_lengths and args.split == "train" and not args.adapter:
        raise SystemExit("No correct samples: increase samples or token budget; do not calibrate on all outputs.")


if __name__ == "__main__":
    main()
