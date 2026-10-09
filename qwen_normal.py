# shared data, rewards, and grpo training for both models
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

MODELS = {
    "qwen": "Qwen/Qwen3-4B-Thinking-2507",
    "qwen-small": "Qwen/Qwen3-1.7B",
}
MARKERS = {
    "qwen": ("<think>", "</think>"),
            "qwen-small": ("<think>", "</" + "think" + ">"),
}
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
MATH_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
ANSWER_RE = re.compile(r"####\s*(-?\d[\d,]*)")
# accept bracketed numbers from older prompts too
FINAL_RE = re.compile(r"####[ \t]*(?:<integer>[ \t]*)?<?[ \t]*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+))[ \t]*>?")


def reference_answer(text):
    matches = ANSWER_RE.findall(text)
    if not matches:
        raise ValueError(f"GSM8K answer without #### integer: {text!r}")
    return int(matches[-1].replace(",", ""))


def marker_ids(marker, tokenizer):
    # encode can drop a closing tag that the tokenizer trie eats whole
    seq = tokenizer.encode(marker, add_special_tokens=False)
    if seq:
        return seq
    specials = {tokenizer.decode([i]): i for i in tokenizer.all_special_ids}
    return [specials[marker]] if marker in specials else []


def thinking_span(ids, tokenizer, model, prompt=""):
    # trl strips special tokens from text, so locate the channel in token ids
    ids = list(ids)
    opening, closing = MARKERS[model]
    start_seq = marker_ids(opening, tokenizer)
    end_seq = marker_ids(closing, tokenizer)

    def find(sequence, start=0):
        return next((i for i in range(start, len(ids) - len(sequence) + 1)
                     if ids[i:i + len(sequence)] == sequence), -1)

    start_at = find(start_seq)
    if start_at >= 0:
        start_at += len(start_seq)
    elif model == "qwen" and opening in prompt:
        start_at = 0
    else:
        # charge all generated tokens when the opening marker is missing
        start_at = 0
    stop_at = find(end_seq, start_at)
    if stop_at < 0:
        return max(0, len(ids) - start_at), None
    return max(0, stop_at - start_at), stop_at + len(end_seq)


def thinking_tokens(ids, tokenizer, model, prompt=""):
    return thinking_span(ids, tokenizer, model, prompt)[0]


def final_answer_ids(ids, tokenizer, model, prompt=""):
    _, answer_start = thinking_span(ids, tokenizer, model, prompt)
    if answer_start is None:
        return None
    final_text = tokenizer.decode(list(ids)[answer_start:], skip_special_tokens=True)
    # only the last nonblank line can hold the final integer answer
    lines = final_text.strip().splitlines()
    match = FINAL_RE.fullmatch(lines[-1].strip()) if lines else None
    return int(match.group(1).replace(",", "")) if match else None


def deepseek_penalty(length, norm, cap, strength, effort):
    return -min(cap, strength * effort * length / norm)


def composer_raw(length, k, q):
    # use the log limit when q is one
    if k <= 0 or q < 1:
        raise ValueError("composer k must be > 0 and q must be >= 1")
    if q == 1:
        return math.log1p(k * length) / k
    return math.expm1((1 - q) * math.log1p(k * length)) / (k * (1 - q))


def length_penalty(length, kind, norm, cap, strength, effort, k, q):
    if kind == "deepseek":
        return deepseek_penalty(length, norm, cap, strength, effort)
    # match both costs at norm without changing the composer curve
    anchor = min(cap, strength * effort)
    return -anchor * composer_raw(length, k, q) / composer_raw(norm, k, q)


def build_prompt(question, tokenizer, model):
    messages = [{"role": "user", "content": (
        "Solve the math problem. Think through it, then put the integer answer "
        "on its own final line as, for example, #### 42. Do not use angle brackets.\n\n" + question)}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    return tokenizer.apply_chat_template(messages, **kwargs)


def load_gsm8k(split, count, seed, tokenizer, model):
    from datasets import load_dataset
    data = load_dataset("openai/gsm8k", "main", split=split, revision=GSM8K_REVISION)
    data = data.shuffle(seed=seed).select(range(min(count, len(data))))
    return data.map(lambda row: {"prompt": build_prompt(row["question"], tokenizer, model),
                                 "target": reference_answer(row["answer"])},
                    remove_columns=data.column_names)


MATH_CONFIGS = (
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
)
INTEGER_RE = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)")


def boxed_integer(solution):
    # keep only a plain integer from the last balanced box
    marker = r"\boxed{"
    index = solution.rfind(marker)
    if index < 0:
        return None
    start = index + len(marker)
    depth = 1
    for at in range(start, len(solution)):
        if solution[at] == "{":
            depth += 1
        elif solution[at] == "}":
            depth -= 1
            if depth == 0:
                text = solution[start:at].strip()
                return int(text.replace(",", "")) if INTEGER_RE.fullmatch(text) else None
    return None


def load_math_integer(split, count, seed, tokenizer, model, min_level=1, max_level=5):
    from datasets import concatenate_datasets, load_dataset
    parts = [load_dataset("EleutherAI/hendrycks_math", name, split=split, revision=MATH_REVISION)
             for name in MATH_CONFIGS]
    data = concatenate_datasets(parts)
    # filter first so every arm shuffles the same eligible questions
    data = data.map(lambda row: {"target": boxed_integer(row["solution"])})
    data = data.filter(lambda row: row["target"] is not None and
                       (row["level"] in {f"Level {n}" for n in range(min_level, max_level + 1)}))
    if len(data) < count:
        raise ValueError(f"Only {len(data)} boxed-integer MATH {split} samples, requested {count}")
    data = data.shuffle(seed=seed).select(range(count))
    return data.map(lambda row: {"prompt": build_prompt(row["problem"], tokenizer, model),
                                 "target": row["target"]}, remove_columns=data.column_names)


MATH500_REVISION = None


def load_math500(count, seed, tokenizer, model):
    from datasets import load_dataset
    data = load_dataset("HuggingFaceH4/MATH-500", split="test")
    data = data.map(lambda row: {"target": boxed_integer(row["solution"])})
    data = data.filter(lambda row: row["target"] is not None)
    if len(data) < count:
        raise ValueError(f"Only {len(data)} boxed-integer MATH-500 samples, requested {count}")
    data = data.shuffle(seed=seed).select(range(count))
    return data.map(lambda row: {"prompt": build_prompt(row["problem"], tokenizer, model),
                                 "target": row["target"]}, remove_columns=data.column_names)


def load_data(split, count, seed, tokenizer, model, dataset):
    if dataset == "math500":
        return load_math500(count, seed, tokenizer, model)
    if dataset == "gsm8k":
        return load_gsm8k(split, count, seed, tokenizer, model)
    if dataset == "math-integer":
        return load_math_integer(split, count, seed, tokenizer, model)
    if dataset == "math-integer-hard":
        return load_math_integer(split, count, seed, tokenizer, model, min_level=4)
    if dataset == "math-integer-medium":
        return load_math_integer(split, count, seed, tokenizer, model, min_level=3, max_level=3)
    raise ValueError(f"Unknown dataset {dataset}")


def parser(length=False, model="qwen"):
    p = argparse.ArgumentParser(description="train a grpo baseline or length-penalty run")
    p.add_argument("--model", choices=MODELS, default=model,
                   help="Which qwen variant to train")
    if length:
        p.add_argument("--penalty", choices=("deepseek", "composer"), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-samples", type=int, default=512)
    p.add_argument("--dataset", choices=("gsm8k", "math-integer", "math-integer-hard", "math-integer-medium", "math500"), default="gsm8k")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-completion-length", type=int, default=2048)
    p.add_argument("--norm", type=float, default=512)
    p.add_argument("--cap", type=float, default=0.5)
    p.add_argument("--strength", type=float, default=0.5)
    p.add_argument("--effort", type=float, default=1.0,
                   help="Fixed effort b; k(b) = strength * effort. Hold fixed across arms.")
    p.add_argument("--composer-k", type=float, default=1 / 512)
    p.add_argument("--q", type=float, default=1.5)
    p.add_argument("--correct-reward", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--use-vllm", action="store_true", help="Requires a working compatible vLLM installation/GPU setup")
    p.add_argument("--vllm-mode", choices=("colocate", "server"), default="colocate",
                   help="Colocate shares training GPUs; server needs a separate running vLLM server")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3,
                   help="Fraction reserved by colocated vLLM (TRL default: 0.3)")
    p.add_argument("--vllm-server-base-url", default=None,
                   help="URL of an already running TRL-compatible vLLM server in server mode")
    return p


def validate(args):
    if args.norm <= 0 or args.cap <= 0 or args.strength <= 0 or args.effort <= 0:
        raise ValueError("norm, cap, strength and effort must be positive")
    if args.composer_k <= 0 or args.q < 1 or args.correct_reward <= 0:
        raise ValueError("composer-k and correct-reward must be positive; q >= 1")
    if min(args.train_samples, args.steps, args.max_completion_length, args.lora_r) < 1:
        raise ValueError("train-samples, steps, max-completion-length and lora-r must be positive")
    if not 0 < args.vllm_gpu_memory_utilization <= 1:
        raise ValueError("vllm-gpu-memory-utilization must be in (0, 1]")
    if args.vllm_mode == "server" and not args.use_vllm:
        raise ValueError("vllm-mode server requires --use-vllm")
    if args.vllm_server_base_url and (not args.use_vllm or args.vllm_mode != "server"):
        raise ValueError("vllm-server-base-url requires --use-vllm --vllm-mode server")
    if args.penalty != "none" and args.correct_reward + length_penalty(
            args.max_completion_length, args.penalty, args.norm, args.cap,
            args.strength, args.effort, args.composer_k, args.q) <= 0:
        raise ValueError("length penalty must stay smaller than the correct-answer reward")


def main(model="qwen", length=False):
    args = parser(length, model).parse_args()
    if not length:
        args.penalty = "none"
    validate(args)
    import torch
    from transformers import AutoTokenizer
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    model_id = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    data = load_data("train", args.train_samples, args.seed, tokenizer, args.model, args.dataset)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    def reward_func(completions, completion_ids, prompts, target, **kwargs):
        result = []
        with (out / "reward_samples.jsonl").open("a") as log:
            for ids, prompt, answer in zip(completion_ids, prompts, target):
                n_tokens, answer_start = thinking_span(ids, tokenizer, args.model, prompt)
                parsed_answer = final_answer_ids(ids, tokenizer, args.model, prompt)
                correct = parsed_answer == int(answer)
                penalty = (length_penalty(n_tokens, args.penalty, args.norm, args.cap,
                                          args.strength, args.effort, args.composer_k, args.q)
                           if length and correct else 0.0)
                reward = (args.correct_reward if correct else 0.0) + penalty
                result.append(reward)
                log.write(json.dumps({"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "target": int(answer), "parsed_answer": parsed_answer, "correct": correct,
                    "finished_thinking": answer_start is not None, "thinking_tokens": n_tokens,
                    "completion_tokens": len(ids), "truncated": len(ids) >= args.max_completion_length,
                    "penalty": penalty, "reward": reward}) + "\n")
        return result

    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    peft_config = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r,
                             target_modules=targets, task_type="CAUSAL_LM")
    config = GRPOConfig(
        output_dir=args.output, seed=args.seed, max_steps=args.steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        num_generations=4,
        max_completion_length=args.max_completion_length,
        learning_rate=args.lr, beta=0.0, temperature=0.8,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True, logging_steps=10, save_steps=args.steps,
        report_to="none", remove_unused_columns=False, use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_server_base_url=args.vllm_server_base_url,
        model_init_kwargs={"dtype": "auto"},
    )
    from importlib.metadata import version
    (out / "experiment.json").write_text(json.dumps({**vars(args), "model_id": model_id,
        "packages": {name: version(name) for name in ("torch", "transformers", "trl", "peft", "accelerate", "datasets")},
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "train_row_hashes": [hashlib.sha256(json.dumps(
            {"prompt": row["prompt"], "target": row["target"]}, sort_keys=True).encode()).hexdigest()
            for row in data],
        "dataset_id": ("openai/gsm8k/main" if args.dataset == "gsm8k"
                       else "EleutherAI/hendrycks_math/boxed-integer-level4-5-train"
                       if args.dataset == "math-integer-hard"
                       else "EleutherAI/hendrycks_math/boxed-integer-level3-train"
                       if args.dataset == "math-integer-medium"
                       else "EleutherAI/hendrycks_math/boxed-integer-train"),
        "dataset_revision": GSM8K_REVISION if args.dataset == "gsm8k" else MATH_REVISION,
        "trl_config": "GRPO, 4 generations, beta=0",
        "composer_calibration": "matched cost to DeepSeek at norm tokens"}, indent=2) + "\n")
    trainer = GRPOTrainer(model=model_id, args=config, train_dataset=data,
                          reward_funcs=reward_func, peft_config=peft_config,
                          processing_class=tokenizer)
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main("qwen", length=False)
