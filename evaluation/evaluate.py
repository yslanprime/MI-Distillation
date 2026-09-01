#!/usr/bin/env python3
"""vLLM evaluation on the math-reasoning benchmarks.

Handles one (dataset, model) pair per invocation and writes a JSON with the
overall accuracy, per-group breakdowns and every prediction. See
DATASET_LOADERS for the supported benchmarks and DEFAULT_PATHS for where each
is expected on disk.

Every loader returns a list of
    {"question": str, "answer": list[str], "groups": dict[str, str]}
where `groups` is optional metadata used for per-group accuracy breakdowns.
"""

import os
import re
import sys
import ast
import json
import argparse
import multiprocessing as mp
from collections import defaultdict
from typing import List, Dict, Any, Callable, Tuple, Optional

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processing.answer_extraction import extract_answer, extract_math_answer, strip_string  # noqa: E402
from eval.eval_script import eval_math  # noqa: E402


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_lora_adapter(path: Optional[str]) -> bool:
    return bool(path) and os.path.isfile(os.path.join(path, "adapter_config.json"))


def _has_lora_weights(path: str) -> bool:
    return any(
        os.path.isfile(os.path.join(path, filename))
        for filename in ("adapter_model.safetensors", "adapter_model.bin")
    )


def _has_tokenizer_files(path: str) -> bool:
    return any(
        os.path.isfile(os.path.join(path, filename))
        for filename in ("tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt")
    )


def _resolve_model_paths(
    model_path: str,
    base_model_path: Optional[str] = None,
    lora_path: Optional[str] = None,
) -> Tuple[str, Optional[str], str]:
    """Return (engine_model_path, resolved_lora_path, tokenizer_path)."""
    resolved_lora_path = lora_path
    if resolved_lora_path is None and _is_lora_adapter(model_path):
        resolved_lora_path = model_path

    if resolved_lora_path is None:
        return model_path, None, model_path

    adapter_config_path = os.path.join(resolved_lora_path, "adapter_config.json")
    if not os.path.isfile(adapter_config_path):
        raise FileNotFoundError(f"LoRA adapter_config.json not found: {adapter_config_path}")
    if not _has_lora_weights(resolved_lora_path):
        raise FileNotFoundError(
            f"LoRA weights not found in {resolved_lora_path}; expected adapter_model.safetensors or adapter_model.bin"
        )

    adapter_config = _read_json(adapter_config_path)
    engine_model_path = base_model_path or adapter_config.get("base_model_name_or_path")
    if not engine_model_path:
        raise ValueError(
            "A LoRA adapter was provided, but no base model was found. "
            "Pass --base_model_path or set base_model_name_or_path in adapter_config.json."
        )

    tokenizer_path = resolved_lora_path if _has_tokenizer_files(resolved_lora_path) else engine_model_path
    return engine_model_path, resolved_lora_path, tokenizer_path


# Grader hangs on some OlympiadBench answers because sympy.simplify / parse_latex
# can run unbounded on complex LaTeX. We run eval_math in a forked subprocess and
# kill it past the deadline, treating timeouts as incorrect.
def _grader_worker(eval_item, output_queue):
    try:
        result = eval_math(eval_item, pred_key="prediction")
    except Exception:
        result = False
    output_queue.put(bool(result))


def _eval_math_with_timeout(eval_item: Dict, timeout_s: float) -> Tuple[bool, bool]:
    pred = eval_item.get("prediction")
    ans = eval_item.get("answer")
    # Skip the subprocess for trivial exact matches.
    if isinstance(pred, str) and isinstance(ans, str) and pred and pred == ans:
        return True, False
    if isinstance(pred, list) and isinstance(ans, list) and pred and pred == ans:
        return True, False

    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_grader_worker, args=(eval_item, q))
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join(1.0)
        if p.is_alive():
            p.kill()
            p.join()
        return False, True
    try:
        return bool(q.get(timeout=1.0)), False
    except Exception:
        return False, False


# =========================================================================
# Dataset loaders
# =========================================================================

_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_CHOICE_LETTERS = set("ABCD")
_CHOICE_PHRASE_RE = re.compile(
    r"(?:final\s+answer|correct\s+(?:answer|option|choice)|answer|option|choice)"
    r"\s*(?:is|:|=)?\s*[\(\[]?\s*([A-D])\s*[\)\]]?",
)
_CHOICE_STRICT_RE = re.compile(
    r"(?:final\s+answer|correct\s+(?:answer|option|choice)|answer|option|choice)"
    r"\s*(?:is|:|=)?\s*[\(\[]?\s*([A-Da-d])\s*[\)\]]?"
    r"(?=\s*(?:$|[.,;:]))",
    re.IGNORECASE,
)
_CHOICE_LINE_RE = re.compile(r"(?:^|[\s({\[])([A-Da-d])(?:[\s.)}\]]*)$")


def _safe_strip(ans: str) -> str:
    try:
        return strip_string(ans)
    except Exception:
        return ans


def _strip_choice_wrappers(text: str) -> str:
    text = str(text).strip().strip("$").strip()
    text = re.sub(r"\\(?:text|mathrm|mathbf)\{([^{}]+)\}", r"\1", text)
    return text.strip().strip("{}").strip()


def _normalize_choice_answer(text: str) -> str:
    text = _strip_choice_wrappers(text)
    if not text:
        return ""

    simple = text.strip().strip("()[]").strip().rstrip(".").strip()
    if len(simple) == 1 and simple.upper() in _CHOICE_LETTERS:
        return simple.upper()

    m = re.match(r"^(?:option|choice|answer)\s*[:=\-]?\s*([A-Da-d])(?:\s*[\).,:;]|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Accept explicit option-label forms such as `D. ...` or `D) ...`.
    # Bare lowercase text must use punctuation, so `a planet` is not choice A.
    m = re.match(r"^([A-D])(?:\s|[\).,:;])", text)
    if m:
        return m.group(1).upper()
    m = re.match(r"^([a-d])[\).,:;]", text)
    if m:
        return m.group(1).upper()

    return ""


def _extract_choice_prediction(response: str) -> List[str]:
    candidates = []

    for ans in extract_answer(response, exhaust=True):
        choice = _normalize_choice_answer(ans)
        if choice:
            candidates.append(choice)
    if candidates:
        return [candidates[-1]]

    matches = _CHOICE_PHRASE_RE.findall(response)
    if matches:
        return [matches[-1].upper()]

    matches = _CHOICE_STRICT_RE.findall(response)
    if matches:
        return [matches[-1].upper()]

    for line in reversed(response.splitlines()):
        line = line.strip()
        if not line:
            continue
        m = _CHOICE_LINE_RE.search(line)
        if m:
            return [m.group(1).upper()]

    return []


def _eval_choice_exact(prediction, ground_truth) -> bool:
    if isinstance(prediction, list):
        preds = [_normalize_choice_answer(p) for p in prediction]
    else:
        preds = [_normalize_choice_answer(prediction)]
    preds = [p for p in preds if p]

    if isinstance(ground_truth, list):
        answers = {_normalize_choice_answer(a) for a in ground_truth}
    else:
        answers = {_normalize_choice_answer(ground_truth)}
    answers.discard("")

    return bool(preds and answers and preds[-1] in answers)


def _split_top_level(s: str) -> List[str]:
    # Split on commas that are at zero depth w.r.t. (), [], {}.
    # Used to unpack OlympiadBench multi-answer strings like
    # '(1,8,19),(2,7,13),(4,5,7)' or '$f(x)=-1$,$f(x)=x+1$' into
    # individual answers so the matcher can do set-based matching.
    parts: List[str] = []
    depth = 0
    buf: List[str] = []
    for ch in s:
        if ch in "([{":
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            piece = "".join(buf).strip()
            if piece:
                parts.append(piece)
            buf = []
        else:
            buf.append(ch)
    piece = "".join(buf).strip()
    if piece:
        parts.append(piece)
    return parts


def _import_pandas():
    import pandas as pd  # local import so missing parquet deps only matter when used
    return pd


def load_aime24(path: str) -> List[Dict[str, Any]]:
    pd = _import_pandas()
    df = pd.read_parquet(path)
    items = []
    for _, row in df.iterrows():
        sol = str(row["solution"])
        matches = _BOXED_RE.findall(sol)
        ans = matches[-1] if matches else sol.strip()
        items.append({
            "question": str(row["problem"]),
            "answer": [_safe_strip(ans)],
            "groups": {},
        })
    return items


def load_aime25(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            items.append({
                "question": obj["problem"],
                "answer": [_safe_strip(str(obj.get("answer", "")))],
                "groups": {},
            })
    return items


def load_amc23(path: str) -> List[Dict[str, Any]]:
    pd = _import_pandas()
    df = pd.read_parquet(path)
    items = []
    for _, row in df.iterrows():
        items.append({
            "question": str(row["question"]),
            "answer": [_safe_strip(str(row["answer"]))],
            "groups": {},
        })
    return items


def load_gpqa_diamond(path: str) -> List[Dict[str, Any]]:
    pd = _import_pandas()
    df = pd.read_parquet(path)
    items = []
    for _, row in df.iterrows():
        answer = _normalize_choice_answer(row.get("answer", ""))
        if not answer:
            raise ValueError(f"Invalid GPQA-Diamond answer: {row.get('answer')!r}")
        items.append({
            "question": str(row["question"]).strip(),
            "answer": [answer],
            "groups": {},
        })
    return items


def load_gsm8k(path: str) -> List[Dict[str, Any]]:
    pd = _import_pandas()
    df = pd.read_parquet(path)
    items = []
    for _, row in df.iterrows():
        ans_full = str(row["answer"])
        # gsm8k format: "...\n#### 18"
        if "####" in ans_full:
            gold = ans_full.split("####")[-1].strip().replace(",", "")
        else:
            gold = ans_full.strip()
        items.append({
            "question": str(row["question"]),
            "answer": [_safe_strip(gold)],
            "groups": {},
        })
    return items


def load_math500(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            items.append({
                "question": obj["problem"],
                "answer": [_safe_strip(str(obj.get("answer", "")))],
                "groups": {
                    "level":   str(obj.get("level", "")),
                    "subject": str(obj.get("subject", "")),
                },
            })
    return items


def load_minervamath(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            items.append({
                "question": obj["question"],
                "answer": [_safe_strip(str(obj.get("answer", "")))],
                "groups": {},
            })
    return items


def load_olympiadbench(path: str) -> List[Dict[str, Any]]:
    pd = _import_pandas()
    df = pd.read_parquet(path)
    items = []
    for _, row in df.iterrows():
        # `final_answer` is a stringified list literal, e.g. "['2']" or "['x=1', 'x=-1']"
        raw = row.get("final_answer", "[]")
        try:
            parsed = ast.literal_eval(str(raw))
            if not isinstance(parsed, list):
                parsed = [parsed]
        except Exception:
            parsed = [str(raw)]

        # Multi-answer rows pack all answers into a single string (e.g.
        # "(1,8,19),(2,7,13),(4,5,7)" or "$69$,$84$"). Split them so the
        # matcher's double-list set-matching path is actually exercised.
        if bool(row.get("is_multiple_answer", False)) and len(parsed) == 1:
            split = _split_top_level(str(parsed[0]))
            if len(split) > 1:
                parsed = split

        gold = [_safe_strip(str(a)) for a in parsed if str(a).strip() != ""]
        if not gold:
            gold = [""]

        items.append({
            "question": str(row["question"]),
            "answer": gold,
            "groups": {
                "subfield":    str(row.get("subfield", "")),
                "answer_type": str(row.get("answer_type", "")),
            },
        })
    return items


DATASET_LOADERS: Dict[str, Callable[[str], List[Dict[str, Any]]]] = {
    "aime24":         load_aime24,
    "aime25":         load_aime25,
    "amc23":          load_amc23,
    "gpqa_diamond":   load_gpqa_diamond,
    "gsm8k":          load_gsm8k,
    "math500":        load_math500,
    "minervamath":    load_minervamath,
    "olympiadbench":  load_olympiadbench,
}

# Default data paths (relative to the project root, where the script is run from)
DEFAULT_PATHS: Dict[str, str] = {
    "aime24":         "data/raw/aime24/test-00000-of-00001.parquet",
    "aime25":         "data/raw/aime25/test.jsonl",
    "amc23":          "data/raw/amc23/test-00000-of-00001.parquet",
    "gpqa_diamond":   "data/raw/GPQA-Diamond/test/gpqa_diamond.parquet",
    "gsm8k":          "data/raw/gsm8k/main/test-00000-of-00001.parquet",
    "math500":        "data/raw/MATH-500/test.jsonl",
    "minervamath":    "data/raw/minervamath/test.jsonl",
    "olympiadbench":  "data/raw/olympiadbench/test.parquet",
}


# =========================================================================
# Prompt / extraction / metrics
# =========================================================================

def format_prompt(question: str, tokenizer, dataset: str = None) -> str:
    if dataset == "gpqa_diamond":
        instruction = "Please reason step by step, and put only the final option letter (A, B, C, or D) within \\boxed{}."
    else:
        instruction = "Please reason step by step, and put your final answer within \\boxed{}."
    content = f"{question}\n{instruction}"
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_prediction(question: str, response: str, dataset: str = None):
    if dataset == "gpqa_diamond":
        return _extract_choice_prediction(response)
    return extract_math_answer(question, response, task="math")


def normalize_prediction_for_answer(prediction, ground_truth, dataset: str = None):
    if dataset == "gpqa_diamond":
        if isinstance(prediction, list):
            return [_normalize_choice_answer(p) for p in prediction if _normalize_choice_answer(p)]
        choice = _normalize_choice_answer(prediction)
        return [choice] if choice else []

    if (
        isinstance(prediction, list)
        and isinstance(ground_truth, list)
        and len(ground_truth) > 1
        and len(prediction) == 1
    ):
        split = _split_top_level(str(prediction[0]))
        if len(split) > 1:
            normalized = [_safe_strip(part) for part in split if str(part).strip() != ""]
            if normalized:
                return normalized
    return prediction


def score_prediction(dataset: str, prediction, ground_truth, grader_timeout: float) -> Tuple[bool, bool]:
    if dataset == "gpqa_diamond":
        return _eval_choice_exact(prediction, ground_truth), False
    return _eval_math_with_timeout({"prediction": prediction, "answer": ground_truth}, timeout_s=grader_timeout)


def compute_group_metrics(results: List[Dict], group_key: str) -> Dict:
    groups = defaultdict(list)
    for r in results:
        key = r.get("groups", {}).get(group_key, "")
        if key == "":
            continue
        groups[key].append(r)

    metrics = {}
    for key in sorted(groups.keys()):
        items = groups[key]
        correct = sum(1 for it in items if it["correct"])
        total = len(items)
        total_tokens = sum(it["generated_tokens"] for it in items)
        metrics[key] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total > 0 else 0,
            "accuracy_percent": f"{correct / total * 100:.2f}%" if total > 0 else "N/A",
            "avg_tokens": total_tokens / total if total > 0 else 0,
        }
    return metrics


def print_table(title: str, metrics: Dict, key_label: str = "Group"):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  {key_label:<25s} {'Accuracy':>12s} {'Correct':>10s} {'Total':>8s} {'AvgTokens':>12s}")
    print(f"  {'-' * 25} {'-' * 12} {'-' * 10} {'-' * 8} {'-' * 12}")
    for key, m in metrics.items():
        print(f"  {str(key):<25s} {m['accuracy_percent']:>12s} {m['correct']:>10d} {m['total']:>8d} {m['avg_tokens']:>12.1f}")
    print(f"{'=' * 70}")


# =========================================================================
# Main eval loop
# =========================================================================

def evaluate(
    dataset: str,
    model_path: str,
    test_data: List[Dict],
    base_model_path: str = None,
    lora_path: str = None,
    lora_name: str = None,
    max_lora_rank: int = 16,
    output_file: str = None,
    max_samples: int = None,
    max_new_tokens: int = 8192,
    max_model_len: int = 16384,
    temperature: float = 0.6,
    top_p: float = 0.95,
    tensor_parallel_size: int = 4,
    gpu_memory_utilization: float = 0.9,
    seed: int = None,
    grader_timeout: float = 5.0,
):
    if max_samples:
        test_data = test_data[:max_samples]

    engine_model_path, resolved_lora_path, tokenizer_path = _resolve_model_paths(
        model_path=model_path,
        base_model_path=base_model_path,
        lora_path=lora_path,
    )

    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        format_prompt(item["question"], tokenizer, dataset=dataset)
        for item in tqdm(test_data, desc="Formatting Prompts")
    ]

    print(f"\nExample Prompt:\n{'=' * 50}")
    print(prompts[0])
    print(f"{'=' * 50}\n")

    print(f"Initializing vLLM with model: {engine_model_path}")
    if resolved_lora_path:
        print(f"Applying LoRA adapter: {resolved_lora_path}")

    llm_kwargs = dict(
        model=engine_model_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
    )
    if resolved_lora_path:
        llm_kwargs.update(
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            max_loras=1,
        )
    llm = LLM(**llm_kwargs)

    sp_kwargs = dict(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )
    if seed is not None:
        sp_kwargs["seed"] = seed
    sampling_params = SamplingParams(**sp_kwargs)

    print("Generating responses...")
    if resolved_lora_path:
        from vllm.lora.request import LoRARequest

        request_name = lora_name or os.path.basename(resolved_lora_path.rstrip("/"))
        outputs = llm.generate(
            prompts,
            sampling_params,
            lora_request=LoRARequest(request_name, 1, resolved_lora_path),
        )
    else:
        outputs = llm.generate(prompts, sampling_params)

    correct = 0
    total = 0
    total_tokens = 0
    grader_timeouts = 0
    results = []

    print("\nEvaluating...")
    for i, item in enumerate(tqdm(test_data, desc="Evaluating")):
        question = item["question"]
        ground_truth = item["answer"]

        output = outputs[i]
        response = output.outputs[0].text
        generated_tokens = len(output.outputs[0].token_ids)
        total_tokens += generated_tokens

        prediction = extract_prediction(question, response, dataset=dataset)
        prediction = normalize_prediction_for_answer(prediction, ground_truth, dataset=dataset)

        is_correct, timed_out = score_prediction(dataset, prediction, ground_truth, grader_timeout=grader_timeout)
        if timed_out:
            grader_timeouts += 1
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "groups": item.get("groups", {}),
            "prediction": prediction,
            "response": response,
            "generated_tokens": generated_tokens,
            "correct": is_correct,
            "grader_timeout": timed_out,
        })

    accuracy = correct / total if total > 0 else 0
    avg_tokens = total_tokens / total if total > 0 else 0
    overall_metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "accuracy_percent": f"{accuracy * 100:.2f}%",
        "avg_tokens": avg_tokens,
        "grader_timeouts": grader_timeouts,
        "grader_timeout_s": grader_timeout,
    }

    print(f"\n{'=' * 70}")
    print(f"  {dataset.upper()} Overall Results")
    print(f"{'=' * 70}")
    print(f"  Accuracy:       {overall_metrics['accuracy_percent']} ({correct}/{total})")
    print(f"  Average Tokens: {avg_tokens:.1f}")
    print(f"  Grader Timeouts:{grader_timeouts:>4d} / {total}  (>{grader_timeout:g}s, counted as wrong)")
    print(f"{'=' * 70}")

    # Per-group breakdowns
    by_group: Dict[str, Dict] = {}
    if results and results[0].get("groups"):
        for gkey in results[0]["groups"].keys():
            gm = compute_group_metrics(results, gkey)
            if gm:
                print_table(f"{dataset.upper()} Results by {gkey}", gm, key_label=gkey)
                by_group[gkey] = {str(k): v for k, v in gm.items()}

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        output_data = {
            "dataset": dataset,
            "model_path": model_path,
            "base_model_path": engine_model_path if resolved_lora_path else None,
            "lora_path": resolved_lora_path,
            "overall": overall_metrics,
            "by_group": by_group,
            "results": results,
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed results saved to: {output_file}")

    return overall_metrics, by_group


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Generic vLLM evaluator for math-reasoning datasets")
    parser.add_argument("--dataset", type=str, required=True, choices=sorted(DATASET_LOADERS.keys()))
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, default=None,
                        help="Base model path for LoRA evaluation. If omitted, adapter_config.json is used when possible.")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Optional LoRA adapter path. If --model_path itself is a LoRA adapter, this is inferred.")
    parser.add_argument("--lora_name", type=str, default=None,
                        help="Optional runtime name for the LoRA adapter.")
    parser.add_argument("--max_lora_rank", type=int, default=16)
    parser.add_argument("--test_data_path", type=str, default=None,
                        help="Custom data path; falls back to DEFAULT_PATHS[dataset] if omitted")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=None,
                        help="Sampling seed; required for reproducible multi-run averaging")
    parser.add_argument("--grader_timeout", type=float, default=5.0,
                        help="Per-sample grader (sympy) timeout in seconds; timed-out samples count as wrong")
    args = parser.parse_args()

    test_data_path = args.test_data_path or DEFAULT_PATHS[args.dataset]

    if args.output_file is None:
        model_name_path = args.lora_path or args.model_path
        model_name = os.path.basename(model_name_path.rstrip("/"))
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        args.output_file = os.path.join(results_dir, f"{args.dataset}_{model_name}.json")

    print("=" * 60)
    print(f"{args.dataset.upper()} Evaluation Configuration:")
    print("=" * 60)
    print(f"  Dataset:        {args.dataset}")
    print(f"  Model:          {args.model_path}")
    print(f"  Base Model:     {args.base_model_path if args.base_model_path else 'auto/none'}")
    print(f"  LoRA Adapter:   {args.lora_path if args.lora_path else 'auto/none'}")
    print(f"  Test Data:      {test_data_path}")
    print(f"  Output File:    {args.output_file}")
    print(f"  Max Samples:    {args.max_samples if args.max_samples else 'All'}")
    print(f"  Max New Tokens: {args.max_new_tokens}")
    print(f"  Max Model Len:  {args.max_model_len}")
    print(f"  Temperature:    {args.temperature}")
    print(f"  Top-p:          {args.top_p}")
    print(f"  TP Size:        {args.tensor_parallel_size}")
    print(f"  Seed:           {args.seed if args.seed is not None else 'random'}")
    print(f"  Grader Timeout: {args.grader_timeout}s")
    print("=" * 60)

    loader = DATASET_LOADERS[args.dataset]
    print(f"Loading {args.dataset} from: {test_data_path}")
    test_data = loader(test_data_path)
    print(f"Loaded {len(test_data)} samples")

    evaluate(
        dataset=args.dataset,
        model_path=args.model_path,
        test_data=test_data,
        base_model_path=args.base_model_path,
        lora_path=args.lora_path,
        lora_name=args.lora_name,
        max_lora_rank=args.max_lora_rank,
        output_file=args.output_file,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        top_p=args.top_p,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        grader_timeout=args.grader_timeout,
    )


if __name__ == "__main__":
    main()
