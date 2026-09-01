#!/usr/bin/env python3
"""Select student-aligned CoT trajectories with SeqLSS (Section 5.2).

Filters the candidate pool by answer correctness and length, scores every
surviving trajectory under the student with Eq. (7)-(10), then keeps the
highest-scoring one per problem:

    S_i    = -log p_s(r_i | x, r_<i)                          surprisal
    U_i    = sum_{v : p_s(v) > p_s(r_i)} p_s(v | x, r_<i)     rank mass
    SeqLSS = sum_i S_i (1 - U_i)^alpha / sum_i S_i

alpha >= 0 is the learnability penalty;
"""

import argparse
import copy
import json
import math
import random
import re
import signal
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm


EVAL_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
DEFAULT_ALPHA = 4.0

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_ROOT = PROJECT_ROOT / "evaluation"
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from data_processing.answer_extraction import extract_math_answer, strip_string  # noqa: E402
from eval.eval_script import eval_math  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select one best CoT per problem using SeqLSS, then sample train data."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the per-lambda CoT JSON files that form the candidate pool.",
    )
    parser.add_argument(
        "--input_glob",
        type=str,
        default="*.json",
        help="Glob pattern applied recursively under input_dir.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Student model. Its likelihoods define SeqLSS, so this must be the model you will distill into.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Exact output directory. When omitted, defaults to "
            "<output_dir_root>/<pool_tag>_SeqLSS_student_<student_tag>."
        ),
    )
    parser.add_argument(
        "--output_dir_root",
        type=str,
        default="data",
        help="Parent directory used when --output_dir is not given.",
    )
    parser.add_argument(
        "--output_stem",
        type=str,
        default=None,
        help=(
            "Output filename stem. Defaults to "
            "<teacher_data_tag>_SeqLSS_student_<student_model_tag>."
        ),
    )
    parser.add_argument("--train_size", type=int, default=7500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument(
        "--instruction",
        type=str,
        default=EVAL_INSTRUCTION,
        help="Instruction appended to each problem.",
    )
    parser.add_argument(
        "--group_key",
        choices=["unique_id", "problem"],
        default="unique_id",
        help="How to identify the same problem across interpolation files.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=(
            "Learnability penalty alpha, the exponent on (1 - U_i). Larger values "
            "favour trajectories that stay inside the student's high-probability region."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, cuda:0, etc. Ignored when --device_map is set.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Optional transformers device_map, e.g. auto.",
    )
    parser.add_argument(
        "--save_all_valid",
        action="store_true",
        help="Save all valid candidates with SeqLSS scores.",
    )
    parser.add_argument(
        "--register_llamafactory",
        action="store_true",
        help="Also register the train file in LlamaFactory/data/dataset_info.json.",
    )
    parser.add_argument(
        "--dataset_info_path",
        type=str,
        default="LlamaFactory/data/dataset_info.json",
        help="dataset_info.json path used by --register_llamafactory.",
    )
    return parser.parse_args()


def discover_input_files(input_dir: Path, pattern: str) -> List[Path]:
    return sorted(path for path in input_dir.rglob(pattern) if path.is_file())


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list.")
    return data

def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

def build_user_content(problem: str, instruction: str) -> str:
    problem = problem.strip()
    return f"{problem}\n{instruction}" if instruction else problem

def build_prompt(tokenizer: Any, problem: str, instruction: str) -> str:
    user_content = build_user_content(problem, instruction)
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": user_content}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{user_content}\n"


def tokenize(tokenizer: Any, text: str) -> List[int]:
    tokenized = tokenizer(text, add_special_tokens=False)
    input_ids = tokenized.get("input_ids")
    if input_ids is None:
        raise ValueError("Tokenizer output has no input_ids.")
    return list(input_ids)


def compute_token_length(
    tokenizer: Any,
    problem: str,
    cot_response: str,
    instruction: str,
) -> int:
    prompt = build_prompt(tokenizer, problem, instruction)
    eos = tokenizer.eos_token or ""
    return len(tokenize(tokenizer, f"{prompt}{cot_response.strip()}{eos}"))


class _CorrectnessTimeout(Exception):
    pass


def _correctness_alarm_handler(signum, frame):  # noqa: ARG001
    raise _CorrectnessTimeout()


CORRECTNESS_TIMEOUT_SECONDS = 5


def is_correct_sample(problem: str, cot_response: str, answer: str) -> bool:
    # sympy.simplify / parse_latex can hang on pathological inputs, so a hard
    # timeout keeps one bad sample from freezing the pipeline.
    old_handler = signal.signal(signal.SIGALRM, _correctness_alarm_handler)
    signal.alarm(CORRECTNESS_TIMEOUT_SECONDS)
    try:
        prediction = extract_math_answer(problem, cot_response, task="math")
        ground_truth = [strip_string(answer)]
        eval_item = {"prediction": prediction, "answer": ground_truth}
        return bool(eval_math(eval_item, pred_key="prediction"))
    except _CorrectnessTimeout:
        return False
    except Exception:
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def source_tag(path: Path) -> str:
    return path.stem


def parse_lambda_from_name(path: Path) -> Dict[str, Optional[float]]:
    """Recover the interpolation coefficient from a generated-CoT filename.

    Accepts either the current ``...lambda0.4...`` convention or the legacy
    ``...instruct0.4_r1distill0.6...`` naming used to produce the released data.
    """
    text = path.stem
    match = re.search(r"lambda([0-9.]+)", text, flags=re.IGNORECASE)
    if match is None:
        match = re.search(r"instruct([0-9.]+)", text, flags=re.IGNORECASE)
    if match is None:
        return {"interpolation_lambda": None}
    return {"interpolation_lambda": float(match.group(1).rstrip("."))}


def problem_key(record: Dict[str, Any], group_key: str) -> Any:
    if group_key == "unique_id" and record.get("unique_id") is not None:
        return record["unique_id"]
    problem = str(record.get("problem", "")).strip()
    if problem:
        return problem
    return None


def normalize_candidate(
    record: Dict[str, Any],
    source_path: Path,
    source_index: int,
    group_key: str,
) -> Optional[Dict[str, Any]]:
    if not isinstance(record, dict):
        return None

    key = problem_key(record, group_key)
    problem = str(record.get("problem", "")).strip()
    cot_response = str(record.get("cot_response", "")).strip()
    answer = str(record.get("answer", "")).strip()
    if key is None or not problem or not cot_response or not answer:
        return None

    candidate = copy.deepcopy(record)
    candidate["problem"] = problem
    candidate["cot_response"] = cot_response
    candidate["answer"] = answer
    candidate["lss_group_key"] = key
    candidate["lss_source_file"] = str(source_path)
    candidate["lss_source_name"] = source_tag(source_path)
    candidate["lss_source_index"] = source_index
    candidate.update(parse_lambda_from_name(source_path))
    return candidate


def load_candidates(input_files: Sequence[Path], group_key: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    candidates: List[Dict[str, Any]] = []
    stats = {
        "files": len(input_files),
        "raw_records": 0,
        "skipped_invalid": 0,
    }

    for path in input_files:
        records = load_json_list(path)
        stats["raw_records"] += len(records)
        for index, record in enumerate(records):
            candidate = normalize_candidate(record, path, index, group_key)
            if candidate is None:
                stats["skipped_invalid"] += 1
                continue
            candidates.append(candidate)

    return candidates, stats


def safe_name(text: str) -> str:
    text = text.strip().replace("/", "_")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def float_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def model_tag(model_path: str) -> str:
    tag = Path(model_path.rstrip("/")).name
    return safe_name(tag or model_path)


def data_tag(input_dir: Path) -> str:
    return safe_name(input_dir.name)


def resolve_output_paths(args: argparse.Namespace, input_dir: Path) -> Tuple[Path, str, str]:
    teacher_tag = data_tag(input_dir)
    student_tag = model_tag(args.model_path)
    default_stem = f"{teacher_tag}_SeqLSS_student_{student_tag}"
    if not math.isclose(args.alpha, DEFAULT_ALPHA):
        default_stem = f"{default_stem}_alpha{float_tag(args.alpha)}"
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.output_dir_root) / default_stem
    output_stem = args.output_stem or default_stem
    return output_dir, output_stem, student_tag


def dtype_from_name(torch: Any, dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_name}")


def device_from_name(torch: Any, device_name: str) -> Any:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_tokenizer(model_path: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(args: argparse.Namespace, tokenizer: Any) -> Tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype_from_name(torch, args.dtype),
    }
    device = device_from_name(torch, args.device)

    if args.device_map:
        model_kwargs["device_map"] = args.device_map
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
        input_device = model.get_input_embeddings().weight.device
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
        model.to(device)
        input_device = device

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return torch, model, input_device


def seq_lss_from_logits(
    torch: Any,
    logits: Any,
    target_ids: Any,
    alpha: float = DEFAULT_ALPHA,
) -> Tuple[Any, Any, Any, Any]:
    """Compute the per-token LSS numerator terms of Eq. (9) plus diagnostics.

    Returns ``(lss_terms, surprisal, higher_mass, target_prob)`` where
    ``lss_terms[i] = S_i * (1 - U_i)^alpha``, ``surprisal[i] = S_i`` and
    ``higher_mass[i] = U_i``.
    """
    logits = logits.to(torch.float32)
    target_ids = target_ids.to(logits.device)
    target_logits = logits.gather(dim=-1, index=target_ids[:, None]).squeeze(-1)

    # Softmax computed from shifted logits to stay numerically stable over a
    # 150k-token vocabulary.
    max_logits = logits.max(dim=-1).values
    exp_logits = torch.exp(logits - max_logits[:, None])
    denom = exp_logits.sum(dim=-1)

    target_prob = torch.exp(target_logits - max_logits) / denom
    # U_i: probability mass on tokens the student ranks strictly above the target.
    higher_mass = exp_logits.masked_fill(logits <= target_logits[:, None], 0.0).sum(dim=-1) / denom
    surprisal = -torch.log(target_prob.clamp_min(torch.finfo(torch.float32).tiny))
    lss_terms = surprisal * (1.0 - higher_mass).clamp(0.0, 1.0).pow(alpha)
    return lss_terms, surprisal, higher_mass, target_prob


def score_candidate_seq_lss(
    torch: Any,
    model: Any,
    tokenizer: Any,
    candidate: Dict[str, Any],
    input_device: Any,
    instruction: str,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, float]:
    prompt = build_prompt(tokenizer, candidate["problem"], instruction)
    cot_response = candidate["cot_response"]
    prompt_ids = tokenize(tokenizer, prompt)
    cot_ids = tokenize(tokenizer, cot_response)
    if not prompt_ids:
        raise ValueError("Prompt has no tokens.")
    if not cot_ids:
        raise ValueError("CoT response has no tokens.")

    with torch.inference_mode():
        input_ids = torch.tensor([prompt_ids + cot_ids], dtype=torch.long, device=input_device)
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits[0]

        # Position t predicts token t+1, so the logits scoring the first CoT
        # token sit at index len(prompt_ids) - 1.
        start = len(prompt_ids) - 1
        end = start + len(cot_ids)
        score_logits = logits[start:end]
        targets = torch.tensor(cot_ids, dtype=torch.long, device=score_logits.device)
        lss_terms, surprisal_all, mass_all, prob_all = seq_lss_from_logits(
            torch=torch,
            logits=score_logits,
            target_ids=targets,
            alpha=alpha,
        )
        surprisal_sum = surprisal_all.sum()
        seq_lss = torch.where(
            surprisal_sum > 0,
            lss_terms.sum() / surprisal_sum,
            torch.zeros((), dtype=surprisal_all.dtype, device=surprisal_all.device),
        )

    surprisal_mean = float(surprisal_all.mean().item())
    try:
        ppl = float(math.exp(surprisal_mean))
    except OverflowError:
        ppl = float("inf")

    return {
        "lss_score": float(seq_lss.item()),
        "lss_alpha": float(alpha),
        "lss_num_tokens": int(lss_terms.numel()),
        "lss_surprisal_mean": surprisal_mean,
        "lss_surprisal_sum": float(surprisal_sum.item()),
        "lss_ppl": ppl,
        "lss_higher_mass_mean": float(mass_all.mean().item()),
        "lss_target_prob_mean": float(prob_all.mean().item()),
        "lss_acceptance_mean": float((1.0 - mass_all).mean().item()),
        "lss_numerator": float(lss_terms.sum().item()),
        "lss_denominator": float(surprisal_sum.item()),
    }


def prefilter_candidates(
    candidates: Iterable[Dict[str, Any]],
    tokenizer: Any,
    instruction: str,
    max_length: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    valid: List[Dict[str, Any]] = []
    stats = {
        "length_pass": 0,
        "correct_pass": 0,
        "skipped_too_long": 0,
        "skipped_incorrect": 0,
    }

    for candidate in tqdm(list(candidates), desc="Correctness/length filtering"):
        token_length = compute_token_length(
            tokenizer=tokenizer,
            problem=candidate["problem"],
            cot_response=candidate["cot_response"],
            instruction=instruction,
        )
        candidate["token_length"] = token_length
        if token_length >= max_length:
            stats["skipped_too_long"] += 1
            continue
        stats["length_pass"] += 1

        if not is_correct_sample(candidate["problem"], candidate["cot_response"], candidate["answer"]):
            stats["skipped_incorrect"] += 1
            continue
        stats["correct_pass"] += 1
        valid.append(candidate)

    return valid, stats


def add_lss_metrics(candidate: Dict[str, Any], metrics: Dict[str, float]) -> Dict[str, Any]:
    scored = copy.deepcopy(candidate)
    scored.update(metrics)
    return scored


def select_best_per_problem(scored: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_key: Dict[Any, Dict[str, Any]] = {}
    for candidate in scored:
        key = candidate["lss_group_key"]
        current = best_by_key.get(key)
        if current is None or float(candidate["lss_score"]) > float(current["lss_score"]):
            best_by_key[key] = candidate

    best = list(best_by_key.values())
    best.sort(key=lambda item: str(item["lss_group_key"]))
    return best


def sample_train(best_records: Sequence[Dict[str, Any]], train_size: int, seed: int) -> List[Dict[str, Any]]:
    if len(best_records) < train_size:
        raise ValueError(f"Need {train_size} best-per-problem records, but only got {len(best_records)}.")
    records = [copy.deepcopy(item) for item in best_records]
    random.Random(seed).shuffle(records)
    return records[:train_size]


def to_llamafactory(records: Sequence[Dict[str, Any]], instruction: str) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for item in records:
        meta = {key: value for key, value in item.items() if key not in {"problem", "cot_response"}}
        converted.append(
            {
                "instruction": build_user_content(item["problem"], instruction),
                "input": "",
                "output": item["cot_response"],
                "system": "",
                "meta": meta,
            }
        )
    return converted


def make_dataset_file_name(dataset_info_path: Path, data_file: Path) -> str:
    try:
        return str(data_file.resolve().relative_to(dataset_info_path.parent.resolve()))
    except ValueError:
        return str(data_file)


def register_llamafactory_dataset(dataset_info_path: Path, dataset_name: str, data_file: Path) -> None:
    if dataset_info_path.exists():
        with dataset_info_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    else:
        data = {}
    data[dataset_name] = {"file_name": make_dataset_file_name(dataset_info_path, data_file)}
    save_json(dataset_info_path, data)


def print_score_summary(label: str, records: Sequence[Dict[str, Any]], score_key: str = "lss_score") -> None:
    if not records:
        print(f"{label}: 0")
        return
    scores = [float(item[score_key]) for item in records]
    print(
        f"{label}: n={len(records)}, "
        f"{score_key} min/mean/max={min(scores):.6f}/{sum(scores) / len(scores):.6f}/{max(scores):.6f}"
    )


def source_counts(records: Sequence[Dict[str, Any]], source_key: str = "lss_source_name") -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for record in records:
        counts[str(record.get(source_key, "unknown"))] += 1
    return dict(sorted(counts.items()))


def print_source_counts(label: str, records: Sequence[Dict[str, Any]]) -> None:
    print(f"{label} source counts:")
    for source, count in source_counts(records).items():
        print(f"  {source}: {count}")


def main() -> None:
    args = parse_args()
    if args.train_size <= 0:
        raise ValueError("--train_size must be positive.")
    if args.max_length <= 0:
        raise ValueError("--max_length must be positive.")
    if args.alpha < 0:
        raise ValueError("--alpha must be non-negative.")

    input_dir = Path(args.input_dir)
    output_dir, output_stem, student_tag = resolve_output_paths(args, input_dir)
    input_files = discover_input_files(input_dir, args.input_glob)
    if not input_files:
        raise ValueError(f"No input files found under {input_dir} with pattern {args.input_glob}.")

    print("=" * 70)
    print("SeqLSS Best-CoT Selection")
    print("=" * 70)
    print(f"Input dir     : {input_dir}")
    print(f"Input files   : {len(input_files)}")
    for path in input_files:
        print(f"  - {path.name}")
    print(f"Student model : {args.model_path}")
    print(f"Student tag   : {student_tag}")
    print(f"Max length    : < {args.max_length}")
    print(f"Train size    : {args.train_size}")
    print(f"Selection     : SeqLSS (alpha={args.alpha:g})")
    print(f"Output dir    : {output_dir}")
    print(f"Output stem   : {output_stem}")
    print("=" * 70)

    print("\n[1/6] Loading candidates ...")
    candidates, load_stats = load_candidates(input_files, args.group_key)
    print(f"Raw records    : {load_stats['raw_records']}")
    print(f"Candidates     : {len(candidates)}")
    print(f"Skipped invalid: {load_stats['skipped_invalid']}")

    print("\n[2/6] Loading tokenizer ...")
    tokenizer = load_tokenizer(args.model_path)

    print("\n[3/6] Filtering by length and correctness ...")
    valid_candidates, filter_stats = prefilter_candidates(
        candidates=candidates,
        tokenizer=tokenizer,
        instruction=args.instruction,
        max_length=args.max_length,
    )
    print(f"Length pass      : {filter_stats['length_pass']}")
    print(f"Correct pass     : {filter_stats['correct_pass']}")
    print(f"Skipped too long : {filter_stats['skipped_too_long']}")
    print(f"Skipped incorrect: {filter_stats['skipped_incorrect']}")
    if not valid_candidates:
        raise ValueError("No valid candidates after length/correctness filtering.")

    print("\n[4/6] Loading student model and scoring SeqLSS ...")
    torch, model, input_device = load_model(args, tokenizer)
    scored_candidates: List[Dict[str, Any]] = []
    skipped_scoring = 0

    for candidate in tqdm(valid_candidates, desc="SeqLSS scoring"):
        try:
            metrics = score_candidate_seq_lss(
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                candidate=candidate,
                input_device=input_device,
                instruction=args.instruction,
                alpha=args.alpha,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"\nWARNING: scoring failed for {candidate.get('unique_id')}: {exc}")
            skipped_scoring += 1
            continue
        except Exception as exc:
            print(f"\nWARNING: scoring failed for {candidate.get('unique_id')}: {exc}")
            skipped_scoring += 1
            continue
        scored_candidates.append(add_lss_metrics(candidate, metrics))

    if not scored_candidates:
        raise ValueError("No candidates were scored successfully.")
    print(f"Scored candidates: {len(scored_candidates)}")
    print(f"Skipped scoring  : {skipped_scoring}")

    print("\n[5/6] Selecting best CoT per problem and sampling train set ...")
    best_records = select_best_per_problem(scored_candidates)
    train_records = sample_train(best_records, args.train_size, args.seed)
    train_llamafactory = to_llamafactory(train_records, args.instruction)
    best_llamafactory = to_llamafactory(best_records, args.instruction)

    print_score_summary("All valid scored", scored_candidates)
    print_score_summary("Best per problem", best_records)
    print_score_summary("Train sampled  ", train_records)
    print_source_counts("Best per problem", best_records)

    print("\n[6/6] Saving outputs ...")
    raw_train_file = output_dir / f"{output_stem}_train{args.train_size}.json"
    train_lf_file = output_dir / f"{output_stem}_train{args.train_size}_llamafactory.json"
    best_file = output_dir / f"{output_stem}_best_per_problem{len(best_records)}.json"
    best_lf_file = output_dir / f"{output_stem}_best_per_problem{len(best_records)}_llamafactory.json"
    summary_file = output_dir / f"{output_stem}_summary.json"

    save_json(raw_train_file, train_records)
    save_json(train_lf_file, train_llamafactory)
    save_json(best_file, best_records)
    save_json(best_lf_file, best_llamafactory)
    if args.save_all_valid:
        all_valid_file = output_dir / f"{output_stem}_all_valid_scored{len(scored_candidates)}.json"
        save_json(all_valid_file, scored_candidates)
    else:
        all_valid_file = None

    summary = {
        "input_dir": str(input_dir),
        "input_files": [str(path) for path in input_files],
        "model_path": args.model_path,
        "student_tag": student_tag,
        "output_stem": output_stem,
        "max_length": args.max_length,
        "train_size": args.train_size,
        "seed": args.seed,
        "selection_method": "SeqLSS",
        "alpha": args.alpha,
        "raw_records": load_stats["raw_records"],
        "candidates": len(candidates),
        "valid_candidates": len(valid_candidates),
        "scored_candidates": len(scored_candidates),
        "best_per_problem": len(best_records),
        "train_records": len(train_records),
        "skipped_invalid": load_stats["skipped_invalid"],
        "skipped_too_long": filter_stats["skipped_too_long"],
        "skipped_incorrect": filter_stats["skipped_incorrect"],
        "skipped_scoring": skipped_scoring,
        "best_source_counts": source_counts(best_records),
        "train_source_counts": source_counts(train_records),
        "raw_train_file": str(raw_train_file),
        "train_llamafactory_file": str(train_lf_file),
        "best_file": str(best_file),
        "best_llamafactory_file": str(best_lf_file),
    }
    if all_valid_file is not None:
        summary["all_valid_scored_file"] = str(all_valid_file)
    save_json(summary_file, summary)

    if args.register_llamafactory:
        dataset_info_path = Path(args.dataset_info_path)
        register_llamafactory_dataset(
            dataset_info_path=dataset_info_path,
            dataset_name=f"{output_stem}_train{args.train_size}",
            data_file=train_lf_file,
        )
        print(f"Registered LlamaFactory dataset: {dataset_info_path}")

    print(f"Raw train       : {raw_train_file}")
    print(f"LlamaFactory    : {train_lf_file}")
    print(f"Best records    : {best_file}")
    print(f"Best LF         : {best_lf_file}")
    print(f"Summary         : {summary_file}")
    if all_valid_file is not None:
        print(f"All valid scored: {all_valid_file}")
    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
