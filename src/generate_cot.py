#!/usr/bin/env python3
"""Sample CoT trajectories from one teacher with vLLM.

Run once per interpolated teacher to build the candidate pool D of Eq. (6).
Results are checkpointed every --save_every samples and keyed by unique_id, so
an interrupted run resumes where it stopped.
"""

import argparse
import json
import os
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


EVAL_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CoT trajectories from a teacher model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Teacher checkpoint: an endpoint model or an interpolated model.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Parquet file, or a directory of parquet files, holding the training problems.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSON path. Defaults to data/cot_generated/<dataset>_<model>.json.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=EVAL_INSTRUCTION,
        help="Instruction appended to each problem. Must match the one used at evaluation time.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=16384, help="Generation budget per sample.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling parameter.")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Number of GPUs for tensor parallelism.")
    parser.add_argument("--max_model_len", type=int, default=20480, help="vLLM context window.")
    parser.add_argument("--max_num_seqs", type=int, default=128, help="vLLM concurrent sequence cap.")
    parser.add_argument(
        "--save_every",
        type=int,
        default=500,
        help="Checkpoint interval, in samples. Also the generation batch size.",
    )
    return parser.parse_args()


def load_problems(data_path: str) -> List[Dict[str, Any]]:
    print(f"Loading problems from: {data_path}")

    if os.path.isdir(data_path):
        parquet_files = [
            os.path.join(data_path, name)
            for name in os.listdir(data_path)
            if name.endswith(".parquet")
        ]
        if not parquet_files:
            nested = os.path.join(data_path, "data")
            if os.path.isdir(nested):
                parquet_files = [
                    os.path.join(nested, name)
                    for name in os.listdir(nested)
                    if name.endswith(".parquet")
                ]
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under {data_path}.")
        df = pd.concat([pd.read_parquet(path) for path in sorted(parquet_files)], ignore_index=True)
    else:
        df = pd.read_parquet(data_path)

    print(f"Total problems: {len(df)}")
    return [
        {
            "unique_id": int(row.get("unique_id", -1)),
            "problem": row["problem"],
            "answer": str(row.get("answer", "")),
            "level": row.get("level", ""),
            "type": row.get("type", ""),
            "solution": row.get("solution", ""),
        }
        for _, row in df.iterrows()
    ]


def format_prompt(question: str, tokenizer: Any, instruction: str) -> str:
    content = f"{question.strip()}\n{instruction}" if instruction else question.strip()
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_existing_results(output_file: str) -> List[Dict[str, Any]]:
    if not os.path.exists(output_file):
        return []
    try:
        with open(output_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    return []


def save_results(results: List[Dict[str, Any]], output_file: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)


def generate(args: argparse.Namespace, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = load_existing_results(args.output_file)
    done_ids = {record["unique_id"] for record in results}
    pending = [item for item in data if item["unique_id"] not in done_ids]
    print(f"Already generated: {len(done_ids)}; remaining: {len(pending)}")
    if not pending:
        print("Nothing to do.")
        return results

    prompts = [
        format_prompt(item["problem"], tokenizer, args.instruction)
        for item in tqdm(pending, desc="Formatting prompts")
    ]
    print(f"\nExample prompt:\n{'=' * 60}\n{prompts[0]}\n{'=' * 60}\n")

    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    batch_size = max(1, args.save_every)
    for start in range(0, len(pending), batch_size):
        batch_items = pending[start: start + batch_size]
        batch_prompts = prompts[start: start + batch_size]
        print(f"\nGenerating [{start + 1} ~ {start + len(batch_items)}] / {len(pending)} ...")
        outputs = llm.generate(batch_prompts, sampling_params)

        for item, output in zip(batch_items, outputs):
            generated = output.outputs[0]
            results.append({
                **item,
                "cot_response": generated.text,
                "generated_tokens": len(generated.token_ids),
                "teacher_model": args.model_path,
            })

        save_results(results, args.output_file)
        print(f"Saved {len(results)} samples to {args.output_file}")

    avg_tokens = sum(r.get("generated_tokens", 0) for r in results) / max(1, len(results))
    print(f"\n{'=' * 60}")
    print("CoT generation complete")
    print(f"  Total samples : {len(results)}")
    print(f"  Avg tokens    : {avg_tokens:.1f}")
    print(f"  Output file   : {args.output_file}")
    print(f"{'=' * 60}")
    return results


def main() -> None:
    args = parse_args()

    if args.output_file is None:
        dataset_name = os.path.basename(os.path.abspath(args.data_path.rstrip("/")))
        model_name = os.path.basename(args.model_path.rstrip("/"))
        args.output_file = os.path.join("data", "cot_generated", f"{dataset_name}_{model_name}.json")

    print("=" * 60)
    print("CoT generation configuration")
    print("=" * 60)
    print(f"  Model          : {args.model_path}")
    print(f"  Data           : {args.data_path}")
    print(f"  Output         : {args.output_file}")
    print(f"  Max new tokens : {args.max_new_tokens}")
    print(f"  Temperature    : {args.temperature}")
    print(f"  Top-p          : {args.top_p}")
    print(f"  TP size        : {args.tensor_parallel_size}")
    print("=" * 60)

    generate(args, load_problems(args.data_path))


if __name__ == "__main__":
    main()
