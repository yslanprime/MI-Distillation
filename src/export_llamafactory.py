#!/usr/bin/env python3
"""Convert selected CoT records into LLaMA-Factory Alpaca-format SFT data.

Appends the evaluation instruction to each question by default, so the training
prompt matches the one used at generation and evaluation time. Optionally
registers the output in a LLaMA-Factory dataset_info.json.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


EVAL_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CoT JSON to LLaMA-Factory SFT data.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file (a list of records).")
    parser.add_argument("--output_file", type=str, required=True, help="Output Alpaca-format JSON path.")
    parser.set_defaults(append_eval_instruction=True)
    parser.add_argument(
        "--append-eval-instruction",
        dest="append_eval_instruction",
        action="store_true",
        help="Append the evaluation instruction to each question (default).",
    )
    parser.add_argument(
        "--no-append-eval-instruction",
        dest="append_eval_instruction",
        action="store_false",
        help="Leave questions untouched.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=EVAL_INSTRUCTION,
        help="Instruction text appended to each question.",
    )
    parser.add_argument(
        "--dataset_info",
        type=str,
        default=None,
        help="Path to a LLaMA-Factory dataset_info.json. When given, the output file is registered in it.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Name to register the dataset under. Defaults to the output filename stem.",
    )
    return parser.parse_args()


def load_records(input_file: Path) -> List[Dict[str, Any]]:
    with input_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{input_file} must contain a JSON list of records.")
    return data


def convert_records(
    records: List[Dict[str, Any]],
    append_eval_instruction: bool,
    instruction: str,
) -> Tuple[List[Dict[str, Any]], int]:
    converted: List[Dict[str, Any]] = []
    skipped = 0

    for item in records:
        if not isinstance(item, dict):
            skipped += 1
            continue

        question = str(item.get("problem", "")).strip()
        response = str(item.get("cot_response", "")).strip()
        if not question or not response:
            skipped += 1
            continue

        prompt = f"{question}\n{instruction}" if append_eval_instruction else question
        # Everything except the text itself is retained as provenance metadata
        # (source lambda, SeqLSS score, token length, ...).
        meta = {key: value for key, value in item.items() if key not in {"problem", "cot_response"}}
        converted.append({
            "instruction": prompt,
            "input": "",
            "output": response,
            "system": "",
            "meta": meta,
        })

    return converted, skipped


def register_dataset(dataset_info_path: Path, dataset_name: str, data_file: Path) -> None:
    dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset_info_path.exists():
        with dataset_info_path.open("r", encoding="utf-8") as handle:
            info = json.load(handle)
    else:
        info = {}

    # LLaMA-Factory resolves file_name relative to dataset_info.json.
    try:
        file_name = str(data_file.resolve().relative_to(dataset_info_path.parent.resolve()))
    except ValueError:
        file_name = str(data_file.resolve())

    info[dataset_name] = {"file_name": file_name}
    with dataset_info_path.open("w", encoding="utf-8") as handle:
        json.dump(info, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)

    records = load_records(input_file)
    converted, skipped = convert_records(
        records,
        append_eval_instruction=args.append_eval_instruction,
        instruction=args.instruction,
    )
    if not converted:
        raise ValueError(f"No usable records found in {input_file}.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(converted, handle, ensure_ascii=False, indent=2)

    print(f"Input samples : {len(records)}")
    print(f"Output samples: {len(converted)}")
    print(f"Skipped       : {skipped}")
    print(f"Saved to      : {output_file}")

    if args.dataset_info:
        dataset_name = args.dataset_name or output_file.stem
        dataset_info_path = Path(args.dataset_info)
        register_dataset(dataset_info_path, dataset_name, output_file)
        print(f"Registered    : {dataset_name} -> {dataset_info_path}")


if __name__ == "__main__":
    main()
