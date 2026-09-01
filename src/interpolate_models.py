#!/usr/bin/env python3
"""Build an interpolated teacher on the Instruct-Reasoning spectrum, Eq. (3):

    Theta_MI(lambda) = lambda * Theta_Ins + (1 - lambda) * Theta_Thi

lambda = 1.0 gives the pure Instruct teacher, lambda = 0.0 the pure Reasoning
teacher. The endpoints must share an architecture and parameter shapes.
"""

import argparse
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Linearly interpolate a reasoning teacher with an instruct teacher."
    )
    parser.add_argument(
        "--reasoning_model_path",
        type=str,
        required=True,
        help="Reasoning-oriented endpoint, e.g. pretrained_model/QwQ-32B.",
    )
    parser.add_argument(
        "--instruct_model_path",
        type=str,
        required=True,
        help="Instruction-oriented endpoint, e.g. pretrained_model/Qwen2.5-32B-Instruct.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory to write the interpolated checkpoint to.",
    )
    parser.add_argument(
        "--lam",
        "--lambda",
        "--alpha",
        dest="lam",
        type=float,
        default=0.4,
        help=(
            "Interpolation coefficient lambda in [0, 1], the weight on the Instruct "
            "endpoint. The Reasoning endpoint receives 1 - lambda. "
            "(--alpha is a deprecated alias.)"
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=sorted(DTYPE_MAP),
        help="Precision used to load and save the checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device holding the weights during interpolation. Use cpu unless both models fit in VRAM.",
    )
    return parser.parse_args()


def load_model(path: str, torch_dtype: torch.dtype, device: str) -> AutoModelForCausalLM:
    print(f"Loading model: {path}")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device if device != "cpu" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    return model


def linear_interpolate(
    reasoning_model: AutoModelForCausalLM,
    instruct_model: AutoModelForCausalLM,
    lam: float,
) -> AutoModelForCausalLM:
    """Interpolate in place and return the reasoning model as the carrier."""
    reasoning_state = reasoning_model.state_dict()
    instruct_state = instruct_model.state_dict()

    mixed_state = {}
    keys = list(reasoning_state.keys())
    print(f"Interpolating {len(keys)} tensors at lambda={lam} ...")
    for key in tqdm(keys, desc="Interpolating"):
        if key not in instruct_state:
            print(f"[warn] {key} missing from the instruct model; keeping the reasoning weights.")
            mixed_state[key] = reasoning_state[key]
            continue

        reasoning_param = reasoning_state[key]
        instruct_param = instruct_state[key]
        if reasoning_param.shape != instruct_param.shape:
            raise ValueError(
                f"Shape mismatch for {key}: {reasoning_param.shape} vs {instruct_param.shape}. "
                "The two endpoints must share an architecture."
            )
        # Accumulate in fp32 so bfloat16 rounding does not bias the midpoints.
        mixed = lam * instruct_param.float() + (1.0 - lam) * reasoning_param.float()
        mixed_state[key] = mixed.to(reasoning_param.dtype)

    reasoning_model.load_state_dict(mixed_state)
    return reasoning_model


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.lam <= 1.0:
        raise ValueError(f"lambda must lie in [0, 1], got {args.lam}.")

    print("=" * 60)
    print(f"Reasoning endpoint : {args.reasoning_model_path}  (weight {1 - args.lam:g})")
    print(f"Instruct endpoint  : {args.instruct_model_path}  (weight {args.lam:g})")
    print(f"lambda             : {args.lam:g}")
    print(f"Output             : {args.output_path}")
    print(f"Precision / device : {args.dtype} / {args.device}")
    print("=" * 60)

    torch_dtype = DTYPE_MAP[args.dtype]
    reasoning_model = load_model(args.reasoning_model_path, torch_dtype, args.device)
    instruct_model = load_model(args.instruct_model_path, torch_dtype, args.device)

    mixed_model = linear_interpolate(reasoning_model, instruct_model, args.lam)

    os.makedirs(args.output_path, exist_ok=True)
    print(f"\nSaving interpolated checkpoint to {args.output_path}")
    mixed_model.save_pretrained(args.output_path, safe_serialization=True)

    # The instruct tokenizer/chat template is used for generation at every lambda,
    # so that prompt formatting is identical across the spectrum.
    tokenizer = AutoTokenizer.from_pretrained(args.instruct_model_path)
    tokenizer.save_pretrained(args.output_path)

    print(f"Done: {args.output_path}")


if __name__ == "__main__":
    main()
