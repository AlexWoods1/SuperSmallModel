"""STLM command-line entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="stlm", description="Super Tiny Language Model"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train a model from a YAML config")
    train_p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/smoke.yaml"),
        help="Path to training config YAML",
    )
    gen_p = sub.add_parser("generate", help="Generate text from a model checkpoint")
    gen_p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    gen_p.add_argument(
        "--prompt",
        type=str,
        default="\n",
        help="Prompt to generate text from",
    )
    gen_p.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate",
    )
    gen_p.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for sampling",
    )
    gen_p.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling",
    )
    gen_p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run model on",
    )
    args = parser.parse_args()
    if args.command == "train":
        from stlm.train import train

        train(args.config)
    elif args.command == "generate":
        from stlm.generate import generate, load_checkpoint

        model, tok = load_checkpoint(args.checkpoint, args.device)
        text = generate(
            model,
            tok,
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(text)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
