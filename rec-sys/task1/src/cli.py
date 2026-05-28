from __future__ import annotations

import argparse
import json
from pathlib import Path

from models import MFConfig
from pipeline import Paths, analyze_dataset, describe_dataset, evaluate, predict_test


def _task_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=["baseline", "mf", "blend", "residual", "ensemble"], default="baseline")
    parser.add_argument("--factors", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.008, help="learning rate for matrix factorization")
    parser.add_argument("--reg", type=float, default=0.05, help="regularization for matrix factorization")
    parser.add_argument("--blend-weight", type=float, default=0.7, help="MF weight when --model blend")
    parser.add_argument("--residual-weight", type=float, default=1.0, help="residual correction weight when --model residual")
    parser.add_argument("--bias-iterations", type=int, default=5, help="alternating bias update iterations for baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shrinkage", type=float, default=10.0, help="baseline bias shrinkage")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RecSys task1 training/evaluation pipeline")
    parser.add_argument("--task-root", type=Path, default=_task_root())

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("stats", help="print dataset statistics")
    subparsers.add_parser("analysis", help="print train/test coverage and cold-start analysis")

    evaluate_parser = subparsers.add_parser("evaluate", help="train on a split and report validation RMSE")
    _add_common_model_args(evaluate_parser)
    evaluate_parser.add_argument("--validation-ratio", type=float, default=0.2)

    predict_parser = subparsers.add_parser("predict", help="train on full train.txt and predict test.txt")
    _add_common_model_args(predict_parser)
    predict_parser.add_argument("--output", type=Path, default=_task_root() / "outputs" / "predictions.txt")
    predict_parser.add_argument("--float-output", action="store_true", help="write floating point scores instead of rounded integers")

    return parser


def _mf_config(args: argparse.Namespace) -> MFConfig:
    return MFConfig(
        factors=args.factors,
        epochs=args.epochs,
        learning_rate=args.lr,
        regularization=args.reg,
        blend_weight=args.blend_weight,
        residual_weight=args.residual_weight,
        bias_iterations=args.bias_iterations,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = Paths.from_task_root(args.task_root)

    if args.command == "stats":
        print(json.dumps(describe_dataset(paths), ensure_ascii=False, indent=2))
        return 0

    if args.command == "analysis":
        print(json.dumps(analyze_dataset(paths), ensure_ascii=False, indent=2))
        return 0

    if args.command == "evaluate":
        result = evaluate(
            paths=paths,
            model_name=args.model,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            mf_config=_mf_config(args),
            shrinkage=args.shrinkage,
        )
        print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
        return 0

    if args.command == "predict":
        count, train_seconds, predict_seconds = predict_test(
            paths=paths,
            model_name=args.model,
            output_path=args.output,
            mf_config=_mf_config(args),
            shrinkage=args.shrinkage,
            round_scores=not args.float_output,
        )
        print(
            json.dumps(
                {
                    "model": args.model,
                    "predictions": count,
                    "output": str(args.output),
                    "train_seconds": train_seconds,
                    "predict_seconds": predict_seconds,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    parser.error(f"unknown command {args.command!r}")
    return 2
