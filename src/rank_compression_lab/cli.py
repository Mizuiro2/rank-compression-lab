import argparse
import json
from pathlib import Path

import torch

from .compression import load_checkpoint
from .data import make_dataset
from .experiment import ExperimentConfig, run_experiment, summary_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, measure, and deploy a CPU ranking-model compression experiment.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="run all five model variants from one seeded dataset")
    defaults = ExperimentConfig()
    for name in ("seed", "queries", "candidates", "epochs", "finetune_epochs", "train_batch_queries", "benchmark_batch_queries", "threads", "warmup", "repetitions"):
        train.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in ("learning_rate", "temperature", "kd_alpha", "keep_fraction"):
        train.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    train.add_argument("--report", type=Path, default=Path("reports/local_cpu.json"))
    train.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints"))
    infer = commands.add_parser("infer", help="load an exported checkpoint and score raw feature groups")
    infer.add_argument("--checkpoint", type=Path, required=True)
    source = infer.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help='JSON with a "features" array [query, candidate, 40]')
    source.add_argument("--demo", action="store_true", help="score one seeded synthetic query")
    infer.add_argument("--output", type=Path)
    infer.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.command == "train":
        config = ExperimentConfig(**{name: getattr(args, name) for name in ExperimentConfig.__dataclass_fields__})
        report = run_experiment(config, args.report, args.checkpoint_dir)
        print(summary_table(report))
        print(f"\nRaw report: {args.report}")
        return
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    model, payload = load_checkpoint(args.checkpoint)
    raw = make_dataset(20, 32, seed=42).features[:1] if args.demo else torch.tensor(json.loads(args.input.read_text())["features"], dtype=torch.float32)
    if raw.ndim != 3 or raw.shape[-1] != payload["architecture"]["input_dim"] or 0 in raw.shape or not torch.isfinite(raw).all():
        parser.error("input features must be a finite nonempty [query, candidate, 40] array")
    with torch.inference_mode():
        scores = model((raw - payload["feature_mean"]) / payload["feature_std"])
    result = {
        "scores": scores.tolist(),
        "ranked_candidate_indices": torch.argsort(scores, dim=-1, descending=True, stable=True).tolist(),
        "quantized_engine": payload["quantized_engine"],
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
