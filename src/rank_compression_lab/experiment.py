from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .compression import prune_hidden_channels, quantize_int8, save_checkpoint
from .data import QueryDataset, fit_standardizer, make_dataset, split_queries, standardize
from .metrics import ranking_metrics
from .model import Reranker


@dataclass
class ExperimentConfig:
    seed: int = 2026
    queries: int = 960
    candidates: int = 32
    epochs: int = 40
    finetune_epochs: int = 15
    train_batch_queries: int = 32
    benchmark_batch_queries: int = 8
    threads: int = 1
    learning_rate: float = 0.002
    temperature: float = 2.0
    kd_alpha: float = 0.6
    keep_fraction: float = 0.5
    warmup: int = 50
    repetitions: int = 200

    def validate(self) -> None:
        for name in ("epochs", "finetune_epochs", "train_batch_queries", "benchmark_batch_queries", "threads", "warmup", "repetitions"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.temperature <= 0 or not 0 <= self.kd_alpha <= 1 or self.learning_rate <= 0:
            raise ValueError("invalid optimizer or distillation configuration")


def supervised_loss(scores: torch.Tensor, grades: torch.Tensor) -> torch.Tensor:
    target = torch.softmax(grades.float() * 1.5, dim=-1)
    return -(target * torch.log_softmax(scores, dim=-1)).sum(-1).mean()


def distillation_loss(scores: torch.Tensor, teacher_scores: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.kl_div(
        torch.log_softmax(scores / temperature, dim=-1),
        torch.softmax(teacher_scores.detach() / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature ** 2


def train_model(model: Reranker, train: QueryDataset, validation: QueryDataset, config: ExperimentConfig, epochs: int, teacher: nn.Module | None = None) -> tuple[Reranker, dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(config.seed + 10)
    best_score = float("-inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    history = []
    # Precomputed only on training examples; the teacher never receives test labels.
    teacher_logits = None
    if teacher is not None:
        teacher.eval()
        with torch.inference_mode():
            teacher_logits = teacher(train.features).clone()
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train.query_ids), generator=generator)
        running_loss = 0.0
        for indices in order.split(config.train_batch_queries):
            optimizer.zero_grad(set_to_none=True)
            scores = model(train.features[indices])
            loss = supervised_loss(scores, train.relevance[indices])
            if teacher_logits is not None:
                loss = (1 - config.kd_alpha) * loss + config.kd_alpha * distillation_loss(scores, teacher_logits[indices], config.temperature)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(indices)
        model.eval()
        with torch.inference_mode():
            validation_score = ranking_metrics(model(validation.features), validation.relevance)["ndcg_at_10"]
        history.append({"epoch": epoch + 1, "training_loss": running_loss / len(train.query_ids), "validation_ndcg_at_10": validation_score})
        if validation_score > best_score:
            best_score = validation_score
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
    model.load_state_dict(best_state)
    return model.eval(), {"seconds": time.perf_counter() - started, "selected_epoch": best_epoch, "best_validation_ndcg_at_10": best_score, "epochs": history}


def benchmark(model: nn.Module, features: torch.Tensor, warmup: int, repetitions: int) -> dict:
    """Resident tensor -> scores only. No data loading, preprocessing, or RPC timing."""
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(features)
        samples = []
        for _ in range(repetitions):
            started = time.perf_counter_ns()
            model(features)
            samples.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "scope": "one CPU forward pass on pre-normalized resident tensors; no network or preprocessing",
        "batch_queries": features.shape[0],
        "candidates_per_query": features.shape[1],
        "candidate_rows": features.shape[0] * features.shape[1],
        "warmup": warmup,
        "repetitions": repetitions,
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "samples_ms": samples,
    }


def score_query_batches(model: nn.Module, features: torch.Tensor, batch_queries: int) -> torch.Tensor:
    """Keep int8 quality evaluation at the deployment workload's activation scale scope."""
    if batch_queries < 1:
        raise ValueError("batch_queries must be positive")
    with torch.inference_mode():
        return torch.cat([model(batch.contiguous()) for batch in features.split(batch_queries)])


def run_experiment(config: ExperimentConfig, report_path: Path, checkpoint_dir: Path) -> dict:
    config.validate()
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    original = split_queries(make_dataset(config.queries, config.candidates, config.seed), config.seed)
    mean, std = fit_standardizer(original["train"])
    splits = {name: standardize(data, mean, std) for name, data in original.items()}
    train, validation, test = (splits[name] for name in ("train", "validation", "test"))
    teacher, teacher_training = train_model(Reranker(hidden=(256, 128)), train, validation, config, config.epochs)
    # Same student initialization and minibatch order, isolating the loss change.
    torch.manual_seed(config.seed + 1)
    initial_student = Reranker(hidden=(64, 32))
    supervised, supervised_training = train_model(deepcopy(initial_student), train, validation, config, config.epochs)
    distilled, distilled_training = train_model(deepcopy(initial_student), train, validation, config, config.epochs, teacher)
    pruned, pruning = prune_hidden_channels(distilled, config.keep_fraction)
    pruned, pruning_training = train_model(pruned, train, validation, config, config.finetune_epochs, teacher)
    int8, engine = quantize_int8(pruned)
    models = {
        "teacher_fp32": (teacher, None, teacher_training),
        "student_supervised_fp32": (supervised, None, supervised_training),
        "student_distilled_fp32": (distilled, None, distilled_training),
        "student_pruned_fp32": (pruned, None, pruning_training),
        "student_pruned_int8": (int8, engine, None),
    }
    batch = test.features[:config.benchmark_batch_queries].contiguous()
    variants = {}
    for name, (model, variant_engine, training) in models.items():
        architecture = teacher.architecture() if name == "teacher_fp32" else pruned.architecture() if "pruned" in name else distilled.architecture()
        checkpoint = checkpoint_dir / f"{name}.pt"
        stored_bytes = save_checkpoint(model, checkpoint, architecture, mean, std, variant_engine)
        quality = ranking_metrics(score_query_batches(model, test.features, config.benchmark_batch_queries), test.relevance)
        quality["evaluation_batch_queries"] = config.benchmark_batch_queries
        variants[name] = {
            "architecture": architecture,
            "float_equivalent_parameters": sum(p.numel() for p in (pruned if variant_engine else model).parameters()),
            "checkpoint_bytes": stored_bytes,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "quantized_engine": variant_engine,
            "test_metrics": quality,
            "latency": benchmark(model, batch, config.warmup, config.repetitions),
            "training": training,
        }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "quantized_engine": engine, "supported_quantized_engines": list(torch.backends.quantized.supported_engines),
            "device": "cpu", "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_build": torch.__config__.show(),
        },
        "dataset": {
            "kind": "seeded synthetic grouped ranking; no real users, clicks, or product data",
            "feature_groups": ["user_8", "query_8", "item_8", "user_item_product_8", "query_item_product_8"],
            "grade_rule": "top 12.5% grade 3, next 12.5% grade 2, next 25% grade 1, remaining grade 0 per query",
            "relevant_for_recall": "grade >= 2",
            "normalizer_fit_split": "train",
            "split_query_ids": {name: data.query_ids.tolist() for name, data in splits.items()},
            "split_counts": {name: len(data.query_ids) for name, data in splits.items()},
        },
        "pruning": pruning,
        "variants": variants,
        "limitations": [
            "Synthetic task, not evidence of e-commerce relevance, production traffic, or LLM compression.",
            "Validation selects checkpoints; test metrics are computed once after all training.",
            "A single seed and one CPU benchmark run; latency distributions are repetitions, not independent host runs.",
            "Fixed-order variant timing can retain host drift; small CPU kernels often make quantization slower.",
            "Checkpoint bytes include metadata and serialization overhead, not process RSS or GPU memory.",
            "No claim that distillation, pruning, or int8 improves every quality or latency metric.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def summary_table(report: dict) -> str:
    lines = ["| Variant | NDCG@10 | Recall@10 | Checkpoint KiB | p50 ms | p95 ms |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, result in report["variants"].items():
        quality, latency = result["test_metrics"], result["latency"]
        lines.append(f"| {name} | {quality['ndcg_at_10']:.4f} | {quality['recall_at_10']:.4f} | {result['checkpoint_bytes'] / 1024:.2f} | {latency['p50_ms']:.4f} | {latency['p95_ms']:.4f} |")
    return "\n".join(lines)
