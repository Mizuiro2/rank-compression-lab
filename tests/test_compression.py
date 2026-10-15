from pathlib import Path

import pytest
import torch
from torch import nn

from rank_compression_lab.compression import load_checkpoint, prune_hidden_channels, quantize_int8, save_checkpoint
from rank_compression_lab.experiment import distillation_loss, score_query_batches
from rank_compression_lab.model import Reranker


def test_physical_pruning_matches_zeroed_hidden_activations():
    torch.manual_seed(123)
    model = Reranker(hidden=(16, 8)).eval()
    compact, info = prune_hidden_channels(model, 0.5)
    features = torch.randn(3, 10, 40)
    mask1, mask2 = torch.zeros(16), torch.zeros(8)
    mask1[info["kept_hidden1"]] = 1
    mask2[info["kept_hidden2"]] = 1
    with torch.inference_mode():
        first = torch.relu(model.fc1(features)) * mask1
        second = torch.relu(model.fc2(first)) * mask2
        masked_scores = model.out(second).squeeze(-1)
        torch.testing.assert_close(compact(features), masked_scores)
    assert compact.fc1.weight.shape == (8, 40)
    assert compact.fc2.weight.shape == (4, 8)
    assert compact.out.weight.shape == (1, 4)
    assert sum(p.numel() for p in compact.parameters()) < sum(p.numel() for p in model.parameters())


def test_no_pruning_is_equivalent_and_bad_fraction_is_rejected():
    model = Reranker()
    compact, _ = prune_hidden_channels(model, 1.0)
    features = torch.randn(2, 40)
    torch.testing.assert_close(model(features), compact(features))
    with pytest.raises(ValueError):
        prune_hidden_channels(model, 0)


@pytest.mark.parametrize("quantized", [False, True])
def test_checkpoint_reload_preserves_scores_and_metadata(tmp_path: Path, quantized: bool):
    torch.set_num_threads(1)
    model = Reranker().eval()
    architecture = model.architecture()
    engine = None
    if quantized:
        if set(torch.backends.quantized.supported_engines) <= {"none"}:
            pytest.skip("CPU PyTorch build without an int8 engine")
        model, engine = quantize_int8(model)
        assert isinstance(model.fc1, torch.ao.nn.quantized.dynamic.Linear)
        assert not isinstance(model.fc1, nn.Linear)
    inputs = torch.randn(3, 32, 40)
    path = tmp_path / "model.pt"
    count = save_checkpoint(model, path, architecture, torch.zeros(40), torch.ones(40), engine)
    restored, metadata = load_checkpoint(path)
    with torch.inference_mode():
        assert restored(inputs).shape == (3, 32)
        torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)
    assert count == path.stat().st_size
    assert metadata["quantized_engine"] == engine


def test_distillation_matches_querywise_kl_and_detaches_teacher():
    student = torch.tensor([[0.2, 1.0, -0.3], [0.1, 0.3, 0.9]], requires_grad=True)
    teacher = torch.tensor([[0.5, 1.8, -0.2], [0.2, 0.7, 0.5]], requires_grad=True)
    temperature = 2.0
    teacher_probabilities = torch.softmax(teacher.detach() / temperature, -1)
    manual = (teacher_probabilities * (teacher_probabilities.log() - torch.log_softmax(student / temperature, -1))).sum(-1).mean() * temperature ** 2
    actual = distillation_loss(student, teacher, temperature)
    torch.testing.assert_close(actual, manual)
    actual.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_batched_scoring_includes_last_partial_query_batch():
    model = Reranker().eval()
    features = torch.randn(11, 32, 40)
    scores = score_query_batches(model, features, 4)
    assert scores.shape == (11, 32)
    with torch.inference_mode():
        torch.testing.assert_close(scores, model(features))
