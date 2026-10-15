import math

import pytest
import torch

from rank_compression_lab.data import fit_standardizer, make_dataset, split_queries, standardize
from rank_compression_lab.metrics import ranking_metrics


def test_grouped_split_is_deterministic_and_disjoint():
    data = make_dataset(80, 32, seed=17)
    second = make_dataset(80, 32, seed=17)
    assert torch.equal(data.features, second.features)
    splits = split_queries(data, seed=12)
    again = split_queries(data, seed=12)
    ids = [set(part.query_ids.tolist()) for part in splits.values()]
    assert sum(len(group) for group in ids) == 80
    assert len(set.union(*ids)) == 80
    for name, part in splits.items():
        assert torch.equal(part.query_ids, again[name].query_ids)
        assert part.features.shape[1:] == (32, 40)
        assert torch.equal(part.features, data.features[part.query_ids])
    assert all(not a & b for i, a in enumerate(ids) for b in ids[i + 1:])
    mean, std = fit_standardizer(splits["train"])
    normalized = standardize(splits["train"], mean, std).features.flatten(0, 1)
    assert torch.allclose(normalized.mean(0), torch.zeros(40), atol=1e-6)


def test_ndcg_uses_graded_gains_and_recall_uses_threshold_two():
    # Model puts grade 0 first, grade 3 second, grade 2 outside top 10.
    grades = torch.tensor([[0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2]])
    scores = torch.arange(11, 0, -1).unsqueeze(0).float()
    metrics = ranking_metrics(scores, grades)
    expected_dcg = 7 / math.log2(3)
    expected_idcg = 7 + 3 / math.log2(3)
    assert metrics["ndcg_at_10"] == pytest.approx(expected_dcg / expected_idcg)
    assert metrics["recall_at_10"] == 0.5
    perfect = ranking_metrics(grades.float(), grades)
    assert perfect["ndcg_at_10"] == pytest.approx(1.0)
    assert perfect["recall_at_10"] == 1.0


def test_metrics_macro_average_and_zero_relevance():
    grades = torch.tensor([[3, 0, 0], [0, 0, 0]])
    result = ranking_metrics(torch.tensor([[3., 2., 1.], [3., 2., 1.]]), grades)
    assert result["ndcg_at_10"] == 0.5
    assert result["recall_at_10"] == 0.5
    assert result["per_query_ndcg_at_10"] == [1.0, 0.0]
    with pytest.raises(ValueError):
        ranking_metrics(torch.ones(3), torch.ones(3))
    with pytest.raises(ValueError):
        ranking_metrics(torch.tensor([[float("nan")]]), torch.ones(1, 1))


def test_metric_names_follow_requested_cutoff():
    result = ranking_metrics(torch.tensor([[3., 1., 2.]]), torch.tensor([[3, 2, 0]]), k=1)
    assert result["ndcg_at_1"] == 1.0
    assert result["recall_at_1"] == 0.5
    assert "ndcg_at_10" not in result
