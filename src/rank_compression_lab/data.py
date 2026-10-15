"""Seeded query groups. Labels are generated before splitting; no fitted scorer is used."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class QueryDataset:
    features: torch.Tensor  # [query, candidate, feature]
    relevance: torch.Tensor  # integer grades 0..3
    query_ids: torch.Tensor

    def take(self, indices: np.ndarray) -> "QueryDataset":
        return QueryDataset(self.features[indices], self.relevance[indices], self.query_ids[indices])


def make_dataset(queries: int = 960, candidates: int = 32, seed: int = 2026) -> QueryDataset:
    """Generate independent query groups with user/query/item features and interactions.

    Each query has a distinct user context. The unknown scoring function is nonlinear,
    with Gaussian noise, and grades are assigned by within-query rank. This is a
    controlled compression workload, not a simulation of real user click behavior.
    """
    if queries < 20 or candidates < 10:
        raise ValueError("need at least 20 queries and 10 candidates for a grouped top-10 evaluation")
    rng = np.random.default_rng(seed)
    user = rng.normal(size=(queries, 1, 8)).astype(np.float32)
    query = rng.normal(size=(queries, 1, 8)).astype(np.float32)
    item = rng.normal(size=(queries, candidates, 8)).astype(np.float32)
    user = np.broadcast_to(user, item.shape)
    query = np.broadcast_to(query, item.shape)
    user_item = user * item
    query_item = query * item
    features = np.concatenate((user, query, item, user_item, query_item), axis=-1)
    # Fixed domain function; train/validation/test all draw from the same distribution.
    latent = (
        0.6 * query_item.sum(-1)
        + 0.4 * user_item.sum(-1)
        + 1.2 * np.sin(item[..., 0] + query[..., 1])
        + 0.8 * np.maximum(item[..., 2] * user[..., 3], 0)
        - 0.5 * item[..., 4] ** 2
        + rng.normal(scale=0.15, size=(queries, candidates))
    )
    order = np.argsort(np.argsort(latent, axis=1, kind="stable"), axis=1, kind="stable")
    percentile = order / candidates
    grades = np.select((percentile >= 0.875, percentile >= 0.75, percentile >= 0.5), (3, 2, 1), 0)
    return QueryDataset(torch.from_numpy(features.copy()), torch.tensor(grades, dtype=torch.long), torch.arange(queries))


def split_queries(data: QueryDataset, seed: int = 2026) -> dict[str, QueryDataset]:
    """Split whole queries, retaining all candidates together, in 70/15/15 proportions."""
    indices = np.random.default_rng(seed).permutation(len(data.query_ids))
    train_end = int(len(indices) * 0.7)
    validation_end = int(len(indices) * 0.85)
    return {name: data.take(idx) for name, idx in (
        ("train", indices[:train_end]),
        ("validation", indices[train_end:validation_end]),
        ("test", indices[validation_end:]),
    )}


def fit_standardizer(train: QueryDataset) -> tuple[torch.Tensor, torch.Tensor]:
    flat = train.features.flatten(0, 1)
    return flat.mean(0), flat.std(0).clamp_min(1e-6)


def standardize(data: QueryDataset, mean: torch.Tensor, std: torch.Tensor) -> QueryDataset:
    return QueryDataset((data.features - mean) / std, data.relevance, data.query_ids)
