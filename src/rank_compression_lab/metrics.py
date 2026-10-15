import torch


def ranking_metrics(scores: torch.Tensor, relevance: torch.Tensor, k: int = 10) -> dict:
    """Macro query average; gains 2**grade-1, relevant means grade >= 2.

    Tied scores retain candidate order. A query without relevant documents receives
    recall 0; a query with zero ideal DCG receives NDCG 0.
    """
    if scores.ndim != 2 or scores.shape != relevance.shape:
        raise ValueError("scores and relevance must have identical [query, candidate] shape")
    if scores.shape[0] == 0 or scores.shape[1] == 0 or k < 1:
        raise ValueError("nonempty query groups and positive k are required")
    if not torch.isfinite(scores).all() or not torch.isfinite(relevance).all() or (relevance < 0).any():
        raise ValueError("scores must be finite and relevance must be nonnegative")
    requested_k = k
    k = min(k, scores.shape[1])
    indices = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
    ranked = torch.gather(relevance, 1, indices).float()
    ideal = torch.sort(relevance, dim=1, descending=True).values[:, :k].float()
    discount = torch.log2(torch.arange(k, device=scores.device, dtype=torch.float32) + 2)
    dcg = ((2.0 ** ranked - 1) / discount).sum(1)
    idcg = ((2.0 ** ideal - 1) / discount).sum(1)
    ndcg = torch.where(idcg > 0, dcg / idcg.clamp_min(1e-12), 0)
    total_relevant = (relevance >= 2).sum(1)
    recall = (ranked >= 2).sum(1).float() / total_relevant.clamp_min(1)
    return {
        f"ndcg_at_{requested_k}": ndcg.mean().item(),
        f"recall_at_{requested_k}": recall.mean().item(),
        "queries": scores.shape[0],
        f"per_query_ndcg_at_{requested_k}": ndcg.tolist(),
        f"per_query_recall_at_{requested_k}": recall.tolist(),
    }
