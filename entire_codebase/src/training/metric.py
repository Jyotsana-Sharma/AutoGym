from __future__ import annotations

import math

import pandas as pd


def compute_ranking_metric(
    frame: pd.DataFrame,
    *,
    group_key: str,
    label_column: str,
    score_column: str,
) -> dict[str, float]:
    ndcg_total = 0.0
    group_count = 0

    for _, group in frame.groupby(group_key, sort=False):
        labels = group.sort_values(score_column, ascending=False)[label_column].astype(float).tolist()
        ideal_labels = sorted(labels, reverse=True)

        dcg = 0.0
        ideal_dcg = 0.0
        for rank, label in enumerate(labels[:10]):
            dcg += ((2.0 ** label) - 1.0) / math.log2(rank + 2.0)
        for rank, label in enumerate(ideal_labels[:10]):
            ideal_dcg += ((2.0 ** label) - 1.0) / math.log2(rank + 2.0)

        ndcg_total += dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        group_count += 1

    return {f"ndcg_at_10": ndcg_total / group_count if group_count > 0 else 0.0}
