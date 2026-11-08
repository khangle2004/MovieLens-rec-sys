import numpy as np

def precision_at_k(recommended, relevant, k):
    return len(set(recommended[:k]) & set(relevant)) / k

def recall_at_k(recommended, relevant, k):
    return len(set(recommended[:k]) & set(relevant)) / len(relevant)

def ndcg_at_k(
    recommended,
    relevant,
    k=10
):

    if len(recommended) == 0:

        return 0

    rel = [
        1 if item in relevant else 0
        for item in recommended[:k]
    ]

    dcg = np.sum([
        r / np.log2(i + 2)
        for i, r in enumerate(rel)
    ])

    idcg = np.sum([
        1 / np.log2(i + 2)
        for i in range(
            min(len(relevant), k)
        )
    ])

    if idcg == 0:

        return 0

    return dcg / idcg



def catalogue_coverage(
    recommendations,
    total_items
):

    recommended_items = set()

    for recs in recommendations.values():

        recommended_items.update(recs)

    return (
        len(recommended_items)
        / total_items
    )
