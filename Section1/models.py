import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

from evaluation.metric import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    catalogue_coverage
)


# ====================================================
# USER-ITEM MATRIX
# ====================================================

def build_user_item_matrix(train):

    return train.pivot_table(
        index='user_id',
        columns='movie_id',
        values='rating',
        fill_value=0
    )


# ====================================================
# SIMILARITIES
# ====================================================

def build_cosine_similarity(user_item):

    sim = cosine_similarity(
        user_item.T
    )

    return pd.DataFrame(
        sim,
        index=user_item.columns,
        columns=user_item.columns
    )


def build_pearson_similarity(user_item):

    return (
        user_item
        .corr(method='pearson')
        .fillna(0)
    )


def build_jaccard_similarity(user_item):

    binary = (
        user_item > 0
    ).astype(bool)

    sim = 1 - pairwise_distances(
        binary.to_numpy().T,
        metric='jaccard'
    )

    return pd.DataFrame(
        sim,
        index=user_item.columns,
        columns=user_item.columns
    )

def build_adjusted_cosine_similarity(train):

    user_means = (
        train
        .groupby('user_id')['rating']
        .mean()
    )

    centered = train.copy()

    centered['adjusted_rating'] = (
        centered['rating']
        - centered['user_id'].map(user_means)
    )

    user_item = centered.pivot_table(
        index='user_id',
        columns='movie_id',
        values='adjusted_rating',
        fill_value=0
    )

    sim = cosine_similarity(
        user_item.T
    )

    return pd.DataFrame(
        sim,
        index=user_item.columns,
        columns=user_item.columns
    )

# ====================================================
# POPULARITY
# ====================================================

def build_popularity_all(train):

    return (
        train
        .groupby('movie_id')
        .size()
        .sort_values(ascending=False)
    )


def build_popularity_positive(train):

    return (
        train[train['rating'] >= 4]
        .groupby('movie_id')
        .size()
        .sort_values(ascending=False)
    )


# ====================================================
# POPULARITY RECOMMENDER
# ====================================================

def recommend_popular(
    user_id,
    train,
    popularity,
    n=10
):

    watched = set(
        train[
            train['user_id'] == user_id
        ]['movie_id']
    )

    recommendations = []

    for movie in popularity.index:

        if movie not in watched:

            recommendations.append(movie)

        if len(recommendations) >= n:

            break

    return recommendations


# ====================================================
# ITEM KNN
# ====================================================

def recommend_knn(
    user_id,
    train,
    similarity_matrix,
    k=20,
    n=10
):

    watched = set(
        train[
            train['user_id'] == user_id
        ]['movie_id']
    )

    if len(watched) == 0:

        return []

    scores = {}

    for movie in watched:

        if movie not in similarity_matrix.columns:

            continue

        neighbors = (
            similarity_matrix[movie]
            .sort_values(ascending=False)
            .iloc[1:k+1]
        )

        for neighbor, sim in neighbors.items():

            if neighbor in watched:

                continue

            scores[neighbor] = (
                scores.get(neighbor, 0)
                + sim
            )

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        movie
        for movie, _
        in ranked[:n]
    ]



# ====================================================
# EVALUATION
# ====================================================

def evaluate(
    recommender,
    users,
    relevant_items,
    total_items
):

    precisions = []
    recalls = []
    ndcgs = []

    recommendations = {}

    for user in users:

        if user not in relevant_items:

            continue

        relevant = relevant_items[user]

        if len(relevant) == 0:

            continue

        recs = recommender(user)

        if len(recs) == 0:

            continue

        recommendations[user] = recs

        precisions.append(
            precision_at_k(
                recs,
                relevant,
                10
            )
        )

        recalls.append(
            recall_at_k(
                recs,
                relevant,
                10
            )
        )

        ndcgs.append(
            ndcg_at_k(
                recs,
                relevant,
                10
            )
        )

    if len(precisions) == 0:

        return {
            'Precision@10': 0,
            'Recall@10': 0,
            'NDCG@10': 0,
            'Coverage': 0
        }

    return {
        'Precision@10': np.mean(precisions),
        'Recall@10': np.mean(recalls),
        'NDCG@10': np.mean(ndcgs),
        'Coverage': catalogue_coverage(
            recommendations,
            total_items
        )
    }