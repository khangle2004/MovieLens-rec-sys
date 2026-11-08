import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
sys.path.append(ROOT)

from datasets import dataset_processing
from analysis import explore_ratings, save_figure
from models import (
    build_adjusted_cosine_similarity,
    build_cosine_similarity,
    build_jaccard_similarity,
    build_pearson_similarity,
    build_popularity_all,
    build_popularity_positive,
    build_user_item_matrix,
    evaluate,
    recommend_knn,
    recommend_popular,
)

POSITIVE_RATING = 4
K_VALUES = [1, 10, 20, 50, 100, 200, 500]


def load_ratings():
    ratings_path = os.path.join(ROOT, 'datasets', 'ratings.dat')
    return dataset_processing.Rating_Dataset(ratings_path).dataset


def split_by_time(ratings, daily_activity):
    first_cutoff = ratings['timestamp'].quantile(0.8)
    second_cutoff = ratings['timestamp'].quantile(0.9)

    train = ratings[ratings['timestamp'] < first_cutoff]
    validation = ratings[
        (ratings['timestamp'] >= first_cutoff)
        & (ratings['timestamp'] < second_cutoff)
    ]
    test = ratings[ratings['timestamp'] >= second_cutoff]

    for split_name, split in (
        ('TRAIN', train),
        ('VALIDATION', validation),
        ('TEST', test),
    ):
        print(f'\n{split_name}')
        print(split['datetime'].min())
        print(split['datetime'].max())
        print(len(split))

    axis = daily_activity.plot(figsize=(12, 5))
    train_end = train['datetime'].max()
    validation_end = validation['datetime'].max()
    axis.axvline(
        train_end,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Train End ({train_end.date()})',
    )
    axis.axvline(
        validation_end,
        color='green',
        linestyle='--',
        linewidth=2,
        label=f'Validation End ({validation_end.date()})',
    )
    plt.xlabel('Time')
    plt.ylabel('Number of Ratings')
    plt.title('Temporal Split')
    plt.legend()
    plt.tight_layout()
    save_figure('temporal_split.png')
    plt.show()
    return train, validation, test


def print_positive_feedback_stats(splits):
    stats = pd.DataFrame({
        'Split': ['Train', 'Validation', 'Test'],
        'Original': [len(split) for split in splits],
        f'Positive (>={POSITIVE_RATING})': [
            len(split[split['rating'] >= POSITIVE_RATING])
            for split in splits
        ],
    })
    positive_column = f'Positive (>={POSITIVE_RATING})'
    stats['Removed'] = stats['Original'] - stats[positive_column]
    stats['Retained (%)'] = 100 * stats[positive_column] / stats['Original']
    stats['Removed (%)'] = 100 * stats['Removed'] / stats['Original']

    print('\n==================================')
    print('POSITIVE FEEDBACK FILTERING')
    print('==================================')
    print(stats.round(2))


def evaluate_popularity_baseline(
    train,
    users,
    relevant_items,
    total_items,
    positive_only=False,
):
    if positive_only:
        name = 'POPULARITY (POSITIVE RATINGS)'
        popularity = build_popularity_positive(train)
    else:
        name = 'POPULARITY (ALL RATINGS)'
        popularity = build_popularity_all(train)

    print(f'\n==================================\n{name}\n==================================')
    result = evaluate(
        lambda user: recommend_popular(user, train, popularity),
        users,
        relevant_items,
        total_items,
    )
    print(result)


def evaluate_item_knn(train_positive, users, relevant_items, total_items):
    user_item = build_user_item_matrix(train_positive)
    similarities = {
        'Cosine': build_cosine_similarity(user_item),
        'Pearson': build_pearson_similarity(user_item),
        'Jaccard': build_jaccard_similarity(user_item),
        'Adjusted Cosine': build_adjusted_cosine_similarity(train_positive),
    }

    for name, similarity_matrix in similarities.items():
        print(f'\n==================================\n{name}\n==================================')
        for k in K_VALUES:
            result = evaluate(
                lambda user: recommend_knn(
                    user,
                    train_positive,
                    similarity_matrix,
                    k=k,
                ),
                users,
                relevant_items,
                total_items,
            )
            print(f'k={k}', result)


def main():
    ratings = load_ratings()
    daily_activity = explore_ratings(ratings)
    train, validation, test = split_by_time(ratings, daily_activity)

    train_positive = train[train['rating'] >= POSITIVE_RATING]
    validation_positive = validation[
        validation['rating'] >= POSITIVE_RATING
    ]
    test_positive = test[test['rating'] >= POSITIVE_RATING]
    relevant_items = (
        validation_positive.groupby('user_id')['movie_id']
        .apply(set)
        .to_dict()
    )
    users = validation['user_id'].unique()
    total_movies = ratings['movie_id'].nunique()

    evaluate_popularity_baseline(
        train,
        users,
        relevant_items,
        total_movies,
    )
    print_positive_feedback_stats((train, validation, test))
    evaluate_popularity_baseline(
        train,
        users,
        relevant_items,
        total_movies,
        positive_only=True,
    )
    evaluate_item_knn(
        train_positive,
        users,
        relevant_items,
        total_movies,
    )


if __name__ == '__main__':
    main()