import matplotlib.pyplot as plt
import pandas as pd

from analysis.plotting import save_figure
from analysis.results import write_json


def plot_long_tail(interactions, item_name, ranking_label, filename):
    sorted_interactions = interactions.sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    plt.plot(sorted_interactions.values)
    plt.yscale('log')
    plt.xlabel(f'{item_name}s Ranked by {ranking_label}')
    plt.ylabel('Number of Ratings')
    plt.title(f'{item_name} {ranking_label} Long Tail')
    plt.tight_layout()
    save_figure(filename)
    plt.show()


def explore_ratings(ratings):
    user_count = ratings['user_id'].nunique()
    movie_count = ratings['movie_id'].nunique()
    rating_count = len(ratings)
    density = rating_count / (user_count * movie_count)

    print('==================================')
    print('DATASET STATISTICS')
    print('==================================')
    print(f'Users     : {user_count}')
    print(f'Movies    : {movie_count}')
    print(f'Ratings   : {rating_count}')
    print(f'Density   : {density:.2%}')
    print(f'Sparsity  : {1 - density:.2%}')

    user_interactions = ratings.groupby('user_id').size()
    print('\nUSER INTERACTION STATISTICS')
    print(user_interactions.describe())
    plot_long_tail(
        user_interactions,
        'User',
        'Activity',
        'user_activity_long_tail.png',
    )

    movie_interactions = ratings.groupby('movie_id').size()
    print('\nMOVIE INTERACTION STATISTICS')
    print(movie_interactions.describe())
    plot_long_tail(
        movie_interactions,
        'Movie',
        'Popularity',
        'movie_popularity_long_tail.png',
    )

    rating_distribution = ratings['rating'].value_counts().sort_index()
    print('\nRATING DISTRIBUTION')
    print(rating_distribution)
    rating_distribution.plot(kind='bar', figsize=(6, 4))
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.title('Rating Distribution')
    plt.tight_layout()
    save_figure('rating_distribution.png')
    plt.show()

    ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
    print('\nTIME SPAN')
    print(ratings['datetime'].min())
    print(ratings['datetime'].max())

    daily_activity = (
        ratings.set_index('datetime')
        .resample('D')
        .size()
    )
    write_json('explore_ratings.json', {
        'dataset': {
            'users': int(user_count),
            'movies': int(movie_count),
            'ratings': int(rating_count),
            'density': float(density),
            'sparsity': float(1 - density),
        },
        'user_interactions': {
            key: float(value)
            for key, value in user_interactions.describe().items()
        },
        'movie_interactions': {
            key: float(value)
            for key, value in movie_interactions.describe().items()
        },
        'rating_distribution': {
            str(int(rating)): int(count)
            for rating, count in rating_distribution.items()
        },
        'time_span': {
            'start': ratings['datetime'].min().isoformat(),
            'end': ratings['datetime'].max().isoformat(),
        },
        'daily_activity': {
            day.strftime('%Y-%m-%d'): int(count)
            for day, count in daily_activity.items()
        },
    })
    daily_activity.plot(figsize=(10, 5))
    plt.xlabel('Time')
    plt.ylabel('Number of Ratings')
    plt.title('Rating Activity Over Time')
    plt.tight_layout()
    save_figure('rating_activity_over_time.png')
    plt.show()
    return daily_activity