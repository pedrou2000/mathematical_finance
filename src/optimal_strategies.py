import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_weights_optimal_pure_open(N, a, weights_by_rank):
    a_N = a[:N]  # Take the first N elements of the vector 'a'
    a_1_N = a_N.sum()  # Compute the sum of the first N elements
    term = 2 - a_1_N  # Compute the term (2 - 𝑎̅_1^N)
    sum_topN_market_weights = weights_by_rank.iloc[:, :N].sum(axis=1)
    topN_weights_by_rank = weights_by_rank.iloc[:, :N].copy()
    Yk_divided_by_sumY1N = topN_weights_by_rank.div(sum_topN_market_weights, axis=0)
    optimal_strategy_percentage = 0.5 * (a_N + Yk_divided_by_sumY1N.mul(term, axis=0))
    return optimal_strategy_percentage

def compute_weights_optimal_mixed_open(N, a, weights_by_rank):
    a_N = a[:N]  # Take the first N elements of the vector 'a'
    # a_1_d = 0  # Compute the sum of all elements in 'a'
    a_1_d = a.sum()  # Compute the sum of all elements in 'a'
    a_N_plus_1_d = a[N:].sum()  # Compute the sum of the elements from a_{N+1} to a_d
    term = 2 - a_1_d  # Compute the term (2 - 𝑎̅_1^d)

    # Compute the optimal strategy for k = 1, ..., N
    topN_weights_by_rank = weights_by_rank.iloc[:, :N].copy()
    optimal_strategy_percentage_topN = 0.5 * (a_N + topN_weights_by_rank.mul(term, axis=0))

    # Compute the optimal strategy for k = N+1, ..., d
    remaining_weights_by_rank = weights_by_rank.iloc[:, N:].copy()
    Y_N_plus_1_d = remaining_weights_by_rank.sum(axis=1)
    optimal_strategy_percentage_remaining = 0.5 * remaining_weights_by_rank.mul((a_N_plus_1_d / Y_N_plus_1_d + term), axis=0)

    # Combine the two results
    optimal_strategy_percentage = pd.concat([optimal_strategy_percentage_topN, optimal_strategy_percentage_remaining], axis=1)

    return optimal_strategy_percentage

def compute_weights_optimal_open(setting, N, a, weights_by_rank):
    if setting == 'pure':
        return compute_weights_optimal_pure_open(N=N, a=a, weights_by_rank=weights_by_rank)
    elif setting == 'mixed':
        return compute_weights_optimal_mixed_open(N=N, a=a, weights_by_rank=weights_by_rank)
    else:
        print('Not Implemented')


def plot_strategy(strategy, date, data_path = None):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(strategy.loc[date])
    ax.set_title('Weights of the Optimal Strategy')
    ax.set_xlabel("Rank")
    ax.set_ylabel("Proportion of Wealth")
    
    if data_path is None:
        plt.show()
    else:
        plt.savefig(f'{data_path}.png', dpi=300)
        



if __name__ == "__main__":
    from dataframe_construction import *
    from a_estimation import *

    N = 1000
    date = '2000-03-31'

    df = Data(datapath='data/')
    a = estimate_a(df.weights_by_rank)
    strategy = compute_weights_optimal_open(setting='pure', N=N, a=a, weights_by_rank=df.weights_by_rank)

    plot_strategy(strategy, date)


