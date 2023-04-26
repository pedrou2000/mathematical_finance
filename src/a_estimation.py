import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
import scipy.ndimage.filters as filters


def smooth_time_series(time_series, smooth_method, smooth_parameter):
    time_series = pd.Series(time_series)
    if smooth_method == 'moving_average':
        smoothed = time_series.rolling(window=smooth_parameter).mean()
    elif smooth_method == 'exponential_moving_average':
        smoothed = time_series.ewm(alpha=smooth_parameter).mean()
    elif smooth_method == 'gaussian':
        smoothed = pd.Series(filters.gaussian_filter(time_series, smooth_parameter))
    else:
        raise ValueError("Invalid smoothing method. Available methods are 'moving_average', 'exponential_moving_average', 'gaussian', 'savitzky_golay', and 'lowess'.")

    return np.array(smoothed)


def compute_a_bar_direct_log(weights_by_rank_df):
    log_weights_by_rank = np.log(weights_by_rank_df)
    log_diff = -log_weights_by_rank.diff(axis=1).iloc[:, 1:]
    mean_diff = log_diff.mean(axis=0)
    a_bar = 1/mean_diff
    a_bar_1 = 0
    a_bar = np.insert(a_bar.values, 0, a_bar_1)
    return a_bar

def compute_a_given_a_bar(vector):
    ret = np.zeros(len(vector))
    ret[:-1] = -np.diff(vector)
    ret[-1] = vector[-1]
    return ret

def compute_a_direct_log(weights_by_rank_df, smooth_method='exponential_moving_average', smooth_parameter=0.01):
    a_bar = compute_a_bar_direct_log(weights_by_rank_df=weights_by_rank_df)
    smoothed_a_bar = smooth_time_series(a_bar, smooth_method=smooth_method, smooth_parameter=smooth_parameter)
    a = compute_a_given_a_bar(smoothed_a_bar)
    smoothed_a = smooth_time_series(a, smooth_method=smooth_method, smooth_parameter=smooth_parameter)
    return np.array(smoothed_a)

def compute_a_moment_matching(weights_by_rank_df, smooth_method='exponential_moving_average', smooth_parameter=0.01):
    # Compute Expectation of Market Weights
    y_k = weights_by_rank_df.mean(axis=0)
    y_k = np.array(y_k)

    a = np.zeros(len(y_k))

    # Calculate a_1
    a[0] = -1 - (y_k[1] / (y_k[0] - y_k[1]))

    # Calculate a_d
    a[-1] = y_k[-1] / (y_k[-2] - y_k[-1])

    # Calculate a_k for k = 2, ..., d - 1
    for k in range(1, len(y_k) - 1):
        a[k] = (y_k[k] / (y_k[k - 1] - y_k[k])) - (y_k[k + 1] / (y_k[k] - y_k[k + 1]))

    smoothed_a = smooth_time_series(a, smooth_method=smooth_method, smooth_parameter=smooth_parameter)

    return np.array(smoothed_a)

def estimate_a(weights_by_rank_df, method='direct_log', smooth_method='exponential_moving_average', smooth_parameter=0.01):
    if method == 'direct_log':
        return compute_a_direct_log(weights_by_rank_df, smooth_method=smooth_method, smooth_parameter=smooth_parameter)
    elif method == 'moment_matching':
        return compute_a_moment_matching(weights_by_rank_df, smooth_method=smooth_method, smooth_parameter=smooth_parameter)
        

def get_a_given_rank(a, rank):
    return a[rank-1]


def plot_a(vector):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(vector)
    ax.set_title('a given rank')
    ax.set_xlabel("rank")
    ax.set_ylabel("a")
    plt.show()



if __name__ == "__main__":
    from dataframe_construction import MonthlyData

    df = MonthlyData(datapath='data/')

    a = estimate_a(df.weights_by_rank, smooth_method='moving_average', smooth_parameter=30)
    plot_a(a)


