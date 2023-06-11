import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
from scipy.optimize import curve_fit
from scipy.ndimage import filters


def smooth_time_series(time_series, smooth_method, smooth_parameter):
    time_series = pd.Series(time_series)
    if smooth_method == 'moving_average':
        smoothed = time_series.rolling(window=smooth_parameter).mean()
    elif smooth_method == 'exponential_moving_average':
        smoothed = time_series.ewm(alpha=smooth_parameter).mean()
    elif smooth_method == 'gaussian':
        smoothed = pd.Series(filters.gaussian_filter(time_series, smooth_parameter))
    elif smooth_method == 'polynomial':
        x_values = np.arange(len(time_series))
        poly_coeffs = np.polyfit(x_values, time_series, smooth_parameter)
        poly_fit = np.poly1d(poly_coeffs)
        smoothed = pd.Series(poly_fit(x_values))
    elif smooth_method == 'exponential_decay':
        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        x_values = np.arange(len(time_series))
        popt_exp_decay, _ = curve_fit(exp_decay, x_values, time_series, p0=(1, 0.1, 1))
        smoothed = exp_decay(x_values, *popt_exp_decay)
    elif smooth_method == 'exponential_decay_2':
        def exp_decay(x, a, b):
            return a * (1 - np.exp(-b * x))
        
        x_values = np.arange(len(time_series))
        popt_exp_decay, _ = curve_fit(exp_decay, x_values, time_series, p0=(1, 0.1))
        smoothed = exp_decay(x_values, *popt_exp_decay)
    else:
        raise ValueError("Invalid smoothing method. Available methods are 'moving_average', 'exponential_moving_average', 'gaussian', and 'polynomial'.")

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

def compute_a_direct_log(weights_by_rank_df, smooth_method='exponential_moving_average', smooth_parameter=0.01, smooth_a = False):
    a_bar = compute_a_bar_direct_log(weights_by_rank_df=weights_by_rank_df)
    smoothed_a_bar = smooth_time_series(a_bar, smooth_method=smooth_method, smooth_parameter=smooth_parameter)
    a = compute_a_given_a_bar(smoothed_a_bar)
    if smooth_a:
        a = smooth_time_series(a, smooth_method=smooth_method, smooth_parameter=smooth_parameter)
    return np.array(a)

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

def estimate_a(weights_by_rank_df, method='direct_log', smooth_method='exponential_moving_average', smooth_parameter=0.01, smooth_a=False):
    if method == 'direct_log':
        return compute_a_direct_log(weights_by_rank_df, smooth_method=smooth_method, smooth_parameter=smooth_parameter, smooth_a=smooth_a)
    elif method == 'moment_matching':
        return compute_a_moment_matching(weights_by_rank_df, smooth_method=smooth_method, smooth_parameter=smooth_parameter)
        

def get_a_given_rank(a, rank):
    return a[rank-1]


def plot_a(vector, data_path = None):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(vector)
    ax.set_title('a estimation')
    ax.set_xlabel("k")
    ax.set_ylabel("a_k")

    if data_path is None:
        plt.show()
    else:
        plt.savefig(f'{data_path}.png', dpi=300)




if __name__ == "__main__":
    from dataframe_construction import *

    df = Data(datapath='data/')

    bar_a = compute_a_bar_direct_log(df.weights_by_rank)
    plot_a(bar_a)

    a = estimate_a(df.weights_by_rank)#, smooth_method='moving_average', smooth_parameter=30)
    plot_a(a)


