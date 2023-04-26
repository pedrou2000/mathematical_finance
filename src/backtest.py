import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Backtest:
    def __init__(self, strategy, weights_permno_by_rank, returns_by_permno):
        self.strategy = strategy 
        self.weights_permno_by_rank = weights_permno_by_rank
        self.returns_by_permno = returns_by_permno
        self.dates = self.strategy.index.tolist()
        self.   s = []
        self.cumulative_rets = []
        self.first_date = strategy.index[0].date()
        self.last_date = strategy.index[-1].date()

    def ret_optimal_one_period(self, date):
        try:
            total_ret = 0
            for rank in list(self.strategy.columns):
                percentage = self.strategy.loc[date, rank]
                permno = self.weights_permno_by_rank.loc[date, rank][1]
                ret = self.returns_by_permno.loc[date, permno]
                ret = 1 + ret
                total_ret += (percentage * ret)
            return total_ret
        except KeyError:
            print(f"Date {date} not found in DataFrame index")
            return None

    def run(self):
        cumulative_ret = 1 
        for date in self.dates:
            ret = self.ret_optimal_one_period(date)
            self.rets.append(ret)
            cumulative_ret *= ret
            self.cumulative_rets.append(cumulative_ret)
    

    def plot_rets(self):
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(self.dates, self.rets)
        ax.set_xlabel("Date")
        ax.set_ylabel("rets")
        ax.set_title("rets between " + str(self.first_date) + " and " + str(self.last_date))
        plt.show()
        plt.close()
    
    def plot_cumulative_rets(self):
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(self.dates, self.cumulative_rets)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative rets")
        ax.set_title("Cumulative rets between " + str(self.first_date) + " and " + str(self.last_date))
        plt.show()
        plt.close()
    
    def plot_cumulative_rets_years(self, years):
        months = 12 * years
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(self.dates[0:months], self.cumulative_rets[0:months])
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative rets")
        ax.set_title("Cumulative rets during the first " + str(years) + " years.")
        plt.show()
        plt.close()



if __name__ == "__main__":
    from dataframe_construction import *
    from a_estimation import *
    from optimal_strategies import *

    N = 10
    date = '2000-03-31'

    df = Data(datapath='data/')
    a = estimate_a(df.weights_by_rank)
    strategy = compute_weights_optimal_open(setting='pure', N=N, a=a, weights_by_rank=df.weights_by_rank)

    backtest = Backtest(strategy=strategy, weights_permno_by_rank=df.weights_permno_by_rank,
                        returns_by_permno=df.returns_by_permno)
    
    backtest.run()
    backtest.plot_rets()
    backtest.plot_cumulative_rets()
