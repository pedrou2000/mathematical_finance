import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Backtest:
    def __init__(self, strategy, weights_permno_by_rank, returns_by_permno):
        self.strategy = strategy 
        self.weights_permno_by_rank = weights_permno_by_rank
        self.returns_by_permno = returns_by_permno
        self.dates = self.strategy.index.tolist()
        self.interests = []
        self.cumulative_interests = []

    def interest_optimal_one_period(self, date):
        try:
            total_interest = 0
            for rank in list(self.strategy.columns):
                percentage = self.strategy.loc[date, rank]
                permno = self.weights_permno_by_rank.loc[date, rank][1]
                ret = self.returns_by_permno.loc[date, permno]
                interest = 1 + ret
                total_interest += (percentage * interest)
            return total_interest
        except KeyError:
            print(f"Date {date} not found in DataFrame index")
            return None

    def run(self):
        cumulative_interest = 1 
        for date in self.dates:
            interest = self.interest_optimal_one_period(date)
            self.interests.append(interest)
            cumulative_interest *= interest
            self.cumulative_interests.append(cumulative_interest)
    

    def plot_interests(self):
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(self.dates, self.interests)
        ax.set_xlabel("Date")
        ax.set_ylabel("Interests")
        ax.set_title("Interests over Time")
        plt.show()
        plt.close()
    
    def plot_cumulative_interests(self):
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(self.dates, self.cumulative_interests)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Interests")
        ax.set_title("Cumulative Interests over 1973-2022")
        plt.show()
        plt.close()
    
    def plot_cumulative_interests_years(self, years):
        months = 12 * years
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(self.dates[0:months], self.cumulative_interests[0:months])
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Interests")
        ax.set_title("Cumulative Interests during the first " + str(years) + " years.")
        plt.show()
        plt.close()



if __name__ == "__main__":
    from dataframe_construction import *
    from a_estimation import *
    from optimal_strategies import *

    N = 10
    date = '2000-03-31'

    df = MonthlyData(datapath='data/')
    a = estimate_a(df.weights_by_rank)
    strategy = compute_weights_optimal_open(setting='pure', N=N, a=a, weights_by_rank=df.weights_by_rank)

    backtest = Backtest(strategy=strategy, weights_permno_by_rank=df.weights_permno_by_rank,
                        returns_by_permno=df.returns_by_permno)
    
    backtest.run()
    backtest.plot_interests()
    backtest.plot_cumulative_interests()
