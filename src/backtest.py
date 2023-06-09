import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Backtest:
    def __init__(self, strategy, weights_permno_by_rank, returns_by_permno, risk_free_rate):
        self.strategy = strategy 
        self.weights_permno_by_rank = weights_permno_by_rank
        self.weights_by_rank = self.weights_permno_by_rank.applymap(lambda x: x[0])
        self.returns_by_permno = returns_by_permno
        self.dates = self.strategy.index.tolist()
        self.first_date = strategy.index[0].date()
        self.last_date = strategy.index[-1].date()
        self.risk_free_rate = risk_free_rate

        self.average_return_percentage = None
        self.average_return = None
        self.sd_return = None
        self.end_period_cumulative_wealth = None
        self.sharpe_ratio = None

        self.dX_t_div_X_t = None
        self.covariance_matrix = None
        self.flag_negative_return = False
    
    def set_strategy(self, new_strategy):
        self.strategy = new_strategy 
        self.dates = self.strategy.index.tolist()


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
        self.flag_negative_return = False
        self.rets = []
        self.cumulative_rets = []
        cumulative_ret = 1 

        for date in self.dates:
            ret = self.ret_optimal_one_period(date)
            self.rets.append(ret)
            cumulative_ret *= ret
            if cumulative_ret <= 0:
                self.flag_negative_return = True
            self.cumulative_rets.append(cumulative_ret)
        
        excess_returns = np.array(self.rets) - 1.0 - self.risk_free_rate
        self.average_return_percentage = round((np.mean(excess_returns))*100, 3)
        self.average_return = round(np.mean(excess_returns), 4)
        self.sd_return = round(np.std(excess_returns), 4)
        self.end_period_cumulative_wealth = round(self.cumulative_rets[-1], 4)
        self.sharpe_ratio = self.average_return / self.sd_return

        self.max_drawdown = np.max(np.maximum.accumulate(self.cumulative_rets) - self.cumulative_rets) # Check: /np.maximum.accumulate(self.cumulative_rets)
        self.calmar_ratio = self.average_return / self.max_drawdown

    def compute_dX_t_div_X_t(self):
        # Compute the percentage change between rows
        percentage_change = self.weights_by_rank.pct_change().fillna(0)
        self.dX_t_div_X_t = percentage_change
        
    def compute_covariance_matrix(self):
        # Compute the difference matrix
        diff_matrix = self.weights_by_rank.diff().fillna(0)

        # Compute the outer product of the difference matrix with itself for each row
        result = {}
        for index, row in diff_matrix.iterrows():
            outer_product = np.outer(row, row)
            result[index] = outer_product

        self.covariance_matrix = result
   
    def backtest_log_wealth(self):
        self.log_wealths = []
        self.cumulative_log_wealths = []

        if self.dX_t_div_X_t is None:
            self.compute_dX_t_div_X_t()
        if self.covariance_matrix is None:
            self.compute_covariance_matrix()

        # Term 1: Initialize an empty dataframe to store the results and iterate over the dates
        term_1 = pd.DataFrame(index=self.strategy.index, columns=['dot_product'])
        for idx, date in enumerate(self.strategy.index):
            if idx == 0:
                term_1.loc[date, 'dot_product'] = 0
            else:
                strategy_prev_t = self.strategy.iloc[idx - 1]
                dX_t_div_X_t = self.dX_t_div_X_t.loc[date].iloc[:len(strategy_prev_t)]

                dot_product = strategy_prev_t.dot(dX_t_div_X_t)
                term_1.loc[date, 'dot_product'] = dot_product

        # Term 2: Initialize an empty dataframe to store the results and iterate over the dates
        term_2 = pd.DataFrame(index=self.strategy.index, columns=['result'])
        for idx, date in enumerate(self.strategy.index):
            if idx == 0:
                term_2.loc[date, 'result'] = 0
            else:
                strategy_prev_t = self.strategy.iloc[idx - 1]
                rank_indices = strategy_prev_t.index - 1
                covariance_matrix_t = self.covariance_matrix[date][rank_indices, :][:, rank_indices]
                result_t = strategy_prev_t.T @ covariance_matrix_t @ strategy_prev_t
                term_2.loc[date, 'result'] = result_t

        # Final Computation of the log Wealth: Iterate over the dates
        sum_log_wealth = 0
        for date in term_1.index:
            term_1_t = term_1.loc[date, 'dot_product']
            term_2_t = term_2.loc[date, 'result']
            result_t = term_1_t - 0.5 * term_2_t
            sum_log_wealth += result_t
            self.log_wealths.append(result_t)
            self.cumulative_log_wealths.append(sum_log_wealth)




    def calculate_theta_X_diff(self):
        # Calculate X_t - X_t-1
        X_diff = self.weights_by_rank.diff()

        # Shift the strategy one period forward to align it with the current weights
        theta_shifted = self.strategy.shift(1)

        # Multiply the result by theta(t-1) using np.dot and sum across the rows
        result = theta_shifted.apply(lambda row: np.dot(row, X_diff.loc[row.name]), axis=1)

        # Drop the first date since it's NaN
        result = result.dropna()

        return result


    def backtest_log_wealth_2(self):
        self.log_wealths = []
        self.cumulative_log_wealths = []

        # Initialize sum_log_wealth
        sum_log_wealth = 0

        # Compute theta_X_diff only once
        theta_X_diff = self.calculate_theta_X_diff()

        # Iterate over the dates
        for date in theta_X_diff.index:
            # Compute term 1: theta_{t-1}^T * (X_t - X_{t-1})
            term_1 = theta_X_diff.loc[date]

            # Compute term 2: 0.5 * (theta_{t-1}^T * (X_t - X_{t-1}))^2
            term_2 = 0.5 * (term_1 ** 2)

            # Compute log wealth at time t and update the total log wealth
            log_wealth_t = term_1 - term_2
            sum_log_wealth += log_wealth_t

            # Store the results
            self.log_wealths.append(log_wealth_t)
            self.cumulative_log_wealths.append(sum_log_wealth)








    def plot_rets(self, data_path=None):
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(self.dates, self.rets)
        ax.set_xlabel("Date")
        ax.set_ylabel("rets")
        ax.set_title("rets between " + str(self.first_date) + " and " + str(self.last_date))
        
        if data_path is None:
            plt.show()
        else:
            plt.savefig(f'{data_path}.png', dpi=300)
        
        plt.close()

    def plot_cumulative_rets(self, data_path=None):
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(self.dates, self.cumulative_rets)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative rets")
        ax.set_title("Cumulative Returns between " + str(self.first_date) + " and " + str(self.last_date))
        
        if data_path is None:
            plt.show()
        else:
            plt.savefig(f'{data_path}.png', dpi=300)
        
        plt.close()

    def plot_cumulative_rets_years(self, year_start, year_end, data_path=None):
        months_start = 12 * year_start
        months_end = 12 * year_end
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(self.dates[months_start:months_end], self.cumulative_rets[months_start:months_end])
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative rets")
        ax.set_title(f"Cumulative Returns from {year_start} years to {year_end} years")
        
        if data_path is None:
            plt.show()
        else:
            plt.savefig(f'{data_path}.png', dpi=300)
        
        plt.close()

    def plot_log_wealth(self, data_path=None):
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(self.dates, self.cumulative_log_wealths)
        ax.set_xlabel("Date")
        ax.set_ylabel("Log Wealth Variation")
        ax.set_title("Log Wealth Variation between " + str(self.first_date) + " and " + str(self.last_date))
        
        if data_path is None:
            plt.show()
        else:
            plt.savefig(f'{data_path}.png', dpi=300)
        
        plt.close()

    def plot_cumulative_log_wealth(self, data_path=None):
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(self.dates, self.cumulative_log_wealths)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative log Wealth")
        ax.set_title("Cumulative log Wealth between " + str(self.first_date) + " and " + str(self.last_date))
        
        if data_path is None:
            plt.show()
        else:
            plt.savefig(f'{data_path}.png', dpi=300)
        
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
    
    dX_t_div_X_t = backtest.compute_dX_t_div_X_t()
    print(dX_t_div_X_t.head(2))
    covariance_matrix = backtest.compute_covariance_matrix()
    print(covariance_matrix[ pd.Timestamp('1965-01-29')])
    
    # backtest.run()
    # backtest.plot_rets()
    # backtest.plot_cumulative_rets()
