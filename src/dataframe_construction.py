import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
import wrds


class Data:
    def __init__(self, d=2000, start_date='1965-01-01', datapath='../data/', filename='CRSP_monthly_cleaned.h5', 
        caps_table='mthprevcap', returns_table='mthret',):
        self.d = d 
        self.start_date = start_date 
        self.datapath = datapath 
        self.filepath = self.datapath + filename 
        self.caps_table = caps_table 
        self.returns_table = returns_table

        self.caps_by_permno = self.create_caps_by_permno_df()
        self.returns_by_permno = self.create_returns_by_permno_df()
        self.weights_by_permno = self.create_weights_by_permno_df()
        self.weights_by_rank = self.create_weights_by_rank_df()
        self.weights_permno_by_rank = self.create_weights_permno_by_rank()
    
    ### Construct Dataframes ###
    def create_caps_by_permno_df(self):
        with pd.HDFStore(self.filepath) as store:
            if '/df' in store.keys():
                raw_returns = store['df'][self.caps_table]
            else:
                print('df not in store.keys()')
                return None
        caps_by_permno = raw_returns[raw_returns.index >= self.start_date]
        # caps_by_permno = caps_by_permno.shift(-1).iloc[:-1]
        return caps_by_permno

    def create_returns_by_permno_df(self):
        with pd.HDFStore(self.filepath) as store:
            if '/df' in store.keys():
                raw_returns = store['df'][self.returns_table]
            else:
                print('df not in store.keys()')
                return None
        returns = raw_returns[raw_returns.index >= self.start_date]
        # returns = returns.shift(-1).iloc[:-1]
        return returns

    def create_weights_by_permno_df(self):
        # Step 1: Create a copy of the original DataFrame
        filtered_caps_by_permno = self.caps_by_permno.copy()

        # Step 2: For each row, sort the values and set values not in the top d to 0
        for date, row in filtered_caps_by_permno.iterrows():
            # Sort the values in descending order
            sorted_row = row.sort_values(ascending=False)
            
            # Get the index of the top d values
            top_d_index = sorted_row.iloc[:self.d].index
            
            # Set values not in the top d to 0
            row[~row.index.isin(top_d_index)] = np.nan

        # Step 3: Reconstruct the DataFrame with the modified values
        filtered_caps_by_permno = pd.DataFrame(filtered_caps_by_permno)

        weights_by_permno = filtered_caps_by_permno.div(filtered_caps_by_permno.sum(axis=1), axis=0)
        weights_by_permno = weights_by_permno.fillna(0)
        
        return weights_by_permno   

    def create_weights_by_rank_df(self):
        weights_by_rank = self.weights_by_permno.copy()
        weights_by_rank = weights_by_rank.apply(lambda x: x.sort_values(ascending=False).head(self.d).reset_index(drop=True), axis=1)
        weights_by_rank.columns = np.arange(len(weights_by_rank.columns)) + 1
        return weights_by_rank

    def create_weights_permno_by_rank(self):
        # Create a copy of the original dataframe
        weights_permno_by_rank = self.weights_by_permno.copy()

        # For each row, sort the market weights in descending order, then create tuples of (market_weight, permno) and keep only the top d
        weights_permno_by_rank = weights_permno_by_rank.apply(lambda x: sorted(list(zip(x.values, x.index)), reverse=True)[:self.d], axis=1)

        # Convert the sorted list of tuples into a dataframe
        weights_permno_by_rank = pd.DataFrame(weights_permno_by_rank.tolist(), index=self.weights_by_permno.index)

        # Rename the columns to be the rank
        weights_permno_by_rank.columns = np.arange(len(weights_permno_by_rank.columns)) + 1

        return weights_permno_by_rank


    ### Easy Access Methods ###
    def get_permno_given_rank(self, date, rank):
        return self.weights_permno_by_rank.loc[date, rank][1]

    def get_weight_given_rank(self, date, rank):
        return self.weights_permno_by_rank.loc[date, rank][0]

    def get_return_given_rank(self, date, rank):
        permno = self.get_permno_given_rank(date, rank)
        return self.returns_by_permno.loc[date, permno]

    def get_rank_given_permno(self, date, permno):
        # Get the row corresponding to the given date
        row = self.weights_permno_by_rank.loc[date]
        
        # Iterate through the columns (ranks) and find the column where the permno matches the given permno
        for rank, value in row.items():
            if value[1] == permno:
                return rank
                
        # If the given permno is not found in the row, return None
        return None

    def get_weight_given_permno(self, date, permno):
        return self.weights_by_permno.loc[date, permno]

    def get_return_given_permno(self, date, permno):
        return self.returns_by_permno.loc[date, permno]



if __name__ == "__main__":

    df = Data(datapath='data/')

    rank = 6
    date = '2015-09-30'
    permno = df.get_permno_given_rank(date, rank)
    print(permno)
    print(df.get_weight_given_rank(date, rank))
    print(df.get_return_given_rank(date, rank))
    print(df.get_rank_given_permno(date, permno))
    print(df.get_weight_given_permno(date, permno))
    print(df.get_return_given_permno(date, permno))
