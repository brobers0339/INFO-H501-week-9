import pandas as pd


class GroupEstimate:
    def __init__(self, estimate='mean'):
        self.estimate = estimate
        self.group_estimates = None
    
   def fit(self, X: pd.DataFrame, y):
        df = X.copy()
        df['y'] = y
        agg_type = 'mean' if self.estimate == 'mean' else 'median'
        self.group_estimates = \
            df.groupby(list(X.columns))['y'] \
            .agg(agg_type) \
            .reset_index()
        
        return self

    def predict(self, X):
        X = pd.DataFrame(X, columns=self.group_estimates.columns[:-1])
        merged_df = X.merge(self.group_estimates, how='left',on=list(X.columns))
        
        return merged_df['y'].to_numpy()
