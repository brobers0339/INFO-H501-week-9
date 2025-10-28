import pandas as pd


class GroupEstimate:
    def __init__(self, estimate='mean'):
        if estimate not in ['mean', 'median']:
            raise ValueError("Estimate must be 'mean' or 'median'")
        
        self.estimate = estimate
        self.group_estimates = None
    
    def fit(self, X: pd.DataFrame, y):
        '''
        Fit the GroupEstimate model through calculating group-wise estimates.
        
        Parameters
        -----------
        X : pd.DataFrame
            A DataFrame containing only categorical features. Each column represents a category.
        y : array
            A 1-D array of target values corresponding to each row in the X variable.
            
        Returns
        -------
        self : GroupEstimate
            The fitted model with group estimates is stored in self.group_estimates variable.
        '''
        
        df = X.copy()
        df['y'] = y
        agg_type = 'mean' if self.estimate == 'mean' else 'median'
        self.group_estimates = \
            df.groupby(list(X.columns))['y'] \
            .agg(agg_type) \
            .reset_index()
        
        return self

    def predict(self, X):
        '''
        Use the fitted GroupEstimates model to predict the estimates for the categorical y column.
        
        Parameters
        -----------
        X : GroupEstimate object
            A GroupEstimate instance containing a data frame with estimates
            (mean or median depending on estimate type) for each categorical variable.

        Returns
        -------
        merged_df : numpy array
            A numpy array from the merged data frame that contains the y categorical
            variables and their estimates.
        '''
        X = pd.DataFrame(X, columns=self.group_estimates.columns[:-1])
        merged_df = X.merge(self.group_estimates, how='left',on=list(X.columns))
        
        return merged_df['y'].to_numpy()
