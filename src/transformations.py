import numpy as np
import pandas as pd
from scipy.stats import norm as invPhi
from abc import ABCMeta, abstractmethod

class Transformation(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self): return
    
    @abstractmethod
    def fit(self, X:pd.DataFrame): return
    
    @abstractmethod
    def transform(self, X:pd.DataFrame) -> pd.DataFrame: return
    
    @abstractmethod
    def fit_transform(self, X:pd.DataFrame) -> pd.DataFrame: return
    
    @abstractmethod
    def inverse_transform(self, X:pd.DataFrame) -> pd.DataFrame: return

    


class DimTransform(Transformation):
    ''' 
        Static class that converts the electricity-price time-series
        to the actual targets that will be predicted and vice-versa.
        
        Converts from:
            X1: datetime x price_ahead_per_zone
        to:
            X2: date x price_ahead_per_zone_per_hour
        
        and vice versa.
    '''
    
    # Only for compatibility with 
    def __init__(self): return
    def fit(self):      return
    
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.transform(X)
    
    @staticmethod
    def transform(X: pd.DataFrame) -> pd.DataFrame:
        ''' Transforms from X1 to X2 '''
        
        records = []
        for date, data in X.groupby(X.index.date):
            data.index   = data.index.time
            data         = data.stack().reset_index()
            data.columns = ['time', 'variable', 'value']
            data['date'] = date
            records.append(data)

        X = pd.concat(records, axis = 0)
        X = X.pivot_table(index = 'date',  columns = ['variable', 'time'])
        X = X.droplevel(0, axis = 1)
        
        return X
    
    
    @staticmethod
    def inverse_transform(X: pd.DataFrame) -> pd.DataFrame:
        ''' Transforms from X2 to X1 '''
        
        XInv  = X.unstack().reset_index()
        dates = pd.to_datetime(XInv['date'], format = '%Y-%m-%d')
        times = pd.to_timedelta(XInv['time'].astype(str))
        XInv.index = dates + times
        XInv.drop(['date', 'time'], axis = 1, inplace = True)
        XInv = XInv.pivot(columns = 'variable')
        
        return XInv
    

class DiffTransform(Transformation):
    ''' Convenienace class for diff and inverse diff 
        transformation of dataframes 
    '''
    
    def __init__(self):
        ''' Initialisation method'''
        
        self.initValues = None # Holds initial values for each column of a dataframe
        return 
    
    
    def fit(self, X:pd.DataFrame) -> None: 
        ''' Fits the transformer'''
        
        self.initValues = X.iloc[0, :].values
        return
    
    
    def transform(self, X:pd.DataFrame, fillValue:float = 0.0) -> pd.DataFrame: 
        ''' Implements the transformation (diff) operation'''
        
        Xnew = X.diff(axis = 0)
        Xnew.iloc[0, :] = fillValue
        return Xnew
    
    
    def fit_transform(self, X:pd.DataFrame) -> pd.DataFrame: 
        ''' Convenience method implementing both fit() and transform() methods '''
        
        self.fit(X)
        return self.transform(X)
    
    
    def inverse_transform(self, X:pd.DataFrame) -> pd.DataFrame:
        ''' Inverse transformation operation (reverse diff) '''
        
        X.iloc[0,:] = self.initValues
        return X.cumsum(axis = 0)



class InverseSinhTransform(Transformation):
    
    '''
    Computation of the Median absolute deviation: C * median(|x_i - median(x)|)
    Short explanation is provided here: https://aakinshin.net/posts/unbiased-mad/ (accessed: 05/07/2022)
    
    The scaling factor (C) is estimated according to the method of:
        Park, Chanseok, Haewon Kim, and Min Wang. “Investigation of finite-sample properties of robust location and scale estimators.” 
        Communications in Statistics-Simulation and Computation (2020): 1-27. DOI: https://doi.org/10.1080/03610918.2019.1699114
    The implementation is valid for sample sizes > 100, and is computed according to:
        Hayes, Kevin. “Finite-sample bias-correction factors for the median absolute deviation.” 
        Communications in Statistics-Simulation and Computation 43, no. 10 (2014): 2205-2212.
        DOI: https://doi.org/10.1080/03610918.2012.748913
    '''
    
    
    def __init__(self):
        ''' Initialisation function'''
        
        # Initialised upon calling fit()
        self.medAbsDev   = None
        self.medianVal   = None
        
        return
    
    
    def fit(self, x:pd.DataFrame) -> None:
        ''' Computes necessary properties in order
            to perform the transformation.
        '''
        
        self.medianVal = x.median(axis = 0)
        self.medAbsDev = (x - self.medianVal).abs().median() * self._scaleFactor(x.shape[0])
        if any(self.medAbsDev == 0):
            raise ValueError('Zero median absolute deviation encountered.')
        
        return
    
    
    ''' Transforms an input matrix (no.samples x no.features)'''
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        
        xStd = (x - self.medianVal) / self.medAbsDev
        return np.log( xStd + np.sqrt(xStd * xStd + 1) )
    
    
    ''' Inverses the transformation (applies the sinh function) '''
    def inverse_transform(self, x:pd.DataFrame) -> pd.DataFrame:
        
        xNew = 1/2 * ( np.exp(x) - np.exp(-x) )
        return xNew * self.medAbsDev + self.medianVal
    
    
    ''' Fits properties on the input matrix and returns the transformed matrix '''
    def fit_transform(self, x:pd.DataFrame) -> pd.DataFrame:
        
        self.fit(x)
        return self.transform(x)
    
    
    @staticmethod
    def _scaleFactor(n: int) -> float:
        ''' Computes scaling factor based on the sample size n '''
        
        A = - 0.76213 / n - 0.86413 / n ** 2
        return 1 / ( invPhi.ppf(0.75) * (1 + A) )

    



class PipeTransform(Transformation):
    ''' Implements an ordered series of given transformations.
        The transformations must implement the following methods:
            - fit
            - transform
            - fit_transform
            - inverse_transform
    '''
    
    
    def __init__(self, steps:list) -> None:
        ''' Initialisation '''
        
        self.transformList = steps
        return
    
    
    ''' Fits all individual transformations '''
    def fit(self, X: pd.DataFrame) -> None:
        
        for transformation in self.transformList:
            # Each transformation must be performed in order to fit
            # the next transformation class in the sequence
            X = transformation.fit_transform(X)
        
        return
    
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        ''' Performs all transformations '''
        
        for transformation in self.transformList:
            X = transformation.transform(X) 
        
        return X
    
    
    ''' Convenience function implementing both the fit and transform methods'''
    def fit_transform(self, X:pd.DataFrame) -> pd.DataFrame:
        
        self.fit(X)
        return self.transform(X)
    
    
    ''' Performs the inverse transformations (in reverse order)'''
    def inverse_transform(self, X:pd.DataFrame) -> pd.DataFrame:
        
        for transformation in reversed(self.transformList):
            X = transformation.inverse_transform(X)
            
        return X