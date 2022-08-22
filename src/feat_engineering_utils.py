import config
from scipy.optimize import minimize
import numpy as np
from collections import defaultdict 
import pandas as pd


''' Fits a bump function to a time-series '''
class BumpInterpolator(object):
    
    def __init__(self, 
                 bounds : list,   # Parameter bounds
                 method : str ,   # Optimisation method (check docs of scipy.optimize.minimize)
                 tol    : float): # Termination tolerance
        
        self.bounds = bounds
        self.method = method
        self.tol    = tol
    
        return
    
    
    ''' Evaluates bump function w/ given parameters '''
    @staticmethod
    def bump(x, params, eps = 1e-12):
        
        w, s = params
        
        f = np.zeros_like(x)
        c = np.abs(x) < 1 / s

        f[c]  = w  * np.exp(- 1 / (1 - (s * x[c] + eps) ** 2) )

        return f

    
    ''' Evaluates the goodness of fit between the data and a bump function with the given parameters '''
    @staticmethod
    def objective(values, params):

        x   = np.linspace(-1, 1, values.shape[0]) # Make x-vector
        f   = BumpInterpolator.bump(x, params)    # Compute values of the RBF
        err = np.mean(np.abs(values - f))         # Evaluate goodness of fit (MAE)

        return err
    
    
    ''' Fits a bump to a time-series'''
    def __call__(self, values):
        
        res = minimize(
            fun    = lambda x: BumpInterpolator.objective(values, x), 
            x0     = [values.max(), 1.0],
            bounds = self.bounds,
            method = self.method, 
            tol    = self.tol)
        
        return pd.Series(res.x, index = ['w', 's'])
        

''' Wraps the Bump Interpolator to fit bumps in a column of the dataframe '''
def fitBump(dfCol):
    
    b = BumpInterpolator(
        bounds = config.BUMP_FIT_BOUNDS,
        method = config.BUMB_FIT_METHOD,
        tol    = config.BUMP_FIT_TOL)
    
    newX = dfCol.groupby(dfCol.index.date).\
            apply(b).\
            reset_index(level = 1).\
            pivot(columns = 'level_1', values = dfCol.name)
    
    newX.columns = dfCol.name + '_' + newX.columns
    
    return newX


''' Fits a Chebyshev Polynomial of a given order to 
    all daily timeseries of a given dataframe
'''
def fitChebyshevPolynomials(df, order):
    
    d = defaultdict()

    feats = []
    for date, datedf in df.groupby(df.index.date):

        for colName in datedf.columns:

            y = datedf[colName].values
            x = np.arange(y.shape[0])

            coeffs = np.polynomial.chebyshev.Chebyshev.fit(x, y, deg = order).coef

            for no, coeff in enumerate(coeffs):
                name = colName + '_' + str(no)
                d[name] = coeff


        feats.append(pd.Series(d, name = date))

    return pd.DataFrame(feats)