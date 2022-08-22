import numpy as np

DATA_DIR = './data/' # Directory containing all relevant data

# Feature extraction parameters
BUMP_FIT_BOUNDS          = [(1e-3, 1e9), 
                            (1e-3, 1e9)]  # Parameter bounds (bump function interpolation on the solar generation columns)
BUMB_FIT_METHOD          = 'Nelder-Mead'  # Optimisation method (bump function interpolation on the solar generation columns)
BUMP_FIT_TOL             = 1e-3           # Stopping criterion tolerance (bump function interpolation on the solar generation columns)
CHEBYSHEV_ORDER_WIND_FIT = 3              # Order for Chebyshev polynomial interpolation of the wind generation features
CHEBYSHEV_ORDER_LOAD_FIT = 6              # Order for Chebyshev polynomial interpolation of the load forecasts

# Dataset generator (preprocessing) parameters    
FILL_VALUE         = 0.0                                  # Value to fill-in the nans produced in the first lagged values if they do not exist
NEIGHBOR_COUNTRIES = 3                                    # No. of closest countries to the bidding zone for which the dataset is being generated.
ELECTR_PRICE_LAGS  = np.array([2, 3, 4, 5, 6, 7, 14, 21]) # Electricty price lag values to be included in the predictors for forecasting

# Modelling parameters
NO_FOLDS    = 4                                   # Number of folds for cross-validation
ELM_NEURONS = np.logspace(2, 3.5, 20).astype(int) # Array with no. neurons on the hidden layers to be tested during cross-validation
ELM_REG     = np.logspace(-6, 2, 20)              # Array of regularisation parameters to be tested during cross-validation
KRR_GAMMA   = np.logspace(-6, 3, 20)              # Array of kernel hyperparameters to be tested during cross-validation
KRR_REG     = np.logspace(-6, 2, 20)              # Array of regularisation parameters to be tested during cross-validation
