# %% [code]
import numpy as np
from scipy.spatial.distance import cdist
import torch
from torch import nn
from abc import ABCMeta, abstractmethod
import config


def TSCVSplit(noSamples: int, noFolds: int):
    ''' Returns start/end train/test indices for time-series CV'''
    
    testSize   = noSamples // (noFolds + 1)
    testStarts = np.arange(noSamples - noFolds * testSize, noSamples, testSize)
    
    for testStart in testStarts: 
        trainEnd = testStart - 1
        testEnd  = testStart + testSize - 1
        yield(0, trainEnd, testStart, testEnd)


class Learner(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self): return
    
    @abstractmethod
    def fit(self, X:np.ndarray, y:np.ndarray): return
    
    @abstractmethod
    def predict(self, X:np.ndarray) -> np.ndarray: return


class Model(Learner):
    ''' Class that combines all 'base learners'. For one bidding zone, which corresponds to 24 targets
        (electricity prices for each hour of any given day), it will fit all models and store the best model for each target. 
        The predictions will then be computed with the best model for each target and combined into one vector.
    '''
    
    def __init__(self, models: list):
        ''' Initialisation function 
            Inputs:
                models: list of initialised Learners
        '''
        
        self.models     = models
        self.bestModels = None
        
        return

    
    def fit(self, X: np.ndarray, y:np.ndarray):
        ''' Trains all base learners, and stores the best one for each target.
            Inputs:
                X:  N x d x O matrix of predictors
                y:  N x O     matrix of targets
        '''

        [model.fit(X, y) for model in self.models]; # Fit all models

        # Find the best model for each target (the one with lowest CV MAE) from the models already
        # fitted (i.e. KRR w/ best hyperparameters, ELM w/ best hyperparameters, etc. etc.)
        cvErrors        = [model.maeBest for model in self.models]
        bestModelIndex  = np.vstack(cvErrors).argmin(axis = 0)
        self.bestModels = [self.models[i] for i in bestModelIndex]

        return
    
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        ''' Predicts on unseen data.
            Inputs:
                X: N x d x O matrix of predictors
            where:
                N: No. datapoints
                d: Input space dimensionality
                O: Output space dimensionality
            
            Outputs: N x d matrix of predictions
        '''

        noSamples, _, noTargets = X.shape
        preds = np.empty(shape = (noSamples, noTargets))

        for targetNo, model in enumerate(self.bestModels):
            yhat = model.predict(X)
            preds[:, targetNo] = yhat[:, targetNo]

        return preds
        
    

class ELM(Learner):
    '''
    Pytorch Implementation of the Extreme Learning Machine according to:

    Huang, Guang-Bin, Qin-Yu Zhu, and Chee-Kheong Siew. "Extreme learning machine: theory and applications." 
        Neurocomputing 70.1-3 (2006): 489-501.
    Huang, Guang-Bin, et al. "Extreme learning machine for regression and multiclass classification." 
        IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics) 42.2 (2011): 513-529.
    Huang, Guang-Bin, Qin-Yu Zhu, and Chee-Kheong Siew. "Extreme learning machine: a new learning scheme of feedforward neural networks." 
        2004 IEEE international joint conference on neural networks (IEEE Cat. No. 04CH37541). Vol. 2. Ieee, 2004.
        
    Objects of this class are instantiated and used in the ELM class that implements a stacked ELM.
    '''


    def __init__(self, 
                 hiddenLayer: np.ndarray = config.ELM_NEURONS, 
                 lambdas    : np.ndarray = config.ELM_REG, 
                 noFolds    : int        = config.NO_FOLDS, 
                 device     : str        = 'cpu',
                 activation : nn         = nn.Sigmoid()):
        ''' Initialisation function.
            Inputs:
                device:     Device for tensor placement
                activation: Activation function for hidden layer
                hidden:     Array of no. neurons on the hidden layers to be tested during cross-validation
                lambdas:    Array of regularisation parameters to be tested during cross-validation
                noFolds:    Number of time-series cross validation to be conducted
        '''
        
        self.act          = activation  # Activation function
        self.dev          = device      # Device for tensors
        self.hiddenLayers = hiddenLayer # List of hidden layers for hyperparameter tuning
        self.lambdas      = torch.tensor(lambdas, dtype = torch.double) # List of regularisation parameters
        self.noFolds      = noFolds     # No. of folds for model selection and error estimation
        self.maeBest      = None        # numpy array of MAEs
        self.hiddenBest   = None        # tensor of hidden layer sizes 
        self.lambdaBest   = None        # tensor of regularisation parameters 
        self.layerTensors = []          # hidden/output layer weight/bias tensors
        
        return

    
    def _fit(self, 
             X: torch.tensor, 
             y: torch.tensor, 
             W: torch.tensor, 
             b: torch.tensor, 
             l: torch.tensor) -> torch.tensor:
        ''' Computes output layer weights on a given dataset, 
            with given hidden layer weights and biases.
            Used in the fit() method.
            Inputs:
                X:  N x d x O matrix of predictors
                y:  N x O     matrix of targets
                W:  d x h x O matrix of weights for the hidden layer
                b:  1 x h x O matrix of biases for the hidden layer
                l:  Regularisation factor            
            Outpus: O x h     matrix of weights for the output layer
            where:
                N:  No. datapoints
                d:  Input space dimensionality
                h:  No. neurons in the hidden layer
                O:  Output space dimensionality
        '''
        
        H  = self.act(torch.einsum('ijm, jkm -> ikm', X, W) + b)
        Ht = torch.transpose(H, 0, 1)
        I  = torch.eye(W.shape[1], dtype = torch.double)[None, ...]
        A  = torch.einsum('ijm, jkm -> mik', Ht, H) + l * I
        B  = torch.einsum('ijk, jk -> ki', Ht, y)

        return torch.linalg.solve(A, B)

    
    def fit(self, X: np.ndarray, y: np.ndarray):
        ''' Fits the model using time-series cross-validation.
            Inputs: 
                X: N x d x O matrix of predictors
                y: N x O     matrix of targets
            where:
                N: No. datapoints
                d: Input space dimensionality
                O: Output space dimensionality
        '''
        
        X, y    = torch.tensor(X).to(self.dev), torch.tensor(y).to(self.dev)    
        N, d, O = X.shape
        dtype   = X.dtype
        
        # Anonymous function to initialise weights/biases tensors of a given size
        # from a U~[-0.5, 0.5) function
        tensorInit = lambda size: torch.rand(size = size, dtype = dtype, device = self.dev) - 0.5
        
        # Initialise best hyperparameter placeholders
        self.maeBest    = np.full((O,), float("Inf"))
        self.hiddenBest = torch.empty((O,), dtype = torch.int)
        self.lambdaBest = torch.empty((O,), dtype = torch.float)

        for DH in self.hiddenLayers:
            W, b = tensorInit(size = (d, DH, O)), tensorInit(size = (1, DH, O))

            for lamda in self.lambdas:
                mae = np.zeros((O,)) # Test set MAE vector

                for ls, le, ts, te in TSCVSplit(N, self.noFolds):

                    Xl, yl = X[ls:le, ...], y[ls:le, :]
                    Xt, yt = X[ts:te, ...], y[ts:te, :]
                    beta   = self._fit(Xl, yl, W, b, lamda)
                    H      = self.act(torch.einsum('ijm, jkm -> ikm', Xt, W) + b)
                    yhat   = torch.einsum('ijk, kj -> ik', H, beta)
                    mae   += torch.abs(yhat - yt).mean(axis = 0).cpu().numpy()

                # Store new hyperparams on improvement
                i = mae < self.maeBest #  Target index with improvements
                self.maeBest[i]    = mae[i]
                self.hiddenBest[i] = DH
                self.lambdaBest[i] = lamda

        # Refit on the entire dataset with the best hyperpars and store weights
        for i in range(O):

            h, lamda = self.hiddenBest[i], self.lambdaBest[i]
            Xc, yc   = X[..., i].unsqueeze(dim = -1), y[..., i].unsqueeze(dim = -1)
            W, b     = tensorInit(size = (d, h, O)), tensorInit(size = (1, h, O))
            beta     = self._fit(Xc, yc, W, b, lamda)
            self.layerTensors.append((W, b, beta))
            
        return
    
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        ''' Predicts on unseen data.
            Inputs:
                X: N x d x O matrix of predictors
            where:
                N: No. datapoints
                d: Input space dimensionality
                O: Output space dimensionality
            
            Outputs: N x d matrix of predictions
        '''
        
        X       = torch.tensor(X).to(self.dev)
        N, _, O = X.shape
        yhat    = torch.empty((N, O), dtype = X.dtype)

        # Loop over all datasets and compute predictions
        for i in range(O):
            W, b, beta = self.layerTensors[i]
            Xcur       = X[:, :, i].unsqueeze(dim = -1)
            H          = self.act(torch.einsum('ijm, jkm -> ikm', Xcur, W) + b)
            yhat[:, i] = torch.einsum('ijk, kj -> ik', H, beta)[:, 0]
        
        return yhat.cpu().numpy()
    
    

class KRR(Learner):
    ''' Kernel ridge regressor.
        Kernel Regularized Least Squares: Reducing Misspecification Bias with a 
        Flexible and Interpretable Machine Learning Approach.
        Hainmueller, J. and Hazlett, C., Political Analysis (2014) 22: 143-168
    '''
    
    
    def __init__(self, 
                 gammas : np.ndarray = config.KRR_GAMMA, 
                 lambdas: np.ndarray = config.KRR_REG, 
                 noFolds: int        = config.NO_FOLDS):
        ''' Initialisation method.
            Inputs:
                lambdas: Array of regularisation parameters to be tested during cross-validation
                gammas:  Array of kernel hyperparameters to be tested during cross-validation
                noFolds: No. folds for cross-validation
        '''
        
        self.weights    = None     # Array of fitted weights w/ best hyperparameters
        self.xTrain     = None     # Training set
        self.gammaBest  = None     # Best kernel hyperparameter
        self.lambdaBest = None     # Best regularisation hyperparameter
        self.maeBest    = None     # Best MAE found from CV
        self.gammas     = gammas   # Array of kernel hyperparameters to be tested for CV
        self.lambdas    = lambdas  # Array of regularisation hyperparameters to be tested for CV
        self.noFolds    = noFolds  # Number of folds for cross-validation
        
        return 
    
    
    def _initHyperparameters(self):
        ''' Initialisation of several properties for training '''
        
        noTargets       = self.xTrain.shape[-1]
        self.maeBest    = np.full(shape = noTargets, fill_value = np.inf)
        self.gammaBest  = np.empty(shape = noTargets)
        self.lambdaBest = np.empty(shape = noTargets)
        
        return
    
    
    def _updateBest(self, gamma: float, lamda: float, mae: np.ndarray):
        ''' Stores the best hyperparameters found so far during training. 
            Used in the fit() method.
            Inputs:
                gamma:  Kernel hyperparamter
                lambda: Regularisation coefficient
                mae:    f-dimensional Vector of validation errors for all targets 
                        when KRR is trained with the input gamma, lambda values on all folds.
                        f: No. folds
        '''
        
        i = mae < self.maeBest # Target indices for which lower MAE was found with given gamma, lambda
        
        self.maeBest[i]    = mae[i]
        self.gammaBest[i]  = gamma
        self.lambdaBest[i] = lamda
        
        return
    
    
    def _refitDataset(self, D:np.ndarray, y:np.ndarray):
        ''' Refit on the entire dataset with best hyperparameters.
            Used in the fit() method.
            Inputs:
                D: d x N x N Distance matrix of the train set
                y: d x N     matrix containing the target values
                where:
                    N: No training samples
                    d: Dimensionality of the output space
        '''
        
        # Dims: noTargets x noTrainSamples x noTrainSamples
        K  = np.exp(-self.gammaBest[:, None, None] * D)
        lI = self.lambdaBest[:, None, None] * np.eye(N = D.shape[-1])
        
        self.weights = np.linalg.solve(K + lI, y) # Dims: noTargets x noTrainSamples
        
        return
    
    
    @staticmethod
    def _computeDistance(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        ''' Computation of euclidean distance.
            Used in the fit() and predict() methods.
            Inputs: 
                x1: N1 x d x o matrix containing the predictors of the prediction set
                x1: N2 x d x o matrix containing the predictors of the training set
                where:
                    d:  Dimensionality of the input space
                    o:  Dimensionality of the output space
                    N1: No. samples of the prediction set
                    N2: No. samples of the training set
        '''
        
        D = np.stack( # strangely enough this is faster than np.linalg.norm
                [cdist(x1[:, :, i], x2[:, :, i], 'sqeuclidean') for i in range(x1.shape[-1])], 
                axis = 0)
        
        return D
        
        
    def fit(self, x: np.ndarray, y: np.ndarray):
        ''' Fits the model by performing time-series cross validation.
            Inputs:
                x: N x d x o matrix of predictors
                y: N x o     matrix of targets
                where:
                    d:  Dimensionality of the input space
                    o:  Dimensionality of the output space
                    N1: No. samples of the prediction set
                    N2: No. samples of the training set
        '''
        
        self.xTrain = x
        self._initHyperparameters()
        
        noSamples, _, noTargets = x.shape
        D = self._computeDistance(x, x)
        y = y.T # Align dims for linalg.solve

        for gamma in self.gammas:
            K = np.exp(-gamma * D) # Exponential kernel

            for lamda in self.lambdas:
                mae = np.zeros(shape = noTargets, dtype = np.float64)

                for ls, le, ts, te in TSCVSplit(noSamples, self.noFolds):

                    # Split train/test set
                    Kl, Kt = K[:, ls:le, ls:le], K[:, ts:te, ls:le]
                    yl, yt = y[:, ls:le], y[:, ts:te]

                    # Fit: compute weights by solving (K + lamda I) @ w = y
                    I = np.eye(N = (le - ls))
                    w = np.linalg.solve(Kl + lamda * I, yl)

                    # Predict on the test set and compute MAE
                    yh  = np.einsum('ijk, ik -> ij', Kt, w)
                    mae += np.abs(yt - yh).mean(axis = -1)

                self._updateBest(gamma, lamda, mae)   
        self._refitDataset(D, y)
        
        return
    
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        ''' Predict on unseen data
            Inputs:
                x: N x d x o matrix of predictors
            Outputs:
                y: N x o     matrix of targets
                
            where:
                d:  Dimensionality of the input space
                o:  Dimensionality of the output space
                N: No. samples to predict
        '''
        
        D = self._computeDistance(x, self.xTrain)
        K = np.exp(-self.gammaBest[:, None, None] * D)

        return np.einsum('ijk, ik -> ji', K, self.weights)