import numpy as np
import pandas as pd
from countryinfo import CountryInfo
from geopy import distance
import pycountry
from itertools import combinations
import config


class NeighborCountryFinder(object):
    ''' Computes the <n> closest countries to any given country in the dataset.
        Proximity is assumed to be the geodesic distance between the capital cities of two countries. 
    '''
    
    
    def __init__(self, codes: list):
        ''' Initialisation function.
            Inputs:
                countryCodes: List of all Alpha-2 country codes that will be queried
        '''
        self.df = self._computeDistances(codes)
        return
    

    def __call__(self, countryFrom:str, nClosest:int = config.NEIGHBOR_COUNTRIES) -> list:
        ''' Computes and returns the <nClosest> countries to 
            country <countryFrom>.
            Inputs:
                countryFrom: A2-code of the query country
                nClosest   : Number of countries to return
        '''
        
        rows    = (self.df['country1'] == countryFrom) | (self.df['country2'] == countryFrom)
        distCur = self.df[rows].nsmallest(nClosest, 'distance')
        closest = distCur.drop('distance', axis = 1).melt()['value'].unique().tolist()
        closest.remove(countryFrom)
        
        return closest

    
    def _computeDistances(self, countryCodes: list) -> pd.DataFrame:
        ''' Computes all pairwise distances between the 
            countries in the <countryCodes> list
            Inputs:
                countryCodes: list of A2 country codes
        '''
        
        combos       = list(combinations(countryCodes, 2))
        records      = [( a2_1, a2_2, self._getCapitalDistance(a2_1, a2_2) ) for a2_1, a2_2 in combos]
        distances    = pd.DataFrame.from_records(data = records, columns = ['country1', 'country2', 'distance'])
        
        return distances
    
    
    def _getCapitalDistance(self, a2code1:str, a2code2:str) -> float:
        ''' Computes distance between the capital cities of 
            the countries designated by the two A2 codes
        '''
        return distance.distance(self._getCoords(a2code1), self._getCoords(a2code2)).km
    
    
    @staticmethod
    def _getCoords(a2code:str) -> tuple:
        ''' Computes lat-long coordinates of the capital city
            of the counry designated by its A2 code
        '''
        name = pycountry.countries.get(alpha_2 = a2code).name

        # countryinfo fails for two countries
        if   name == 'Czechia'   : latlong = (50.0755, 14.4378)
        elif name == 'Montenegro': latlong = (42.4304, 19.2594)
        else                     : latlong = CountryInfo(name).capital_latlng()
        
        return latlong
    
    
class LagValueGenerator(object):
    ''' Generator of lag-values of the electricity prices
        of all bidding zones in the dataset.
    '''
    
    @staticmethod
    def __call__(x         : pd.DataFrame, 
                 lagList   : np.ndarray   = config.ELECTR_PRICE_LAGS, 
                 fillValue : float        = config.FILL_VALUE) -> pd.DataFrame:
        ''' Computes and returns a dataframe containing the lag
            values of the electricity prices of a bidding zone
            Inputs:
                x:        Day-ahead forecasts of the electricity prices
                lagList:  array of lags to be computed [in days]
                fillVale: Value to fill-in the nans that are produced
                          in the first lagged values if they do not exist
            Outputs:      Dataframe (w/ dims: date x lags)
        '''
        
        x = LagValueGenerator._toLong(x)
        x = LagValueGenerator._getLags(x, lagList, fillValue)
        x = LagValueGenerator._toWide(x)
        
        return x
    
    
    @staticmethod
    def _toLong(x: pd.DataFrame) -> pd.DataFrame:
        ''' Converts target dataframe to long format: datetime x electricity price 
            (required to correctly compute lags).
            Inputs:
                x:        Day-ahead forecasts of the electricity prices
                Outputs:  Dataframe in long format
        '''
        
        x       = x.unstack(level = 0).reset_index()
        dates   = pd.to_datetime(x['date'], format = '%Y-%m-%d')
        times   = pd.to_timedelta(x['time'].astype(str))
        x.index = dates + times
        x.drop(['date', 'time'], axis = 1, inplace = True)

        if x.shape[1] > 1:
            x = x.pivot(columns = 'variable').droplevel(level = None, axis = 1)

        return x
    
    
    @staticmethod
    def _getLags(x, lagList: np.ndarray, fillValue: float = 0.0) -> pd.DataFrame:
        ''' Computes lag values from a single column electricity 
            price dataframe of shape datetime x electricity price.
            Inputs:
                x:        Day-ahead forecasts of the electricity prices
                lagList:  array of lags to be computed [in days]
                fillVale: Value to fill-in the nans that are produced
                          in the first lagged values if they do not exist
            Outputs:      A dataframe of shape: date x lag values
        '''
        
        shiftedFrames = []
        for lagVal in lagList:
            xShift = x.shift(periods = lagVal)
            xShift.columns = [str(col) + '_lag_' + str(lagVal) for col in xShift.columns]
            shiftedFrames.append(xShift)

        return pd.concat(shiftedFrames, axis = 1).fillna(fillValue)
    
    
    @staticmethod
    def _toWide(xOld: pd.DataFrame) -> pd.DataFrame:
        ''' Converts dataframe w/ lags of the electricity 
            prices to wide format: date x lag values.
        '''

        records = []
        for date, df in xOld.groupby(xOld.index.date):
            df.index   = df.index.time
            df         = df.stack().reset_index()
            df.columns = ['time', 'variable', 'value']
            df['date'] = date
            records.append(df)

        x = pd.concat(records, axis = 0)
        x = x.pivot_table(index = 'date', columns = ['time', 'variable'])
        x = x.droplevel(0, axis = 1)
        x.index = pd.to_datetime(x.index)

        return x
    


class DatasetGenerator(object):
    ''' Generator of predictor and target datasets for the prediction
        of the day ahead electricity prices of a bidding zone.
    '''
    
    
    def __init__(self, xCols: pd.Index, yCols: pd.Index):
        ''' Initialisation of necessary properties '''
        
        cols   = xCols.tolist() + yCols.get_level_values('variable').tolist()
        codes  = DatasetGenerator._getCodes(cols)
        
        self.targetNames   = yCols.get_level_values('variable').unique()
        self._getNeighbors = NeighborCountryFinder(codes)
        self._getLags      = LagValueGenerator()
        
        return 
    
    
    def __call__(self, 
                 X         : pd.DataFrame, 
                 y         : pd.DataFrame, 
                 targetNo  : int, 
                 lags      : np.ndarray = config.ELECTR_PRICE_LAGS, 
                 fillValue : float      = config.FILL_VALUE, 
                 nClosest  : int        = config.NEIGHBOR_COUNTRIES) -> (np.ndarray, pd.DataFrame):
        ''' Generates a set of predictors for the prediction of target number <targetNo>.
            This target corresponds to the one-day ahead electricity prices for a specific
            bidding zone at an hourly interval (i.e. the prediction of 24 different quantities).
            The predictors extracted include:
                - Lag-values <lags> of the electricity prices in all bidding zones.
                - Representation of solar generation, load forecast from ENTSOE transparency, 
                  wind generation. Representation is extracted in the 'EDA' notebooks.
                - If available, a representation of onshore wind/solar generation is specifically extracted.
                - National / religion-related holidays.
                      The above are extracted for the country that the target bidding zones belongs to, 
                      as well as for the <nClosest> countries (proximity is measured as the geodesic 
                      distance between the centers of the capital cities),
                - Week number and day number (of the week)

            Inputs:
                X:        External predictors
                y:        Day-ahead electricity prices
                lags:     Lag-values of day ahead electricity prices
                nClosest: No. of closest countries for which X, y will be considered
                targetNo: ID of bidding zone for which the dataset will be generated

            Outputs: X, y datasets for the corresponding target
        '''
        
        tCol     = self.targetNames[targetNo]                  # Grab specific target
        country  = tCol.split('_')[0]                          # A2 code of <targetCol>
        cols     = self._getCountryCols(country, nClosest)     # Get predictor columns of interest for this target
        external = X.iloc[:, X.columns.str.contains(country)]  # External predictors for the <country>
        lagsdf   = self._getLags(y[cols], lags, fillValue)     # Prices lag values of all bid zones of <country>
        yCur     = y[tCol]                                     # Current targets
        xCur     = []                                          # Empty list to hold datasets (one for each hour)
        
        for hour in yCur.columns: # Make predictor dataset for each hour (targets = electricity prices of <targetCol> at <hour>)

            hCols = lagsdf.columns.get_level_values('time') == hour          # Columns corresponding to this specific hour
            hLags = lagsdf.iloc[:, hCols].droplevel(level = 0, axis = 1)     # Lags of electricity price for <targetCol> @ <hour>
            xNew  = pd.concat([external, hLags], axis = 1)                   # Merge lags with the external predictors
            xNew['day'], xNew['week'] = self._getDayWeekNo(y.index.tolist()) # Add weekday and week number as predictors

            xCur.append(xNew)

        return np.stack(xCur, axis = 2), yCur
    
    
    @staticmethod
    def _getDayWeekNo(dateList: list) -> (np.ndarray, np.ndarray):
        ''' Returns the day and week number of a list of dates given in [dd/mm/yyyy] format
        '''
        
        dayNo  = [date.weekday() for date in dateList]           # weekdays of the dataset
        weekNo = [int(date.strftime("%V")) for date in dateList] # weeknumbers of the dataset
        
        return DatasetGenerator._normalise(dayNo), DatasetGenerator._normalise(weekNo)
    
    
    def _getCountryCols(self, country: str, n: int) -> list:
        ''' Returns the column names that appear in the target dataset
            that refer to <country>, or its <n> closest countries
        '''

        cCols = [col for col in self.targetNames if country in col] # Columns referring to <country>

        if n > 0:
            for pCountry in self._getNeighbors(country, n): # Countries closest to <country>

                # Add columns referring to <pCountry>
                pCols = [col for col in self.targetNames if col.startswith(pCountry)]
                if any(pCols): cCols.extend(pCols)

        return list(set(cCols + pCols))
    
    
    @staticmethod
    def _normalise(x: list) -> np.ndarray:
        ''' Standardisation of a one-dimensional iterable '''
        return (x - np.mean(x)) / (np.max(x) - np.min(x))
    
    
    @staticmethod 
    def _getCodes(l: list) -> set:
        ''' Extracts a set of A2 country codes from a list of column names '''
        return set([name.split('_')[0] for name in l])