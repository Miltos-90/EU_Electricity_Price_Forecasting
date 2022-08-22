import numpy as np
import pandas as pd
import datetime as dt


def parseHolidays(filename: str, mostCommon: int) -> pd.DataFrame:
    ''' Parses the excel file containing the holiday dates and names 
        of all countries of interest.
        Inputs:
            filename:   Name of the excel file
            mostCommon: No. of most common holidays to indicate
        Outputs:
            df: Dataframe containing the holidays
    '''

    # Read and parse holidays (returns dict with sheet names as keys)
    hDict = pd.read_excel(filename,
                          names  = ['date', 'day', 'holiday', 'type', 'note'],
                          header = None, sheet_name = None, index_col = None)

    # Convert dict to list of dfs
    df = []
    for country, dfCur in hDict.items():

        # Keep holiday name and date (index) only
        dfCur  = dfCur['type']
        # Drop nan and duplicates
        dfCur  = dfCur[~dfCur.isnull()]
        dfCur  = dfCur[~dfCur.index.duplicated(keep = 'first')]
        # Get top most common values
        common = dfCur.value_counts()[:mostCommon].to_dict()
        # Replace most common values with their occurence and set 1 to the rest
        dfCur  = dfCur.map(dict(common)).fillna(1)
        # Rename series
        dfCur.name = country + '_holiday'
        # Add to list
        df.append(dfCur)

    # concatenate and convert to categorical
    df = pd.concat(df, axis = 1).fillna(0).astype(int).apply(lambda x: pd.Categorical(x))
    df.index = pd.to_datetime(df.index)
    return df


def trainTestSplit(X, y, trainRatio):
    ''' Simple time-series train test split'''
    
    noDays = (X.index.max() - X.index.min()).days
    trainEndDate = str(X.index.min() + dt.timedelta(days = int(noDays * trainRatio)))

    Xtrain = X[X.index < trainEndDate]
    Xtest  = X[X.index >= trainEndDate]

    ytrain = y[y.index < trainEndDate]
    ytest = y[y.index >= trainEndDate]
    
    return Xtrain, Xtest, ytrain, ytest
