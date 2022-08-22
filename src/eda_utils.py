'''
    Collection of helper function for the EDA notebooks
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pycountry


''' Returns the pairs of variables sorted according to their correlation '''
def getCorrPairs(corr):
    
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    corr[mask] = np.nan
    pairs = corr.abs().unstack()
    pairs = pairs.sort_values(ascending = False)
    
    return pairs


''' Imputes a predictor timeSeries'''
def imputeTS(timeSeries):
    
    if 'capacity' in timeSeries.name:
        res = _imputeCapacity(timeSeries)
    else:
        res = _imputeGeneric(timeSeries)
        
    return res


''' Imputes a generic time-series by interpolation or year-ahead, year-prior values ''' 
def _imputeGeneric(timeSeries,
                  hoursInWeek = 24 * 1,
                  hoursInYear = 24 * 364):
    
    # Interpolate at most 1 week forwards/backwards in time
    timeSeries = timeSeries.interpolate(
        method          = 'time', 
        limit           = hoursInWeek, 
        limit_area      = 'inside',
        limit_direction = 'both')

    # Roll-back one year and impute remaining blocks (fills in gaps mostly at the beginning of the time-series)
    timeSeries = timeSeries.combine_first(timeSeries.shift(-hoursInYear))

    # Roll-forward one year and impute (fills in gaps mostly at the end of the time-series)
    timeSeries = timeSeries.combine_first(timeSeries.shift(hoursInYear))

    # Re-interpolate any nans remaining
    timeSeries = timeSeries.interpolate(
        method          = 'time',
        limit_area      = 'inside',
        limit_direction = 'both')
    
    return timeSeries


''' Imputes capacity timeseries by padding'''
def _imputeCapacity(timeSeries):
    
    return timeSeries.fillna(method = 'pad')


''' Plots original / imputed time-series'''
def plotImputation(originalTS, imputedTS, withMean = False, hoursInMonth = 24 * 7 * 4):
    
    
    imputedTS[~originalTS.isnull()] = np.nan

    plt.figure(figsize = (15, 3))
    plt.plot(originalTS, linewidth = 0.5)
    plt.plot(imputedTS,  linewidth = 0.5)
    
    if withMean:
        monthMean = imputedTS.rolling(hoursInMonth).mean()
        plt.plot(monthMean, color = 'k')
        plt.legend(['Original', 'Imputed', 'Monthly avg. (rolling)'], ncol = 3);
    else:
        plt.legend(['Original', 'Imputed'], ncol = 2);
        
    plt.title(originalTS.name + ' Imputed');
    
    return


''' Fixes information for the areas.csv dataframe '''
def makeAreaMetadata(df):

    df = df.where(pd.notnull(df), None)
    countries, a2Codes, mapCodes, pAreas, bZones, cAreas, mAreas = [], [], [], [], [], [], []

    for _, row in df.iterrows():

        a2code = row['area ID'].split('_')[0]

        if a2code == 'CS': country = 'SerbiaMontenegro' # Does not exist in pycountry
        else:              country = pycountry.countries.get(alpha_2 = a2code).name

        mapcode      = a2code
        primary_area = country + '_default'
        bidZone      = country + '_default'
        control_area = country + '_default'
        market_area  = country + '_default'

        if row['country'] is None: countries.append(country)
        else:                      countries.append(row['country'])

        if row['ISO 3166-1 alpha-2'] is None: a2Codes.append(a2code)
        else:                                 a2Codes.append(row['ISO 3166-1 alpha-2'])

        if row['MapCode ENTSO-E'] is None: mapCodes.append(mapcode)
        else:                              mapCodes.append(row['MapCode ENTSO-E'])

        if row['primary AreaName ENTSO-E'] is None: pAreas.append(primary_area)
        else:                                       pAreas.append(row['primary AreaName ENTSO-E'])

        if row['bidding zone'] is None: bZones.append(bidZone)
        else:                           bZones.append(row['bidding zone'])

        if row['control area'] is None: cAreas.append(control_area)
        else:                           cAreas.append(row['control area'])

        if row['market balance area'] is None: mAreas.append(market_area)
        else:                                  mAreas.append(row['market balance area'])


    df['country']                  = countries
    df['ISO 3166-1 alpha-2']       = a2Codes
    df['MapCode ENTSO-E']          = mapCodes
    df['primary AreaName ENTSO-E'] = pAreas
    df['bidding zone']             = bZones
    df['control area']             = cAreas
    df['market balance area']      = mAreas
    
    return df


''' Returns areaIDs per concept type'''
def _getAreas(primaryConcept, df):
    return df[df['primary concept'] == primaryConcept]['area ID'].unique().tolist()


''' Checks if a column name appears in a list of area codes and returns area code'''
def areaID(fieldName, conceptType, df):

    for area in _getAreas(conceptType, df):

        if isinstance(area, str):
            if area in fieldName:
                return area
    
    return None