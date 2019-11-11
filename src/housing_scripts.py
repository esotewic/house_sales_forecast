import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from FormatScripts import hello



def string(x):
    return str(x)

def add_unit(x):
    if x != 0:
        return ' ' + str(x)
    else:
        return ''

def lower(x):
    return x.lower()

def roound(x):
    if isinstance(x,str)==True:
        return x
    else:
        return round(x)

def abr_suf(x):
    if x == 'Street':
        return ' st'
    elif x == 'Avenue':
        return ' ave'
    elif x == 'Boulevard':
        return ' blvd'
    elif x == 'Drive':
        return ' dr'
    elif x == 'Way':
        return ' wy'
    elif x == 'Place' or x == 'place':
        return ' pl'
    elif x == 'Lane':
        return ' ln'
    elif x == 'Court':
        return ' ct'
    elif x == 'Parkway':
        return ' pkwy'
    elif x == 'Road':
        return ' rd'
    else:
        return ''

def prop_type_update(x):
    if x == 'Duplex' or x == 'Triplex' or x == 'Duadruplex':
        return 'Multi'
    elif x == 'Studio' or x == 'Loft' or x == 'Condominium':
        return 'Condominium'
    elif x == 'Townhouse':
        return 'Townhouse'
    else:
        return 'Single Family Residence'

def yn_impute(x):
    if x == True:
        return 1
    else:
        return 0

def impute_features(df, feature_list):
    for feature in feature_list:
        df[feature] = df[feature].apply(yn_impute)

def wall_clean(x):
    if x == 'No Common Walls' or x == 'End Unit' or x == 'End Unit, No Common Walls' or x == 'No Common Walls, End Unit':
        return 1
    else:
        return 0

def impute_livingarea(cols):
    LivingArea = cols[0]
    PropertySubType = cols[1]
    if pd.isnull(LivingArea):
        if PropertySubType == 'Single Family Residence':
            return np.mean(master[master.PropertySubType == 'Single Family Residence']['LivingArea'])
        elif PropertySubType == 'Condominium':
            return np.mean(master[master.PropertySubType == 'Condominium']['LivingArea'])
        elif PropertySubType == 'Townhouse':
            return np.mean(master[master.PropertySubType == 'Townhouse']['LivingArea'])
        else:
            return np.mean(master[master.PropertySubType == 'Multi']['LivingArea'])
    else:
        return LivingArea

def impute_LotSizeSquareFeet(cols):
    LotSizeSquareFeet = cols[0]
    PropertySubType = cols[1]
    if pd.isnull(LotSizeSquareFeet):
        if PropertySubType == 'Single Family Residence':
            return np.mean(master[master.PropertySubType == 'Single Family Residence']['LotSizeSquareFeet'])
        elif PropertySubType == 'Condominium':
            return np.mean(master[master.PropertySubType == 'Condominium']['LotSizeSquareFeet'])
        elif PropertySubType == 'Townhouse':
            return np.mean(master[master.PropertySubType == 'Townhouse']['LotSizeSquareFeet'])
        else:
            return np.mean(master[master.PropertySubType == 'Multi']['LotSizeSquareFeet'])
    else:
        return LotSizeSquareFeet

def impute_YearBuilt(cols):
    YearBuilt = cols[0]
    PropertySubType = cols[1]
    if pd.isnull(YearBuilt):
        if PropertySubType == 'Single Family Residence':
            return np.mean(master[master.PropertySubType == 'Single Family Residence']['YearBuilt'])
        elif PropertySubType == 'Condominium':
            return np.mean(master[master.PropertySubType == 'Condominium']['YearBuilt'])
        elif PropertySubType == 'Townhouse':
            return np.mean(master[master.PropertySubType == 'Townhouse']['YearBuilt'])
        else:
            return np.mean(master[master.PropertySubType == 'Multi']['YearBuilt'])
    else:
        return YearBuilt

def impute_BathroomsTotalInteger(cols):
    BathroomsTotalInteger = cols[0]
    PropertySubType = cols[1]
    if pd.isnull(BathroomsTotalInteger):
        if PropertySubType == 'Single Family Residence':
            return np.mean(master[master.PropertySubType == 'Single Family Residence']['BathroomsTotalInteger'])
        elif PropertySubType == 'Condominium':
            return np.mean(master[master.PropertySubType == 'Condominium']['BathroomsTotalInteger'])
        elif PropertySubType == 'Townhouse':
            return np.mean(master[master.PropertySubType == 'Townhouse']['BathroomsTotalInteger'])
        else:
            return np.mean(master[master.PropertySubType == 'Multi']['BathroomsTotalInteger'])
    else:
        return BathroomsTotalInteger

def impute_BedroomsTotal(cols):
    BedroomsTotal = cols[0]
    PropertySubType = cols[1]
    if pd.isnull(BedroomsTotal):
        if PropertySubType == 'Single Family Residence':
            return np.mean(master[master.PropertySubType == 'Single Family Residence']['BedroomsTotal'])
        elif PropertySubType == 'Condominium':
            return np.mean(master[master.PropertySubType == 'Condominium']['BedroomsTotal'])
        elif PropertySubType == 'Townhouse':
            return np.mean(master[master.PropertySubType == 'Townhouse']['BedroomsTotal'])
        else:
            return np.mean(master[master.PropertySubType == 'Multi']['BedroomsTotal'])
    else:
        return BedroomsTotal

def WinterIsComing(x):
    if x in [3,4,5,6,7,8]:
        return 0
    else:
        return 1

def impute_bondprice(cols):
    Bond10Year = cols[0]
    CloseYear = cols[1]
    if pd.isnull(Bond10Year):
        if CloseYear == 2016:
            return np.mean(master[master['CloseYear'] == 2016]['Bond10Year'])
        elif CloseYear == 2017:
            return np.mean(master[master['CloseYear'] == 2017]['Bond10Year'])
        elif CloseYear == 2018:
            return np.mean(master[master['CloseYear'] == 2018]['Bond10Year'])
        else:
            return np.mean(master[master['CloseYear'] == 2019]['Bond10Year'])
    else:
        return Bond10Year


def get_housing():
    full_list = []
    for i in range(1,131):
        x = pd.read_csv('houses/Full ({}).csv'.format(i))
        full_list.append(x)

    MLS = pd.concat(full_list,ignore_index=True)
    MLS = MLS[['ClosePrice','ParcelNumber','LotSizeAcres','PrivateRemarks',
            'OriginalListPrice',
            'PublicRemarks','Appliances',
            'AppliancesYN','MLSAreaMajor',
            'AssociationAmenities',
            'AssociationFee','BathroomsTotalInteger','BathroomsFull',
            'BedroomsTotal','City', 'CommonWalls','Cooling','CoolingYN',
            'CountyOrParish','CumulativeDaysOnMarket','CurrentPrice',
            'FireplaceYN','HeatingYN','Latitude','Longitude','LaundryYN','OriginalListPrice',
            'LotSizeSquareFeet','ParkingTotal', 'NumberOfUnitsTotal','OccupantType',
            'OnMarketTimestamp','OpenHouseCount','ParkingYN','PatioYN','PoolPrivateYN',
            'PricePerSquareFoot','PropertyType','PropertySubType','RoomType',
            'BuyerAgencyCompensation','UnitNumber','LivingArea','StateOrProvince',
            'StoriesTotal','StreetName','StreetNumberNumeric','StreetSuffix',
            'SyndicationRemarks','ViewYN','YearBuilt','PostalCode','Zoning',
            'StandardStatus','CloseDate']]



    MLS.UnitNumber.fillna(0,inplace=True)
    MLS['addy'] = MLS['StreetNumberNumeric'].apply(lambda x: str(x)) + ' ' + MLS.StreetName.apply(lower) + MLS.StreetSuffix.apply(abr_suf) + MLS.UnitNumber.apply(add_unit)

    merge = MLS
    # Prepping Data For Modeling
    daddy = merge
    master = daddy[['addy','City','PostalCode',
                    'ClosePrice','CloseDate',
                    'CurrentPrice','YearBuilt','LivingArea',
                    'BedroomsTotal','BathroomsTotalInteger',
                    'LotSizeAcres','LotSizeSquareFeet',
                    'Latitude','Longitude',
                    'StoriesTotal','PropertyType',
                    'PropertySubType','RoomType',
                    'AppliancesYN','CoolingYN', 'FireplaceYN',
                    'HeatingYN','LaundryYN','ParkingYN',
                    'PatioYN', 'PoolPrivateYN','CommonWalls','ViewYN','OriginalListPrice']]
    master.to_csv('westcoastbestcoast.csv')
    return master



def remove_outliers(master):
    master[master.BathroomsTotalInteger>15].head(3)

    master['CloseDate'] = pd.to_datetime(master['CloseDate'])

    master = master[(master['ClosePrice'] > 400000) & (master['ClosePrice']<20000000) & (master['StoriesTotal']<=4)]
    master = master[master['Latitude'] > 33.90]
    master = master[master['Longitude'] < -118.20]
    master = master[master['YearBuilt'] > 0]
    master = master[master['BedroomsTotal'] > 0]
    master = master[master['BedroomsTotal'] < 10]
    master = master[master['BathroomsTotalInteger'] > 0]
    master = master[master['BathroomsTotalInteger'] < 16]
    master = master[(master['LotSizeSquareFeet'] < 100000) & (master['LotSizeSquareFeet'] > 0)]
    master = master[(master['LivingArea'] < 14000) & (master['LivingArea'] > 0)]
    master.PropertySubType = master.PropertySubType.apply(prop_type_update)
    return master



def get_dummies(master):
    prop_type_df = pd.get_dummies(master.PropertySubType)
    master = pd.concat([master,prop_type_df],axis=1)



    master = master[['ClosePrice','YearBuilt',
                     'OriginalListPrice',
                    'PostalCode','PropertySubType',
                    'BedroomsTotal','BathroomsTotalInteger',
                    'Latitude','Longitude','LivingArea','StoriesTotal',
                    'LotSizeSquareFeet','CloseDate','CommonWalls',
                    'Single Family Residence', 'Condominium','Multi','Townhouse',
                    'AppliancesYN','CoolingYN', 'FireplaceYN','HeatingYN',
                    'LaundryYN','ParkingYN','PatioYN', 'PoolPrivateYN','ViewYN']]

    #impute missing values
    master.LivingArea = master[['LivingArea','PropertySubType']].apply(impute_livingarea,axis=1)
    master.LotSizeSquareFeet = master[['LotSizeSquareFeet','PropertySubType']].apply(impute_LotSizeSquareFeet,axis=1)
    master.YearBuilt = master[['YearBuilt','PropertySubType']].apply(impute_YearBuilt,axis=1)
    master.BathroomsTotalInteger = master[['BathroomsTotalInteger','PropertySubType']].apply(impute_BathroomsTotalInteger,axis=1)
    master.BedroomsTotal = master[['BedroomsTotal','PropertySubType']].apply(impute_BedroomsTotal,axis=1)

    master.StoriesTotal.fillna(1, inplace=True)
    master.StoriesTotal.replace(0,1, inplace=True)

    master.CommonWalls = master.CommonWalls.map(wall_clean)
    return master


def amen_impute(master):
    col_list = ['AppliancesYN','CoolingYN', 'FireplaceYN','HeatingYN',
                 'LaundryYN','ParkingYN','PatioYN', 'PoolPrivateYN','CommonWalls','ViewYN']

    master = impute_features(master, col_list)
    return master

def feature_engineering(master):
    master['Age'] = 2020 - master['YearBuilt']
    master['CloseMonth']=pd.to_datetime(master['CloseDate']).dt.month
    master['CloseYear']=pd.to_datetime(master['CloseDate']).dt.year
    master['WinterIsComing']=master['CloseMonth'].apply(WinterIsComing)
    #Remove old data
    master=master[master['CloseYear']>2015]

    #Add 10 year bond Price
    master['BondDate']=pd.to_datetime(master['CloseDate']).dt.date
    master['BondDate'] = master['BondDate'].apply(lambda x: (x - datetime.timedelta(1*365/12)))

    bond_prices = pd.read_csv('DGS10.csv')
    bond_prices.DGS10 = bond_prices.DGS10.replace('.','1.77')
    bond_dic = dict(zip(bond_prices.DATE, bond_prices.DGS10))
    master['Bond10Year'] = pd.to_datetime(master['BondDate']).astype(str)
    master['Bond10Year'] = master['Bond10Year'].map(bond_dic)
    master['Bond10Year'] = master['Bond10Year'].apply(float)
    # master['Bond10Year'] = master['Bond10Year'].fillna(np.mean(master.Bond10Year))
    master.Bond10Year = master[['Bond10Year','CloseYear']].apply(impute_bondprice,axis=1)

    return master
