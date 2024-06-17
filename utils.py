from enum import Enum
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

def qualityLookup(value, quality_enum):
    return quality_enum[value].value

class LivingRoomsQuality(Enum):
    Excellent = 2
    Good = 1
    Poor = 0

class KitchenQuality(Enum):
    Excellent = 2
    Good = 1
    Poor = 0

class BathroomsQuality(Enum):
    Excellent = 2
    Good = 1
    Poor = 0  

class BedroomsQuality(Enum):
    Excellent = 2
    Good = 1
    Poor = 0

class HeatingType(Enum):
    gas = 2
    oil = 1
    electric = 0

class Location(Enum):
    Suburban = 2
    Rural = 1
    Urban = 0

class PoolQuality(Enum):
    Excellent = 3
    Good = 2
    Poor = 1
    NoPool = 0

class HouseColor(Enum):
    Green = 3
    Yellow = 2
    Gray = 1
    White = 0

class WindowModelNames(Enum):
    Steel = 2
    Wood = 1
    Aluminum = 0

def predictBedrooms(data):
    train_data = data.dropna(subset=['Bedrooms'])
    test_data = data[data['Bedrooms'].isna()]
    
    X_train = train_data[['SquareFootageHouse']]
    y_train = train_data['Bedrooms']
    
    X_predict = test_data[['SquareFootageHouse']]
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    
    predicted_bedrooms = clf.predict(X_predict)
    test_data.loc[:, 'Bedrooms'] = predicted_bedrooms
    return pd.concat([train_data, test_data])

def predictBathrooms(data):
    target_column = "Bathrooms"
    predictor_columns = ['SquareFootageHouse', 'HeatingCosts', 'Bedrooms']
    train_data = data.dropna(subset=[target_column])
    test_data = data[data[target_column].isna()]

    X_train = train_data[predictor_columns]
    y_train = train_data[target_column]
    X_predict = test_data[predictor_columns]

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    
    predicted_bedrooms = clf.predict(X_predict)
    test_data.loc[:, 'Bathrooms'] = predicted_bedrooms
    return pd.concat([train_data, test_data])

def predictHouseColor(data):
    train_data = data.dropna(subset=['HouseColor'])
    test_data = data[data['HouseColor'].isna()]
    X_train = train_data[['PreviousOwnerRating']]
    y_train = train_data['HouseColor']

    X_predict = test_data[['PreviousOwnerRating']]
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    predicted_house_colors = clf.predict(X_predict)

    test_data.loc[:, 'HouseColor'] = predicted_house_colors
    return pd.concat([train_data, test_data])

def predictHeatingCosts(data):
    target_column = 'HeatingCosts'
    predictor_columns = ['SquareFootageHouse', 'HeatingType']
    train_data = data.dropna(subset=[target_column])
    test_data = data[data[target_column].isna()]

    X_train = train_data[predictor_columns]
    y_train = train_data[target_column]

    X_predict = test_data[predictor_columns]
    model = LinearRegression()
    model.fit(X_train, y_train)

    predicted_heating_costs = model.predict(X_predict)
    test_data[target_column] = predicted_heating_costs
    return pd.concat([train_data, test_data])

def map_model_type(model_type):
    if 'steel' in model_type.lower():
        return WindowModelNames.Steel.value
    elif 'wood' in model_type.lower():
        return WindowModelNames.Wood.value
    elif 'aluminum' in model_type.lower():
        return WindowModelNames.Aluminum.value
    else:
        return None
    
def map_house_color(color_type):
    if 'green' in color_type.lower():
        return HouseColor.Green.value
    elif 'yellow' in color_type.lower():
        return HouseColor.Yellow.value
    elif 'gray' in color_type.lower():
        return HouseColor.Gray.value
    elif 'white' in color_type.lower():
        return HouseColor.White.value
    else:
        return None

def map_heating_type(heating_type):
    if 'oil' in heating_type.lower():
        return HeatingType.oil.value
    elif 'electric' in heating_type.lower():
        return HeatingType.electric.value
    elif 'gas' in heating_type.lower():
        return HeatingType.gas.value
    else:
        return None
    
def map_location_type(location_type):
    if 'suburban' in location_type.lower():
        return Location.Suburban.value
    elif 'rural' in location_type.lower():
        return Location.Rural.value
    elif 'urban' in location_type.lower():
        return Location.Urban.value
    else:
        return None
    
def map_poolQuality_type(quality_type):
    if 'excellent' in quality_type.lower():
        return PoolQuality.Excellent.value
    elif 'good' in quality_type.lower():
        return PoolQuality.Good.value
    elif 'poor' in quality_type.lower():
        return PoolQuality.Poor.value
    elif 'none' in quality_type.lower():
        return PoolQuality.NoPool.value
    else:
        return None