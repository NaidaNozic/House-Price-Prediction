import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
from utils import qualityLookup, KitchenQuality, BathroomsQuality, BedroomsQuality, LivingRoomsQuality
from utils import map_heating_type, map_location_type, map_poolQuality_type, map_model_type, map_house_color
from utils import predictHouseColor, predictHeatingCosts, predictBedrooms, predictBathrooms

data = pd.read_csv('data_7.csv')
# There are 5 rows that are completely empty so we will remove them
data = data.dropna(how='all')
# We have decided to remove the Previous Owner Name due to lack of correlation
data = data.drop(columns=['PreviousOwnerName'])

# IMPUTATIONS
data['Location'] = data['Location'].fillna(data['Location'].mode()[0])
data['PoolQuality'] = data['PoolQuality'].fillna('None')
data = predictHouseColor(data)

data.loc[data['HeatingCosts'] < 0, 'HeatingCosts'] = np.nan
data['HeatingType'] = data['HeatingType'].map(lambda x: map_heating_type(x))
data = predictHeatingCosts(data)

data = predictBedrooms(data)
data = predictBathrooms(data)

data['Age'] = data['Age'].fillna(data['Age'].mean())
data['HasPhotovoltaics'] = data['HasPhotovoltaics'].astype(bool).fillna(data['HasPhotovoltaics'].mode()[0])

# DATA TRANSFORMATION
data['DateSinceForSale'] = pd.to_datetime(data['DateSinceForSale'])
reference_date = pd.Timestamp('1970-01-01')
data['DateSinceForSale'] = (data['DateSinceForSale'] - reference_date).dt.days
data['KitchensQuality'] = data['KitchensQuality'].map(lambda x: qualityLookup(x, KitchenQuality))
data['BathroomsQuality'] = data['BathroomsQuality'].map(lambda x: qualityLookup(x, BathroomsQuality))
data['BedroomsQuality'] = data['BedroomsQuality'].map(lambda x: qualityLookup(x, BedroomsQuality))
data['LivingRoomsQuality'] = data['LivingRoomsQuality'].map(lambda x: qualityLookup(x, LivingRoomsQuality))
data['Location'] = data['Location'].map(lambda x: map_location_type(x))
data['PoolQuality'] = data['PoolQuality'].map(lambda x: map_poolQuality_type(x))
data['HasPhotovoltaics'] = data['HasPhotovoltaics'].astype(int)
data['HouseColor'] = data['HouseColor'].map(lambda x: map_house_color(x))
data['WindowModelNames'] = data['WindowModelNames'].map(lambda x: map_model_type(x))

# REMOVE OUTLIERS
lof = LocalOutlierFactor(n_neighbors=20)
outlier_predictions = lof.fit_predict(data)
data_cleaned = data[outlier_predictions == 1]
data_cleaned.to_csv('new_data.csv', index=False)