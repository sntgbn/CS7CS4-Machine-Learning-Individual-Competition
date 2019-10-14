# General purpose imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
# data_processing library for custom code/functions
import custom_data_processing as cdp
# Machine Learning models from SKLEARN
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split 
from catboost import CatBoostRegressor

# Importing CSV and generating a pandas dataframe from it
training_data = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
prediction_data = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')
submission_data = pd.read_csv('tcd ml 2019-20 income prediction submission file.csv')
# Deleting instance as we are not using it
del training_data['Instance']
del prediction_data['Instance']

# MinMax Scaler for Data Normalization
min_max_scaler = preprocessing.MinMaxScaler()

# Gender
# We must take NaN, 0, other and unknown, merge into other gender
training_data = cdp.merge_column_values(training_data, 'Gender', ['other', 'unknown', 0, '0', np.nan], 'other gender')
prediction_data = cdp.merge_column_values(prediction_data, 'Gender', ['other', 'unknown', 0, '0', np.nan], 'other gender')
# One-Hot encoding of Gender, now three colums for gender
training_data = training_data.join(pd.get_dummies(training_data['Gender']))
del training_data['Gender']
prediction_data = prediction_data.join(pd.get_dummies(prediction_data['Gender']))
del prediction_data['Gender']

# Year of Record 
# Stays mostly the same
# Take all nan values and replace with the mean
imputer = SimpleImputer(strategy="mean")
training_data['Year of Record'] = imputer.fit_transform(training_data['Year of Record'].values.reshape(-1,1))
training_data['Year of Record'] = training_data['Year of Record'].astype(int)
prediction_data['Year of Record'] = imputer.fit_transform(prediction_data['Year of Record'].values.reshape(-1,1))
prediction_data['Year of Record'] = prediction_data['Year of Record'].astype(int)

# Age
# Replace NaN with average/mean
# Training data
imputer = SimpleImputer(strategy="mean")
training_data['Age'] = imputer.fit_transform(training_data['Age'].values.reshape(-1,1))
training_data['Age'] = training_data['Age'].astype(int)
# Prediction data
prediction_data['Age'] = imputer.fit_transform(prediction_data['Age'].values.reshape(-1,1))
prediction_data['Age'] = prediction_data['Age'].astype(int)

# Country
# Country stays the same
# No NaNs found

# Size of City
# Normalize data using min_max_scaler
# training_data
city_size_values = training_data[['Size of City']].values.astype(float)
city_size_scaled = min_max_scaler.fit_transform(city_size_values)
training_data['Size of City'] = pd.DataFrame(city_size_scaled)
# prediction_data
city_size_values = prediction_data[['Size of City']].values.astype(float)
city_size_scaled = min_max_scaler.fit_transform(city_size_values)
prediction_data['Size of City'] = pd.DataFrame(city_size_scaled)

# Profession
# Removing NaN for not specified/unknown
imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='Unknown')
training_data['Profession'] = imputer.fit_transform(training_data['Profession'].values.reshape(-1,1))
prediction_data['Profession'] = imputer.fit_transform(prediction_data['Profession'].values.reshape(-1,1))

# University Degree 
# Remove NaN from training/testing data
# Change University degree NAN with value of preceeding instance (closest below)
imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='No')
training_data['University Degree'] = imputer.fit_transform(training_data['University Degree'].values.reshape(-1,1))
prediction_data['University Degree'] = imputer.fit_transform(prediction_data['University Degree'].values.reshape(-1,1))
imputer = SimpleImputer(missing_values='0', strategy='constant', fill_value='No')
training_data['University Degree'] = imputer.fit_transform(training_data['University Degree'].values.reshape(-1,1))
prediction_data['University Degree'] = imputer.fit_transform(prediction_data['University Degree'].values.reshape(-1,1))
training_data = training_data.join(pd.get_dummies(training_data['University Degree']))
prediction_data = prediction_data.join(pd.get_dummies(prediction_data['University Degree']))
del training_data['University Degree']
del prediction_data['University Degree']

# Glasses
# Removed as there is no correlation (thanks contact lenses!)
del training_data['Wears Glasses']
del prediction_data['Wears Glasses']

# Hair Color
# Removed as it has no good correlation
del training_data['Hair Color']
del prediction_data['Hair Color']

# Body Height
# Training Data
upper_lim = training_data['Body Height [cm]'].quantile(.95)
lower_lim = training_data['Body Height [cm]'].quantile(.05)
training_data.loc[(training_data['Body Height [cm]'] > upper_lim), 'Body Height [cm]'] = upper_lim
training_data.loc[(training_data['Body Height [cm]'] < lower_lim), 'Body Height [cm]'] = lower_lim
# Prediction data
upper_lim = prediction_data['Body Height [cm]'].quantile(.95)
lower_lim = prediction_data['Body Height [cm]'].quantile(.05)
prediction_data.loc[(prediction_data['Body Height [cm]'] > upper_lim), 'Body Height [cm]'] = upper_lim
prediction_data.loc[(prediction_data['Body Height [cm]'] < lower_lim), 'Body Height [cm]'] = lower_lim

training_x = training_data.copy()
training_y = training_data['Income in EUR'].copy()
del training_x['Income in EUR']

x_train, x_test, y_train, y_test = train_test_split(training_x, training_y, train_size=0.7, random_state=905732)

#Only categorical variables are Profession and Country
category_indexes = np.where(training_data.dtypes == np.object)[0] 

# Different iterations, depth and learning rate garnered different results
model=CatBoostRegressor(iterations=10000, depth=5, learning_rate=0.1)
model.fit(x_train, y_train, cat_features=category_indexes, eval_set=(x_test, y_test),plot=True)
submission_data['Income'] = model.predict(prediction_data)
submission_data.to_csv('tcd_ml_final_submission.csv', index=False)