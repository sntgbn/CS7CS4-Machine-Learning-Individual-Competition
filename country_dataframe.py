import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb
from sklearn import metrics
from sklearn import preprocessing
# data_processing library for custom code/functions
import custom_data_processing as cdp
# Machine Learning models from SKLEARN
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression


# Country Dataframe
country_continents_csv = "https://datahub.io/JohnSnowLabs/country-and-continent-codes-list/r/country-and-continent-codes-list-csv.csv"
country_continents = pd.read_csv(country_continents_csv)
country_continents = country_continents.drop(['Continent_Code', 'Three_Letter_Country_Code', 'Country_Number'], axis='columns')
country_continents['Country_Name'] = country_continents['Country_Name'].replace('Namibia, Republic of','Namibia')
country_continents['Two_Letter_Country_Code'] = country_continents['Two_Letter_Country_Code'].replace(np.nan,'NA')
country_continents['Country_Name'] = country_continents['Country_Name'].replace('Palestinian Territory, Occupied','State of Palestine')
country_codes_csv = "https://datahub.io/core/country-list/r/data.csv"
country_codes = pd.read_csv(country_codes_csv)
country_codes['Code'] = country_codes['Code'].replace(np.nan, 'NA')


for continent_index, continent_element in country_continents.iterrows():
    continent_name_replacement = country_codes.loc[country_codes['Code'] == continent_element['Two_Letter_Country_Code']]
    if len(continent_name_replacement)== 1:
        country_continents['Country_Name'][continent_index] = continent_name_replacement['Name'].values[0]
    elif len(continent_name_replacement) == 0:
        print('Warning - Country %s not found'%country_continents['Country_Name'][continent_index])
    else:
        print('Error - this should not happen')