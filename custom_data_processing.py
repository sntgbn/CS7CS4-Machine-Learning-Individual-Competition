# Data processing file
# Includes functions that might be used multiple times
import pandas as pd
import numpy as np

def merge_column_values(dataFrame, column_name, values_to_merge, new_value):
    """
    Function used to merge/clean up 
    """
    dataFrame[column_name] = dataFrame[column_name].replace(values_to_merge, new_value)

    return dataFrame


def get_country_continents():
    # List of countries and continents
    country_continents_csv = "https://datahub.io/JohnSnowLabs/country-and-continent-codes-list/r/country-and-continent-codes-list-csv.csv"
    country_continents = pd.read_csv(country_continents_csv)
    # Removing irrelevant columns and fixing some of the data
    country_continents = country_continents.drop(['Continent_Code', 'Three_Letter_Country_Code', 'Country_Number'], axis='columns')
    country_continents['Two_Letter_Country_Code'] = country_continents['Two_Letter_Country_Code'].replace(np.nan,'NA')
    # List of countries and their codes
    country_codes_csv = "https://datahub.io/core/country-list/r/data.csv"
    country_codes = pd.read_csv(country_codes_csv)
    # Fixing some of the data/country names
    country_codes['Code'] = country_codes['Code'].replace(np.nan, 'NA')
    country_codes['Name'] = country_codes['Name'].replace('Bolivia, Plurinational State of','Bolivia')
    country_codes['Name'] = country_codes['Name'].replace('Brunei Darussalam','Brunei')
    country_codes['Name'] = country_codes['Name'].replace('Czech Republic','Czechia')
    country_codes['Name'] = country_codes['Name'].replace('Congo, the Democratic Republic of the','DR Congo')
    country_codes['Name'] = country_codes['Name'].replace('Korea, Republic of','South Korea')
    country_codes['Name'] = country_codes['Name'].replace("Korea, Democratic People's Republic of",'North Korea')
    country_codes['Name'] = country_codes['Name'].replace('Macedonia, the Former Yugoslav Republic of','North Macedonia')
    country_codes['Name'] = country_codes['Name'].replace('Micronesia, Federated States of','Micronesia')
    country_codes['Name'] = country_codes['Name'].replace('Moldova, Republic of','Moldova')
    country_codes['Name'] = country_codes['Name'].replace('Namibia, Republic of','Namibia')
    country_codes['Name'] = country_codes['Name'].replace('Palestine, State of','State of Palestine')
    country_codes['Name'] = country_codes['Name'].replace('Syrian Arab Republic','Syria')
    country_codes['Name'] = country_codes['Name'].replace('Tanzania, United Republic of','Tanzania')
    country_codes['Name'] = country_codes['Name'].replace('Venezuela, Bolivarian Republic of','Venezuela')
    country_codes['Name'] = country_codes['Name'].replace('Viet Nam','Vietnam') 

    for continent_index, continent_element in country_continents.iterrows():
        continent_name_replacement = country_codes.loc[country_codes['Code'] == continent_element['Two_Letter_Country_Code']]
        if len(continent_name_replacement)== 1:
            country_continents['Country_Name'][continent_index] = continent_name_replacement['Name'].values[0]
        elif len(continent_name_replacement) == 0:
            print('Warning - Country %s not found'%country_continents['Country_Name'][continent_index])
        else:
            print('Error - this should not happen')
    country_continents = country_continents.drop(['Two_Letter_Country_Code'], axis='columns')
    # Adding missing countries:
    country_continents.loc[len(country_continents)] = ['Asia', 'Laos']
    country_continents.loc[len(country_continents)] = ['Africa', 'Eswatini']
    country_continents.loc[len(country_continents)] = ['Africa', 'Cabo Verde']
    return country_continents