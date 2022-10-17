import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# train test split from sklearn
from sklearn.model_selection import train_test_split
# imputer from sklearn
from sklearn.impute import SimpleImputer

# filter out warnings
import warnings
warnings.filterwarnings('ignore')

# our own acquire script:
import acquire as ac

def prep_iris(df):
    '''
    accepts the untransformed iris data, and returns the data with the transformations above applied.
    '''
    df = df.drop_duplicates()
    #drop species_id and 'mesurement_id'
    df.drop(columns = ['species_id', 'measurement_id'], inplace = True)
    #rename 'species_name' to 'species'
    df.rename(columns = {'species_name':'species'}, inplace = True)
    #get dummies for 
    dummies = pd.get_dummies(df['species'], dummy_na = False, drop_first = True)
    return pd.concat([df, dummies] , axis = 1)

def prep_titanic(df):
    '''
    Takes in a titanic dataframe and returns a cleaned dataframe
    Arguments: df - a pandas dataframe with the expected feature names and columns
    Return: clean_df - a dataframe with the cleaning operations performed on it
    '''
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    # Drop columns 
    columns_to_drop = ['embarked', 'pclass', 'age', 'passenger_id', 'deck']
    df = df.drop(columns = columns_to_drop)
    # encoded categorical variables
    dummy_df = pd.get_dummies(df[['sex', 'class', 'embark_town']], dummy_na=False, drop_first=[True, True])
    df = pd.concat([df, dummy_df], axis=1)
    return df

def prep_telco(df):
    df.drop_duplicates(inplace = True)
    df.drop(
    columns = ['customer_id', 'contract_type_id', 'internet_service_type_id', 'payment_type_id'], 
    inplace = True)
    df.total_charges = df.total_charges.replace(' ', np.nan).astype(float)
    df = df.dropna()
    
    #columns for dummies
    col_dummies2 = [] #for 2 values
    col_dummies = []
    for col in df.columns:
        if df[col].dtype == 'O' and df[col].nunique() == 2:
            col_dummies2.append(col)
        elif df[col].dtype == 'O' and df[col].nunique() > 2:
            col_dummies.append(col)
            
    #create dummies
    #dummies for columns with 2 values
    telco_dummies2 = pd.get_dummies(df[col_dummies2], dummy_na=False, drop_first= True)

    #dummies for columns with 2+ values drop_first = First, because 
    telco_dummies = pd.get_dummies(df[col_dummies], dummy_na=False, drop_first= False)
    
    #concat and return
    return pd.concat([df, telco_dummies2, telco_dummies], axis = 1)

def split_db(df, target):
    '''
    splits DataFrame in 3 df's: train, validate, test
    '''
    train, test = train_test_split(df,
                               train_size = 0.8,
                               stratify = df[target])
    train, validate = train_test_split(train,
                                  train_size = 0.7,
                                  stratify = train[target])
    
    return train, validate, test