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

### SPLIT FUNCTION ###

def split_db(df, target, seed = 2912):
    '''
    splits DataFrame in 3 df's: train, validate, test
    '''
    train, test = train_test_split(df,
                               train_size = 0.8,
                               random_state=seed,
                               stratify = df[target])
    train, validate = train_test_split(train,
                                  train_size = 0.7,
                                  random_state=seed,
                                  stratify = train[target])
    
    return train, validate, test

def full_split(df, target):
    '''
    splits the data frame into:
    X_train, X_validate, X_test, y_train, y_validate, y_test
    '''
    train, validate, test = split_db(df, target)

    #save target column
    y_train = train[target]
    y_validate = validate[target]
    y_test = test[target]

    #remove target column from the sets
    train.drop(columns = target, inplace=True)
    validate.drop(columns = target, inplace=True)
    test.drop(columns = target, inplace=True)

    return train, validate, test, y_train, y_validate, y_test

def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    #split_db class verision with random seed
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test

### IRIS DATA ###
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


### TITANIC DATA ###
def prep_titanic(df):
    '''
    Takes in a titanic dataframe and returns a cleaned df
    Arguments: df - a pandas dataframe with the expected feature names and columns
    Return: clean_df - a dataframe with the cleaning operations performed on it
    '''
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    # Drop columns 
    columns_to_drop = ['embarked', 'pclass', 'passenger_id', 'deck']
    df = df.drop(columns = columns_to_drop)

    return df

def prep_titanic_3(df):
    '''
    Takes in a titanic dataframe and returns 3 data sets
    Arguments: df - a pandas dataframe with the expected feature names and columns
    Return: train, validate and test data sets
    '''
        # Drop duplicates
    df.drop_duplicates(inplace=True)
    # Drop columns 
    columns_to_drop = ['embarked', 'pclass', 'passenger_id', 'deck']
    df = df.drop(columns = columns_to_drop)

    #train, validate, test = split_db(df, 'survived')
    #replace my function with another one to use a random seed
    train, validate, test = train_validate_test_split(df, 'survived', seed=123)
    #impute missing values
    train, validate, test = impute_titanic(train, validate, test)
    
    #create dummies
    train, validate, test = titanic_dummies(train, validate, test)

    return train, validate, test

def prep_titanic_6(df):

    X_train, X_validate, X_test = prep_titanic_3(df)

    y_train = X_train.survived
    y_validate = X_validate.survived
    y_test = X_test.survived

    X_train.drop(columns = ['survived', 'sex', 'class', 'embark_town'], inplace = True)
    X_validate.drop(columns = ['survived', 'sex', 'class', 'embark_town'], inplace = True)
    X_test.drop(columns = ['survived', 'sex', 'class', 'embark_town'], inplace = True)

    return X_train, X_validate, X_test, y_train, y_validate, y_test


def impute_age(train, validate, test):
    '''
    Imputes the mean age of train to all three datasets
    '''
    imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
    imputer = imputer.fit(train[['age']])
    train[['age']] = imputer.transform(train[['age']])
    validate[['age']] = imputer.transform(validate[['age']])
    test[['age']] = imputer.transform(test[['age']])
    return train, validate, test

def impute_embark_town(train, validate, test):
   
    #df['embark_town'] = df.embark_town.fillna(value='Southampton')
    #create an imputer with the strategy mode(most_frequent)
    imputer = SimpleImputer(missing_values = np.nan, strategy='most_frequent')
    #add the value
    imputer = imputer.fit(train[['embark_town']])
    
    #fill the missing values with transform
    train[['embark_town']] = imputer.transform(train[['embark_town']])
    validate[['embark_town']] = imputer.transform(validate[['embark_town']])
    test[['embark_town']] = imputer.transform(test[['embark_town']])
    #df[['embark_town']] = imputer.transform(df[['embark_town']])
    return train, validate, test

def impute_titanic(train, validate, test):
    '''
    returns 3 d frames where missed values of age and embarked and creates dummies for each
    '''
    train, validate, test = impute_age(train, validate, test)
    train, validate, test = impute_embark_town(train, validate, test)
    return train, validate, test

def titanic_dummies(train, validate, test):
        # encoded categorical variables train set
    dummy_df_train = pd.get_dummies(train[['sex', 'class', 'embark_town']], dummy_na=False, drop_first=[True, True])
    train = pd.concat([train, dummy_df_train], axis=1)

    # encoded categorical variables validate set
    dummy_df_validate = pd.get_dummies(validate[['sex', 'class', 'embark_town']], dummy_na=False, drop_first=[True, True])
    validate = pd.concat([validate, dummy_df_validate], axis=1)

    # encoded categorical variables test set
    dummy_df_test = pd.get_dummies(test[['sex', 'class', 'embark_town']], dummy_na=False, drop_first=[True, True])
    test = pd.concat([test, dummy_df_test], axis=1)

    return train, validate, test

### TELCO DATA ###

def prep_telco_dummies(df):
    df.drop_duplicates(inplace=True)
    df.drop(
        columns = ['contract_type_id', 'internet_service_type_id', 'payment_type_id'], 
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

def prep_telco(df):
    # Drop duplicates
    df.drop_duplicates(inplace = True)
    df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'], inplace=True)
       
    # Drop null values stored as whitespace    
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != '']
    
    # Convert to correct datatype
    df['total_charges'] = df.total_charges.astype(float)

    #convert objects to category
    df.gender = df.loc[:, 'gender'].astype('category')
    df.loc[:, 'partner':'dependents'] = df.loc[:, 'partner':'dependents'].astype('category')
    df.loc[:, 'phone_service':'paperless_billing'] = \
        df.loc[:, 'phone_service':'paperless_billing'].astype('category')
    df.loc[:, 'churn':'payment_type'] = \
        df.loc[:, 'churn':'payment_type'].astype('category')
    df.loc[:, 'senior_citizen'] = df.loc[:, 'senior_citizen'].astype('uint8')

    return df

def prep_telco_data(df):
    '''
    creates dummies and
    drops not numerical columns (except 'churn') 
    '''
    df = prep_telco(df)
    
    # Convert binary categorical variables to numeric
    df['gender_encoded'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partner_encoded'] = df.partner.map({'Yes': 1, 'No': 0})
    df['dependents_encoded'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['phone_service_encoded'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['paperless_billing_encoded'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['churn_encoded'] = df.churn.map({'Yes': 1, 'No': 0})

    #change type to uint8
    df.loc[:, 'gender_encoded':'churn_encoded'] = \
        df.loc[:, 'gender_encoded':'churn_encoded'].astype('uint8')
    
    # Get dummies for non-binary categorical variables
    dummy_df = pd.get_dummies(df[['multiple_lines', \
                              'online_security', \
                              'online_backup', \
                              'device_protection', \
                              'tech_support', \
                              'streaming_tv', \
                              'streaming_movies', \
                              'contract_type', \
                              'internet_service_type', \
                              'payment_type']], dummy_na=False, \
                              drop_first=True)
    
    # Concatenate dummy dataframe to original 
    df = pd.concat([df, dummy_df], axis=1)

    #drop unneeded columns
    df.drop(columns = ['gender', 'partner', 'dependents', 'phone_service', \
                    'multiple_lines', 'online_security', 'online_backup',\
                   'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies',\
                   'paperless_billing', 'contract_type', 'internet_service_type', 'payment_type'],
                   inplace = True)
    
    # split the data
    #train, validate, test = split_db(df, 'churn')
    return df
    #return train, validate, test

def split_telco_3(df):
    '''
    splits telco in
    train, validate, test data sets
    '''
    df = prep_telco_data(df)
    return split_db(df, 'churn')

def split_telco_6(df):
    '''
    splits telco if 
    X_train, X_validate, X_test Data Frames and
    y_train, y_validate, y_test Series
    '''
    df = prep_telco_data(df)
    return full_split(df, 'churn')

def split_telco(df, splits=6):
    '''
    by default splits in 
    X_train, X_validate, X_test Data Frames 
    y_train, y_validate, y_test Series

    if you pass splits=3
    splits in train, validate, test data sets
    '''
    if splits == 3:
        return split_telco_3(df)
    else:
        return split_telco_6(df)