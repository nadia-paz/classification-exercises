import os

import pandas as pd

from env import get_db_url

'''
def new_titanic_data():
    url = get_db_url('titanic_db')
    return pd.read_sql('SELECT * FROM passengers', url)
'''



def get_titanic_data():
    '''
    returns a DataFrame from Titanic database
    '''
    
    filename = "titanic.csv"
    url = get_db_url('titanic_db')
    
    
    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        # read the SQL query into a dataframe
        df =  pd.read_sql('SELECT * FROM passengers', url)
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index_label = False)

        # Return the dataframe to the calling code
        return df  

def get_iris_data():
    '''
    returns a DataFrame from iris_db
    with the species name and species id
    '''
    filename = 'iris.csv'
    sql = 'SELECT * FROM species'
    url = get_db_url('iris_db')

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        df = pd.read_sql(sql, url)
        df.to_csv(filename, index_label = False)
        return df

def get_telco_data():
    '''
    returns everythin from telco_churn database
    '''
    filename = 'telco.csv'
    sql = '''
    SELECT *
    FROM customer_details
    LEFT JOIN customer_churn USING (customer_id)
    LEFT JOIN customer_contracts USING (customer_id)
    LEFT JOIN customer_payments USING (customer_id)
    LEFT JOIN customer_signups USING (customer_id)
    LEFT JOIN customer_subscriptions USING (customer_id)
    LEFT JOIN contract_types USING (contract_type_id)
    LEFT JOIN internet_service_types USING (internet_service_type_id)
    LEFT JOIN payment_types USING (payment_type_id)
    '''
    url = get_db_url('telco_churn')

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        df = pd.read_sql(sql, url)
        df.to_csv(filename, index_label = False)
        return df