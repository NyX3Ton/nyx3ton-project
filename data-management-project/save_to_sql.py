###Library imports and global settings
from sqlalchemy import create_engine
import urllib.parse

import numpy as np
np.seterr(divide = 'ignore', invalid='ignore')


import pandas as pd
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 100)

import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
%matplotlib inline
rcParams['figure.figsize'] = 19, 11

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os

from datetime import datetime
today = datetime.today().strftime('%Y-%m-%d')
today_odbc = datetime.today()

cwd = os.getcwd()

os.makedirs('./Output', exist_ok=True) 
os.makedirs('./Input', exist_ok=True)  
os.makedirs('./model_save/model', exist_ok=True) 


### Load CSV data and wrangle some string and int into coategory types
address = f'./Input/ibm_hr_synthetic_data.csv'

df = pd.read_csv(address)
display(df)

cat_cols = [
            'Attrition',
            'Over18',
            'BusinessTravel',
            'Gender',
            'EducationField',
            'EnvironmentSatisfaction',
            'JobInvolvement',
            'JobLevel',
            'JobRole',
            'JobSatisfaction',
            'PerformanceRating',
            'RelationshipSatisfaction',
            'WorkLifeBalance',
            'MaritalStatus',
            'OverTime',
            'Department',
            'DistanceFromHome',
            'Education']

df[cat_cols] = df[cat_cols].astype('category')

### Save to SQL Server database table
server   = '192.168.50.88,1433'
database = 'data_science'
username = 'sa'
password = 'W4rpDr1v3@'

driver = 'ODBC Driver 17 for SQL Server'

driver_encoded = urllib.parse.quote_plus(driver)
password_encoded = urllib.parse.quote_plus(password)

connection_string = (
    f"mssql+pyodbc://{username}:{password_encoded}@{server}/{database}"
    f"?driver={driver_encoded}&Encrypt=no&TrustServerCertificate=yes"
)

print("Connecting with:", connection_string)

engine = create_engine(connection_string)
table_name = 'HR_Synth_Data'
df.to_sql(table_name, con=engine, if_exists='append', index=False, schema='dbo')
#df.info()
###Library imports and global settings
from sqlalchemy import create_engine
import urllib.parse

import numpy as np
np.seterr(divide = 'ignore', invalid='ignore')


import pandas as pd
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 100)

import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
%matplotlib inline
rcParams['figure.figsize'] = 19, 11

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os

from datetime import datetime
today = datetime.today().strftime('%Y-%m-%d')
today_odbc = datetime.today()

cwd = os.getcwd()

os.makedirs('./Output', exist_ok=True) 
os.makedirs('./Input', exist_ok=True)  
os.makedirs('./model_save/model', exist_ok=True) 


### Load CSV data and wrangle some string and int into coategory types
address = f'./Input/ibm_hr_synthetic_data.csv'

df = pd.read_csv(address)
display(df)

cat_cols = [
            'Attrition',
            'Over18',
            'BusinessTravel',
            'Gender',
            'EducationField',
            'EnvironmentSatisfaction',
            'JobInvolvement',
            'JobLevel',
            'JobRole',
            'JobSatisfaction',
            'PerformanceRating',
            'RelationshipSatisfaction',
            'WorkLifeBalance',
            'MaritalStatus',
            'OverTime',
            'Department',
            'DistanceFromHome',
            'Education']

df[cat_cols] = df[cat_cols].astype('category')
display(df)
