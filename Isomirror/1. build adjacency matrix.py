#%% 
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import chi2_contingency
import gower
import pickle

df = pd.read_excel("C:/Users/HopeUgwuoke/anti trafficking/CTDC_global_synthetic_data_v2024.xlsx")

#%%
#Cleaning data and seperating by year
df['yearOfRegistration'] = pd.to_numeric(df['yearOfRegistration'], errors='coerce')
df = df.dropna(subset=['yearOfRegistration'])
df['yearOfRegistration'] = df['yearOfRegistration'].astype(int)
data_by_year = {year: data for year, data in df.groupby('yearOfRegistration')}

#stopping at 2014 for the sake of computational time
data_by_year = dict(list(data_by_year.items())[:13])

Adjacency_matrices= {}
for key,data in data_by_year.items(): 
    #print(data.shape())
    # Compute Gower's distance matrix
    gower_dist_matrix = gower.gower_matrix(data)
    adjacency_matrix = 1-gower_dist_matrix
    Adjacency_matrices[key] = adjacency_matrix
    print(key)

#Saving so it only needs to be run once
with open('adjacency_matrices.pkl', 'wb') as f:
    pickle.dump(Adjacency_matrices, f)

print('saved pickle file')

'''# Mini DataFrame
data = pd.DataFrame({
    'yearOfRegistration': [2014, 2014],
    'gender': ['Man', 'Man'],
    'ageBroad': ['30--38', '30--38'],
    'citizenship': ['UKR', 'UKR'],
    'CountryOfExploitation': ['RUS', 'RUS'],
    'traffickMonths': ['0--12 (0-1 yr)', '0--12 (0-1 yr)'],
    'meansDebtBondageEarnings': [1, 1],
    'meansThreats': [1, 1]
})'''
# %%
