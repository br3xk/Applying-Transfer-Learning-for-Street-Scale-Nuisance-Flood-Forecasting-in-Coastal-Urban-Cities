
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:44:41 2023

@author: binata
"""

"""
4-Hour LSTM Forecasting w/o Transfer Model for Street-Scale Flooding in Norfolk, VA

This script implements an LSTM model to forecast future water depth using rainfall, tide, elevation, TWI, DTW, and past water depth data for flood-prone streets of Norfolk, Virginia.

Model Details:
- Forecast horizon: 4 hours (n_ahead = 4)
- Lookback window: 4 hours (n_back = 4)
- Input features:
    - Static: ELV, TWI, DTW
    - Dynamic: 
        - Rainfall (RH): [t-3, t-2, t-1, t, t+1, t+2, t+3, t+4]
        - Tide (TD): [t-3, t-2, t-1, t, t+1, t+2, t+3, t+4]
        - Water Depth (w_depth): [t-3, t-2, t-1, t]
- Output: Water Depth (w_depth) for [t+1, t+2, t+3, t+4]

Functionality:
- Loads `node_data`, `tide_data`, and `weather_data` from a relational database.
- Uses `lstm_data_tools.py` to preprocess data into 3D tensors for training, validation, and testing.
- Loads the best pre-trained base model and predicts on test data, and writes predictions to CSV files.

To run with different w/o Transfer model based on base model configurations (varying % of source streets):

** w/o Transfer model for 10% base model [Ss=18] **
trial_data ='S_S0.1_E1.0_T_S1.0'
base_model = load_model('best_source_model_S0.1.h5')

** w/o Transfer model for 20% base model [Ss=36] **
trial_data ='S_S0.2_E1.0_T_S1.0' 
base_model = load_model('best_source_model_S0.2.h5')

** w/o Transfer model for 60% base model [Ss=108] ** 
trial_data ='S_S0.6_E1.0_T_S1.0'
base_model = load_model('best_source_model_S0.6.h5')

** w/o Transfer model for 100% base model [Ss=180] **
trial_data ='S_S1.0_E1.0_T_S1.0' 
base_model = load_model('best_source_model_S1.0.h5')

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
#import keras
from tensorflow import keras #gpu
#from keras.models import Sequential
from tensorflow.keras import Sequential #gpu
#from keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.layers import Dense, Dropout, LSTM #gpu
#from keras.models import load_model
from tensorflow.keras.models import load_model #gpu
from sklearn.metrics import mean_squared_error
import math
import random


#use GPU
physical_devices = tf.config.list_physical_devices('GPU')
#nondeterministic GPU
tf.config.experimental.enable_op_determinism()

import lstm_data_tools as ldt

os.getcwd()

os.chdir('../../Transfer_7/')

db = ldt.SLF_Data_Builder(os.getcwd() + '/relational_data_50/')


'define run_name mlflow'
trial_model='LSTM_GPU'
trial_domain ='S180_T180'
trial_seed ='Best'
trial_freeze = 'wo_TL'
trial_data ='S_S0.1_E1.0_T_S1.0' #change to S0.1, S0.2, S0.6 and S1.0 based on base models to be tested

trial_all = '{}_{}_{}_{}_{}'.format(trial_model, trial_domain, trial_seed, trial_freeze, trial_data)
print(trial_all)



'..............................................w/o transfer model............................................'


'train_df and test_df different'
#specify parameters
cols = ['FID_', 'Event', 'DateTime', 'RH', 'TD_HR', 'w_depth', 'ELV', 'DTW', 'TWI']           
print("Data Columns: ", cols)



''''''
#specify events                  
Events =[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]


test_Events=[12,13,14,18,33,38,40]


#specify FIDs for transfer learning
path_FIDs="input/"

#set of streets data
FID_selected=pd.read_csv(path_FIDs+"Target_100.csv") #fixed

FIDs=FID_selected['FID_']

test_nodes = FIDs
test_events = test_Events
 

#full data
data_org = db.get_data(nodes = FIDs, events = Events, columns=cols)

test_data_org = db.get_data(nodes = test_nodes, events = test_events, columns=cols)
test_data_org.head()

cols2scale = ['RH', 'TD_HR', 'w_depth', 'ELV', 'DTW', 'TWI']


db.fit_scaler(test_data_org, columns_to_fit=cols2scale, scaler_type='Standard') #nothing to be seen #nothing to train
test_data = db.scale_data(test_data_org, columns_to_scale=cols2scale)
test_data.head()

print(len(test_events))

lstm_test_data = ldt.SLF_LSTM_Data(test_data)

''''''
n_back = 4
n_ahead = 4
forecast_cols = ['RH', 'TD_HR']
x_cols = ['w_depth','ELV', 'DTW', 'TWI']
y_cols = ['w_depth']

lstm_test_data.build_data(
n_back = n_back,
n_ahead = n_ahead,
forecast_cols = forecast_cols,
y_cols = y_cols,
x_cols = x_cols,
verbose = False
)

test_x1, test_y1 = lstm_test_data.get_lstm_data()

test_x1 = np.asarray(test_x1).astype(np.float32)
test_y1 = np.asarray(test_y1).astype(np.float32)

print('Data Shapes')
print('Test x1:', test_x1.shape)
print('Test y1:', test_y1.shape)

base_model = load_model('best_source_model_S0.1.h5') #change to S0.1, S0.2, S0.6 and S1.0 based on base models to be tested 

hp_model='base_woTL'

rmses = []
mse_scores = []
rmse_scores = []

'test'
preds = base_model.predict(test_x1)

rmse = np.sqrt(np.mean((preds - test_y1)**2))
print(f"{hp_model}_test_rmse: ", rmse)

rmse_df_base = pd.DataFrame({'RMSE': [rmse]})
rmse_df_base.to_csv(f"1.Result_TF/rmse_{hp_model}.csv") 

rmses.append(rmse)
print('rmses:', rmses)

test_data_1=test_data
    
'remap multi-ahead'
for k in range(n_ahead):
    preds_col = pd.Series(preds[:,k], index=lstm_test_data.data_map)
    test_data_1[f'preds_y{k+1}_s'] = preds_col
    test_data_1[f'preds_y{k+1}'] = test_data_1[f'preds_y{k+1}_s'].shift(k)
    del test_data_1[f'preds_y{k+1}_s']

    real_col = pd.Series(test_y1[:,k], index=lstm_test_data.data_map)
    test_data_1[f'real_y{k+1}_s'] = real_col
    test_data_1[f'real_y{k+1}'] = test_data_1[f'real_y{k+1}_s'].shift(k)
    del test_data_1[f'real_y{k+1}_s']

test_data_1_inv = test_data_1.copy()
test_data_1_inv.head()
      
      
cols2scale = ['RH','w_depth', 'preds_y1', 'real_y1', 'preds_y2',
'real_y2', 'preds_y3', 'real_y3', 'preds_y4', 'real_y4']
orig_cols = ['RH','w_depth', 'w_depth', 'w_depth', 'w_depth', 
'w_depth', 'w_depth', 'w_depth', 'w_depth', 'w_depth']

# inverse scale the test data
test_data_1_inv = db.inverse_scale_data(test_data_1_inv, columns_to_scale=cols2scale, orig_col_names=orig_cols)
test_data_1_inv.head()  

'dataframe to csv'
print ('dataframe to csv')   

test_data_1_inv=test_data_1_inv.reset_index(drop=True)
test_data_1_inv['Datetime'] =  pd.to_datetime(test_data_1_inv['DateTime'], format='%b%y_%d_%H')
test_data_1_inv.set_index('Datetime', inplace=True, drop=True)
 
#csv    
test_data_1_inv.to_csv(f"1.Result_TF/All_test_{hp_model}_{trial_all}.csv") 


print("test RMSE for wo_TL:", rmse)

# Evaluate MSE and RMSE
mse_check = mean_squared_error(test_y1, preds)
rmse_check = math.sqrt(mse_check)

# Store scores
mse_scores.append(mse_check)
print(mse_scores)
rmse_scores.append(rmse_check)
print(rmse_scores)
    
mse_scores_df_TL = pd.DataFrame({'MSE': [mse_scores]})
rmse_scores_df_TL = pd.DataFrame({'RMSE': [rmse_scores]})

rmses_df_TL = pd.DataFrame({'RMSE': [rmses]})
    
mse_scores_df_TL.to_csv(f"1.Result_TF/mse_scores_{hp_model}_{trial_all}.csv") 
rmse_scores_df_TL.to_csv(f"1.Result_TF/rmse_scores_{hp_model}_{trial_all}.csv") 

rmses_df_TL.to_csv(f"1.Result_TF/rmses_{hp_model}_{trial_all}.csv")
 

