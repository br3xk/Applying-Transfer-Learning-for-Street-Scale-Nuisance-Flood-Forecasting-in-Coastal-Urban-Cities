
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:44:41 2023

@author: binata
"""

"""
4-Hour LSTM Forecasting Transfer Model [TL-a: Output linear layer is re-trained] 
Street-Scale Flooding in Norfolk, VA

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
- Loads pre-trained base models and re-trains them across 5 random seeds, saves each model, and writes predictions to CSV files.
- Verifies whether weights, recurrent weights, and biases are correctly frozen or retrained in the transfer model compared to base model.
- Selects the best-performing model based on minimum MSE.


To run with different trasnfer model configurations (varying % of source streets and varying % of target events):

    
** 10% base model [Ss=18] ** & ** 10% local model [ET=3] **
trial_data ='S_S0.1_E1.0_T_S1.0_E0.1'
train_Events=[10, 26, 36]
base_model = load_model('best_source_model_S0.1.h5')
pretrained_model = load_model('best_source_model_S0.1.h5')

** 10% base model [Ss=18] ** & ** 20% local model [ET=6] **
trial_data ='S_S0.1_E1.0_T_S1.0_E0.2'
train_Events=[10,11,26,28,36,43]
base_model = load_model('best_source_model_S0.1.h5')
pretrained_model = load_model('best_source_model_S0.1.h5')

** 10% base model [Ss=18] ** & ** 60% local model [ET=18] ** 
trial_data ='S_S0.1_E1.0_T_S1.0_E0.6'
train_Events=[1,2,4,7,10,11,21,24,26,28,29,34,35,36,39,41,43,44]
base_model = load_model('best_source_model_S0.1.h5')
pretrained_model = load_model('best_source_model_S0.1.h5')    

** 10% base model [Ss=18] ** & ** 100% local model [ET=30] ** 
trial_data ='S_S0.1_E1.0_T_S1.0_E1.0'
train_Events=[1,2,3,4,5,7,8,9,10,11,15,16,19,20,21,22,24,26,28,29,30,32,34,35,36,39,41,42,43,44]
base_model = load_model('best_source_model_S0.1.h5')
pretrained_model = load_model('best_source_model_S0.1.h5')    
    
    
    
** 20% base model [Ss=36] ** & ** 10% local model [ET=3] **
trial_data ='S_S0.2_E1.0_T_S1.0_E0.1'
train_Events=[10, 26, 36]
base_model = load_model('best_source_model_S0.2.h5')
pretrained_model = load_model('best_source_model_S0.2.h5')

** 20% base model [Ss=36] ** & ** 20% local model [ET=6] **
trial_data ='S_S0.2_E1.0_T_S1.0_E0.2'
train_Events=[10,11,26,28,36,43]
base_model = load_model('best_source_model_S0.2.h5')
pretrained_model = load_model('best_source_model_S0.2.h5')

** 20% base model [Ss=36] ** & ** 60% local model [ET=18] ** 
trial_data ='S_S0.2_E1.0_T_S1.0_E0.6'
train_Events=[1,2,4,7,10,11,21,24,26,28,29,34,35,36,39,41,43,44]
base_model = load_model('best_source_model_S0.2.h5')
pretrained_model = load_model('best_source_model_S0.2.h5')    

** 20% base model [Ss=36] ** & ** 100% local model [ET=30] ** 
trial_data ='S_S0.2_E1.0_T_S1.0_E1.0'
train_Events=[1,2,3,4,5,7,8,9,10,11,15,16,19,20,21,22,24,26,28,29,30,32,34,35,36,39,41,42,43,44]
base_model = load_model('best_source_model_S0.2.h5')
pretrained_model = load_model('best_source_model_S0.2.h5')    



** 60% base model [Ss=108] ** & ** 10% local model [ET=3] **
trial_data ='S_S0.6_E1.0_T_S1.0_E0.1'
train_Events=[10, 26, 36]
base_model = load_model('best_source_model_S0.6.h5')
pretrained_model = load_model('best_source_model_S0.6.h5')

** 60% base model [Ss=108] ** & ** 20% local model [ET=6] **
trial_data ='S_S0.6_E1.0_T_S1.0_E0.2'
train_Events=[10,11,26,28,36,43]
base_model = load_model('best_source_model_S0.6.h5')
pretrained_model = load_model('best_source_model_S0.6.h5')

** 60% base model [Ss=108] ** & ** 60% local model [ET=18] ** 
trial_data ='S_S0.6_E1.0_T_S1.0_E0.6'
train_Events=[1,2,4,7,10,11,21,24,26,28,29,34,35,36,39,41,43,44]
base_model = load_model('best_source_model_S0.6.h5')
pretrained_model = load_model('best_source_model_S0.6.h5')    

** 60% base model [Ss=108] ** & ** 100% local model [ET=30] ** 
trial_data ='S_S0.6_E1.0_T_S1.0_E1.0'
train_Events=[1,2,3,4,5,7,8,9,10,11,15,16,19,20,21,22,24,26,28,29,30,32,34,35,36,39,41,42,43,44]
base_model = load_model('best_source_model_S0.6.h5')
pretrained_model = load_model('best_source_model_S0.6.h5')    



** 100% base model [Ss=180] ** & ** 10% local model [ET=3] **
trial_data ='S_S1.0_E1.0_T_S1.0_E0.1'
train_Events=[10, 26, 36]
base_model = load_model('best_source_model_S1.0.h5')
pretrained_model = load_model('best_source_model_S1.0.h5')

** 100% base model [Ss=180] ** & ** 20% local model [ET=6] **
trial_data ='S_S1.0_E1.0_T_S1.0_E0.2'
train_Events=[10,11,26,28,36,43]
base_model = load_model('best_source_model_S1.0.h5')
pretrained_model = load_model('best_source_model_S1.0.h5')

** 100% base model [Ss=180] ** & ** 60% local model [ET=18] ** 
trial_data ='S_S1.0_E1.0_T_S1.0_E0.6'
train_Events=[1,2,4,7,10,11,21,24,26,28,29,34,35,36,39,41,43,44]
base_model = load_model('best_source_model_S1.0.h5')
pretrained_model = load_model('best_source_model_S1.0.h5')    

** 100% base model [Ss=180] ** & ** 100% local model [ET=30] ** 
trial_data ='S_S1.0_E1.0_T_S1.0_E1.0'
train_Events=[1,2,3,4,5,7,8,9,10,11,15,16,19,20,21,22,24,26,28,29,30,32,34,35,36,39,41,42,43,44]
base_model = load_model('best_source_model_S1.0.h5')
pretrained_model = load_model('best_source_model_S1.0.h5')    

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
trial_seed ='BTS5'
trial_freeze = 'w_TLa'
trial_data ='S_S0.1_E1.0_T_S1.0_E1.0' #change

trial_all = '{}_{}_{}_{}_{}'.format(trial_model, trial_domain, trial_seed, trial_freeze, trial_data)
print(trial_all)



'..............................................transfer model............................................'


'train_df and test_df different'
#specify parameters
cols = ['FID_', 'Event', 'DateTime', 'RH', 'TD_HR', 'w_depth', 'ELV', 'DTW', 'TWI']           
print("Data Columns: ", cols)

#specify events                   
Events =[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]

  
#train_Events=[10, 26, 36] #change
#train_Events=[10,11,26,28,36,43] #change
#train_Events=[1,2,4,7,10,11,21,24,26,28,29,34,35,36,39,41,43,44] #change
train_Events=[1,2,3,4,5,7,8,9,10,11,15,16,19,20,21,22,24,26,28,29,30,32,34,35,36,39,41,42,43,44] #change
validation_Events=[6,17,23,25,27,31,37]
test_Events=[12,13,14,18,33,38,40]

  

#specify FIDs for transfer learning
path_FIDs="input/"

#set of streets data
FID_selected=pd.read_csv(path_FIDs+"Target_100.csv") #fixed

FIDs=FID_selected['FID_']

train_nodes = FIDs
train_events = train_Events

validation_nodes = FIDs
validation_events = validation_Events

test_nodes = FIDs
test_events = test_Events


#full data
data_org = db.get_data(nodes = FIDs, events = Events, columns=cols)

train_data_org = db.get_data(nodes = train_nodes, events = train_events, columns=cols)
train_data_org.head()

validation_data_org = db.get_data(nodes = validation_nodes, events = validation_events, columns=cols)
validation_data_org.head()

test_data_org = db.get_data(nodes = test_nodes, events = test_events, columns=cols)
test_data_org.head()

cols2scale = ['RH', 'TD_HR', 'w_depth', 'ELV', 'DTW', 'TWI']


db.fit_scaler(train_data_org, columns_to_fit=cols2scale, scaler_type='Standard')
train_data = db.scale_data(train_data_org, columns_to_scale=cols2scale)
train_data.head()

validation_data = db.scale_data(validation_data_org, columns_to_scale=cols2scale)
validation_data.head()

test_data = db.scale_data(test_data_org, columns_to_scale=cols2scale)
test_data.head()

print(len(train_events), len(validation_events), len(test_events))

lstm_train_data = ldt.SLF_LSTM_Data(train_data)
lstm_validation_data = ldt.SLF_LSTM_Data(validation_data)
lstm_test_data = ldt.SLF_LSTM_Data(test_data)

n_back = 4
n_ahead = 4
forecast_cols = ['RH', 'TD_HR']
x_cols = ['w_depth','ELV', 'DTW', 'TWI']
y_cols = ['w_depth']

lstm_train_data.build_data(
n_back = n_back,
n_ahead = n_ahead,
forecast_cols = forecast_cols,
y_cols = y_cols,
x_cols = x_cols,
verbose = False
)

lstm_validation_data.build_data(
n_back = n_back,
n_ahead = n_ahead,
forecast_cols = forecast_cols,
y_cols = y_cols,
x_cols = x_cols,
verbose = False
)

lstm_test_data.build_data(
n_back = n_back,
n_ahead = n_ahead,
forecast_cols = forecast_cols,
y_cols = y_cols,
x_cols = x_cols,
verbose = False
)

train_x1, train_y1 = lstm_train_data.get_lstm_data()
validation_x, validation_y = lstm_validation_data.get_lstm_data()
test_x1, test_y1 = lstm_test_data.get_lstm_data()

#float32
train_x1 = np.asarray(train_x1).astype(np.float32)
train_y1 = np.asarray(train_y1).astype(np.float32)

validation_x = np.asarray(validation_x).astype(np.float32)
validation_y = np.asarray(validation_y).astype(np.float32)

test_x1 = np.asarray(test_x1).astype(np.float32)
test_y1 = np.asarray(test_y1).astype(np.float32)

print('Data Shapes')
print('Train x1:', train_x1.shape)
print('Train y1:', train_y1.shape)

print('Validation x:', validation_x.shape)
print('Validation y:', validation_y.shape)

print('Test x1:', test_x1.shape)
print('Test y1:', test_y1.shape)


# Function to set random seeds
def set_random_seeds(seed):
    tf.keras.utils.set_random_seed(seed)
    print(f"Random Seed: {seed}")
    
# Number of bootstrap samples
num_bootstraps = 5

# Lists to store models, MSE and RMSE for each model
models = []
times = []
epochs = []
rmses = []
mse_scores = []
rmse_scores = []

for bootstrap_index in range(num_bootstraps):
    print(f"Training Bootstrap Model {bootstrap_index + 1}")
    
    # Set a different random seed for each bootstrap
    set_random_seeds(bootstrap_index + 1)
    
    # Load the pre-trained model
    base_model = load_model('best_source_model_S0.1.h5') #change
    pretrained_model = load_model('best_source_model_S0.1.h5') #change
    
    #layers of the loaded model
    pretrained_model.layers
    
    #rename the layers from the pre-trained model with unique names
    for layer in pretrained_model.layers:
        layer._name = f'trainable_{layer.name}'
        print (layer.name)
    

    #time_base
    start_base = time.time()


    'TL techniques'
    
    'TL - a'
    # weights (W) and bias (B) of the output linear dense layer were allowed to be updated
    # W, B

    #'''''''''''''''''''weightwise''''''''''''''''''''
    # Freeze input weights (W) for LSTM layers
    for layer in pretrained_model.layers[0:4]:
        for weight in layer.weights:
            if 'kernel' in weight.name and 'recurrent_kernel' not in weight.name:
                weight._trainable = False
                print(f"weight name: {weight.name}, Trainable: {weight.trainable}")            
                
                
    # Freeze recurrent weights (U) for LSTM layers
    for layer in pretrained_model.layers[0:4]:
        for weight in layer.weights:
            if 'recurrent_kernel' in weight.name:
                weight._trainable = False
                print(f"weight name: {weight.name}, Trainable: {weight.trainable}") 
                
    # Freeze bias (B) for LSTM layers
    for layer in pretrained_model.layers[0:4]:
        for weight in layer.weights:
            if 'bias' in weight.name:
                weight._trainable = False
                print(f"weight name: {weight.name}, Trainable: {weight.trainable}") 
                
 
    # check weights and bias for each LSTM layers
    for layer in pretrained_model.layers:
        for weight in layer.weights:
            print(f"weight name: {weight.name}, Trainable: {weight.trainable}")
        
        
    
    optimizer='Nadam'
    
    lr=1e-3

    
    optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
    
    # Compile the model
    pretrained_model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    
    #callback
    
    mc_filename = f'best_model_TL_bootstrap_{bootstrap_index + 1}.h5'
    
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode='min', factor=0.10, patience=10, min_lr=1e-4, verbose=1)
    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    mc = keras.callbacks.ModelCheckpoint(mc_filename, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    #time_TL
    start_TL = time.time()
    
    
    # Fit the model
    historyTL = pretrained_model.fit(train_x1, train_y1, validation_data=(validation_x, validation_y), epochs=150, verbose=1, batch_size=512, callbacks=[reduce_lr, es, mc])
    
    #load the saved model
    modelTL = load_model(mc_filename)
    
       
       
    'weight check'
    base_model_weights_1st = base_model.layers[0].get_weights()
    transfer_model_weights_1st = modelTL.layers[0].get_weights()

       
    base_model_weights_2nd = base_model.layers[2].get_weights()
    transfer_model_weights_2nd = modelTL.layers[2].get_weights()

       
    base_model_weights_3rd = base_model.layers[4].get_weights()
    transfer_model_weights_3rd = modelTL.layers[4].get_weights()

       
       
    'weight check'
    if np.array_equal(base_model_weights_1st[0], transfer_model_weights_1st[0]):
        print("Weights in the first LSTM layer of the base model and transfer model are the same.")

    else:
        print("Weights in the first LSTM layer of the base model and transfer model are different.")

       
    if np.array_equal(base_model_weights_2nd[0], transfer_model_weights_2nd[0]):
        print("Weights in the second LSTM layer of the base model and transfer model are the same.")

    else:
        print("Weights in the second LSTM layer of the base model and transfer model are different.")

    if np.array_equal(base_model_weights_3rd[0], transfer_model_weights_3rd[0]):
        print("Weights in the third Dense layer of the base model and transfer model are the same.")

    else:
        print("Weights in the third Dense layer of the base model and transfer model are different.")

        
       
    'recurrent weight check'
    if np.array_equal(base_model_weights_1st[1], transfer_model_weights_1st[1]):
        print("Recurrent weights in the first LSTM layer of the base model and transfer model are the same.")
    else:
        print("Recurrent weights in the first LSTM layer of the base model and transfer model are different.")
       
    if np.array_equal(base_model_weights_2nd[1], transfer_model_weights_2nd[1]):
        print("Recurrent weights in the second LSTM layer of the base model and transfer model are the same.")
    else:
        print("Recurrent weights in the second LSTM layer of the base model and transfer model are different.")
       
       
    'bias check'
    if np.array_equal(base_model_weights_1st[2], transfer_model_weights_1st[2]):
        print("Biases in the first LSTM layer of the base model and transfer model are the same.")
    else:
        print("Biases in the first LSTM layer of the base model and transfer model are different.")
       
    if np.array_equal(base_model_weights_2nd[2], transfer_model_weights_2nd[2]):
        print("Biases in the second LSTM layer of the base model and transfer model are the same.")

    else:
        print("Biases in the second LSTM layer of the base model and transfer model are different.")

    if np.array_equal(base_model_weights_3rd[1], transfer_model_weights_3rd[1]):
        print("Biases in the third Dense layer of the base model and transfer model are the same.")

    else:
        print("Biases in the third Dense layer of the base model and transfer model are different.") 

     
    
    hp_model='transfer'
    
    #time
    end_TL = time.time()
    time_TL = (end_TL - start_TL)/60
    print(f"{hp_model}_{bootstrap_index + 1}_pred_time: ", time_TL)

    time_df_TL = pd.DataFrame({'Time': [time_TL]})
    time_df_TL.to_csv(f"1.Result_TF/time_{hp_model}_{bootstrap_index + 1}.csv") 
    
    times.append(time_TL)
    
    
    #epoch
    epochs_TL = len(historyTL.epoch)
    print(f"{hp_model}_{bootstrap_index + 1}_epochs: ", epochs_TL)

    epochs_df_TL = pd.DataFrame({'Epoch': [epochs_TL]})
    epochs_df_TL.to_csv(f"1.Result_TF/epoch_{hp_model}_{bootstrap_index + 1}.csv") 
    
    epochs.append(epochs_TL)
   
    
    #plot loss
    plt.figure(1, figsize=(5, 3))
    plt.subplot(111)
    plt.plot(historyTL.history['loss'])
    plt.plot(historyTL.history['val_loss'])
    plt.title('TL Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig(f"1.Result_TF/{hp_model}_{bootstrap_index + 1}_epoch_loss.png")
    plt.show()

    historyTL_df_TL=pd.DataFrame(historyTL.history)
    historyTL_df_TL.to_csv(f"1.Result_TF/history_{hp_model}_{bootstrap_index + 1}.csv")
    
    modelTL.summary()
    
    # Save the model
    modelTL.save(f"1.Result_TF/{hp_model}_{bootstrap_index + 1}.h5")
    print(f"Saved {hp_model} {bootstrap_index + 1} to disk")
    
    models.append(modelTL)
    
    

    'test'
    preds = modelTL.predict(test_x1)
    
    rmse = np.sqrt(np.mean((preds - test_y1)**2))
    print(f"{hp_model}_{bootstrap_index + 1}_test_rmse: ", rmse)

    rmse_df_base = pd.DataFrame({'RMSE': [rmse]})
    rmse_df_base.to_csv(f"1.Result_TF/rmse_{hp_model}_{bootstrap_index + 1}.csv") 
    
    rmses.append(rmse)
    print('rmses:', rmses)
    print('times:', times)
    print('epochs:', epochs)
    
    
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
    test_data_1_inv.to_csv(f"1.Result_TF/All_test_{hp_model}_{bootstrap_index + 1}_{trial_all}.csv")  
    

    # Evaluate MSE and RMSE
    mse_check = mean_squared_error(test_y1, preds)
    rmse_check = math.sqrt(mse_check)
    
    # Store scores
    mse_scores.append(mse_check)
    print('mse', mse_scores)
    rmse_scores.append(rmse_check)
    print('rmse', rmse_scores)
    
mse_scores_df_TL = pd.DataFrame({'MSE': [mse_scores]})
rmse_scores_df_TL = pd.DataFrame({'RMSE': [rmse_scores]})
rmses_df_TL = pd.DataFrame({'RMSE': [rmses]})
epochs_df_TL = pd.DataFrame({'Epoch': [epochs]})
times_df_TL = pd.DataFrame({'Time': [times]})
    
mse_scores_df_TL.to_csv(f"1.Result_TF/mse_scores_{hp_model}_{trial_all}.csv") 
rmse_scores_df_TL.to_csv(f"1.Result_TF/rmse_scores_{hp_model}_{trial_all}.csv") 
rmses_df_TL.to_csv(f"1.Result_TF/rmses_{hp_model}_{trial_all}.csv")
epochs_df_TL.to_csv(f"1.Result_TF/epochs_{hp_model}_{trial_all}.csv") 
times_df_TL.to_csv(f"1.Result_TF/times_{hp_model}_{trial_all}.csv") 
 
 
# model with the best MSE
best_model_index = np.argmin(mse_scores)
best_model_index = np.argmin(rmse_scores)

print(f"Best Model based on MSE: Bootstrap Model {best_model_index + 1}")
print(f"MSE of the Best Model: {mse_scores[best_model_index]}")
print(f"RMSE of the Best Model: {rmse_scores[best_model_index]}")

models[best_model_index].save("best_target_model.h5")
models[best_model_index].save(f"1.Result_TF/{hp_model}_best_BTS5.h5")
print("Saved best target model to disk")



'bootstrap 5 mean ensemble'
# ensemble prediction
ensemble_preds = np.mean([model.predict(test_x1) for model in models], axis=0)


rmse_ensemble = np.sqrt(np.mean((ensemble_preds - test_y1)**2))
print(f"{hp_model}_BTS5_test_rmse: ", rmse_ensemble)

rmse_ensemble_df_base = pd.DataFrame({'RMSE': [rmse_ensemble]})
rmse_ensemble_df_base.to_csv(f"1.Result_TF/rmse_{hp_model}_BTS5.csv") 
  


test_data_1=test_data
    
'remap multi-ahead'
for k in range(n_ahead):
    ensemble_preds_col = pd.Series(ensemble_preds[:,k], index=lstm_test_data.data_map)
    test_data_1[f'ensemble_preds_y{k+1}_s'] = ensemble_preds_col
    test_data_1[f'ensemble_preds_y{k+1}'] = test_data_1[f'ensemble_preds_y{k+1}_s'].shift(k)
    del test_data_1[f'ensemble_preds_y{k+1}_s']

    real_col = pd.Series(test_y1[:,k], index=lstm_test_data.data_map)
    test_data_1[f'real_y{k+1}_s'] = real_col
    test_data_1[f'real_y{k+1}'] = test_data_1[f'real_y{k+1}_s'].shift(k)
    del test_data_1[f'real_y{k+1}_s']

test_data_1_inv = test_data_1.copy()
test_data_1_inv.head()
      
      
cols2scale = ['RH','w_depth', 'ensemble_preds_y1', 'real_y1', 'ensemble_preds_y2',
'real_y2', 'ensemble_preds_y3', 'real_y3', 'ensemble_preds_y4', 'real_y4']
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
test_data_1_inv.to_csv(f"1.Result_TF/All_test_{hp_model}_BTS5_{trial_all}.csv") 

print("test RMSE for TL BTS5:", rmse_ensemble)
