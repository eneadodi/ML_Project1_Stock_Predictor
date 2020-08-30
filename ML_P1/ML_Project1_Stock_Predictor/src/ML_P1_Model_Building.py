import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pickle import HIGHEST_PROTOCOL
from pandas.tests.reshape.test_pivot import dropna
from numpy import NaN
import ML_P1_data_extraction as e
#import tensorflow as tf
#from tensorflow.keras import layers
from sklearn.svm import SVC
import ML_p1_data_preperation as p

#Keras imports
#from keras.models import Sequential , Dropout, Dense, LSTM
#from keras.models import LSTM
#from keras.models import Dropout
#from keras.models import Dense

#From Google Machine Learning Crash Course
def plot_the_loss_curve(epochs, mse):
  """Plot a curve of loss vs. epoch."""

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Mean Squared Error")

  plt.plot(epochs, mse, label="Loss")
  plt.legend()
  plt.ylim([mse.min()*0.95, mse.max() * 1.05])
  plt.show()  



'''
LSTM needs dada in a specific format
'''
def prepare_LSTM_data(df):
    vpw, ppw, cv = p.filtered_columns(df) 
    
    time_related_cols = [col for col in df if col not in cv]
    
    time_related_df = df[time_related_cols]
    
    time_series_df_by_ticker = []
    
    col_names = time_related_df.columns
    col_names_length = len(col_names)
    
    '''first, we make a list of dataframes where each dataframe is a ticker
        where the rows are the Dates and the columns are price and volume at 
        that date. 
    '''
    for i, r in time_related_df.iterrows():
        
        j = 0 
        tdf = pd.DataFrame(columns=['Price', 'Volume'])
        tdf.name = r.name
        
        while j < col_names_length-1:   
            close = r[col_names[j]]
            volume = r[col_names[j+1]]
            date = col_names[j][:16]
            tdf.loc[date] = [close, volume]
            j+=2
        time_series_df_by_ticker.append(tdf)
        print('f')
        #print('close', close , ' volume' , volume, ' date' , date)
    
    e.save_obj(time_series_df_by_ticker,'ticker_time_series_low_criteria')
    
    pass



'''
To create the model I will be using Dropout as the main regularization technique
Dropout in every iteration goes through the layers of a network and sets a probability of removing a node at each layer.
This'll cause all the ingoing an doutgoing links from the removed node to also be removed. Thisl'll lead to a diminished network
and force the netwrok to never compeltely rely on any one node as it can be removed at any time. This'll also prevent scenarios
where layers co-adapt to correct mistakes from prior layers. Dropout regularization works in synch with backwards propogation.

Due to Dropout regularization, more nodes will be necassary on the layers.
There will be three LSTM layers with 60 , 30 , 60 units respectively. 

DOWNFALLS
LSTM often works best 200-400 time steps however I only have data with 100 time steps. Thus
Subdividing to 10 sub-sequences of 10 time steps may not be the best approach. Maybe I should've
used the daily data.

LSTM shape is [Samples, Time Steps, Features]
'''
def create_LSTM_model(learning_rate, feature_layer):
    model = Sequential()
    
    df = pd.read_pickle('../data/2YStockDFLowCriteria.pkl')
    
    
    #layer 1
    model.add(LSTM(units=120,return_sequences=True,input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.3))
    
    #layer 2
    model.add(LSTM(units=30,return_sequences=True))
    model.add(Dropout(0.1))
    
    
    #layer 3
    model.add(LSTM(units=60,return_sequences=True))
    model.add(Dropout(0.2))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.MeanSquaredError()])
    
    #model.compile(optimizer='adam',loss='mean_squared_error')
    
    return model


#From Google Machine Learning Crash Course
def train_LSTM_model(model, dataset, epochs, label_name):
    
  #implement dataset morphing here 
  
  
  features = {name:np.array(value) for name, value in dataset.items()}
  label = np.array(features.pop(label_name))
  history = model.fit(x=features, y=label, batch_size=batch_size,
                      epochs=epochs, shuffle=True) 


  # The list of epochs is stored separately from the rest of history.
  epochs = history.epoch
    
  # To track the progression of training, gather a snapshot
  # of the model's mean squared error at each epoch. 
  hist = pd.DataFrame(history.history)
  mse = hist["mean_squared_error"]
    
  return epochs, mse
    

def main():
    df = pd.read_pickle('../data/2YStockDFLowCriteria.pkl')
    #prepare_LSTM_data(df)
    q = e.load_obj('ticker_time_series_low_criteria')
    
if __name__ == '__main__':
    main()
