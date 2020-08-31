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
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.svm import SVC
import ML_p1_data_preperation as p
import copy

#Keras imports
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

'''
Sadly, when i lognormally scaled the prices and volumes,
I had not saved the scalar so I can perform an inverse_transform
to compare the predicted prices with the real prices.
Good news for me. I had saved the pre-scaling prices.
Thus I can rebuild the scalar here, scale prices, then use
the inverse to inverse the model predicted prices.

However, I faced issues with broadcasting shapes. 
Thus I had to do a very spaghetti fix to it.
I may have been able to create a more elequent fix
then this but for the purpose I think it's good enough.
'''
def compare_LSTM_results(model,x_test,y_test,ticker_names):
    df = pd.read_pickle('../data/Low_Criteria_Pre_price_v_scale.pkl')
    #df.to_excel('real_price.xlsx')
    vpw,ppw,cv = p.filtered_columns(df)
    p.print_full(ppw.head())
    
    df = ppw
    
    for column in df:
        df.loc[:,column] = np.log(df[column])
    

    '''Because of broadcast shape issues i had to do this part slightly spaghetti'''
    
    
    #first I make new dataframe with only test values. Note log is already calculated.
    df4 = df.loc[ticker_names]
    
    #then i split the semi and last column:
    semi_last_col = copy.deepcopy(df4.iloc[:,-2])
    last_col = copy.deepcopy(df4.iloc[:,-1])
    
    #then I run the predict.
    x_test = numpify_LSTM_data(x_test)
    y_test = numpify_LSTM_data(y_test)

    predicted_stock_prices = model.predict(x_test,batch_size = 1)
    
    scaler2 = StandardScaler()
    
    #then i standard scale
    df4 = pd.DataFrame(columns = df4.columns,index=df4.index,data=scaler2.fit_transform(df4.values.T).T)
    
    #then i place the predicted price as last column
    df4.iloc[:,-1] = predicted_stock_prices
    
    #then i inverse transform.
    df4 = pd.DataFrame(columns=df4.columns,index=df4.index,data=scaler2.inverse_transform(df4.values.T).T)
    for column in df4:
        df4.loc[:,column] = np.exp(df4[column])
    
    
    compare_df = pd.concat([np.exp(semi_last_col),np.exp(last_col),df4.iloc[:,-1]],axis=1,sort=False)
    compare_df.columns = ['Previous Price', 'Real Next Price','Predicted Next Price']
    compare_df['Error %'] = compare_df.apply(lambda row: (np.abs((row.iloc[1]-row.iloc[2]))/np.abs((row.iloc[1]+row.iloc[2])/2))*100, axis=1)
    compare_df['Real % Change'] = compare_df.apply(lambda row: ((row.iloc[1] - row.iloc[0])/np.abs(row.iloc[0]))*100,axis=1)
    compare_df['Predicted % Change'] = compare_df.apply(lambda row: ((row.iloc[2] - row.iloc[0])/np.abs(row.iloc[0]))*100,axis=1)
    compare_df['Difference in % Change'] = compare_df.apply(lambda row: (row.loc['Real % Change'] - row.loc['Predicted % Change']),axis=1)
    # to see if model predicted atleast 3 % increase when there was a 3% increase.
    compare_df['Real Increase 3%'] = compare_df.apply(lambda row: (True if (row.loc['Real % Change'] >= 3.00) else False),axis=1)
    compare_df['Predicted Increase 3%'] = compare_df.apply(lambda row: (True if  (row.loc['Predicted % Change'] >= 3.00) else False),axis=1)
    
    #Could've done this as a if real and predicted increase 3% then True else false but I had implemented this before those two.
    compare_df['Right Investment Prediction'] = compare_df.apply(lambda row: (True if ((row.loc['Real % Change'] >= 3.00) and (row.loc['Predicted % Change'] >= 3.00)) else False),axis=1)
    
    
    compare_df.to_excel('LSTM_SPP_results2.xlsx')
    
    #Some statistics:
    true_count_p = compare_df['Predicted Increase 3%'].sum()
    true_count_r = compare_df['Real Increase 3%'].sum()
    true_count_t = compare_df['Right Investment Prediction'].sum()
    mean_error_pred = compare_df['Error %'].mean()
    mean_perc_change = compare_df['Real % Change'].mean()
    mean_p_perc_change = compare_df['Predicted % Change'].mean()
    mean_diff_perc = compare_df['Difference in % Change'].mean()
    
    print('True count predicted 3% increase: ',true_count_p)
    print('True count real 3% increase: ', true_count_r)
    print('True count predicted and real 3% increase', true_count_t)
    print('Mean error percentage between real and predicted',mean_error_pred)
    print('Mean real % change: ',mean_perc_change)
    print('Mean predicted % change: ', mean_p_perc_change)
    print('Mean difference in % change: ',mean_diff_perc)
    
'''From Google Machine Learning Crash Course'''
def plot_the_loss_curve(epochs, mse):
  """Plot a curve of loss vs. epoch."""

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Mean Squared Error")

  plt.plot(epochs, mse, label="Loss")
  plt.legend()
  plt.ylim([mse.min()*0.95, mse.max() * 1.03])
  plt.savefig('LossCurve.png')
  plt.show()  
  
  


def plot_test_results_to_real(real,predicted):
    plt.plot(real, color = 'black', label = 'Real Stock Prices')
    plt.plot(predicted, color = 'green', label = 'Predicted Stock Prices')

    plt.legend()
    plt.show()



'''
LSTM needs dada in a specific format
'''
def prepare_LSTM_data(df):
    vpw, ppw, cv = p.filtered_columns(df) 
    
    time_related_cols = [col for col in df if col not in cv]
    
    time_related_df = df[time_related_cols]
    
    time_series_df_by_ticker = []
    ticker_names = []
    y_values_tsdbt = []
    
    col_names = time_related_df.columns
    col_names_length = len(col_names)
    g = 0
    '''first, we make a list of dataframes where each dataframe is a ticker
        where the rows are the Dates and the columns are price and volume at 
        that date. 
    '''
    for i, r in time_related_df.iterrows():
        print(g)
        g+=1
        
        j = 0 
        tdf = pd.DataFrame(columns=['Price', 'Volume'])
        ticker_names.append(r.name)
        
        while j < col_names_length-1:   
            close = r[col_names[j]]
            volume = r[col_names[j+1]]
            date = col_names[j][:16]
            tdf.loc[date] = [close, volume]
            j+=2
            
        y_row = tdf.iloc[-1]
        tdf = tdf[:-1]
        
        time_series_df_by_ticker.append(tdf)
        y_values_tsdbt.append(y_row)
        
        #print('close', close , ' volume' , volume, ' date' , date)
    
    
    e.save_obj(time_series_df_by_ticker,'x_ticker_time_series_low_criteria')
    e.save_obj(y_values_tsdbt,'y_ticker_time_series_low_criteria')
    return ticker_names



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
def create_LSTM_model(learning_rate,x_train):
    model = Sequential()
    
    df = pd.read_pickle('../data/2YStockDFLowCriteria.pkl')
    
    
    #layer 1
    model.add(LSTM(units=64,return_sequences=True,input_shape=x_train.shape[1:]))
    model.add(Dropout(0.3))
    
    #layer 2
    model.add(LSTM(units=64,return_sequences=True))
    model.add(Dropout(0.1))
    
    #layer 3
    model.add(LSTM(units=64,return_sequences=False))
    model.add(Dropout(0.2))
    
    
    model.add(Dense(units=1))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.MeanSquaredError()])
    
    #model.compile(optimizer='adam',loss='mean_squared_error')
    
    return model


#From Google Machine Learning Crash Course
def train_LSTM_model(model, x_vals, y_vals,batchsize, epochs,v_split=0.2):
    
  history = model.fit(x=x_vals, y=y_vals, batch_size=batchsize,
                      epochs=epochs, shuffle=False,validation_split=v_split) 


  # The list of epochs is stored separately from the rest of history.
  epochs = history.epoch
    
  # To track the progression of training, gather a snapshot
  # of the model's mean squared error at each epoch. 
  hist = pd.DataFrame(history.history)
  mse = hist["mean_squared_error"]
    
  return epochs, mse, hist
    
def numpify_LSTM_data(data):
    np_l = []
    for i in data:
        np_l.append(i.to_numpy())
    return np.array(np_l)

def fix_y_values(data):
    for i in data:
        i.drop(labels='Volume', inplace=True)
    return data

def main():
    #df = pd.read_pickle('../data/2YStockDFLowCriteria.pkl')
    #ticker_names = prepare_LSTM_data(df)
    #e.save_obj(ticker_names,'ticker_names')
    
    LSTM_data_X = e.load_obj('../data/x_ticker_time_series_low_criteria')
    LSTM_data_Y = fix_y_values(e.load_obj('../data/y_ticker_time_series_low_criteria'))
    ticker_names = e.load_obj('ticker_names')
    
    x_train = LSTM_data_X[0:3041]
    x_validate = LSTM_data_X[2366:3041]
    x_test = LSTM_data_X[3041:]
    
    y_train = LSTM_data_Y[0:3041]
    y_validate = LSTM_data_Y[2366:3041]
    y_test = LSTM_data_Y[3041:]
    
    ticker_names = ticker_names[3041:]
    
    x_train = numpify_LSTM_data(x_train)
    y_train = numpify_LSTM_data(y_train)
    print('ready to create model')
    model = create_LSTM_model(0.005,x_train)
    print('ready to train model')
    epochs,mse,history = train_LSTM_model(model, x_train, y_train, 64, 20,v_split=0.2)
    print('model trained.')
    list_of_metrics_to_plot = ['accuracy'] 
    plot_the_loss_curve(epochs, mse)

    compare_LSTM_results(model,x_test,y_test,ticker_names)
    
if __name__ == '__main__':
    main()
