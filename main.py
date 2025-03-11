# Importing libraries
import sklearn
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib as plt
import seaborn as sns
# import os
from datetime import datetime
import warnings
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# py -3.11 pip install 

# Read our stocks data into a DataFrame object
#  Specify delimiter to ',' to match CSV format
#  Skip errors in our CSV file because I do not want to look through that file for every little error :)
# date,open,high,low,close,volume,Name
data = pd.read_csv('stocks_5yrs.csv', delimiter=',', on_bad_lines='skip') 

print(data.shape)
print(data.sample(5))

data.info()

# Convert the date column to a datetime object
#  This will allow us to plot the data over time
data['date'] = pd.to_datetime(data['date'])
data.info()
# date vs open 
# date vs close

# Define the list of companies you want to plot
companies = ['AAPL', 'AMD', 'FB', 'GOOGL', 'AMZN', 'NVDA', 'EBAY', 'CSCO', 'IBM']

# Plotting the open and close prices of the stocks
# Create a 3x3 grid of plots to display the data for each company
plt.figure(figsize=(15, 8)) 
for index, company in enumerate(companies, 1): 
    plt.subplot(3, 3, index) 
    c = data[data['Name'] == company] 
    plt.plot(c['date'], c['close'], c="r", label="close", marker="+") 
    plt.plot(c['date'], c['open'], c="g", label="open", marker="^") 
    plt.title(company) 
    plt.legend() 
    plt.tight_layout()

# Plotting the volume of stocks being traded
plt.figure(figsize=(15, 8))
for index, company in enumerate(companies, 1):
    plt.subplot(3, 3, index)
    c = data[data['Name'] == company]
    plt.plot(c['date'], c['volume'], c='purple', marker='*')
    plt.title(f"{company} Volume")
    plt.tight_layout()
    
# Plotting the high and low prices of the stocks for Apple    
apple = data[data['Name'] == 'AAPL']
prediction_range = apple.loc[(apple['date'] > datetime(2013,1,1))
 & (apple['date']<datetime(2018,1,1))]
plt.plot(apple['date'],apple['close'])
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Apple Stock Prices")
plt.show()

close_data = apple.filter(['close'])
dataset = close_data.values
training = int(np.ceil(len(dataset) * .95))
print(training)

# Scale the data
# This is done to normalize the data so that the model can learn better
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:int(training), :]
# prepare feature and labels
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=64,
                            return_sequences=True,
                            input_shape=(x_train.shape[1], 1)))
model.add(keras.layers.LSTM(units=64))
model.add(keras.layers.Dense(32))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))
model.summary()

# Compile the model
#  We use the adam optimizer and mean squared error loss function
model.compile(optimizer='adam',
              loss='mean_squared_error')
history = model.fit(x_train,
                    y_train,
                    epochs=10)

# Test the model
#  We will test the model on the last 5% of the data
test_data = scaled_data[training - 60:, :]
x_test = []
y_test = dataset[training:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# predict the testing data
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# evaluation metrics
mse = np.mean(((predictions - y_test) ** 2))
print("MSE", mse)
print("RMSE", np.sqrt(mse))

# Plot the data
train = apple[:training]
test = apple[training:]
test['Predictions'] = predictions

plt.figure(figsize=(10, 8))
plt.plot(train['date'], train['close'])
plt.plot(test['date'], test[['close', 'Predictions']])
plt.title('Apple Stock Close Price')
plt.xlabel('Date')
plt.ylabel("Close")
plt.legend(['Train', 'Test', 'Predictions'])
