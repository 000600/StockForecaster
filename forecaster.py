# Imports
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# Ticker (change to have the model predict on a different listed security)
ticker = 'BRK-B'

# Download raw data
data = yf.download(tickers = ticker, period = 'max', interval = '1d')

# Collect closing prices as data
close = data[['Close']]
df = close.values

# View data (stock price)
plt.plot(df)
plt.title(f'{ticker} Closing Price Over Time')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.show()

scaler = MinMaxScaler()
df = scaler.fit_transform(np.array(df).reshape(-1, 1))

# Split data into training and testing data
train_percent = 0.7 # Percent of data allocated to the train set
train_size = int(len(df) * train_percent)
test_size = len(df) - train_size

# Create train and test datasets
train = df[:train_size]
test = df[train_size:]

# Get total number of closing prices within each dataset
total_prices_train = len(train)
total_prices_test = len(test)

# Create datasets
seq_len = 5 # Number of previous closing prices included in a single input

x_raw_train = []
y_raw_train = []
x_raw_test = []
y_raw_test = []

# Training data
for index in range(0, total_prices_train - seq_len, 1): # Loop through the list of closing prices and pair seq_len input closing values with one output closing value
  input = train[index: index + seq_len] # Get a list of seq_len prices to create the x value
  output = train[index + seq_len] # Get the next closing price for the y value

  # Add values to corresponding dataset lists
  x_raw_train.append(input)
  y_raw_train.append(output[0])

# Testing data
for index in range(0, total_prices_test - seq_len, 1): # Loop through the list of closing prices and pair seq_len input closing values with one output closing value
  input = test[index: index + seq_len] # Get a list of seq_len prices to create the x value
  output = test[index + seq_len] # Get the next closing price for the y value

  # Add values to corresponding dataset lists
  x_raw_test.append(input)
  y_raw_test.append(output[0])

# Reshape x-values
x_train = np.array(x_raw_train)
x_test = np.array(x_raw_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] , 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] , 1)

# Turn y-values into arrays
y_train = np.array(y_raw_train)
y_test = np.array(y_raw_test)

# Initialize optimizer
opt = Adam(learning_rate = 0.001)

# Get input shape
input_shape = (x_train.shape[1], x_train.shape[2])

# Create model
model = Sequential()
model.add(BatchNormalization())

# Input LSTM layer
model.add(LSTM(50, input_shape = input_shape, return_sequences = True, activation = 'tanh'))
model.add(Dropout(0.2))

# Hidden layers
model.add(LSTM(50, activation = 'tanh'))
model.add(Dropout(0.2))

# Output layer
model.add(Dense(1))

# Configure early stopping
early_stopping = EarlyStopping(min_delta = 0.001, patience = 10, restore_best_weights = True)

# Set epochs and batch size
epochs = 100
batch_size = 64

# Compile and train model
model.compile(optimizer = opt, loss = 'mse')
history = model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, validation_data = (x_test, y_test), callbacks = [early_stopping])

# Visualize loss and validation loss
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']

plt.plot(loss, label = 'Loss')
plt.plot(val_loss, label = 'Validation Loss')
plt.title('Validation and Training Loss Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# View model's predictions compared to actual values
pred_train = model.predict(x_train) # Get model's predictions on test dataset
pred_test = model.predict(x_test) # Get model's predictions on train dataset

# Normalize scaled numbers
pred_train = scaler.inverse_transform(pred_train)
pred_test = scaler.inverse_transform(pred_test)
df_normalized = scaler.inverse_transform(df)

# Reformat pred_test so that it shows up correctly on the graph
filler = np.array([np.NaN for i in range(len(train))])
pred_test_filled = np.append(filler, pred_test)

# Visualize predictions and actual values
plt.figure(figsize = (8, 4))
plt.plot(df_normalized, label = 'Actual Closing Price') # Plot actual values
plt.plot(pred_train, label = 'Predicted Closing Price (Train Data)') # Plot predictions on training data
plt.plot(pred_test_filled, label = 'Predicted Closing Price (Test Data)') # Plot predictions on testing data
plt.xlabel('Time')
plt.ylabel(f'{ticker} Closing Price')
plt.title(f"Model's Predicted Closing Price Compared to Actual Closing Price of {ticker}")
plt.legend()
plt.show()

# View prediction sequences on random inputs

# Generate random index
index = np.random.randint(0, len(x_raw_test) - 1)

# Create seed
seed = x_test[index]
print("\nSeed:")
print(seed)
predictions = []
num_iterations = 100 # Change this number to have the model predict more values

# Loop through num_iterations times and add model's prediction to inputs each time
for iter in range(num_iterations):
  input = np.reshape(seed, (1, len(seed), 1)) # Create input
  prediction = model.predict(input) # Get model's prediction
  predictions.append(prediction[0])
  seed = np.append(seed, prediction) # Add model's prediction to seed so that it is taken into account in the next iteratior
  seed = seed[1: ] # Shift seed forward so that it maintains the correct shape (the shape is dictated by seq_len)

# Normalize scaled values
predictions = scaler.inverse_transform(predictions)

# Visualize model's predictions
plt.figure(figsize = (8, 4))
plt.plot(predictions)
plt.xlabel('Time')
plt.ylabel('Projected Price')
plt.title(f"Model's Projected Closing Price of {ticker} From a Random Seed")
plt.show()

# View model's predicted outlook on the stock after the dataset

# Create seed
seed = x_test[-1]
proj_predictions = []
num_iterations = 100 # Change this number to have the model forecast farther less far into the future

# Loop through num_iterations times and add model's prediction to inputs each time
for iter in range(num_iterations):
  input = np.reshape(seed, (1, len(seed), 1)) # Create input
  prediction = model.predict(input) # Get model's prediction
  proj_predictions.append(prediction[0])
  seed = np.append(seed, prediction) # Add model's prediction to seed so that it is taken into account in the next iteratior
  seed = seed[1: ] # Shift seed forward so that it maintains the correct shape (the shape is dictated by seq_len)

# Get total predictions on both training and testing sets
pred_total = np.append(pred_train, pred_test)

# Add filler values so that the projections appear in the right spot on the graph
filler = np.array([np.NaN for i in range(len(x_test) + len(x_train))])
proj_predictions = scaler.inverse_transform(proj_predictions) # Normalize scaled values
projected = np.append(filler, proj_predictions)

# Visualize previous predictions and new projections
plt.figure(figsize = (8, 4))
plt.plot(df_normalized, label = 'Actual Closing Price') # Plot actual values
plt.plot(pred_total, label = 'Predicted Closing Price') # Plot predictions
plt.plot(projected, label = 'Projected Closing Price') # Plot projections
plt.xlabel('Time')
plt.ylabel(f'{ticker} Closing Price')
plt.title(f"Model's Projected Closing Price of {ticker}")
plt.legend()
plt.show()
