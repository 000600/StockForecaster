# Stock Forecaster

## The Neural Network

This neural network is an RNN that forecasts the next day's opening price of a publicly listed stock based on the previous *n* days (*n* is named **seq_len** in the file has a default value of 5). The model will predict a value a scaled stock price, which can be converted into a realistic stock price with the line **scaler.inverse_transform(model.predict(x))** where *x* is a list of the previous **seq_len** closing prices. Since the model in a regression algorithm, it uses a mean squared error loss function and has 1 output neuron. The model uses a standard Adam optimizer with a learning rate of 0.001, and has an architecture consisting of:
- 1 Batch Normalization layer
- 1 LSTM layer (with 50 neurons, an input shape of 5, and a Tanh activation function)
- 1 Hidden layer (an LSTM layer with 50 neurons and a Tanh activation function)
- 2 Dropout layers (one after each hidden layer and input layer and each with a dropout rate of 0.2)
- 1 Output layer (with 1 output neuron and no activation function)

Feel free to further tune the hyperparameters or build upon the model!

## The Dataset
The dataset consists of input and output value sets for each day. For any given day, the input value is [*closing price 5 days ago*, *closing price 4 days ago*, *closing price 3 days ago*, *closing price 2 days ago*, *closing price 1 day ago*] and output value is the closing price of the stock that day. The number of values in the input list is determined by the **seq_len** variable. The data is downloaded from the Yahoo Finance API, yfinance, and concerns the entire stock price timeframe (since the stock's IPO or SPAC), with closing prices collected every day.

## Libraries
This neural network was created with the help of the Tensorflow, Imbalanced-Learn, and Scikit-Learn libraries.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install

## Disclaimer
Please note that I do not recommend, endorse, or encourage the use of any of my work here in actual medical use or application in any way. 
