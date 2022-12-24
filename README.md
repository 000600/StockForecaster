# Stock Forecaster

## The Neural Network

This neural network is an RNN that forecasts the closing price of a publicly listed stock based on the stock's closing price on the previous *n* days (*n* is named **seq_len** in the file and has a default value of 5). The model will predict a scaled stock price, which can be converted into a realistic (normalized) stock price with the line **scaler.inverse_transform(model.predict(x))**, where *x* is a list of the previous **seq_len** closing prices. Since the model is a regression algorithm, it uses a mean squared error loss function and has 1 output neuron. The model uses a standard Adam optimizer with a learning rate of 0.001, and has an architecture consisting of:
- 1 Batch Normalization layer
- 1 LSTM layer (with 50 neurons, an input shape of 5, and a Tanh activation function)
- 1 Hidden layer (an LSTM layer with 50 neurons and a Tanh activation function)
- 2 Dropout layers (one after each hidden layer and input layer and each with a dropout rate of 0.2)
- 1 Output layer (with 1 output neuron and no activation function)

Feel free to further tune the hyperparameters or build upon the model!

## The Dataset
The dataset consists of input and output value sets for each day. For any given day, the input value is [*closing price 5 days ago*, *closing price 4 days ago*, *closing price 3 days ago*, *closing price 2 days ago*, *closing price 1 day ago*] and the output value is the closing price of the stock that day: [*closing price today*]. The number of values in the input list is determined by the **seq_len** variable. The data is downloaded from the Yahoo Finance API (the Python package is yfinance) and contains the closing price of a stock every day since the stock's first public offering (either through an IPO or SPAC). In order to change which stock's closing price is represented in the dataset, simply change the ticker specified near the top of the code.

Please note that the data is scaled with Scikit-Learn's **MinMaxScaler()**, although all values are rescaled before they are graphed our outputted.

## Libraries
This neural network was created with the help of the Tensorflow, Scikit-Learn, and Yahoo Finance API libraries.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install
- Scikit-Learn's Website: https://scikit-learn.org/stable/
- Scikit-Learn's Installation Instructions: https://scikit-learn.org/stable/install.html
- yfinance installation instructions: https://pypi.org/project/yfinance/

## Disclaimer
Please note that I do not recommend, endorse, or encourage the use of any of my work here in actual financial use or application in any way. Nothing here is meant to advise or guide financial decisions.
