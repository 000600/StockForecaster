# Stock Forecaster

## The Neural Network

This neural network forecasts the next day's opening price of a publicly listed stock based on the previous *n* days (*n* has a default value of 5). The model will predict a value a scaled stock price, which can be converted into a realistic stock price with the line **scaler.inverse_transform(model.predict(x))** where *x* is a list of the previous five closing prices. Since the model in a regression algorithm, it uses a  mean squared error loss function and has 1 output neuron. The model uses a standard Adam optimizer with a learning rate of 0.001, and has an architecture consisting of:
- 1 Batch Normalization layer
- 1 Input layer (with 128 input neurons and a ReLU activation function)
- 2 Hidden layers (each with 64 neurons and a ReLU activation function)
- 3 Dropout layers (one after each hidden layer and input layer and each with a dropout rate of 0.4)
- 1 Output layer (with 1 output neuron and a sigmoid activation function)

Feel free to further tune the hyperparameters or build upon the model!

## The Dataset
The dataset can be found at this link: https://www.kaggle.com/datasets/jillanisofttech/brain-stroke-dataset. Credit for the dataset collection goes to **whxna-0615**, **stpete_ishii**, and others on *Kaggle*. It describes whether or not a person will have a stroke (encoded as 0 or 1) based on multiple factors, including:
- Age
- Hypertension (0 : no patient hypertension, 1 : hypertension within patient)
- Average glucose level 
- Body mass index (BMI)
- Smoking status

Note that the initial dataset is biased (this statistic can be found on the data's webpage); it contains a higher representation of non-stroke cases (encoded as 0's in this model) than stroke cases (encoded as 1's in this model). This issue is addressed within the classifier file using 
Imbalanced-Learn's **SMOTE()**, which oversamples the minority class within the dataset.

## Libraries
This neural network was created with the help of the Tensorflow, Imbalanced-Learn, and Scikit-Learn libraries.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install

## Disclaimer
Please note that I do not recommend, endorse, or encourage the use of any of my work here in actual medical use or application in any way. 
