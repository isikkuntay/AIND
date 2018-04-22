import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    x_s = []
    y_s = 0
    for n in range(len(series)-window_size):
        for i in range(window_size):
            x_s.append(series[n+i])
        y_s = series[n+window_size]
        X.append(x_s)
        x_s = []
        y.append(y_s)
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    hidden_size = 5
    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(window_size,1)))
    model.add(Dense(1))
    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    
    for char in text:
        if char not in punctuation:
            text = text.replace(char, ' ')
            
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    input_s = []
    output_s = []
    idx = 0
    while (idx < len(text)-window_size):
        for i in range(window_size):
            input_s.append(text[idx+i])
        output_s = text[idx+window_size]
        inputs.append(''.join(input_s))
        input_s = []
        outputs.append(output_s)
        idx += step_size
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    hidden_size = 200
    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='softmax'))
    return model
