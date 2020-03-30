from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import InputLayer

def model_0():
    model = Sequential()
    model.add(Dense(units=12, activation='sigmoid', input_shape=(12,)))
    model.add(Dense(48, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(48, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(15, activation='sigmoid'))
    return model

def model_1():
    model = Sequential()
    model.add(Dense(units=12, activation='sigmoid', input_shape=(12,)))
    model.add(Dense(48, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(12, activation='sigmoid'))
    return model


def model_2():
    model = Sequential()
    model.add(Dense(units=12, activation='sigmoid', input_shape=(12,)))
    model.add(Dense(48, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='sigmoid'))
    return model

def model_3():
    model = Sequential()
    model.add(InputLayer(input_shape=(4, 12)))
    model.add(Flatten())
    model.add(Dense(48, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(36, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(15, activation='sigmoid'))
    return model