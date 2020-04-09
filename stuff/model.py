from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import InputLayer
from keras.layers import LSTM

# Defines all the model architectures


def model_0():  # first attempt, using chroma aggregated to beats
    model = Sequential()
    model.add(Dense(units=12, activation='sigmoid', input_shape=(12,)))
    model.add(Dense(48, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(15, activation='sigmoid'))
    return model


def model_1():  # second attempt, key only
    model = Sequential()
    model.add(Dense(units=12, activation='sigmoid', input_shape=(12,)))
    model.add(Dense(48, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(12, activation='sigmoid'))
    return model


def model_2():  # second attempt, minor/major only
    model = Sequential()
    model.add(Dense(units=12, activation='sigmoid', input_shape=(12,)))
    model.add(Dense(48, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='sigmoid'))
    return model


def model_3():  # third attempt, using semiquaver chroma grouped into beats
    model = Sequential()
    model.add(InputLayer(input_shape=(4, 12)))
    model.add(Flatten())
    model.add(Dense(48, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(15, activation='sigmoid'))
    return model


def model_4():  # fourth attempt, using semiquaver chroma again, using RNN as it is time series data
    model = Sequential()
    model.add(LSTM(100, input_shape=(4, 12)))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(15, activation='sigmoid'))
    return model


def model_5():  # fifth attempt, RNN semiquavers key only
    model = Sequential()
    model.add(LSTM(100, input_shape=(4, 12)))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(12, activation='sigmoid'))
    return model


def model_6():  # fifth attempt, RNN semiquavers minor/major only
    model = Sequential()
    model.add(LSTM(100, input_shape=(4, 12)))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    return model