from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from dsp_preprocess import chroma_process
from keras.optimizers import Adam

from sklearn.preprocessing import MultiLabelBinarizer


print('Build model...')
model = Sequential()
model.add(Dense(units=12, activation='sigmoid', input_shape=(12,)))
model.add(Dense(48, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(15, activation='sigmoid'))

model.load_weights('model_test.h5')

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

x_data = chroma_process(librosa.load("test/Soviet_Union_National_Anthem.wav"))
preds = model.predict(x_data)

labels = [["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "NC", "M", "m"]]
mlb = MultiLabelBinarizer()
mlb.fit(labels)
for p in preds:
    print(mlb.inverse_transform(p))
