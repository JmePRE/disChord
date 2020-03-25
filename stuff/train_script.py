from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from dsp_preprocess import label_process
from keras.optimizers import Adam
import os
'''
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")

args = vars(ap.parse_args())
'''
EPOCHS = 30
INIT_LR = 1e-3
BS = 20
print('Build model...')
model = Sequential()
model.add(Dense(units=12, activation='sigmoid', input_shape=(12,)))
model.add(Dense(48, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(15, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

xt, yt = np.zeros(12), np.zeros(15)
for(root, dir, file) in os.walk('data/labels/'):
    for f in file:
        x0, y0 = label_process(("data/labels/"+f), ("data/soundfiles/"+f[:-4]+".wav"))

        print(xt.shape)
        print(yt.shape)
        print(x0.shape)
        print(y0.shape)
        xt = np.vstack([x0, xt])
        yt = np.vstack([y0, yt])
print(xt.shape)
print(yt.shape)
# train the network
print("[INFO] training network...")
H = model.fit(
    xt, yt, validation_split=0.2,
	epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save("model_test.h5")


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.show()
