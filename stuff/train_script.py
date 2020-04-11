
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
import pickle
from model import *


def train(model_id, retrain=False, EPOCHS=50):
    if model_id == 0:
        model = model_0()
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        if (not os.path.exists('xt.npy')) or retrain:  # pre processed data not present
            xt, yt = np.zeros(12), np.zeros(15)  # initialize x train and y train
            donefiles = []
            print('die')
        else:
            donefiles = pickle.load(open('donefiles.pkl', 'rb'))  # load previously preprocessed data
            xt, yt = np.load('xt.npy'), np.load('yt.npy')
            print(xt.shape)
            print(yt.shape)

        for (root, dir, file) in os.walk('data/labels/'):
            for f in file:
                if (f not in donefiles) or retrain:  # process unprocessed data or all data if retrain flag set to true
                    x0, y0 = label_process(("data/labels/" + f), ("data/soundfiles/" + f[:-4] + ".wav"))
                    donefiles.append(f)
                    xt = np.vstack([x0, xt])
                    yt = np.vstack([y0, yt])

        pickle.dump(donefiles, open('donefiles.pkl', 'wb'))  # save list of already processed files
        np.save('xt.npy', xt)  # save x train values as numpy binary file
        np.save('yt.npy', yt)  # same for y train

        # train the network
        print("[INFO] training network...")
        H = model.fit(
            xt, yt, validation_split=0.2,
            epochs=EPOCHS, verbose=2)

        # save the model to disk
        print("[INFO] serializing network...")
        model.save("model_test.h5")

        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        N = EPOCHS

        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss_original")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss_original")
        plt.plot(np.arange(0, N), H.history["acc"], label="train_acc_original")
        plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc_original")

        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper left")
        plt.show()

    elif model_id == 1:
        model = model_1()
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        if (not os.path.exists('xt.npy')) or retrain:
            xt, yt = np.zeros(12), np.zeros(15)
            donefiles = []
            print('die')
        else:
            donefiles = pickle.load(open('donefiles.pkl', 'rb'))
            xt, yt = np.load('xt.npy'), np.load('yt.npy')
            print(xt.shape)
            print(yt.shape)

        for (root, dir, file) in os.walk('data/labels/'):
            for f in file:
                if (f not in donefiles) or retrain:
                    x0, y0 = label_process(("data/labels/" + f), ("data/soundfiles/" + f[:-4] + ".wav"))
                    donefiles.append(f)
                    xt = np.vstack([x0, xt])
                    yt = np.vstack([y0, yt])

        pickle.dump(donefiles, open('donefiles.pkl', 'wb'))
        np.save('xt.npy', xt)
        np.save('yt.npy', yt)
        yt1 = yt[:, :12]
        # train the network
        print("[INFO] training network...")
        H = model.fit(
            xt, yt1, validation_split=0.2,
            epochs=EPOCHS, verbose=2)

        # save the model to disk
        print("[INFO] serializing network...")
        model.save("model_1_test.h5")

        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        N = EPOCHS

        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss_key")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss_key")
        plt.plot(np.arange(0, N), H.history["acc"], label="train_acc_key")
        plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc_key")

        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper left")
        plt.show()

    elif model_id == 2:
        model = model_2()
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        if (not os.path.exists('xt.npy')) or retrain:
            xt, yt = np.zeros(12), np.zeros(15)
            donefiles = []
            print('die')
        else:
            donefiles = pickle.load(open('donefiles.pkl', 'rb'))
            xt, yt = np.load('xt.npy'), np.load('yt.npy')
            print(xt.shape)
            print(yt.shape)

        for (root, dir, file) in os.walk('data/labels/'):
            for f in file:
                if (f not in donefiles) or retrain:
                    x0, y0 = label_process(("data/labels/" + f), ("data/soundfiles/" + f[:-4] + ".wav"))
                    donefiles.append(f)
                    xt = np.vstack([x0, xt])
                    yt = np.vstack([y0, yt])

        pickle.dump(donefiles, open('donefiles.pkl', 'wb'))
        np.save('xt.npy', xt)
        np.save('yt.npy', yt)
        yt2 = yt[:, 12:]
        # train the network
        print("[INFO] training network...")
        H = model.fit(
            xt, yt2, validation_split=0.2,
            epochs=EPOCHS, verbose=2)

        # save the model to disk
        print("[INFO] serializing network...")
        model.save("model_2_test.h5")

        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        N = EPOCHS

        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss_mm")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss_mm")
        plt.plot(np.arange(0, N), H.history["acc"], label="train_acc_mm")
        plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc_mm")

        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper left")
        plt.show()

    elif model_id == 3 or model_id == 4:
        if model_id == 4:
            model = model_4()
        else:
            model = model_3()
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        if (not os.path.exists('xt_alt.npy')) or retrain:
            xt, yt = np.zeros((1, 4, 12)), np.zeros(15)
            donefiles = []
            print('die')
        else:
            donefiles = pickle.load(open('donefiles_alt.pkl', 'rb'))
            xt, yt = np.load('xt_alt.npy'), np.load('yt_alt.npy')
            print(xt.shape)
            print(yt.shape)

        for (root, dir, file) in os.walk('data/labels/'):
            for f in file:
                if (f not in donefiles) or retrain:
                    x0, y0 = label_process(("data/labels/" + f), ("data/soundfiles/" + f[:-4] + ".wav"), split=True)
                    donefiles.append(f)
                    print(xt.shape)
                    print(x0.shape)
                    xt = np.concatenate((x0, xt))
                    yt = np.vstack([y0, yt])

        np.set_printoptions(threshold=np.inf)
        print(xt[1:21])
        print(yt[1:21])
        pickle.dump(donefiles, open('donefiles_alt.pkl', 'wb'))
        np.save('xt_alt.npy', xt)
        np.save('yt_alt.npy', yt)

        # train the network
        print("[INFO] training network...")
        H = model.fit(
            xt, yt, validation_split=0.2,
            epochs=EPOCHS, verbose=2)

        # save the model to disk
        print("[INFO] serializing network...")

        if model_id == 4:
            model.save("model_4_test.h5")
        else:
            model.save("model_3_test.h5")
        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        N = EPOCHS

        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss_alt")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss_alt")
        plt.plot(np.arange(0, N), H.history["acc"], label="train_acc_alt")
        plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc_alt")

        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper left")
        plt.show()

    elif model_id == 5:
        model = model_5()
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        if (not os.path.exists('xt_alt.npy')) or retrain:
            xt, yt = np.zeros((1, 4, 12)), np.zeros(15)
            donefiles = []
            print('die')
        else:
            donefiles = pickle.load(open('donefiles_alt.pkl', 'rb'))
            xt, yt = np.load('xt_alt.npy'), np.load('yt_alt.npy')
            print(xt.shape)
            print(yt.shape)

        for (root, dir, file) in os.walk('data/labels/'):
            for f in file:
                if (f not in donefiles) or retrain:
                    x0, y0 = label_process(("data/labels/" + f), ("data/soundfiles/" + f[:-4] + ".wav"), split=True)
                    donefiles.append(f)
                    print(xt.shape)
                    print(x0.shape)
                    xt = np.concatenate((x0, xt))
                    yt = np.vstack([y0, yt])

        np.set_printoptions(threshold=np.inf)
        pickle.dump(donefiles, open('donefiles_alt.pkl', 'wb'))
        np.save('xt_alt.npy', xt)
        np.save('yt_alt.npy', yt)
        yt1 = yt[:, :12]
        # train the network
        print("[INFO] training network...")
        H = model.fit(
            xt, yt1, validation_split=0.2,
            epochs=EPOCHS, verbose=2)

        # save the model to disk
        print("[INFO] serializing network...")
        model.save("model_5_test.h5")

        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        N = EPOCHS

        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss_key")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss_key")
        plt.plot(np.arange(0, N), H.history["acc"], label="train_acc_key")
        plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc_key")

        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper left")
        plt.show()

    elif model_id == 6:
        model = model_6()
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        if (not os.path.exists('xt_alt.npy')) or retrain:
            xt, yt = np.zeros((1, 4, 12)), np.zeros(15)
            donefiles = []
            print('die')
        else:
            donefiles = pickle.load(open('donefiles_alt.pkl', 'rb'))
            xt, yt = np.load('xt_alt.npy'), np.load('yt_alt.npy')
            print(xt.shape)
            print(yt.shape)

        for (root, dir, file) in os.walk('data/labels/'):
            for f in file:
                if (f not in donefiles) or retrain:
                    x0, y0 = label_process(("data/labels/" + f), ("data/soundfiles/" + f[:-4] + ".wav"), split=True)
                    donefiles.append(f)
                    print(xt.shape)
                    print(x0.shape)
                    xt = np.concatenate((x0, xt))
                    yt = np.vstack([y0, yt])

        np.set_printoptions(threshold=np.inf)
        pickle.dump(donefiles, open('donefiles_alt.pkl', 'wb'))
        np.save('xt_alt.npy', xt)
        np.save('yt_alt.npy', yt)
        yt1 = yt[:, 12:]
        # train the network
        print("[INFO] training network...")
        H = model.fit(
            xt, yt1, validation_split=0.2,
            epochs=EPOCHS, verbose=2)

        # save the model to disk
        print("[INFO] serializing network...")
        model.save("model_6_test.h5")

        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        N = EPOCHS

        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss_mm")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss_mm")
        plt.plot(np.arange(0, N), H.history["acc"], label="train_acc_mm")
        plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc_mm")

        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper left")
        plt.show()
    return


train(5, EPOCHS=10)
