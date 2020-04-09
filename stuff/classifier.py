import librosa
import numpy as np
from dsp_preprocess import chroma_process
from dsp_preprocess import chroma_process_split
from model import *
import csv

from sklearn.preprocessing import MultiLabelBinarizer

# Call classify on a loaded sound file to get an array of chord predictions and corresponding timestamps in song


def classify(y0, sr0, mode=5):

    #  Select model based on mode flag, make predictions for sound file
    if mode == 0 or mode == 1:

        x_data, frame_times = chroma_process(y0, sr0)
        if(mode == 0):
            model = model_0()
            model.load_weights('model_test.h5')
            preds = model.predict(x_data)
        else:
            model0 = model_1()
            model1 = model_2()
            model0.load_weights('model_1_test.h5')
            model1.load_weights('model_2_test.h5')
            preds_0 = model0.predict(x_data)
            preds_1 = model1.predict(x_data)
            preds = np.concatenate([preds_0, preds_1], axis=1)
    elif(mode == 3):
        x_data, frame_times = chroma_process_split(y0, sr0)
        model3 = model_3()
        model3.load_weights('model_3_test.h5')
        preds = model3.predict(x_data)
    elif (mode == 4):
        x_data, frame_times = chroma_process_split(y0, sr0)
        model4 = model_4()
        model4.load_weights('model_4_test.h5')
        preds = model4.predict(x_data)
    elif (mode==5):
        x_data, frame_times = chroma_process_split(y0, sr0)
        model0 = model_5()
        model1 = model_6()
        model0.load_weights('model_5_test.h5')
        model1.load_weights('model_6_test.h5')
        preds_0 = model0.predict(x_data)
        preds_1 = model1.predict(x_data)
        preds = np.concatenate([preds_0, preds_1], axis=1)

    labels = [["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "NC", "M", "m"]]
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)
    pr = np.zeros(preds.shape)
    pi = 0
    for p in preds:
        root = np.where(p == max(p[:12]))  # find index of key with highest confidence
        mmn = np.where(p == max(p[12:]))  # same for minor/major
        p = np.zeros(p.shape)
        p[root] = 1
        p[mmn] = 1
        pr[pi] = p
        pi += 1
    label_arr = mlb.inverse_transform(pr)  # returns array of text labels corresponding to prediction
    return label_arr, frame_times  # returns labels and corresponding beat timings


def classify_to_csv(csv_file, y0, sr0, mode=5):  # exactly what it says
    label_arr, frames = classify(y0, sr0, mode=mode)
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(label_arr)):
            row = [label_arr[i][0], label_arr[i][1], frames[i]]
            print(row)
            writer.writerow(row)
    print('done')
    return


y, sr = librosa.load('test/Maroon_5_-_Memories.wav')
classify_to_csv('preds.csv', y, sr)
