import librosa
import numpy as np
from dsp_preprocess import chroma_process
from dsp_preprocess import chroma_process_split
from model import model_0
from model import model_1
from model import model_2
from model import model_3

from sklearn.preprocessing import MultiLabelBinarizer

def classify(y0, sr0, mode=0):
    print('Build model...')
    if(mode == 0 or mode == 1):

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
        model3.load_weights('model_test_alt.h5')
        preds = model3.predict(x_data)

    labels = [["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "NC", "M", "m"]]
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)
    pr = np.zeros(preds.shape)
    pi = 0
    for p in preds:
        root = np.where(p == max(p[:12]))
        mmn = np.where(p == max(p[12:]))
        p = np.zeros(p.shape)
        p[root] = 1
        p[mmn] = 1
        pr[pi] = p
        pi += 1
    label_arr = mlb.inverse_transform(pr)
    return label_arr, frame_times
