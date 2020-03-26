
import numpy as np
import sounddevice as sd
import time
import librosa
from classifier import classify
import csv


def play_predicts(file):
    y, sr = librosa.load(file)
    labels, beats = classify(y, sr)
    cbeat = 0
    st = time.time()
    sd.play(y, sr)
    while(time.time()<st+beats[-1]):
        if(time.time()>st+beats[cbeat]):
            print(labels[cbeat])
            cbeat += 1
    sd.stop()
    return


def play_csv(sf, cf):
    labels = []
    with open(cf, newline='') as csvfile:
        r = csv.reader(csvfile)
        for row in r:
            for i in range(int(row[2])):
                # y_train_key[ri][chdict.get(row[0])] = 1
                # y_train_min_maj[ri][mmdict.get(row[1])] = 1
                # print(row[:2])
                labels.append(tuple(row[:2]))
    y, sr = librosa.load(sf)
    HOP_LENGTH = 512
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    while(beat_frames[-1]<(len(y)//HOP_LENGTH)):
        beat_frames = np.append(beat_frames, (beat_frames[-1]+int(np.mean([(beat_frames[i] - beat_frames[i - 1]) for i in range(1, len(beat_frames))]))))
    beats = librosa.frames_to_time(beat_frames, sr)
    cbeat = 0
    st = time.time()
    sd.play(y, sr)
    while (time.time() < st + beats[-1]):
        if (time.time() > st + beats[cbeat]):
            print(labels[cbeat])
            cbeat += 1
    sd.stop()
    return


play_predicts("test/Soviet_Union_National_Anthem.wav")
