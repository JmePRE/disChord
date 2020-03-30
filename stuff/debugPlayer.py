
import numpy as np
import sounddevice as sd
import time
import librosa
from classifier import classify
import csv


def play_predicts(file, mode=0):
    y, sr = librosa.load(file)
    labels, beats = classify(y, sr, mode)
    cbeat = 0
    print(len(beats))
    st = time.time()
    sd.play(y, sr)
    while(time.time()<st+beats[-1]):
        if(time.time()>st+beats[cbeat]):
            print(labels[cbeat])
            cbeat += 1
    sd.stop()
    return


def play_check(sf, cf, fast=False, mode=0):
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
    labels0, beats0 = classify(y, sr, mode)

    HOP_LENGTH = 512
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    while (beat_frames[-1] < (len(y) // HOP_LENGTH)):
        beat_frames = np.append(beat_frames, (beat_frames[-1] + int(
            np.mean([(beat_frames[i] - beat_frames[i - 1]) for i in range(1, len(beat_frames))]))))
    beats = librosa.frames_to_time(beat_frames, sr)
    cbeat = 0
    print(len(beats0))
    st = time.time()
    if not fast:
        sd.play(y, sr)
        while (time.time() < st + beats[-1]):
            if (time.time() > st + beats[cbeat]):
                try:
                    print("actual: " + str(labels[cbeat]) + " / pred:" + str(labels0[cbeat]))
                except IndexError:
                    pass
                cbeat += 1
        sd.stop()
    else:
        for c in range(len(labels)):
            try:
                print("actual: " + str(labels[c]) + " / pred:" + str(labels0[c]))
            except IndexError:
                pass

    print(len(labels))
    print(sum([labels[i] == labels0[i] for i in range(min([len(labels), len(labels0)]))]))
    return


def play_csv(sf, cf):
    labels = []
    sum_beats = 0
    with open(cf, newline='') as csvfile:
        r = csv.reader(csvfile)
        for row in r:
            sum_beats += int(row[2])
            for i in range(int(row[2])):
                # y_train_key[ri][chdict.get(row[0])] = 1
                # y_train_min_maj[ri][mmdict.get(row[1])] = 1
                # print(row[:2])
                labels.append(tuple(row[:2]))
    y, sr = librosa.load(sf)
    HOP_LENGTH = 512
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    while (beat_frames[-1] < (len(y) // HOP_LENGTH)):
        beat_frames = np.append(beat_frames, (beat_frames[-1] + int(
            np.mean([(beat_frames[i] - beat_frames[i - 1]) for i in range(1, len(beat_frames))]))))
    beats = librosa.frames_to_time(beat_frames, sr)
    cbeat = 0
    print(len(beats))
    print(sum_beats)
    st = time.time()
    sd.play(y, sr)
    while (time.time() < st + beats[-1]):
        if (time.time() > st + beats[cbeat]):
            try:
                print("actual: " + str(labels[cbeat]))
            except IndexError:
                pass
            cbeat += 1
    sd.stop()
    return


# play_predicts('test/Maroon_5_-_Memories.wav', mode=3)
# play_check("data/losing_my_religion.wav", "data/losing_my_religion.csv", fast=True, mode=3)

play_check('data/All_Of_Me.wav','data/All_Of_Me.csv', fast=True, mode=3)
