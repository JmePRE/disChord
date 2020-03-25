
import numpy as np
import sounddevice as sd
import time
import librosa
from classifier import classify


y, sr = librosa.load("test/Soviet_Union_National_Anthem.wav")
labels, beats = classify(y, sr)
cbeat = 0
st = time.time()
sd.play(y, sr)
while(time.time()<st+beats[-1]):
    if(time.time()>st+beats[cbeat]):
        print(labels[cbeat])
        cbeat += 1
sd.stop()
