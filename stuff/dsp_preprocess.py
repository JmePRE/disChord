
import librosa.display
import matplotlib.pyplot as plt

from sklearn.preprocessing import MultiLabelBinarizer
import librosa
import numpy as np
import scipy
import csv
# y0, sr0 = librosa.load(librosa.util.example_audio_file())
def chroma_process(y, sr):
    HOP_LENGTH = 512
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=8)
    # Beat track on the percussive signal
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)

    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, bins_per_octave=12*3, hop_length=HOP_LENGTH)
    chroma_filter = np.minimum(chroma,
                               librosa.decompose.nn_filter(chroma,
                                                           aggregate=np.median,
                                                           metric='cosine'))
    chroma_smooth = scipy.ndimage.median_filter(chroma_filter, size=(1, 9))
    # beat_frames = np.pad(beat_frames, 1, 'constant', constant_values=0)
    print(beat_frames)
    print(len(chroma[0]))
    beat_frames = np.append(beat_frames, (beat_frames[-1]+int(np.mean([(beat_frames[i] - beat_frames[i - 1]) for i in range(1, len(beat_frames))]))))
    print(tempo)

    beat_chroma = librosa.util.sync(chroma_smooth, beat_frames, aggregate=np.median)
    chroma_each_beat = np.transpose(beat_chroma)
    # np.set_printoptions(threshold=np.inf)
    # print(chroma_each_beat[-16:])


    # bc = np.split(np.transpose(chroma_smooth), beat_frames)
    # print(len(chroma_each_beat))
    return chroma_each_beat


# bc = chroma_process(y0, sr0)


def label_process(label_csv, sound_file):
    # root, number of beats
    y0, sr0 = librosa.load(sound_file)
    x_train = chroma_process(y0, sr0)
    labels = [["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "NC", "M", "m"]]
    # y_train_key = np.zeros((len(x_train), 13))
    # y_train_min_maj = np.zeros((len(x_train), 3))
    y_train = np.zeros((len(x_train), len(labels[0])))
    mlb = MultiLabelBinarizer()

    chdict = {"A": 0, "A#": 1, "Bb": 1, "B": 2, "C": 3, "C#": 4, "Db": 4, "D": 5, "D#": 6,
              "Eb": 6, "E": 7, "F": 8, "F#": 9, "Gb": 9, "G": 10, "G#": 11, "Ab": 11, "NC":12}
    # mmdict = {"M": 0, "m": 1, "NC": 2}
    mlb.fit(labels)
    print(mlb.classes_)
    with open(label_csv, newline='') as csvfile:
        r = csv.reader(csvfile)

        ri = 0
        for row in r:
            for i in range(int(row[2])):
                # y_train_key[ri][chdict.get(row[0])] = 1
                # y_train_min_maj[ri][mmdict.get(row[1])] = 1
                # print(row[:2])
                row[0] = labels[0][chdict.get(row[0])]
                y_train[ri] = mlb.transform([tuple(row[:2])])
                ri += 1
    return x_train, y_train


x,y = label_process("data/chordtest0.csv", "data/chordtest0.wav")
print(x[0].shape)
print(len(y))
# print(bc.shape)
'''
plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma_smooth, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram')
plt.tight_layout()
plt.show()'''