
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
    # print(len(beat_frames))
    while(beat_frames[-1]<(len(y)//HOP_LENGTH)):
        beat_frames = np.append(beat_frames, (beat_frames[-1]+int(np.mean([(beat_frames[i] - beat_frames[i - 1]) for i in range(1, len(beat_frames))]))))

    # print([(beat_frames[i] - beat_frames[i - 1]) for i in range(1, len(beat_frames))])
    beat_chroma = librosa.util.sync(chroma_smooth, beat_frames, aggregate=np.median)
    chroma_each_beat = np.transpose(beat_chroma)
    # print(len(chroma_each_beat))
    return chroma_each_beat, librosa.frames_to_time(beat_frames, sr)


def label_process(label_csv, sound_file, split=False):
    print(sound_file)
    # root, number of beats
    y0, sr0 = librosa.load(sound_file)
    if split:
        x_train, beat_frames = chroma_process_split(y0, sr0)
    else:
        x_train, beat_frames = chroma_process(y0, sr0)
    labels = [["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "NC", "M", "m"]]

    y_train = np.zeros((len(x_train), len(labels[0])))
    mlb = MultiLabelBinarizer()
    chdict = {"A": 0, "A#": 1, "Bb": 1, "B": 2, "C": 3, "C#": 4, "Db": 4, "D": 5, "D#": 6,
              "Eb": 6, "E": 7, "F": 8, "F#": 9, "Gb": 9, "G": 10, "G#": 11, "Ab": 11, "NC":12}
    mlb.fit(labels)
    with open(label_csv, newline='') as csvfile:
        r = csv.reader(csvfile)
        ri = 0
        for row in r:
            for i in range(int(row[2])):
                try:
                    row[0] = labels[0][chdict.get(row[0])]
                    y_train[ri] = mlb.transform([tuple(row[:2])])
                    ri += 1
                except IndexError:
                    print(ri)
                    ri += 1
                    pass
        print(y_train)
    return x_train, y_train



def chroma_process_split(y, sr):
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
    while(beat_frames[-1]<(len(y)//HOP_LENGTH)):
        beat_frames = np.append(beat_frames, (beat_frames[-1]+int(np.mean([(beat_frames[i] - beat_frames[i - 1]) for i in range(1, len(beat_frames))]))))

    n_seg = 4
    split_beat_frames = split_beats(beat_frames, n_segments=n_seg)
    split_chroma = librosa.util.sync(chroma_smooth, split_beat_frames, aggregate=np.median)

    if (split_chroma.shape[1]%n_seg != 0):
        split_chroma = split_chroma[:, :(-1*(split_chroma.shape[1]%n_seg))]
        beat_frames = beat_frames[:-1]
    chroma_each_beat = np.split(split_chroma, (split_chroma.shape[1]/4), axis=1)
    for i, chroma in enumerate(chroma_each_beat):
        chroma0 = np.transpose(chroma)
        chroma_each_beat[i] = chroma0
    chroma_each_beat = np.stack(chroma_each_beat)
    return chroma_each_beat, librosa.frames_to_time(beat_frames, sr)

def split_beats(beats, n_segments=4):
    new_beats = []
    sub = 0
    for i in range(len(beats)-1):
        space = beats[i+1]-beats[i]
        sub = space/n_segments
        for j in range(n_segments):
            new_beats.append(beats[i]+int(j*sub))
    for j in range(n_segments):
        new_beats.append(beats[len(beats)-1]+int(j*sub))
    new_beats = np.array(new_beats)
    return new_beats


# x,y = label_process("data/chordtest0.csv", "data/chordtest0.wav", split=True)
# print(x[0].shape)
# print(len(y))
# print(bc.shape)
'''
plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma_smooth, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram')
plt.tight_layout()
plt.show()'''