
import librosa.display
from sklearn.preprocessing import MultiLabelBinarizer
import librosa
import numpy as np
import scipy
import csv


def chroma_process(y, sr):
    HOP_LENGTH = 512  # Number of samples per window for processing
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=8)  # Separate track into percussive and harmonic elements
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)  # Track beats and return frames
    # Audio is cut into HOP_LENGTH long frames and indexed from 0, beat onset frame indices are an array

    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, bins_per_octave=12*3, hop_length=HOP_LENGTH)  # Chroma

    chroma_filter = np.minimum(chroma,
                               librosa.decompose.nn_filter(chroma,
                                                           aggregate=np.median,
                                                           metric='cosine'))  # Some filtering to remove noise
    chroma_smooth = scipy.ndimage.median_filter(chroma_filter, size=(1, 9))  # More filtering
    # print(len(beat_frames))
    while beat_frames[-1] < (len(y)//HOP_LENGTH):  # Add extra frames to the end to avoid the end parts being missed
        beat_frames = np.append(beat_frames, (beat_frames[-1]+int(np.mean([(beat_frames[i] - beat_frames[i - 1])
                                                                           for i in range(1, len(beat_frames))]))))

    beat_chroma = librosa.util.sync(chroma_smooth, beat_frames, aggregate=np.median)  # Average chroma each beat
    chroma_each_beat = np.transpose(beat_chroma)  # flip array dimensions so each beat has shape (12, )

    return chroma_each_beat, librosa.frames_to_time(beat_frames, sr)  # return chroma and beat timings in seconds


def label_process(label_csv, sound_file, split=False):  # split returns split chroma
    print(sound_file)
    y0, sr0 = librosa.load(sound_file)
    if split:
        x_train, beat_frames = chroma_process_split(y0, sr0)
    else:
        x_train, beat_frames = chroma_process(y0, sr0)
    labels = [["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "NC", "M", "m"]]  # Possible labels

    y_train = np.zeros((len(x_train), len(labels[0])))  # Initialize array
    mlb = MultiLabelBinarizer()  # Does multi 1 hot encoding
    chdict = {"A": 0, "A#": 1, "Bb": 1, "B": 2, "C": 3, "C#": 4, "Db": 4, "D": 5, "D#": 6,
              "Eb": 6, "E": 7, "F": 8, "F#": 9, "Gb": 9, "G": 10, "G#": 11, "Ab": 11, "NC": 12}  # Dict for flat labels
    mlb.fit(labels)
    with open(label_csv, newline='') as csvfile:  # read csv labels
        r = csv.reader(csvfile)
        ri = 0  # beat index variable
        for row in r:
            for i in range(int(row[2])):  # for i number of beats that the chord lasts
                try:
                    row[0] = labels[0][chdict.get(row[0])]
                    y_train[ri] = mlb.transform([tuple(row[:2])])  # one hot encoded label array
                    ri += 1
                except IndexError:  # Sometimes there are more beats than labels, indexError will happen
                    print(ri)
                    ri += 1
                    pass
        print(y_train)
    return x_train, y_train


def chroma_process_split(y, sr):
    HOP_LENGTH = 512
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=8)  # percussive harmonic split
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)  # beat frames

    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, bins_per_octave=12*3, hop_length=HOP_LENGTH)  # chroma

    chroma_filter = np.minimum(chroma,
                               librosa.decompose.nn_filter(chroma,
                                                           aggregate=np.median,
                                                           metric='cosine'))
    chroma_smooth = scipy.ndimage.median_filter(chroma_filter, size=(1, 9))  # Filtering

    while beat_frames[-1] < (len(y)//HOP_LENGTH):  # beat frame padding to avoid beats less than labeled beats
        beat_frames = np.append(beat_frames, (beat_frames[-1]+int(np.mean([(beat_frames[i] - beat_frames[i - 1])
                                                                           for i in range(1, len(beat_frames))]))))

    n_seg = 4  # number of subdivisions per beat, 4 = semiquavers
    split_beat_frames = split_beats(beat_frames, n_segments=n_seg)  # beat frames of semiquavers
    split_chroma = librosa.util.sync(chroma_smooth, split_beat_frames, aggregate=np.median)

    if split_chroma.shape[1] % n_seg != 0:  # Cut off excess chroma if last segment is not 4 semiquavers
        split_chroma = split_chroma[:, :(-1*(split_chroma.shape[1] % n_seg))]
        beat_frames = beat_frames[:-1]

    chroma_each_beat = np.split(split_chroma, (split_chroma.shape[1]/4), axis=1)  # Cut into 4 semiquaver sections
    for i, chroma in enumerate(chroma_each_beat):
        chroma0 = np.transpose(chroma)  # flip dimensions to get (12, 4) array for each semiquaver
        chroma_each_beat[i] = chroma0
    chroma_each_beat = np.stack(chroma_each_beat)  # from list of array to array
    return chroma_each_beat, librosa.frames_to_time(beat_frames, sr)  # return subdivided chroma and beat timings in seconds


def split_beats(beats, n_segments=4):  # inserts additional beat indices such that each beat is now n_seg sub beats
    new_beats = []
    sub = 0
    for i in range(len(beats)-1):
        space = beats[i+1]-beats[i]  # number of frames in beat
        sub = space/n_segments  # number of frames per sub beat (semiquaver by default)
        for j in range(n_segments):
            new_beats.append(beats[i]+int(j*sub))
    for j in range(n_segments):  # fix off by one without index error
        new_beats.append(beats[len(beats)-1]+int(j*sub))
    new_beats = np.array(new_beats)
    return new_beats  # returns array of subdivided beat frames
