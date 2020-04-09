import librosa
import csv
import os.path as path


def allKeys(soundfile, csvfile):  # Data augmentation by pitch shifting to every available key
    tones = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "NC"]
    chdict = {"A": 0, "A#": 1, "Bb": 1, "B": 2, "C": 3, "C#": 4, "Db": 4, "D": 5, "D#": 6,
              "Eb": 6, "E": 7, "F": 8, "F#": 9, "Gb": 9, "G": 10, "G#": 11, "Ab": 11, "NC": 12}

    y, sr = librosa.load(soundfile)
    for n in range(-6, 6):
        yn = librosa.effects.pitch_shift(y, sr, n_steps=float(n))  # Pitch shifted samples, same sample rate
        csvdir, csvname = path.split(csvfile)
        print(csvdir)
        csvname, csvext = path.splitext(csvname)
        with open(csvfile, newline='') as csvf:
            r = csv.reader(csvf)
            n_csv = path.join('data/labels/', (csvname+'_'+str(n)+csvext))  # writes new labels to data/labels
            print(n_csv)
            with open(n_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                for row in r:
                    row[0] = tones[(chdict.get(row[0])+n) % 12]  # Write transposed chord symbol
                    writer.writerow(row)

        n_sf = path.join('data/soundfiles/', (csvname + '_' + str(n)+'.wav'))  # wave file stored in data/soundfiles
        librosa.output.write_wav(n_sf, yn, sr)
    return


#  Run this to generate all training data

allKeys('data/505.wav', 'data/505.csv')

allKeys('data/All_Of_Me.wav', 'data/All_Of_Me.csv')

allKeys('data/Ave_Verum_Corpus_Mozart.wav', 'data/Ave_Verum_Corpus_Mozart.csv')

allKeys('data/chordtest0.wav', 'data/chordtest0.csv')

allKeys('data/despacito.wav', 'data/despacito.csv')

allKeys('data/greensleeves.wav', 'data/greensleeves.csv')

allKeys('data/losing_my_religion.wav', 'data/losing_my_religion.csv')
