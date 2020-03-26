import librosa
import csv
import os.path as path


def allKeys(soundfile, csvfile):
    tones = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "NC"]

    chdict = {"A": 0, "A#": 1, "Bb": 1, "B": 2, "C": 3, "C#": 4, "Db": 4, "D": 5, "D#": 6,
              "Eb": 6, "E": 7, "F": 8, "F#": 9, "Gb": 9, "G": 10, "G#": 11, "Ab": 11, "NC": 12}
    y, sr = librosa.load(soundfile)
    for n in range(-6, 6):
        yn = librosa.effects.pitch_shift(y, sr, n_steps=float(n))
        csvdir, csvname = path.split(csvfile)
        print(csvdir)
        csvname, csvext = path.splitext(csvname)
        with open(csvfile, newline='') as csvf:
            r = csv.reader(csvf)
            n_csv = path.join('data/labels/', (csvname+'_'+str(n)+csvext))
            print(n_csv)
            with open(n_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                for row in r:
                    row[0] = tones[(chdict.get(row[0])+n)%12]
                    writer.writerow(row)

        n_sf = path.join('data/soundfiles/', (csvname + '_' + str(n)+'.wav'))
        librosa.output.write_wav(n_sf, yn, sr)
    return


allKeys('data/losing_my_religion.wav','data/losing_my_religion.csv')