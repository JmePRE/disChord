import librosa
import csv
import os.path as path
import soundfile as sf

def allKeys(soundfile, csvfile):
    tones = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]

    chdict = {"A": 0, "A#": 1, "Bb": 1, "B": 2, "C": 3, "C#": 4, "Db": 4, "D": 5, "D#": 6,
              "Eb": 6, "E": 7, "F": 8, "F#": 9, "Gb": 9, "G": 10, "G#": 11, "Ab": 11}
    y, sr = librosa.load(soundfile)
    for n in range(-6,6):
        yn, srn = librosa.effects.pitch_shift(y, sr, float(n))
        csvdir, csvname = path.split(csvfile)
        csvname, csvext = path.splitext(csvname)
        with open(csvfile, newline='') as csvf:
            r = csv.reader(csvf)
            n_csv = path.join(csvdir, '/labels/', (csvname+'_'+str(n)+csvext))
            with open(n_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                for row in r:
                    row[0] = tones[(tones.index(row[tones[chdict.get(row[0])]])+n)%12]
                    writer.writerow(row)

        n_sf = path.join(csvdir, '/soundfiles/', (csvname + '_' + str(n)+'.wav'))
        sf.write(n_sf,yn,srn)
    return


allKeys('data/despacito.wav','data/despacito.csv')