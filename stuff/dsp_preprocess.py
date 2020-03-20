
import librosa.display
import matplotlib.pyplot as plt
import librosa
import numpy as np
import scipy
y, sr = librosa.load(librosa.util.example_audio_file())
y_harmonic, y_percussive = librosa.effects.hpss(y, margin=8)
hop_length = 256
# Beat track on the percussive signal
tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, bins_per_octave=12*3)
chroma_filter = np.minimum(chroma,
                           librosa.decompose.nn_filter(chroma,
                                                       aggregate=np.median,
                                                       metric='cosine'))
chroma_smooth = scipy.ndimage.median_filter(chroma_filter, size=(1, 9))

beat_chroma = librosa.util.sync(chroma_smooth,
                                beat_frames,
                                aggregate=np.median)
chroma_each_beat = np.transpose(beat_chroma)
print(beat_frames[:20])
print(beat_chroma.shape)
plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma_smooth, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram')
plt.tight_layout()
plt.show()