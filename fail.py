import librosa
import numpy as np

def load_audio(wave, t_start=None, t_end=None, without_file=False, sr_audio=16000):
    hop_length = 512
    if without_file:
        y = wave
    else:
        y, sr = librosa.load(wave, sr=sr_audio)

    short_y = y[t_start:t_end] if t_start is not None else y
    short_y = short_y.astype(np.float32)
    onset_t = librosa.onset.onset_detect(y=short_y, sr=sr_audio, hop_length=hop_length, units="time")
    return onset_t

def test(audio_path, sr=16000):
    load_audio(audio_path, sr_audio=sr)

if __name__ == '__main__':
    test('BEAT2/beat_english_v2.0.0/wave16k/1_wayne_0_1_1.wav')