import numpy as np
import librosa
class FeatureExtractor:
    def __init__(self):
        pass
    def extract_features(self, audio_data):
        features = {}
        features['mfcc'] = self.get_mfcc(audio_data)
        features['spectral_centroid'] = self.get_spectral_centroid(audio_data)
        features['spectral_bandwidth'] = self.get_spectral_bandwidth(audio_data)
        features['spectral_rolloff'] = self.get_spectral_rolloff(audio_data)
        features['zero_crossing_rate'] = self.get_zero_crossing_rate(audio_data)
        features['rms'] = self.get_rms(audio_data)
        return features
    def get_mfcc(self, audio_data):
        mfccs = librosa.feature.mfcc(y=audio_data, sr=44100, n_mfcc=13)
        return np.mean(mfccs, axis=1)
    def get_spectral_centroid(self, audio_data):
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=44100)[0]
        return np.mean(spectral_centroids)
    def get_spectral_bandwidth(self, audio_data):
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=44100)[0]
        return np.mean(spectral_bandwidth)
    def get_spectral_rolloff(self, audio_data):
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=44100)[0]
        return np.mean(spectral_rolloff)
    def get_zero_crossing_rate(self, audio_data):
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        return np.mean(zcr)
    def get_rms(self, audio_data):
        return np.sqrt(np.mean(np.square(audio_data)))
    def get_spectrogram(self, audio_data):
        n_fft = min(2048, len(audio_data))
        if n_fft < 32:
            n_fft = 32
        n_fft = 2 ** int(np.log2(n_fft))
        D = librosa.stft(audio_data, n_fft=n_fft)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        return S_db