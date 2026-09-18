import numpy as np
from scipy.signal import cheby1, filtfilt
from scipy.linalg import qr, svd

class FBCCA:

    def __init__(self, target_freqs, max_duration_sec=5.0, fs=250, num_harmonics=5, num_bands=5):
        self.target_freqs = target_freqs
        self.fs = fs
        self.max_samples = int(max_duration_sec * fs)
        self.num_harmonics = num_harmonics
        self.num_bands = num_bands
        self.weights = np.array([(n + 1) ** (-1.25) + 0.25 for n in range(num_bands)])
        self.ref_signals = self._generate_reference_signals()

    def _generate_reference_signals(self):
        t = np.arange(self.max_samples) / self.fs
        ref_signals = {}
        for freq in self.target_freqs:
            Y = np.zeros((self.max_samples, 2 * self.num_harmonics))
            for h in range(self.num_harmonics):
                Y[:, 2 * h] = np.sin(2 * np.pi * (h + 1) * freq * t)
                Y[:, 2 * h + 1] = np.cos(2 * np.pi * (h + 1) * freq * t)
            ref_signals[freq] = Y
        return ref_signals

    def _canoncorr(self, X, Y):
        X = X - np.mean(X, axis=0)
        Y = Y - np.mean(Y, axis=0)
        (Qx, _) = qr(X, mode='economic')
        (Qy, _) = qr(Y, mode='economic')
        (_, S, _) = svd(np.dot(Qx.T, Qy))
        return S[0]

    def _filterbank(self, eeg_data):
        (num_channels, num_samples) = eeg_data.shape
        filtered_data = np.zeros((self.num_bands, num_channels, num_samples))
        nyq = self.fs / 2
        for i in range(self.num_bands):
            passband = [8.0 * (i + 1), 90.0]
            Wn = [passband[0] / nyq, passband[1] / nyq]
            (b, a) = cheby1(N=4, rp=0.5, Wn=Wn, btype='bandpass')
            filtered_data[i, :, :] = filtfilt(b, a, eeg_data, axis=1)
        return filtered_data

    def predict(self, eeg_data):
        if eeg_data.ndim != 2:
            raise ValueError(f'输入数据形状必须是 [channels, samples]，当前为 {eeg_data.shape}')
        (num_channels, current_samples) = eeg_data.shape
        if current_samples > self.max_samples:
            raise ValueError(f'输入数据长度 ({current_samples}) 超过了预生成的最大长度 ({self.max_samples})！请在初始化时调大 max_duration_sec 参数。')
        filtered_bands = self._filterbank(eeg_data)
        scores = np.zeros(len(self.target_freqs))
        for (class_idx, freq) in enumerate(self.target_freqs):
            Y = self.ref_signals[freq][:current_samples, :]
            rho_squared_sum = 0.0
            for band_idx in range(self.num_bands):
                X = filtered_bands[band_idx, :, :].T
                rho = self._canoncorr(X, Y)
                rho_squared_sum += self.weights[band_idx] * rho ** 2
            scores[class_idx] = rho_squared_sum
        predicted_idx = np.argmax(scores)
        predicted_freq = self.target_freqs[predicted_idx]
        return (predicted_freq, predicted_idx, scores)
