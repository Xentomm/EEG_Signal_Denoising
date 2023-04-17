import numpy as np
import matplotlib.pyplot as plt
import pyedflib
import sklearn.linear_model as slm
from sklearn import metrics
from statsmodels.tsa.ar_model import AutoReg
import scipy
import scipy.signal as sig
from sklearn.decomposition import FastICA, PCA
import pywt

class EEGP:
    def __init__(self) -> None:
        pass

    class NoiseReduction:

        def autoregression(data, lag):
            signals_number = len(data)
            samples_number = len(data[0])
            output = np.zeros((signals_number, samples_number))

            for i in range(0, signals_number):
                model = AutoReg(data[i], lags=lag)
                model_fit = model.fit()
                predictions = model_fit.predict(start=0, end=samples_number-1, dynamic=False)
                output[i, :samples_number] = predictions
            return output

        def wavelet(linear_array):
            name = 'bior3.1'

            # Create wavelet object and define parameters
            wav = pywt.Wavelet(name)
            max_level = pywt.dwt_max_level(len(linear_array) + 1, wav.dec_len)
            # maxlev = 2 # Override if desired
            print("Maximum level is " + str(max_level))
            threshold = 0.04  # Threshold for filtering

            # Decompose into wavelet components, to the level selected:
            coeffs = pywt.wavedec(linear_array, name, level=5)

            # cA = pywt.threshold(cA, threshold*max(cA))
            plt.figure()
            for i in range(1, len(coeffs)):
                plt.subplot(max_level, 1, i)
                plt.plot(coeffs[i])
                coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
                plt.plot(coeffs[i])
            plt.show()

        def wavelet_all_channels(data):
            output = []
            for c in data:
                output.append(EegProcessing.NoiseReduction.wavelet(c))
            return np.stack(output)
        
        def ica(data, mask=None):
            # maska do wyboru składowych
            reduce_level = [True, True, True, True, True, True, True, True, True]
            # reduce_level[7] = False
            reduce_level[7] = False
            # reduce_level = [True, False]

            if mask is not None:
                reduce_level = mask

            sigT = data.T
            n = data.shape[0]

            # obliczanie ICA
            ica = FastICA(n_components=n)
            sig_ica = ica.fit_transform(sigT)
            # Macierz mmieszania
            A_ica = ica.mixing_
            # Przycięcie macierzy mieszającej, aby odrzucić najmniej znaczące wartości
            A_ica_reduced = A_ica
            sig_ica = sig_ica[:, reduce_level]
            X_reduced = np.dot(sig_ica, A_ica_reduced.T[reduce_level, :]) + ica.mean_
            ica_reconstruct = X_reduced.T
            return ica_reconstruct

        pass

    class Noise:

        def add_uniform_noise(data, low, high, seed=None):
            signals_number = len(data)
            samples_number = len(data[0])
            output = np.zeros((signals_number, samples_number))

            if seed is not None:
                np.random.seed(seed)

            for i in range(signals_number):
                if isinstance(low, str):
                    if low == "min_value":
                        low = min(data[i])

                if isinstance(high, str):
                    if high == "max_value":
                        high = max(data[i])
                noise = np.random.uniform(low, high, samples_number)
                output[i] = data[i] + noise
            return output

        def add_normal_noise(data, mean, std, amplitude=1, seed=None):
            signals_number = len(data)
            samples_number = len(data[0])
            output = np.zeros((signals_number, samples_number))

            if seed is not None:
                np.random.seed(seed)

            for i in range(signals_number):
                noise = np.random.normal(mean, std, samples_number)
                output[i] = data[i] + noise
            return amplitude*output
        
        def add_triangular_noise(data, left, peak, right, seed=None):
            signals_number = len(data)
            samples_number = len(data[0])
            output = np.zeros((signals_number, samples_number))

            if seed is not None:
                np.random.seed(seed)
                
            for i in range(signals_number):
                noise = np.random.triangular(left, peak, right, samples_number)
                output[i] = data[i] + noise
            return output

        pass


def main():
    print("Test")

if __name__ == '__main__':
    main()
