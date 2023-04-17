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

class EEGSignalProcessing:
    def __init__(self) -> None:
        pass

    def read_signal(filename, number_of_samples = None, offset = 0):
        file = pyedflib.EdfReader(filename)
        if number_of_samples is None:
            number_of_samples = file.getNSamples()[0]
        number_of_signals = file.signals_in_file
        signal = np.zeros((number_of_signals, number_of_samples))

        for i in range(number_of_signals):
            signal[i, :] = file.readSignal(i)[offset:offset + number_of_samples]
        
        file.close()
        return signal
    
    def plot_signal(data, sampling_frequency, title, number_of_channels = None, channel_labels = None, yaxis_label = None, xaxis_label = None):
        
        plt.rcParams['font.size'] = '16'
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        lenght = len(data)
        
        if number_of_channels is None:
            number_of_channels_useful = range(0, lenght)
        else:
            if isinstance(number_of_channels, str):
                number_of_channels_useful = range(0, lenght-1)
            else:
                number_of_channels_useful = number_of_channels
        
        for channel in number_of_channels_useful:
            if channel_labels is None:
                label = 'ch' + str(channel + 1)
            else:
                label = channel_labels[channel]

            limit = data[channel, :].size
            x_values = [num/sampling_frequency for num in range(0, limit)]
            ax.plot(x_values, data[channel, :], label = label)
        
        fig.set_size_inches(15,5)
        plt.title(title)
        plt.legend()

        if yaxis_label is not None:
            plt.ylabel(yaxis_label)
        if xaxis_label is not None:
            plt.xlabel(xaxis_label)
        
        plt.show(block = True)

    def channel_desynchronize(data_1d, delay, value = 0):
        number_of_samples = len(data_1d)
        if delay > 0:
            for i in range(number_of_samples - 1, delay - 1, -1):
                data_1d[i] = data_1d[i - delay]
            for i in range(0, delay):
                data_1d[i] = value
        if delay < 0:
            delay = -delay
            for i in range(0, number_of_samples - delay):
                data_1d[i] = data_1d[i + delay]
            for i in range(number_of_samples - delay, number_of_samples):
                data_1d[i] = value        

    def all_channels_desynchronize(data, delay, value = 0):
        for i in range(0, len(data)):
            EEGSignalProcessing.channel_desynchronize(data[i], delay, value)        

    class NoiseReduction:

        def autoregression(data, delay):
            signals_number = len(data)
            samples_number = len(data[0])
            output = np.zeros((signals_number, samples_number))

            for i in range(0, signals_number):
                model = AutoReg(data[i], lags=delay)
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
                output.append(EEGSignalProcessing.NoiseReduction.wavelet(c))
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


def main():
    channels_to_plot = [0,1,2,3,4]
    # signal = EEGSignalProcessing.read_signal(filename = "Subject00_2.edf", number_of_samples=1000)
    # signal = EEGSignalProcessing.read_signal(filename = "Subject00_1.edf", number_of_samples=1000)
    signal = EEGSignalProcessing.read_signal(filename = "rsvp_10Hz_08b.edf", number_of_samples=1000)

    EEGSignalProcessing.plot_signal(signal,sampling_frequency= 2048, title = "Orginalne sygnały EEG", number_of_channels = channels_to_plot,
                                    yaxis_label='Wartosc sygnalu', xaxis_label='Czas [s]')

    low = 0
    high = 20
    sig_noise1 = EEGSignalProcessing.Noise.add_uniform_noise(signal, low=low, high=high, seed=100)
    EEGSignalProcessing.plot_signal(sig_noise1, title="Zaszumiony sygnał 5 kanałów EEG (Uniform distribution Low={}, High={})".format(
        low, high), sampling_frequency=2048, number_of_channels=channels_to_plot, yaxis_label='Wartość sygnału', xaxis_label='Czas [s]')

    mean = 0
    std = 1
    ampl = 1
    sig_noise2 = EEGSignalProcessing.Noise.add_normal_noise(signal, mean=mean, std=std, amplitude=ampl, seed=100)
    EEGSignalProcessing.plot_signal(sig_noise2, title="Zaszumiony sygnał 5 kanałów EEG (Normal Distribution Low={}, High={})".format(
        mean, std), sampling_frequency=2048, number_of_channels=channels_to_plot, yaxis_label='Wartość sygnału', xaxis_label='Czas [s]')

    sig_n3_left = 0
    sig_n3_peak = 5
    sig_n3_right = 15
    sig_noise3 = EEGSignalProcessing.Noise.add_triangular_noise(signal, left=sig_n3_left, peak=sig_n3_peak, right=sig_n3_right, seed=100)
    EEGSignalProcessing.plot_signal(sig_noise3, title="Zaszumiony sygnał 5 kanałów EEG (Triangular Distribution Left={}, Peak={}, High={})".format(
        sig_n3_left, sig_n3_peak, sig_n3_right), sampling_frequency=2048, number_of_channels=channels_to_plot, yaxis_label='Wartość sygnału', xaxis_label='Czas [s]')

if __name__ == '__main__':
    main()