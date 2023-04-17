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
        pass

    class Noise:
        pass


def main():
    channels_to_plot = [0,1,2,3,4]
    # signal = EEGSignalProcessing.read_signal(filename = "Subject00_2.edf", number_of_samples=1000)
    # signal = EEGSignalProcessing.read_signal(filename = "Subject00_1.edf", number_of_samples=1000)
    signal = EEGSignalProcessing.read_signal(filename = "rsvp_10Hz_08b.edf", number_of_samples=1000)

    EEGSignalProcessing.plot_signal(signal,sampling_frequency= 2048, title = "Orginalne sygnały EEG", number_of_channels = channels_to_plot,
                                    yaxis_label='Wartosc sygnalu', xaxis_label='Czas [s]')


if __name__ == '__main__':
    main()