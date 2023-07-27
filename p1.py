#!/usr/bin/python3

from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

audio_files = []
for i in range(12):
    audio_files.append("./voices/v{}.wav".format(i))


##! part 1: 2 -> power spectrum
def plot_power_spectrum(audio_file_path, plot=True):
    ## read audio file
    sample_freq, sound = wavfile.read(audio_file_path)
    ## choose one of channels (left)
    sound = sound[:, 0]
    ## fft
    spec = np.fft.rfft(sound)
    freq = np.fft.rfftfreq(sound.size, d=1./sample_freq)
    ## abs
    spec = np.abs(spec)
    ## add to plot
    # plot_size = len(spec) // 4
    plot_size = 20000
    if plot:
        plt.plot(freq[:plot_size], spec[:plot_size])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
    return spec


##! part 1: 3 -> power spectrum
def find_peak_frequency(audio_file_path):
    ## read audio file
    spec = plot_power_spectrum(audio_file_path, plot=False)
    p = np.where(spec == np.amax(spec))
    return (p[0] // 20)[0]


##! part 1: 4 -> label
## audio_files: list of audio files path
def recon(audio_files):
    ## read audio files and find peaks
    peaks = []
    for audio in audio_files:
        peak = find_peak_frequency(audio)
        peaks.append(peak)

    ## find avr
    avr = np.average(peaks)
    if avr > 180 or avr < 165: avr = 172

    ## woman above average and ...
    res = []
    for i in range(len(audio_files)):
        woman = False
        if peaks[i] >= avr:
            woman = True
        res.append(woman)
        print("audio {} with peak {} Hz is {}".format(audio_files[i], peaks[i], 'women' if woman else 'men'))
    return res


for audio in audio_files:
    plot_power_spectrum(audio)

for audio in audio_files:
    peak_frequency = find_peak_frequency(audio)
    print("audio {} peak frequency: {} Hz".format(audio, peak_frequency))

recon(audio_files)


plt.show()


