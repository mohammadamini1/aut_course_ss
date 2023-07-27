#!/usr/bin/python3

from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import math
import librosa

audio_file = './Test.wav'


def spectral_subtraction(audio_, noise_):
    ##! find audio mean
    audio           = librosa.stft(audio_) ## short time fourier transform
    audio_mean      = np.abs(audio)
    ## for inverse transform
    audio_inverse   = np.exp(1.0j * np.angle(audio))

    ##! find noise mean
    noise           = librosa.stft(noise_) ## short time fourier transform
    noise_mean      = np.mean(np.abs(noise), axis=1)
    noise_mean      = noise_mean.reshape((noise_mean.shape[0], 1))

    ##! subtract
    result = audio_mean - noise_mean

    ##! apply phase information & inverse transform to time domain
    result = result * audio_inverse
    result = librosa.istft(result)

    return result

def awgn(signal, power):
    rang = 2
    sigpower = sum([math.pow(abs(signal[i]), 2)
                   for i in range(len(signal))]) / len(signal)
    noisepower = sigpower / (math.pow(10, power/10))
    noise = math.sqrt(noisepower) * (np.random.uniform(-rang, rang, size=len(signal)))
    return signal + noise, noise

def plott(s): plt.plot(np.linspace(0, 50, len(s)), s)
def plott2(ax, s): ax.plot(np.linspace(0, 10, len(s)), s)



##! load org audio
sound, sample_freq = librosa.load(audio_file, sr=None, mono=True)

fig, ax = plt.subplots(3, 3, figsize=(14, 14))
for axi, power in zip(ax.flat, range(0, 9)):
    ##! add noise with awgn and power
    noisy_sound, noise = awgn(sound, power)

    ##! spectral_subtraction
    denoised_sound = spectral_subtraction(noisy_sound, noise)

    ## save output & plot
    wavfile.write("spectralSub_output/noisy_sound_{}.wav".format(power), sample_freq, noisy_sound)
    wavfile.write("spectralSub_output/denoise_sound_{}.wav".format(power), sample_freq, denoised_sound)
    print("\nnoisy sound    saved as: spectralSub_output/noisy_sound_{}.wav".format(power))
    print("denoised sound saved as: spectralSub_output/denoise_sound_{}.wav".format(power))
    plott2(axi, noisy_sound) ## blue
    plott2(axi, denoised_sound) ## orange
    plott2(axi, sound) ## green
    axi.set_xlabel('power: ' + str(power))



plt.show()

