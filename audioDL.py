from datetime import datetime
import h5py
from numpy import loadtxt, genfromtxt
import numpy as np
import cv2
import ctypes

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
import librosa


fig, axs = plt.subplots(2, 1)

def toSigned32(n):
    n = n & 0xffffffff
    return n | (-(n & 0x80000000))

def crop_image(img, crop_size:int=None):
    w,h=img.shape
    if crop_size>=w or crop_size>=h:
        return img, 1, 1
    crop_image=[]
    dx, dy = int(np.floor(w/crop_size)), int(np.floor(h/crop_size))
    for x in range(0, w - crop_size, crop_size):
        for y in range(0, h - crop_size, crop_size):
            crop_image.append(img[x:x + crop_size, y:y + crop_size])
    return crop_image, dx, dy

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
   waveform = waveform.numpy()
   num_channels, num_frames = waveform.shape
   time_axis = torch.arange(0, num_frames) / sample_rate

   figure, axes = plt.subplots(num_channels, 1)
   if num_channels == 1:
       axes = [axes]
   for c in range(num_channels):
       axes[c].plot(time_axis, waveform[c], linewidth=1)
       axes[c].grid(True)
       if num_channels > 1:
           axes[c].set_ylabel(f"Channel {c+1}")
       if xlim:
           axes[c].set_xlim(xlim)
       if ylim:
           axes[c].set_ylim(ylim)
   figure.suptitle(title)
   plt.show(block=False)

def print_stats(waveform, sample_rate=None, src=None):
   if src:
       print("-"*10)
       print(f"Source: {src}")
       print("-"*10)
   if sample_rate:
       print(f"Sample Rate: {sample_rate}")
   print("Dtype:", waveform.dtype)
   print(f" - Max:     {waveform.max().item():6.3f}")
   print(f" - Min:     {waveform.min().item():6.3f}")
   print(f" - Mean:    {waveform.mean().item():6.3f}")
   print(f" - Std Dev: {waveform.std().item():6.3f}")
   print()
   print(waveform)
   print()

def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
   waveform = waveform.numpy()
   num_channels, num_frames = waveform.shape
   figure, axes = plt.subplots(num_channels, 1)
   if num_channels == 1:
       axes = [axes]
   for c in range(num_channels):
       spec, f, t, im = axes[c].specgram(waveform[c], Fs=sample_rate)
       if num_channels > 1:
           axes[c].set_ylabel(f"Channel {c+1}")
       if xlim:
           axes[c].set_xlim(xlim)
   figure.suptitle(title)

   plt.show(block=False)
   return spec, f, t, im

def get_rir_sample(*, resample=None, processed=False):
   rir_raw, sample_rate = _get_sample(SAMPLE_RIR_PATH, resample=resample)
   if not processed:
       return rir_raw, sample_rate
   rir = rir_raw[:, int(sample_rate*1.01) : int(sample_rate * 1.3)]
   rir = rir / torch.norm(rir, p=2)
   rir = torch.flip(rir, [1])
   return rir, sample_rate

sample_rate = 100000
filename = 'data/CV04ROLLER_3_2023_05_23_14_42_38.h5'
with h5py.File(filename, 'r') as f:
    ultra_origin = f['/ULTRA/I003'][0,:]
    ultra_origin = ultra_origin.astype(int)
    for idx, a in enumerate(ultra_origin):
        ultra_origin[idx]=toSigned32(a)
    scaled =ultra_origin - np.mean(ultra_origin)
    # scaled = np.int16(ultra_dc/ 12.5 * 32767)
    wavfile.write('test.wav', sample_rate, scaled)


# time_axis = torch.arange(0, num_frames) / sample_rate
waveform, sample_rate = torchaudio.load("test.wav")
time_axis = torch.arange(0, 300000) / sample_rate
axs[0].plot(time_axis, waveform[0], linewidth=1)
axs[0].grid(True)
axs[0].set_xlim([0, time_axis[-1]])
plot_waveform(waveform, sample_rate, title="original")
spec, f, t, im = plot_specgram(waveform, sample_rate, title="original spectogram")
_ = axs[1].imshow(im.get_array())
im_array = im.get_array()

plt.show()
# crop_images = crop_image(im_array, 129)






