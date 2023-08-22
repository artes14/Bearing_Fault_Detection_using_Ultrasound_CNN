
import matplotlib.pyplot as plt
import numpy as np

def toSigned32(n):
    n = n & 0xffffffff
    return n | (-(n & 0x80000000))

def crop_image(img, crop_size:int=None):
    w,h=img.shape
    if crop_size>=w and crop_size>=h:
        return img, 1, 1
    crop_image=[]
    dx, dy = int(np.floor(w/crop_size)), int(np.floor(h/crop_size))
    for x in range(0, w - crop_size+1, crop_size):
        for y in range(0, h - crop_size+1, crop_size):
            crop_image.append(img[x:x + crop_size, y:y + crop_size])
    return crop_image, dx, dy

def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
   waveform = waveform.numpy()
   num_channels, num_frames = waveform.shape
   # figure, axes = plt.subplots(num_channels, 1)
   spec, f, t, im = plt.specgram(waveform[0], Fs=sample_rate)
   plt.cla()
   return spec, f, t, im