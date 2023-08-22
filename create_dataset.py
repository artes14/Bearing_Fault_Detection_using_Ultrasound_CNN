import glob
import itertools
from datetime import datetime
import h5py
from numpy import loadtxt, genfromtxt
import numpy as np
import cv2
import ctypes
import os

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

savefolder = 'dataset_img'
if not os.path.exists(savefolder):
    os.mkdir(savefolder)

train_data_path = 'dataset/train'
test_data_path = 'dataset/test'

train_image_paths = [] #to store image paths in list
test_image_paths = [] #to store image paths in list
classes = [] #to store class values

# train images
for data_path in glob.glob(train_data_path+'/*'):
    classes.append(data_path.split('/')[-1].split('\\')[-1])
    train_image_paths.append(glob.glob(data_path + '/*'))
print(classes)
print(train_image_paths)
train_image_paths = list(itertools.chain(*train_image_paths))
sample_rate = 100000
for filename in train_image_paths:
    with h5py.File(filename, 'r') as f:
        ultra_origin = f['/ULTRA/I003'][0, :]
        ultra_origin = ultra_origin.astype(int)
        for idx, a in enumerate(ultra_origin):
            ultra_origin[idx] = toSigned32(a)
        scaled = ultra_origin - np.mean(ultra_origin)
        # scaled = np.int16(ultra_dc/ 12.5 * 32767)
        wavfile.write('test.wav', sample_rate, scaled)
        waveform, sample_rate = torchaudio.load("test.wav")
        spec, f, t, im = plot_specgram(waveform, sample_rate, title="original spectogram")
        im_array = im.get_array()
        tmp=np.interp(im_array, (im_array.min(), im_array.max()), (0,255))
        crop_images, dx, dy = crop_image(tmp, 129)
        idx=0
        for i in range(dx*dy):
            image=crop_images[i]
            savename = savefolder+'/'+filename.split('/')[1].split('.')[0]+'_'+str(idx)+'.jpg'
            cv2.imwrite(savename, image)
            idx+=1

# train images
for data_path in glob.glob(test_data_path+'/*'):
    classes.append(data_path.split('/')[-1].split('\\')[-1])
    test_image_paths.append(glob.glob(data_path + '/*'))
print(classes)
print(test_image_paths)
test_image_paths = list(itertools.chain(*test_image_paths))
sample_rate = 100000
for filename in test_image_paths:
    with h5py.File(filename, 'r') as f:
        ultra_origin = f['/ULTRA/I003'][0, :]
        ultra_origin = ultra_origin.astype(int)
        for idx, a in enumerate(ultra_origin):
            ultra_origin[idx] = toSigned32(a)
        scaled = ultra_origin - np.mean(ultra_origin)
        # scaled = np.int16(ultra_dc/ 12.5 * 32767)
        wavfile.write('test.wav', sample_rate, scaled)
        waveform, sample_rate = torchaudio.load("test.wav")
        spec, f, t, im = plot_specgram(waveform, sample_rate, title="original spectogram")
        im_array = im.get_array()
        tmp=np.interp(im_array, (im_array.min(), im_array.max()), (0,255))
        crop_images, dx, dy = crop_image(tmp, 129)
        idx=0
        for i in range(dx*dy):
            image=crop_images[i]
            savename = savefolder+'/'+filename.split('/')[1].split('.')[0]+'_'+str(idx)+'.jpg'
            cv2.imwrite(savename, image)
            idx+=1
print('done.')
