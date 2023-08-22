import glob
import itertools
from datetime import datetime
import h5py
from numpy import loadtxt, genfromtxt
import numpy as np
import cv2
import ctypes
import os
import random
import re

# pytorch
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import torchvision
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import librosa
from utils import *

test_transforms = A.Compose(
    [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 32, 5)
        self.fc1 = nn.Linear(32 * 29 * 29, 144)
        self.fc2 = nn.Linear(144, 72)
        self.fc3 = nn.Linear(72, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

#######################################################
#               Define Dataset Class
#######################################################

class UltraSonicDataset(Dataset):
    def __init__(self, hdf5name:str, transform=False):
        self.image_paths=[]
        self.hdf5name = hdf5name
        self.savefolder=re.split(r'[/\\]',hdf5name)[0]
        sample_rate = 100000
        with h5py.File(hdf5name, 'r') as f:
            ultra_origin = f['/ULTRA/I003'][0, :]
            ultra_origin = ultra_origin.astype(int)
            for idx, a in enumerate(ultra_origin):
                ultra_origin[idx] = toSigned32(a)
            scaled = ultra_origin - np.mean(ultra_origin)
            wavfile.write('test.wav', sample_rate, scaled)
            waveform, sample_rate = torchaudio.load("test.wav")
            spec, f, t, im = plot_specgram(waveform, sample_rate, title="original spectogram")
            im_array = im.get_array()
            tmp = np.interp(im_array, (im_array.min(), im_array.max()), (0, 255))
            crop_images, dx, dy = crop_image(tmp, 129)
            idx = 0
            for i in range(dx * dy):
                image = crop_images[i]
                savename = self.savefolder + '/images/' + re.split(r'[/\\]',hdf5name)[-1].split('.')[0] + '_' + str(idx) + '.jpg'
                cv2.imwrite(savename, image)
                self.image_paths.append(savename)
                idx += 1
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, self.hdf5name

#######################################################
#                  Create Dataset
#######################################################

test_data_path = 'eval'
classes = ['defect', 'normal'] #to store class values

# create dictionary for class indexes
idx_to_class = {i:j for i, j in enumerate(classes)}
class_to_idx = {value:key for key,value in idx_to_class.items()}
print(idx_to_class)
for test_path in glob.glob(test_data_path +'/*'):
    if os.path.isdir(test_path):
        continue
    if re.split(r'[.]',test_path)[-1] != 'h5':
        continue
    test_dataset = UltraSonicDataset(test_path, test_transforms)
    test_loader = DataLoader(
        test_dataset, batch_size=18, shuffle=False
    )

    PATH='weights/ultra_net4.pth'

    dataiter = iter(test_loader)
    images = next(dataiter)

    net = Net()
    net.load_state_dict(torch.load(PATH))

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data, filename in test_loader:
            images = data.type(torch.float32)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            result = np.sum(predicted.numpy())
            b = len(predicted.numpy())
            print(filename[0])
            print(classes[int(result/b)])



