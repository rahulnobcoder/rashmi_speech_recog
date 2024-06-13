
import numpy as np
import librosa
import os
import librosa
import numpy as np
from tqdm import tqdm
import librosa.display
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import random
sr=22050

def f_high(y,sr=sr):
    b,a = signal.butter(10, 2000/(sr/2), btype='highpass')
    yf = signal.lfilter(b,a,y)
    return yf

def extract_features(data,sample_rate=sr):
    # ZCR
    result = np.array([])
#     zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
#     result=np.hstack((result, zcr)) # stacking horizontally

#     # Chroma_stft
#     stft = np.abs(librosa.stft(data))
#     chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
#     result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
#     rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
#     result = np.hstack((result, rms)) # stacking horizontally

#     # MelSpectogram
#     mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
#     result = np.hstack((result, mel)) # stacking horizontally
    
    return result

def get_features(data,sample_rate=sr):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.    
    # without augmentation
    res1 = extract_features(data)
    result = np.array(res1)
    
#     # data with noise
#     noise_data = noise(data)
#     res2 = extract_features(noise_data)
#     result = np.vstack((result, res2)) # stacking vertically
    
#     # data with stretching and pitching
#     new_data = stretch(data)
#     data_stretch_pitch = pitch(new_data, sample_rate)
#     res3 = extract_features(data_stretch_pitch)
#     result = np.vstack((result, res3)) # stacking vertically
    
    return result

def process(audio):
    audio=f_high(audio)
    features=get_features(audio)
    return features