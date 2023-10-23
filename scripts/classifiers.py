# Imports and utility functions
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

import pandas as pd
import librosa
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
import os
import sys
from pathlib import Path
import json
import soundfile as sf
import wave
import ntpath
import sounddevice as sd
import pickle
import glob
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score

# new imports
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

# Setup to import custom modules
if not f'{os.path.dirname(os.path.realpath(__file__))}/../scripts' in sys.path:
    sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/../scripts')
    

# Load audio datasets and get info based upon filenames
import audiodatasets

# Ease of use with my dictionaries
import dictmgmt

# Other miscellaneous utilities (like the timer)
import misc

# Some important directory paths
jar = f'{os.path.dirname(os.path.realpath(__file__))}/../pickles'
model_dir = f'{os.path.dirname(os.path.realpath(__file__))}/../models'
data_dir = f'{os.path.dirname(os.path.realpath(__file__))}/../data'

#### We expose 4 models: SIMPLE_PURE, SIMPLE_RECORDED, SEA and FLAIR

models = {"SIMPLE_PURE": None, "SIMPLE_RECORDED": None, "SEA": None, "FLAIR": None, "GAO_LSTM": None, "CNNLSTM": None}

#######################
#SIMPLE CLASSIFIERS
#######################

# Simple feature extraction

def wrapped_simple(filelist):
    X_vals, _, _, y_vals, _, featureindices = extract_features_simple(filelist)
    return X_vals, y_vals, featureindices

def extract_features_simple(filelist, max_duration=4.0, sample_rate=10000):
    
    max_sample_length = sample_rate * max_duration
                 
    y_values_emo = []
    y_values_speaker = []
    X_mfcc = []
    X_zc = []
    X_mf = []

    index = 0
    feature_indices = {}
    for key in sorted(filelist.keys()):
        file = filelist[key]
        x, sr = librosa.load(file, duration=max_duration, sr=sample_rate)
        
        # Pad with zeroes, if necessary
        x = librosa.util.pad_center(x, max_sample_length, mode='constant')
        
        mfcc = librosa.feature.mfcc(x, sr=sr)
        zc = librosa.feature.zero_crossing_rate(x)
        mf = librosa.feature.melspectrogram(x)
        X_mfcc.append(mfcc)
        X_zc.append(zc)
        X_mf.append(mf)

        # Infer the class from the filename
        head, tail = ntpath.split(file)
        y_values_emo.append(audiodatasets.get_emotion_id(tail) - 1)
        y_values_speaker.append(audiodatasets.get_speaker_id(tail) - 1)
        
        # Finally, update the feature indices
        feature_indices[key] = index
        index += 1
        
    X_mfcc = np.stack(X_mfcc)
    X_mfcc = np.reshape(X_mfcc, (np.shape(X_mfcc)[0], np.shape(X_mfcc)[1]*np.shape(X_mfcc)[2]))

    X_zc = np.stack(X_zc)
    X_zc = np.reshape(X_zc, (np.shape(X_zc)[0], np.shape(X_zc)[1]*np.shape(X_zc)[2]))

    X_mf = np.stack(X_mf)
    X_mf = np.reshape(X_mf, (np.shape(X_mf)[0], np.shape(X_mf)[1]*np.shape(X_mf)[2]))

    return X_mfcc, X_zc, X_mf, y_values_emo, y_values_speaker, feature_indices

# SIMPLE MODELS...load or train
ravdess_files = audiodatasets.list_audio_files(f'{data_dir}/fixed_ravdess')
ravdess_files_acoustic = audiodatasets.list_audio_files(f'{data_dir}/fixed_ravdess_acoustic')

tess_files = audiodatasets.list_audio_files(f'{data_dir}/fixed_tess')
#tess_files_acoustic = audiodatasets.list_audio_files(f'{data_dir}/fixed_tess_acoustic')
tess_files_acoustic = audiodatasets.list_audio_files(f'{data_dir}/fixed_tess_acoustic_1ft')

iemocap_files = audiodatasets.list_audio_files(f'{data_dir}/fixed_iemocap')

#training_data = { "PURE": {**iemocap_files}, "RECORDED": {**ravdess_files_acoustic, **tess_files_acoustic} }
training_data = { "PURE": {**ravdess_files, **tess_files, **iemocap_files}, "RECORDED": {**ravdess_files_acoustic, **tess_files_acoustic} }

for key in training_data:

    if os.path.exists(f'{model_dir}/SIMPLE_{key}_EMOTION_CLASSIFIER.h5'): # and key == "PURE":
        models[f'SIMPLE_{key}'] = keras.models.load_model(f'{model_dir}/SIMPLE_{key}_EMOTION_CLASSIFIER.h5')
#        models[f'SIMPLE_{key}'] = keras.models.load_model(open(f'{model_dir}/SIMPLE_{key}_EMOTION_CLASSIFIER.h5', 'rb'))
    else:
    
        X_values, y_values, _ = wrapped_simple(training_data[key])
        num_classes = 8 #len(set(y_values))

        models[f'SIMPLE_{key}'] = tf.keras.models.Sequential([tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                        tf.keras.layers.Dense(256, activation=tf.nn.relu), 
                                        tf.keras.layers.Dense(256, activation=tf.nn.relu), 
                                        tf.keras.layers.Dense(256, activation=tf.nn.relu), 
                                        tf.keras.layers.Dense(256, activation=tf.nn.relu), 
                                        tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                        tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)])

        models[f'SIMPLE_{key}'].compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        models[f'SIMPLE_{key}'].fit(X_values, np.stack(y_values), epochs=20)
        models[f'SIMPLE_{key}'].save(f'{model_dir}/SIMPLE_{key}_EMOTION_CLASSIFIER.h5')

# Also mking one trained with IEMOCAP
iemocap_trng_data = audiodatasets.list_audio_files(f'{data_dir}/fixed_iemocap')
if os.path.exists(f'{model_dir}/SIMPLE_PURE_EMOTION_CLASSIFIER-IEMOCAP.h5'):
    models[f'SIMPLE_PURE_IEMOCAP'] = keras.models.load_model(f'{model_dir}/SIMPLE_PURE_EMOTION_CLASSIFIER-IEMOCAP.h5')
else:
    X_values, y_values, _ = wrapped_simple(iemocap_trng_data)
    models[f'SIMPLE_PURE_IEMOCAP'] = tf.keras.models.Sequential([tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(256, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(256, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(256, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(256, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(8, activation=tf.nn.softmax)])

    models[f'SIMPLE_PURE_IEMOCAP'].compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    models[f'SIMPLE_PURE_IEMOCAP'].fit(X_values, np.stack(y_values), epochs=20)
    models[f'SIMPLE_PURE_IEMOCAP'].save(f'{model_dir}/SIMPLE_PURE_EMOTION_CLASSIFIER-IEMOCAP.h5')


############################
# SEA...load
################################

### FIX ME!!!!!!!

models[f'SEA'] = None
#models[f'SEA'] = keras.models.load_model(open(f'{model_dir}/Speech-Emotion-Analyzer/saved_models/Emotion_Voice_Detection_Model.h5', 'rb'))

def wrapped_sea(filelist):
    X_vals, y_vals, featureindices = extract_features_sea(filelist)

    X_vals = np.expand_dims(X_vals, axis=2)
    
    return X_vals, y_vals, featureindices

# External classifier feature extraction
def extract_features_sea(file_dict):

    features = []
    labels = []
    featureindices = {}
    
    # Integrate 3rd-party feature extraction logic with my data management scheme
    # 3rd Party Source: https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer/blob/master/final_results_gender_test.ipynb 
    indx = 0
    for key in sorted(file_dict.keys()):
        file = file_dict[key]
       
        # First we will see if this is a valid label for the classifier
        emotion = audiodatasets.get_emotion_id (key)
        if emotion == 2:
            emotion = 'calm'
        elif emotion == 3:
            emotion = 'happy'
        elif emotion == 4:
            emotion = 'sad'
        elif emotion == 5:
            emotion = 'angry'
        elif emotion == 6:
            emotion = 'fearful'
        else:
            emotion = 'INVALID'

        gender = 'female' if audiodatasets.get_speaker_id (key) % 2 == 0 else 'male'

        # If this is a valid emotion for the classifier, continue
        if emotion != 'INVALID':
            labels.append(f'{gender}_{emotion}')
        
            X, sample_rate = librosa.load(file, res_type='kaiser_fast', duration=2.5,sr=22050*2,offset=0.5)
            X = librosa.util.pad_center(X, 2.5*22050*2, mode='constant')
            
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
            
            features.append(mfccs)
            
            # Record the index and increment before the next iteration
            featureindices[key] = indx
            indx += 1
    
    # Finally, convert the lists generated above into np.ndarrays and one-hot encode the labels
    features = np.stack(features)

    lb = LabelEncoder()
    labels_onehot = np.zeros(shape=(np.shape(labels)[0], 10))

    # Match the labels as documented here:
    #       https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer/blob/master/README.md
    indx = 0
    for label in labels:
        if label == 'female_angry':
            labels_onehot[indx][0] = 1
        elif label == 'female_calm':
            labels_onehot[indx][1] = 1
        elif label == 'female_fearful':
            labels_onehot[indx][2] = 1
        elif label == 'female_happy':
            labels_onehot[indx][3] = 1
        elif label == 'female_sad':
            labels_onehot[indx][4] = 1
        elif label == 'male_angry':
            labels_onehot[indx][5] = 1
        elif label == 'male_calm':
            labels_onehot[indx][6] = 1
        elif label == 'male_fearful':
            labels_onehot[indx][7] = 1
        elif label == 'male_happy':
            labels_onehot[indx][8] = 1
        elif label == 'male_sad':
            labels_onehot[indx][9] = 1
        else:
            print(f'Oops: {label}')

        indx += 1
    
    return features, labels_onehot, featureindices

############################
# Flair
############################
# Train FLAIR
emotions={
  1:'neutral',
  2:'calm',
  3:'happy',
  4:'sad',
  5:'angry',
  6:'fearful',
  7:'disgust',
  8:'surprised'
}
#DataFlair - Emotions to observe
observed_emotions=['calm', 'happy', 'fearful', 'disgust']

def extract_features_flair(file_dict):
    features = []
    labels = []
    featureindices = {}
       
    # Integrate 3rd-party feature extraction logic with my data management scheme
    # 3rd Party Source: https://data-flair.training/blogs/python-mini-project-speech-emotion-recognition/
    indx = 0
    
    for key in sorted(file_dict.keys()):
        file = file_dict[key]
       
        # First we will see if this is a valid label for the classifier
        emotion = audiodatasets.get_emotion_id(key)
        
        if emotions[emotion] in observed_emotions:
            
            labels.append(emotion)
            
            # Hack for monotone; not in the original paper, but won't work without this
            Path('/tmp/evaluate').mkdir(parents=True, exist_ok=True)
            x, sr = librosa.load(file_dict[key], mono=True)
            sf.write(f'/tmp/evaluate/foo.wav', x, sr, )
            
            # Reference did not specify if their results were achieved with MFCC, Mel Spectrogram or chroma
            # Tried all 3: MFCC came closest to the reported accuracy
            # Using the same extraction setings as the original code

            with sf.SoundFile('/tmp/evaluate/foo.wav') as sound_file:
                X = sound_file.read(dtype="float32")
                sample_rate=sound_file.samplerate

                mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                features.append(list(mfccs))
                
#                stft=np.abs(librosa.stft(X))
#                chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
#                for val in chroma:
#                    features[-1].append(val)

#                mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
#                for val in mel:
#                    features[-1].append(val)
    
            # Update our dictionary to tie keys to the indices for the features and labels, increment the index and continue
            featureindices[key] = indx
            indx += 1
    
    # Before we return, convert the lists generated above into np.ndarrays
    features = np.stack(features)
    labels = np.stack(labels)
    
    return features, labels, featureindices

# Wrapped FLAIR feature extraction
def wrapped_flair(filelist):
    X_vals, y_vals, featureindices = extract_features_flair(filelist)
    return X_vals, y_vals, featureindices

# Finally, train Flair if we have not done so already
if os.path.exists(f'{model_dir}/FLAIR_EMOTION_CLASSIFIER.pickle'):
    models["FLAIR"] = pickle.load(open(f'{model_dir}/FLAIR_EMOTION_CLASSIFIER.pickle', 'rb'))
else:
    flair_training_filelist = audiodatasets.list_audio_files(f'{data_dir}/fixed_ravdess_flair')
    X, y, _ = extract_features_flair(flair_training_filelist)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
    flair_model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500, random_state=101)
    flair_model.fit(X_train,y_train)
    y_pred = flair_model.predict(X_test)
    accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
    
    models["FLAIR"] = flair_model
    pickle.dump(models["FLAIR"], open(f'{model_dir}/FLAIR_EMOTION_CLASSIFIER.pickle', 'wb'))

# ALSO TRAINING A VARIANT WITH ONLY IEMOCAP
if os.path.exists(f'{model_dir}/FLAIR_EMOTION_CLASSIFIER_IEMOCAP.pickle'):
    models["FLAIR_IEMOCAP"] = pickle.load(open(f'{model_dir}/FLAIR_EMOTION_CLASSIFIER_IEMOCAP.pickle', 'rb'))
else:
    flair_training_filelist = audiodatasets.list_audio_files(f'{data_dir}/fixed_iemocap')    
    X, y, _ = extract_features_flair(flair_training_filelist)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
    flair_model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500, random_state=101)
    flair_model.fit(X_train,y_train)
    y_pred = flair_model.predict(X_test)
    accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
    
    models["FLAIR_IEMOCAP"] = flair_model
    pickle.dump(models["FLAIR_IEMOCAP"], open(f'{model_dir}/FLAIR_EMOTION_CLASSIFIER_IEMOCAP.pickle', 'wb'))


############################
# CNN/LSTM
############################

# CNN/LSTM Classifier using Spectrograms
def getMELspectrogram(audio, sample_rate):
    mel_spec = librosa.feature.melspectrogram(y=audio,
                                  sr=sample_rate,
                                  n_fft=1024,
                                  win_length = 512,
                                  window='hamming',
                                  hop_length = 256,
                                  n_mels=128,
                                  fmax=sample_rate/2
                                             )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def splitIntoChunks(mel_spec,win_size,stride):
    t = mel_spec.shape[1]
    num_of_chunks = int(t/stride)
    chunks = []
    for i in range(num_of_chunks):
        chunk = mel_spec[:,i*stride:i*stride+win_size]
        if chunk.shape[1] == win_size:
            chunks.append(chunk)
    return np.stack(chunks,axis=0)

def wrapped_cnnlstm(filelist):
    SAMPLE_RATE = 48000
    x_val=[]
    y_val=[]
    feature_indices={}
    index = 0
    for key in sorted(filelist.keys()):
        file = filelist[key]
        print(file)
        audio, sample_rate = librosa.load(file, duration=3, offset=0.5, sr=SAMPLE_RATE)
        signal = np.zeros((int(SAMPLE_RATE*3,)))
        signal[:len(audio)] = audio
        mel_spectrogram = getMELspectrogram(signal, sample_rate=SAMPLE_RATE)
        chunks = splitIntoChunks(mel_spectrogram, win_size=128,stride=64)
        chunks = np.expand_dims(chunks,1)
        x_val.append(chunks)
        
        #copy from SIMPLE CLASSIFIERS
        head, tail = ntpath.split(file)
        y_val.append(audiodatasets.get_emotion_id(tail) - 1)
        feature_indices[key] = index
        index += 1
        print("\r Processed {}/{} files".format(index,len(filelist)),end='')
        
        
    x_val = np.stack(x_val,axis=0)
    y_val = np.stack(y_val,axis=0)
    scaler = StandardScaler()
    
    b,t,c,h,w = x_val.shape
    x_val = np.reshape(x_val, newshape=(b,-1))
    x_val = scaler.fit_transform(x_val)
    x_val = np.reshape(x_val, newshape=(b,t,c,h,w))
    return x_val,y_val,feature_indices

#######################
#CNN_LSTM CLASSIFIERS model defination
#######################
class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)
        # squash samples and timesteps into a single axis
        elif len(x.size()) == 3: # (samples, timesteps, inp1)
            x_reshape = x.contiguous().view(-1, x.size(2))  # (samples * timesteps, inp1)
        elif len(x.size()) == 4: # (samples,timesteps,inp1,inp2)
            x_reshape = x.contiguous().view(-1, x.size(2), x.size(3)) # (samples*timesteps,inp1,inp2)
        else: # (samples,timesteps,inp1,inp2,inp3)
            x_reshape = x.contiguous().view(-1, x.size(2), x.size(3),x.size(4)) # (samples*timesteps,inp1,inp2,inp3)

        y = self.module(x_reshape)

        # we have to reshape Y
        if len(x.size()) == 3:
            y = y.contiguous().view(x.size(0), -1, y.size(1))  # (samples, timesteps, out1)
        elif len(x.size()) == 4:
            y = y.contiguous().view(x.size(0), -1, y.size(1), y.size(2)) # (samples, timesteps, out1,out2)
        else:
            y = y.contiguous().view(x.size(0), -1, y.size(1), y.size(2),y.size(3)) # (samples, timesteps, out1,out2, out3)
        return y

class HybridModel(nn.Module):
    def __init__(self,num_emotions):
        super().__init__()
        # conv block
        self.conv2Dblock = nn.Sequential(
            # 1. conv block
            TimeDistributed(nn.Conv2d(in_channels=1,
                                   out_channels=16,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1
                                  )),
            TimeDistributed(nn.BatchNorm2d(16)),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.MaxPool2d(kernel_size=2, stride=2)),
            TimeDistributed(nn.Dropout(p=0.4)),
            # 2. conv block
            TimeDistributed(nn.Conv2d(in_channels=16,
                                   out_channels=32,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1
                                  )),
            TimeDistributed(nn.BatchNorm2d(32)),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.MaxPool2d(kernel_size=4, stride=4)),
            TimeDistributed(nn.Dropout(p=0.4)),
            # 3. conv block
            TimeDistributed(nn.Conv2d(in_channels=32,
                                   out_channels=64,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1
                                  )),
            TimeDistributed(nn.BatchNorm2d(64)),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.MaxPool2d(kernel_size=4, stride=4)),
            TimeDistributed(nn.Dropout(p=0.4)),
            # 4. conv block
            TimeDistributed(nn.Conv2d(in_channels=64,
                                   out_channels=128,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1
                                  )),
            TimeDistributed(nn.BatchNorm2d(128)),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.MaxPool2d(kernel_size=4, stride=4)),
            TimeDistributed(nn.Dropout(p=0.4))
        )
        # LSTM block
        hidden_size = 64
        self.lstm = nn.LSTM(input_size=128,hidden_size=hidden_size,bidirectional=False, batch_first=True)
        self.dropout_lstm = nn.Dropout(p=0.3)
        # Linear softmax layer
        self.out_linear = nn.Linear(hidden_size,num_emotions)
    def forward(self,x):
        conv_embedding = self.conv2Dblock(x)
        conv_embedding = torch.flatten(conv_embedding, start_dim=2) # do not flatten batch dimension and time
        lstm_embedding, (h,c) = self.lstm(conv_embedding)
        lstm_embedding = self.dropout_lstm(lstm_embedding)
        # lstm_embedding (batch, time, hidden_size)
        lstm_output = lstm_embedding[:,-1,:]
        output_logits = self.out_linear(lstm_output)
        output_softmax = nn.functional.softmax(output_logits,dim=1)
        return output_logits, output_softmax

def loss_fnc(predictions, targets):
    return nn.CrossEntropyLoss()(input=predictions,target=targets)

def make_validate_step(model, X_test,Y_test,loss_fnc):
    BATCH_SIZE = 5
    DATASET_SIZE= X_test.shape[0]
    model.eval()
    iters = int(len(X_test)/ BATCH_SIZE)
    val_acc=0
    val_loss=0
    for i in range(iters):
        batch_start = i * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(X_test))
        actual_batch_size = batch_end-batch_start
        X = X_test[batch_start:batch_end,:,:,:,:]
        Y = Y_test[batch_start:batch_end]
        X_tensor = torch.tensor(X,device=device).float()
        Y_tensor = torch.tensor(Y, dtype=torch.long,device=device)

        output_logits, output_softmax = model(X_tensor)

        predictions = torch.argmax(output_softmax,dim=1)
        accuracy = torch.sum(Y_tensor==predictions)/float(len(Y))
        loss = loss_fnc(output_logits,Y_tensor)
        
        val_acc += accuracy*actual_batch_size/len(X_test)
        val_loss += loss*actual_batch_size/len(X_test)

    return val_acc,val_loss


def make_train_step(model, X_train,Y_train,loss_fnc, optimizer):
    model.train()
    BATCH_SIZE = 5
    DATASET_SIZE= X_train.shape[0]
    ind = np.random.permutation(DATASET_SIZE)

    X_train = X_train[ind,:,:,:,:]

    Y_train = Y_train[ind]
    epoch_acc = 0
    epoch_loss = 0
    iters = int(DATASET_SIZE / BATCH_SIZE)
    for i in range(iters):
        batch_start = i * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, DATASET_SIZE)
        actual_batch_size = batch_end-batch_start
        X = X_train[batch_start:batch_end,:,:,:,:]
        Y = Y_train[batch_start:batch_end]
        X_tensor = torch.tensor(X,device=device).float()
        Y_tensor = torch.tensor(Y, dtype=torch.long,device=device)
        optimizer.zero_grad()
        # forward pass
        output_logits, output_softmax = model(X_tensor)
        predictions = torch.argmax(output_softmax,dim=1)
        accuracy = torch.sum(Y_tensor==predictions)/float(len(Y))
        # compute loss
        loss = loss_fnc(output_logits, Y_tensor)
        # compute gradients
        loss.backward()
        # update parameters and zero gradients
        optimizer.step()
        epoch_acc += accuracy*actual_batch_size/DATASET_SIZE
        epoch_loss += loss*actual_batch_size/DATASET_SIZE
    return epoch_acc,epoch_loss

##load data 
#tess=audiodatasets.list_audio_files(f'../data/fixed_tess')
#ravdess=audiodatasets.list_audio_files(f'../data/fixed_ravdess')
#loaddata={**tess, **{x: ravdess[x] for x in ravdess if not audiodatasets.is_variant(x)}}
#training_data = { "simple": {**loaddata}}
#X_values, y_values, _ = wrapped_cnnlstm(training_data['simple'])

#num_classes=8
#EPOCHS=700
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#model = HybridModel(num_emotions=num_classes).to(device)
#OPTIMIZER = torch.optim.SGD(model.parameters(),lr=0.01, weight_decay=1e-3, momentum=0.8)


#for epoch in range(EPOCHS):
#    print(f"\r Epoch {epoch}:")
#    acc,loss=make_train_step(model,X_values, y_values,loss_fnc,OPTIMIZER)
#    acc,loss=make_validate_step(model,X_values, y_values,loss_fnc)


###################################################################
#
# LSTM classifier - Gao, et. al. 2022 
#
###################################################################
if not f'{os.getcwd()}/../models/Ser_LSTM' in sys.path:
    sys.path.append(f'{os.getcwd()}/../models/Ser_LSTM')

import SER
gao_lstm = SER.LSTM()

def extract_features_gao_lstm(filelist, max_duration=4.0, sample_rate=10000):
    our_indices = { 5:"angry", 6:"fear", 3:"happy", 1:"neutral", 4:"sad", 8:"surprise"}
    this_classifier_indices = {"angry":0, "fear":1, "happy":2, "neutral":3, "sad":4, "surprise":5}
    
    X_vals = []
    y_values_emo = []
    feature_indices = {}
    
    for key in sorted(filelist.keys()):
        class_indx = audiodatasets.get_speaker_id(key)
        if class_indx in our_indices:
            new_y = this_classifier_indices[our_indices[class_indx]]
            xval = filelist[key]
            
            feature_indices[key] = len(feature_indices)
            X_vals.append(xval)
            y_values_emo.append(new_y)
    
    return X_vals, y_values_emo, feature_indices

class LSTMDOER:
    def __init__(self):
        return
        
    def predict(self, X):
        predictions = []
        
        for x in X:
            predictions.append(SER.LSTM_predict(gao_lstm, x)[0])

        return predictions

models["GAO_LSTM"] = LSTMDOER()


#########################################
# DEFINE "label getters"; typically good enough to find misclassifications
#########################################
def get_class_labels_sea(actuals, predicted, indx):
    actual_label = np.argmax(actuals[indx])

def get_class_labels_simple(actuals, predicted, indx):
    actual_label = actuals[indx]
    predicted_label = np.argmax(predicted[indx])
    
    return actual_label, predicted_label
    predicted_label = np.argmax(predicted[indx])
    
    return actual_label, predicted_label

def get_class_labels_flair(actuals, predicted, indx):
    actual_label = actuals[indx]
    predicted_label = predicted[indx]
    
    return actual_label, predicted_label

def get_class_labels_gao_lstm(actuals, predicted, indx):
    actual_label = actuals[indx]
    predicted_label = predicted[indx]
    
    return actual_label, predicted_label

# Add them to a list


#########################################
# Simple getters for the calling program
##########################################
classifier_names = ["SIMPLE_PURE", "SIMPLE_RECORDED", "SEA", "FLAIR", "GAO_LSTM", "CNNLSTM"]

class_names = {
                    "SIMPLE_PURE": ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'],

                    "SIMPLE_RECORDED": ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'],

                    "SEA": ['female_angry', 'female_calm', 'female_fearful', 'female_happy', 'female_sad', 
                            'male_angry', 'male_calm', 'male_fearful', 'male_happy', 'male_sad'],
    
                    "FLAIR": ['', 'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'],
    
                    "GAO_LSTM": ["angry", "fear", "happy", "neutral", "sad", "surprise"],
    
                    "CNNLSTM": ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'],

}

feature_extractors = { "SIMPLE_PURE": wrapped_simple, 
                       "SIMPLE_RECORDED": wrapped_simple, 
                       "SEA": wrapped_sea, 
                       "FLAIR": wrapped_flair,
                       "GAO_LSTM": extract_features_gao_lstm,
                       "CNNLSTM": wrapped_cnnlstm
                     }

label_getters = {"SIMPLE_PURE": get_class_labels_simple, 
                 "SIMPLE_RECORDED": get_class_labels_simple, 
                 "SEA": get_class_labels_sea, 
                 "FLAIR": get_class_labels_flair,
                 "GAO_LSTM": get_class_labels_gao_lstm
                }     


#mydict = {'003-001-003-002-002-001-004.wav': '../data/ser_lstm_training/happy/003-001-003-002-002-001-004.wav'}
#model1 = LSTMDOER()
#X1, y1, fi1 = feature_extractors["GAO_LSTM"](mydict)
#res = models["GAO_LSTM"].predict(X1)

#for key in fi1:
#    a, b = label_getters["GAO_LSTM"](y1, res, fi1[key])
#    print(key, a, b)