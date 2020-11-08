import os 
import random
import librosa as lib
import pickle as pk
import numpy as np

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def transform(data_path,dataset_path,transform_type,sampling_rate = 16000):
    os.makedirs(dataset_path, exist_ok=True)
    start = datetime.now()
    artists = [artist for artist in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, artist))]
    for artist in artists:
        print("Accessing Artist: {}".format(artist))
        artist_path = os.path.join(data_path, artist)
        albums = os.listdir(artist_path)
    
        for album in albums:
            print("-- Album: {}".format(album))
            album_path = os.path.join(artist_path, album)
            songs = os.listdir(album_path)
        
            for song in songs:
                print("---- Song: {}".format(song))
                song_path = os.path.join(album_path,song)
                y,sr = lib.load(song_path, sr=sampling_rate)
                
                if transform_type == "mfcc":
                    S = lib.feature.mfcc(y,sr=sr)
                
                elif transform_type == "spectrogram":
                    S = lib.feature.melspectrogram(y,sr=sr)
                    S = lib.power_to_db(S)
                    
                file = artist+"--"+album+"--"+song
                data = (S,artist,song)
                with open(os.path.join(dataset_path,file),'wb') as file_path:
                    pk.dump(data,file_path)
                    
    print("Time taken: {}".format(datetime.now()-start))
 
def load_album(data_path,dataset_path,random_state = 1234):
    random.seed(random_state)
    songs =  os.listdir(dataset_path)
    artists = [artist for artist in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, artist))]
    train, test, val =[], [], []
    
    for artist in artists:
        artist_path = os.path.join(data_path, artist)
        albums = os.listdir(artist_path)
        random.shuffle(albums)
        test.append(artist + "--" + albums.pop(0))
        val.append(artist + "--" + albums.pop(0))
        train = train + [artist + "--" + album for album in albums]
        
    X_train, y_train, s_train = [], [], []
    X_val, y_val, s_val = [], [], []
    X_test, y_test, s_test = [], [], []
    
    for song in songs:
        with open(os.path.join(dataset_path,song),'rb') as file_path:
            load = pk.load(file_path)
        artist, album, song_name = song.split("--")
        artist_album = artist + "--" + album
        
        if artist_album in train:
            X_train.append(load[0])
            y_train.append(load[1])
            s_train.append(load[2])
        elif artist_album in val:
            X_val.append(load[0])
            y_val.append(load[1])
            s_val.append(load[2])
        elif artist_album in test:
            X_test.append(load[0])
            y_test.append(load[1])
            s_test.append(load[2])
    
    return X_train, y_train, s_train, X_val, y_val, s_val, X_test, y_test, s_test
    
def load_songs(data_path,dataset_path,val_size,test_size,random_state = 1234):
    songs =  os.listdir(dataset_path)
    
    X, Y, s =[], [], []
    
    for song in songs:
        with open(os.path.join(dataset_path,song),'rb') as file_path:
            load = pk.load(file_path)
        X.append(load[0])
        Y.append(load[1])
        s.append(load[2])
    
    X_train, X_test, Y_train, Y_test, s_train, s_test = train_test_split(X, Y, s, test_size=test_size, stratify=Y, random_state=random_state)
    
    X_train, X_val, Y_train, Y_val, s_train, s_val = train_test_split(X_train, Y_train, s_train, test_size=val_size, stratify=Y_train, random_state=random_state)
    
    return X_train, y_train, s_train, X_val, y_val, s_val, X_test, y_test, s_test

def slice_song(X,y,S, slice_len):
    spectrograms, artists, songs = [], [], []
    
    for i, spectrogram in enumerate(X):
        slices = spectrogram.shape[1]//slice_len
        for j in range(slices-1):
            spectrograms.append(spectrogram[:,(slice_len*j):(slice_len*(j+1))])
            artists.append(y[i])
            songs.append(S[i])
    
    return np.array(spectrograms), np.array(artists), np.array(songs)
    
def encode(y, label=None, onehot=None):
    y_len = len(y)
    
    if not label:
        label = preprocessing.LabelEncoder()
    y_label = label.fit_transform(y).reshape(y_len,1) 
    
    if not onehot:
        onehot = preprocessing.OneHotEncoder()
    y_onehot = onehot.fit_transform(y_label).toarray()
    
    return y_onehot, label, onehot