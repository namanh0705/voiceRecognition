from scipy.io import wavfile
from pydub import AudioSegment
from pydub.silence import split_on_silence
import soundfile
import librosa
import glob
import os
import numpy as np
from keras.models import model_from_json

fs, data = wavfile.read('/home/hanh/emotion_detection/test.wav')            # reading the file

wavfile.write('/home/hanh/emotion_detection/channel_1.wav', fs, data[:, 0])   # saving first column which corresponds to channel 1
#wavfile.write('/home/hanh/emotion_detection/guitar_channel_2.wav', fs, data[:, 1])

#sound_file = data[:, 0]
sound_file = AudioSegment.from_wav("/home/hanh/emotion_detection/channel_1.wav")
audio_chunks = split_on_silence(sound_file,
    # must be silent for at least half a second
    min_silence_len=600,

    # consider it silent if quieter than -40 dBFS
    silence_thresh=-40
)

for i, chunk in enumerate(audio_chunks):

    out_file = "/home/hanh/emotion_detection/audio/split1/chunk{0}.wav".format(i)
    print("exporting", out_file)
    chunk.export(out_file, format="wav")

def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
    return result

#from keras.utils import CustomObjectScope
json_file = open('/home/hanh/emotion_detection/model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
Loaded_Model = model_from_json(loaded_model_json)
#Load weights into new model
Loaded_Model.load_weights("/home/hanh/emotion_detection/model/model.h5")
print("Loaded")

em = ['neutral', 'happy', 'angry']

neu = 0
hap = 0
ang = 0
for file in glob.glob('/home/hanh/emotion_detection/audio/split1/*.wav'):
  features = np.array(extract_feature(file, mfcc=True, chroma=True, mel=True).reshape(1, -1))
  basename = os.path.basename(file)
    # predict
  f = np.expand_dims(features,axis=2)
  result = Loaded_Model.predict_classes(f)[0]
    # show the result !

  if (result == 0):
    neu = neu+1
  elif (result == 1):
    hap = hap+1
  else: ang = ang +1
  print('name: ', basename)
  print('result: ', em[result])

print('number neutral: ', neu)
print('number happy: ', hap)
print('number angry: ', ang)

cnt = neu + hap + ang
th = cnt/ang
if th<3:
    print('EMOTION UN-NORMAL')
else:
    print('EM0TION NORMAL')
