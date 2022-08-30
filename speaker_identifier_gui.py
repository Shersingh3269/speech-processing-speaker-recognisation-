import os
import librosa
import librosa.display
import pickle
import numpy as np
from IPython.display import display, Audio
from tkinter import *
from tkinter import filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from sklearn.svm import SVC
from playsound import playsound
from tkinter import messagebox as mb
import tensorflow as tf
from tensorflow import keras


SAMPLING_RATE = 16000
BATCH_SIZE = 128
SHUFFLE_SEED = 43
samplingrate = None
filepath = None
audio_signal = None
dataset = None
features_mfcc=None
features_fft = None
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

window = Tk()
window.geometry("1600x850")

#Variables for radio button
type_ = IntVar()
values = {"Speaker Identification with MFCC & SVM" : 1, "Speaker Identification with FFT & CNN" : 2}

window.title('Speaker Identification')


title = Label(window, text="Speaker Identification", font=("Helvetica", 18),relief=GROOVE,height = 2, width = 50,anchor='w', justify='center')
title.config(anchor=CENTER)
title.pack(pady=6)
# View for selecting which type.
def showPredictButton(type_,predict_btn):
    if type_.get()==1:
        predict_btn['text'] = "Predict using SVM"
        predict_btn['command'] = lambda: predict(features_mfcc,predicted_lbl,predicted_lbl2)
    elif type_.get()==2:
        predict_btn['text'] = "Predict using CNN"
        predict_btn['command'] = lambda: predict(features_fft,predicted_lbl,predicted_lbl2)

radiogroup = Frame(window)
radiogroup.pack(pady=6)
for idx,(text, value) in enumerate(values.items()):
    r_btn = Radiobutton(radiogroup, text = text, font=("Helvetica", 16), variable = type_, value = value, width=35,indicator = 0,relief=GROOVE, command=lambda: showPredictButton(type_,predict_btn),background = "light blue")
    r_btn.grid(row=0,column=idx)


#Support functions

def paths_and_labels_to_dataset(audio_paths, labels):
    """Constructs a dataset of audios and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(lambda x: path_to_audio(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))

def path_to_audio(path):
    """Reads and decodes an audio file."""
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    return audio
def audio_to_fft(audio):
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])




def errorBox(message):
    mb.showerror("Error", message)

def openFile(type_,speaker_lbl):
    if type_.get()==0:
        errorBox('You have to select which method is to use.')
        return 0
    filepath = filedialog.askopenfilename(filetypes=(("wave files","*.wav"),))
    speaker = filepath.split('/')[-2]
    global samplingrate
    global audio_signal
    global dataset
    signal, sr = librosa.load(filepath)
    audio_signal = signal
    samplingrate = sr
    ds = paths_and_labels_to_dataset([filepath], [speaker])
    ds = ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(
        BATCH_SIZE
    )
    dataset = ds
    speaker_lbl['text']="You have selected an Audio of {}".format(speaker)
 

def plot(y):
    if y is None:
        errorBox('Open the file again.')
        return 0
    if type_.get()==0:
        errorBox('You have to select which method is to use.')
        return 0        
    frame = Frame(window)
    fig = Figure(figsize = (8, 2), dpi = 100)
    plot1 = fig.add_subplot(111)
    plot1.set_title('Audio Signal')
    plot1.plot(y)
    canvas = FigureCanvasTkAgg(fig, master = frame) 
    toolbar = NavigationToolbar2Tk(canvas, frame)
    toolbar.update()
    frame.place(x=650, y=200)
    canvas.get_tk_widget().pack()
    
    



def plot_features(feature):
    if feature is None:
        errorBox("Compute MFCC before plot")
        return 0
    if type_.get()==0:
        errorBox('You have to select which method is to use.')
        return 0  
    frame = Frame(window)
    fig = Figure(figsize = (8, 2), dpi = 100)
    if type_.get()==1:
        plot1 = fig.add_subplot(111)
        plot1.imshow(feature, interpolation='nearest', origin='lower')
        plot1.set_title('MFCC')

    elif type_.get()==2:
        plot1 = fig.add_subplot(111)
        plot1.set_title('FFT')
        plot1.plot(feature)
    canvas = FigureCanvasTkAgg(fig, master = frame) 
    toolbar = NavigationToolbar2Tk(canvas, frame)
    toolbar.update()
    frame.place(x=650, y=500)
    canvas.get_tk_widget().pack()

    

def computeFeature(audio_signal,dataset,sr):
    if dataset is None or sr is None:
        errorBox("Read the audio file or open an audio")
        return 0
    if type_.get()==0:
        errorBox('You have to select which method is to use.')
        return 0  
    global features_fft
    global features_mfcc

    mfcc = librosa.feature.mfcc(y=audio_signal, sr=sr)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    data = np.empty((20,132))
    data[:,:44] = mfcc
    data[:,44:88] = mfcc_delta
    data[:,88:] = mfcc_delta2
    features_mfcc = data.reshape(data.shape[0]*data.shape[1])
    
    for audios, lbls in dataset.take(1):
        features_fft = audio_to_fft(audios)
        break

    if type_.get()==1:
        plot_features(mfcc)
    elif type_.get()==2:
        plot_features(features_fft[0])


def predict(features,predicted_lbl,predicted_lbl2):
    labels = ['Benjamin_Netanyau', 'Jens_Stoltenberg', 'Julia_Gillard', 'Magaret_Tarcher', 'Nelson_Mandela']
    types = ['MFCC','FFT']
    result = None
    if type_.get()==0:
        errorBox('You have to select which method is to use.')
        return 0
    if features is None:
        errorBox("Please compute {} featurs".format(types[type_.get()-1]))
        return 0
    if type_.get()==1:
        model = pickle.load(open('speaker_identifier.sav', 'rb'))
        result = model.predict(np.array([features]))
        predicted_lbl['text']="Predicted speaker is (SVM): "
    else:
        loaded_model = tf.keras.models.load_model("speaker_identifier_cnn.h5")
        y_pred = loaded_model.predict(features)
        result = np.argmax(y_pred, axis=-1)
        predicted_lbl['text']="Predicted speaker is (CNN): "

    predicted_lbl2['text']=labels[result[0]]

def play(path):
    if path is None:
        errorBox('Please select the audio.')
        return 0
    playsound(path)






load = Button(window,text="Select Audio", font=("Helvetica", 16), height = 2, width = 15,relief=GROOVE, command=lambda: openFile(type_,speaker_lbl))
load.place(x=10, y=200)

play_button = Button(window, text="Play Audio", font=("Helvetica", 16), height = 2, width = 15, relief=GROOVE, command= lambda: play(filepath))
play_button.place(x=350, y=200)

speaker_lbl = Label(window, text="You have not selected any audio", font=("Helvetica", 16),relief=GROOVE,height = 2, width = 50,anchor='w', justify='left')
speaker_lbl.place(x=10, y=300) 

load_lbl = Label(window, text='Click here to plot the signal : ',font=("Helvetica", 12), height = 2, width = 30,anchor='w', justify='left')
load_lbl.place(x=10, y=410)
load = Button(window,text="Plot",font=("Helvetica", 16), height = 2, width = 15,relief=GROOVE,command=lambda: plot(audio_signal))
load.place(x=350, y=400)


mfc_lbl = Label(window, text='Create features and plot features : ',font=("Helvetica", 12), height = 2, width = 30,anchor='w', justify='left')
mfc_lbl.place(x=10, y=510)
mfc = Button(window,text="MFCC/FFT", font=("Helvetica", 16),height = 2, width = 15, relief=GROOVE, command=lambda: computeFeature(audio_signal,dataset,samplingrate))
mfc.place(x=350, y=500)

predict_btn = Button(window,text="Predict",font=("Helvetica", 16), height = 2, width = 50,relief=GROOVE, command=lambda: predict(features_mfcc,predicted_lbl,predicted_lbl2))
predict_btn.place(x=10, y=600)

predicted_lbl = Label(window, text="Predicted speaker is : ", font=("Helvetica", 16), height = 2, width = 40, relief=GROOVE, anchor='w', justify='left')
predicted_lbl.place(x=10, y=700)
predicted_lbl2 = Label(window, text="Not predicted yet", font=("Helvetica", 16), height = 2, width = 20,anchor='w',relief=GROOVE, justify='left', bg="#000", fg="#FFF")
predicted_lbl2.place(x=350, y=700)

window.state('zoomed')
window.mainloop()