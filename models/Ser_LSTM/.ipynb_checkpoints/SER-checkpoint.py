import numpy as np
import os
from tensorflow.keras import utils

from DNN_Model import LSTM_Model
#from ML_Model import SVM_Model
#from ML_Model import MLP_Model
from DNN_Model import CNN_Model

from Utilities import get_feature
from Utilities import get_feature_svm
from Utilities import get_data
from Utilities import load_model
from Utilities import Radar

DATA_PATH = '../../data/ser_lstm_training'
# CLASS_LABELS = ("Angry", "Happy", "Neutral", "Sad")
# CLASS_LABELS = ("Angry", "Fearful", "Happy", "Neutral", "Sad", "Surprise")
CLASS_LABELS = ("angry", "fear", "happy", "neutral", "sad", "surprise")

def LSTM():
    FLATTEN = False
    LOAD_MODEL = 'DNN'
    NUM_LABELS = len(CLASS_LABELS)
    SVM = False
    
    model = LSTM_Model(input_shape = (398, 39), num_classes = NUM_LABELS)

    if os.path.exists('Models/LSTM1.h5'):
#        model.model.load_weights('Models/LSTM1.h5')
        model = load_model(model_name = "LSTM1", load_model = LOAD_MODEL)
    else:
        x_train, x_test, y_train, y_test = get_data(DATA_PATH, class_labels = CLASS_LABELS, flatten = FLATTEN, _svm = SVM)
        y_train = utils.to_categorical(y_train)
        y_test_train = utils.to_categorical(y_test)

        model.train(x_train, y_train, x_test, y_test_train, n_epochs = 100)
        model.evaluate(x_test, y_test)
        model.save_model("LSTM1")

    return model

def LSTM_predict(model, file_path: str):
    
    result = np.argmax(model.predict(np.array([get_feature(file_path)])))
    result_prob = model.predict(np.array([get_feature(file_path)]))[0]

    return result, result_prob

#lstm = LSTM()
#pred, pred_prob = LSTM_predict(lstm, "angry_sample.wav")

#print(pred)
#print(pred_prob)