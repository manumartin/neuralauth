import os
import sys
import pickle

import numpy as np
import pandas as pd

from keras.optimizers import Adam
from keras.models import load_model
from keras import backend as tfbackend
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.callbacks import Callback

from mouse_challenge_utils import * 
from model import *

class StdoutFlusher(Callback):
    def on_train_begin(self, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        sys.stdout.flush()

def main():

    training_sessions, validation_sessions, testing_sessions = load_mouse_dynamics_dataset()
    df_train = sessions_to_dataframe(training_sessions)
    df_val = sessions_to_dataframe(validation_sessions)
    df_train = preprocess_data(df_train)
    df_val = preprocess_data(df_val)

    # SPECIAL, make this in a more generic way
    df_train = df_train.drop(['XButton'], axis = 1) # Drop XButton column from df_train

    seq_size = 300

    train_x, train_y = data_to_machine_learning_examples(df_train, seq_size)
    val_x, val_y = data_to_machine_learning_examples(df_val, seq_size)

    model = load_model('model/model_18.h5')

    cb_check = ModelCheckpoint('model/model_18_checkpoint.h5', monitor='val_loss', verbose=1, period=30)
    cb_reducelr = ReduceLROnPlateau(verbose=1)
    cb_tensorboard = TensorBoard(log_dir='./logs', histogram_freq=30, write_graph=True)

    epochs = 3000 
    batch_size = 30

    hist = model.fit(train_x, train_y, 
                 batch_size, epochs, 2, 
                 validation_data=(val_x, val_y),
                 callbacks =[cb_check, cb_reducelr, cb_tensorboard, StdoutFlusher()])

    model.save('model/model_simple.h5')

    data = {}
    data['history'] = hist.history

    with open('model/model_simple.hist', 'wb') as f:
        pickle.dump(data, f)

    # Uncomment this if this error appears:
    # AttributeError: 'NoneType' object has no attribute 'TF_DeleteStatus'
    # https://github.com/tensorflow/tensorflow/issues/3388
    tfbackend.clear_session()

if __name__ == "__main__":
    main()
