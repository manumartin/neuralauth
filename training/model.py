from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.core import Dense, Activation, Flatten
from keras.regularizers import l2

def create_model_traditional(input_shape, classes, batch_norm_momentum, l2_regularization):

    model = Sequential([
        Conv1D(32, 8, padding='same', strides=2, 
               input_shape=input_shape, kernel_regularizer=l2(l2_regularization)),
        BatchNormalization(momentum=batch_norm_momentum),
        LeakyReLU(),

        Conv1D(64, 5, padding='same', strides=2, kernel_regularizer=l2(l2_regularization)),
        BatchNormalization(momentum=batch_norm_momentum),
        LeakyReLU(),

        Conv1D(128, 3, padding='same', strides=2, kernel_regularizer=l2(l2_regularization)),
        BatchNormalization(momentum=batch_norm_momentum),
        LeakyReLU(),
    
        GlobalAveragePooling1D(),
        Dense(classes, activation='softmax', kernel_regularizer=l2(l2_regularization))
    ])

    return model

def create_model_paper(input_shape, classes, batch_norm_momentum, l2_regularization):
    
    model = Sequential([
            Conv1D(128, 8, padding='same', input_shape=input_shape, kernel_regularizer=l2(l2_regularization)),
            BatchNormalization(momentum=batch_norm_momentum),
            LeakyReLU(),

            Conv1D(256, 5, padding='same', kernel_regularizer=l2(l2_regularization)),
            BatchNormalization(momentum=batch_norm_momentum),
            LeakyReLU(),

            Conv1D(128, 3, padding='same', kernel_regularizer=l2(l2_regularization)),
            BatchNormalization(momentum=batch_norm_momentum),
            LeakyReLU(),

            GlobalAveragePooling1D(),
            Dense(classes, activation='softmax', kernel_regularizer=l2(l2_regularization))
    ])

    return model