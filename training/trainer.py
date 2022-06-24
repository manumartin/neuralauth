import os
import pickle

import numpy as np

from keras.optimizers import Adam
from keras import backend as tfbackend

import mouse_challenge_utils
import model

def cartesian_product(*data):
    """ returns the cartesian product of the arrays/vectors passed as arguments """
    return np.squeeze(np.array(np.meshgrid(*data))).T.reshape(-1, len(data))

def main():

    os.mkdir('results')

    # Load data
    print("[*] Loading Mouse_Dynamics_Dataset data..")
    training_sessions, validation_sessions, testing_sessions = load_mouse_dynamics_dataset()    
    print_stats(training_sessions, 'training')
    print_stats(validation_sessions, 'validation')
    print_stats(testing_sessions, 'testing')
    print()
    print('[?] User 0 represents malicious user activity while User -1 represents an unknown user (unlabeled data)')

    print("[*] Converting data to Pandas dataframes..")
    df_train = sessions_to_dataframe(training_sessions)
    df_val = sessions_to_dataframe(validation_sessions)

    print("[*] Preprocessing data for machine learning consumption.. ")
    df_train = preprocess_data(df_train)
    df_val = preprocess_data(df_val)
    df_train = df_train.drop(['XButton'], axis = 1) # Drop XButton column from df_train

    print("[*] Example of data being fed to the algorithm:")
    print(df_train.head())

    # Create hyperparameter combinations
    combinations = cartesian_product([0.0001, 0.00001, 0.000001], [150, 200, 250, 300, 350], [20, 30, 40], [200])

    print("[*] {} hyperparameter combinations".format(len(combinations)))

    for idx, hyperparameters in enumerate(combinations):

        # Hyperparameters for this training session
        seq_size = int(hyperparameters[1])
        epochs = int(hyperparameters[3])
        batch_size = int(hyperparameters[2]) 
        learning_rate = hyperparameters[0]
        batch_norm_momentum = 0.2
        n_classes = 10
        data_point_dimensionality = 13
        l2_regularization = 0.01

        print("[*] Training model {}, using the following hyperparameters:".format(idx))
        print()
        print("seq_size: {}".format(seq_size))
        print("epochs: {}".format(epochs))
        print("batch_size: {}".format(batch_size))
        print("learning_rate: {}".format(learning_rate))
        print("batch_norm_momentum: {}".format(batch_norm_momentum))
        print("n_classes: {}".format(n_classes))
        print("data_point_dimensionality: {}".format(data_point_dimensionality))
        print("l2_regularization: {}".format(l2_regularization))
        print()

        print("[*] Generating training examples..")
        train_x, train_y = data_to_machine_learning_examples(df_train, seq_size)
        print('[*] Generated traning examples {} and labels {}'.format(train_x.shape, train_y.shape))
        val_x, val_y = data_to_machine_learning_examples(df_val, seq_size)
        print('[*] Generated validation examples {} and labels {}'.format(val_x.shape, val_y.shape))

        model = create_model_paper(input_shape = (seq_size, data_point_dimensionality),
                             classes = n_classes,
                             batch_norm_momentum = batch_norm_momentum,
                             l2_regularization = l2_regularization)

        optimizer = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        #TODO: Usar callbacks para reduceLROnPlateau, Tensorboard y checkpoints

        print('[*] Training the model..')
        history = model.fit(train_x, train_y, 
                         batch_size, epochs, 2, 
                         validation_data=(val_x, val_y))

        print('[*] Model {} trained, saving data.. ')

        # process history data to extract things like validation accuracy variance, and overfitting measure
        # save hyperparameters, model and training history
        model.save('results/model_{}.h5'.format(idx))

        data = {}
        data['history'] = history.history
        data['hyperparameters'] = hyperparameters

        with open('results/model_{}.hist'.format(idx), 'wb') as f:
            pickle.dump(data, f)
    
        # Uncomment this if this error appears:
        # AttributeError: 'NoneType' object has no attribute 'TF_DeleteStatus'
        # https://github.com/tensorflow/tensorflow/issues/3388
        tfbackend.clear_session()

if __name__ == "__main__":
    main()
