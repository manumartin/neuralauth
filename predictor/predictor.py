from timeit import default_timer as timer
import Pyro4
import numpy as np
import pandas as pd
from tensorflow import keras


@Pyro4.expose
class Detector(object):
    # These are needed to standardize the incoming values to the same
    # standard as the training data
    distribution_mean = [6.82225917e-02, 4.26156125e+02, 3.81495767e+02]
    distribution_std = [7.10663732e-02, 3.27463301e+02, 2.25397509e+02]

    def __init__(self):
        self.model = keras.models.load_model('network.h5')
        self._print_model(self.model)
        self.embedding_layer = keras.models.Model(
            inputs=self.model.input, outputs=self.model.get_layer(
                'global_average_pooling1d_1').output)

    # self.graph = tf

    # Workaround with problem in keras: it looks like if the call to this
    # method is made remotely it is made in another thread and the default
    # graph isn't the same as the one where the model was created
    # initially, source: https://github.com/fchollet/keras/issues/2397
    # self.graph = tf..get_default_graph()

    @staticmethod
    def _print_model(model):
        print("[*] Sequential model created with the following layers:")
        for layer in model.layers:
            print("{:30}{} -> {}".format(layer.name, layer.input_shape,
                                         layer.output_shape))

    def prepare_data(self, data):
        # Convert to pandas dataframe for easier manipulation
        df = pd.DataFrame(data)
        df.columns = ['client_dt', 'button', 'state', 'x', 'y']

        # Standardize the incoming values
        df['client_dt'] = df['client_dt'].apply(lambda x: (
                (x - self.distribution_mean[0]) / self.distribution_std[0]))
        df['x'] = df['x'].apply(lambda x: (
                (x - self.distribution_mean[1]) / self.distribution_std[1]))
        df['y'] = df['y'].apply(lambda x: (
                (x - self.distribution_mean[2]) / self.distribution_std[2]))

        # Add state columns
        df['Down'] = 0
        df['Drag'] = 0
        df['Move'] = 0
        df['Pressed'] = 0
        df['Released'] = 0
        df['Up'] = 0

        # Add button columns
        df['Left'] = 0
        df['NoButton'] = 0
        df['Right'] = 0
        df['Scroll'] = 0
        # df['XButton'] = 0 # Omitted for now

        # Find every row with a given state and then set the column
        # corresponding to that state to 1 for those rows
        df.loc[df.state == 'Down', 'Down'] = 1
        df.loc[df.state == 'Drag', 'Drag'] = 1
        df.loc[df.state == 'Move', 'Move'] = 1
        df.loc[df.state == 'Pressed', 'Pressed'] = 1
        df.loc[df.state == 'Released', 'Released'] = 1
        df.loc[df.state == 'Up', 'Up'] = 1
        df.loc[df.button == 'Left', 'Left'] = 1
        df.loc[df.button == 'NoButton', 'NoButton'] = 1
        df.loc[df.button == 'Right', 'Right'] = 1
        df.loc[df.button == 'Scroll', 'Scroll'] = 1
        # df.loc[df.button == 'XButton', 'XButton'] = 1

        df.drop('button', axis=1, inplace=True)
        df.drop('state', axis=1, inplace=True)


        print(df.shape)
        a = np.array(df).reshape((-1, 300, 13))
        print(a.shape)
        return a

    def prepare_and_predict(self, data, screen_width, screen_height):
        return self.predict(
            self.prepare_data(data), screen_width, screen_height).tolist()

    def predict(self, data, screen_width, screen_height):
        """
        Makes a prediction by executing the model

        Expects data with the following format:
        A plain python array with dimensions (n_points, n_features) with the
        following features:

        client_dt: int32 with timestamp in seconds
        button: one of ['NoButton', 'Left', 'Scroll', 'Right']
        state: one of ['Move', 'Released', 'Up', 'Down', 'Pressed', 'Drag']
        (Up and Down are for scrolling)
        x: int32 with x position on screen coordinates (bottom left corner
        is 0,0)
        y: int32 with y position on screen coordinates

        - The numerical features will be normalized to zero mean and unit
        variance by using the mean and the standard deviation of the
        training distribution

        TODO: the x and y positions will be rescaled to an idealized
        TODO: screen of width 1 and height between 0 and 1. The height
        TODO: will be given by the aspect ratio of the screen of the client so
        TODO: that the original aspect ratio is preserved while keeping the
        TODO: values between 0 and 1
        """

        # Workaround with problem in Keras: it looks like if the call to this
        # method is made remotely it is made in another thread and the default
        # graph isn't the same as the one where the model was created
        # initially, source: https://github.com/fchollet/keras/issues/2397
        # with self.graph.as_default():
        start = timer()
        y = self.embedding_layer.predict(data)
        end = timer()
        print("[*] Forward pass executed, time elapsed: {:.2f} secs."
              .format(end - start))

        return y


def main():
    daemon = Pyro4.Daemon()
    ns = Pyro4.locateNS()
    detector = Detector()
    # print(f'${detector.predict(np.random.rand(1, 300, 13), 2, 3)}')
    uri = daemon.register(detector)
    ns.register("neuralauth.detector", uri)

    print(f'[*] Detector object registered at ${uri}')

    daemon.requestLoop()


if __name__ == "__main__":
    main()
