import pickle
import datetime
import random
import itertools
from collections import OrderedDict

import numpy as np

import Pyro4

from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
# in theory prevents matplotlib from wanting an x server to render things which will
# raise a tkinter exception if executed on a server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter

from django.shortcuts import render
from django.core import serializers
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authentication import SessionAuthentication, BasicAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import JSONParser
from rest_framework.decorators import api_view
from rest_framework.decorators import authentication_classes
from rest_framework.decorators import permission_classes
from rest_framework.decorators import parser_classes

def timestamps_to_dts(timestamps):
    """ Converts a list of timestamps to time deltas """
    orig = timestamps[0]
    timestamps = [i-orig for i in timestamps]
    timestamps[1:] = [timestamps[idx+1] - timestamps[idx] for idx, i in enumerate(timestamps[1:])]
    return timestamps

def translate(data):
    """ 
    convert browser input data to the format that the neural network used for training
    (timestamp, e.buttons, type, x, y)
    - we will convert the timestamp to time differences and from milliseconds to seconds
    """

    result = []
    data = np.array(data, dtype=object)
    dts = timestamps_to_dts(data[:,0]) # convert to time deltas
    dts = [i/1000.0 for i in dts] # convert to seconds
    data[:,0] = dts

    buttons_map = {0: 'NoButton', 1: 'Left', 2: 'Right' }
    states_map = {0: 'Move', 1: 'Pressed', 2:'Released', 3: 'Drag'}

    data[:,1] = [buttons_map.get(i, 'Unknown') for i in data[:,1]]
    data[:,2] = [states_map.get(i, 'Unknown') for i in data[:,2]]

    return data.tolist()

# Use pickle as an improvised database
def persist(id, obj):
    with open('state/user_{}_state.pkl'.format(id), 'wb') as tmp:
        pickle.dump(obj, tmp)

def load(id):
    with open('state/user_{}_state.pkl'.format(id), 'rb') as tmp:
        return pickle.load(tmp)

def reset_state(id):
    with open('state/user_{}_state.pkl'.format(id), 'wb') as tmp:
        pickle.dump(OrderedDict(), tmp)

def get_or_create_state(id):
    state = OrderedDict()

    try:
        state = load(id)
    except Exception:
        pass
    
    return state

def plot_confusion_matrix(cm, classes, rangemin,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    #plt.imshow(cm, interpolation='nearest', cmap=cmap, norm=matplotlib.colors.PowerNorm(gamma=3.0, vmin=0.99, vmax=1.0))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=rangemin, vmax=1.0)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = 0.8
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:.5f}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Users')
    plt.xlabel('Users')

@api_view(['GET'])
@authentication_classes((SessionAuthentication, BasicAuthentication))
@permission_classes((IsAuthenticated,))
@parser_classes((JSONParser,))
def compare(request):
    """ Returns a PNG image with a confusion matrix graph that shows how the network compares
    any user against every other user"""

    if request.method == 'GET':

        rangemin = float(request.query_params['rangemin'])

        state = get_or_create_state(request.user.id)

        confusion_matrix = np.zeros((len(state.items()),len(state.items())))
        # this can be done with itertools but it uses lexicographic order?
        combinations = [(x,y) for x in enumerate(state.items()) for y in enumerate(state.items())] 
        for (i, (user1, embedding1)), (j, (user2, embedding2)) in combinations:
            confusion_matrix[i, j] = cosine_similarity(embedding1, embedding2)

        fig = plt.figure()
        plot_confusion_matrix(confusion_matrix, state.keys(), rangemin)
        canvas=FigureCanvas(fig)

        response = HttpResponse(content_type='image/png')
        canvas.print_png(response)
        return response

@api_view(['POST'])
@authentication_classes((SessionAuthentication, BasicAuthentication))
@permission_classes((IsAuthenticated,))
@parser_classes((JSONParser,))
def reset(request):
    """ Resets the saved recognition signatures"""
    if request.method == 'POST':
        reset_state(request.user.id)
        return Response()

@api_view(['POST'])
@authentication_classes((SessionAuthentication, BasicAuthentication))
@permission_classes((IsAuthenticated,))
@parser_classes((JSONParser,))
def recognize(request):
    """
    Sends the recorded input sequence the neural network which computes a 
    signature. The signature is saved and associated with the given user.
    """
    if request.method == 'POST':
        state = get_or_create_state(request.user.id)

        user = request.data['user']
        data = translate(request.data['input'])

        detector = Pyro4.Proxy('PYRONAME:neuralauth.detector')
        embedding = np.array(detector.prepare_and_predict(data, 1024, 768))

        # We asume the average embedding will be representative of all the recorded
        # sequences
        try:
        #   state[user]
        #   print('distance :{}'.format(np.linalg.norm(state[user] - embedding)))
           state[user] = np.mean([state[user], embedding], axis=0) 
        except KeyError:
           state[user] = embedding
        #state[user] = embedding

        persist(request.user.id, state)

        return Response(state[user])
