import threading
import time
import sys
import numpy as np
import pickle

from pymouse import PyMouseEvent
from pykeyboard import PyKeyboardEvent

if len(sys.argv) == 1:
    print("Usage: capture.py <filename> <label> <secs>")
    sys.exit()

filename = sys.argv[1]
label = sys.argv[2]
secs = int(sys.argv[3])

# Dataset:
# timestamp, device, event_type, x, y

events = np.array([], dtype=np.int32).reshape(0, 5)
labels = np.array([], dtype=np.int32).reshape(0, 1)
before = None


class MouseCapturer(PyMouseEvent):
    def __init__(self):
        PyMouseEvent.__init__(self)

    def click(self, x, y, button, press):
        global events, labels
        event_type = 'MOUSE_{}_{}'.format(button, 'DOWN' if press else 'UP')
        event = [time.time(), 'MOUSE', event_type, x, y]
        events = np.vstack([events, event])
        labels = np.vstack([labels, label])
        print(event)

    def move(self, x, y):
        global events, labels, before
        dt = 0
        if before == None:
            before = time.time()
        else:
            now = time.time()
            dt = now - before
            before = now
        event = [dt, 'MOUSE', 'MOUSE_MOVE', x, y]
        events = np.vstack([events, event])
        labels = np.vstack([labels, label])
        print(event)

    def scroll(self, x, y, vertical, horizontal):
        global events, labels
        event_type = 'MOUSE_SCROLL_{}'.format('UP' if vertical == 1 else 'DOWN')
        event = [time.time(), 'MOUSE', event_type, x, y]
        events = np.vstack([events, event])
        labels = np.vstack([labels, label])
        print(event)


class KeyboardCapturer(PyKeyboardEvent):
    def __init__(self):
        PyKeyboardEvent.__init__(self)

    def tap(self, keycode, character, press):
        global events, labels
        event_type = 'KEY_{}_{}'.format(keycode, 'DOWN' if press else 'UP')
        x = 0
        y = 0
        if len(events):
            x = events[-1][-2]
            y = events[-1][-1]
        event = [time.time(), 'KEYBOARD', event_type, x, y]
        events = np.vstack([events, event])
        labels = np.vstack([labels, label])
        print(event)


mouseCapturer = MouseCapturer()
mouseCapturer.start()

keyboardCapturer = KeyboardCapturer()
keyboardCapturer.start()


def serialize(filename, events, labels):
    file = open(filename, 'wb')
    pickle.dump({'x': events, 'y': labels}, file)
    file.close()


try:
    time.sleep(secs)
except KeyboardInterrupt:
    pass
finally:
    serialize(filename, events, labels)

sys.exit(0)
