#!/bin/sh
source venv/bin/activate
pyro4-ns &
python predictor/predictor.py &
python manage.py runserver 0.0.0.0:8000 &
