# Prediction process 

This is the process that loads the neural network and feeds it incoming mouse 
input sequences to generate predictions. The process runs in the background 
and has an API exported with an RPC library called pyro.

A pyro nameserver has to be launched so that clients of the API are able to 
find the service with just a name.

To launch the processes first make sure the virtual environment is active:

```
source venv/bin/activate 
```

Then launch the pyro nameserver:
```
pyro4-ns
```

And finally the predictor process:
```
python predictor.py
```

