# Biometric authentication with neural networks

This project is an exploration of how we can use neural networks
to try to identify a user by using their mouse usage patterns. It's composed
of a web application that reads the mouse input and sends it to a backend
where a convolutional neural network is executed against this input.
The network is pretrained against a dataset of different known users so
that it learns how to tell them apart from one another. The question is if we 
can use the features or representations generated by the network as a kind
of signature so that we can distinguish users or detect anomalous patterns. 

### Background 

Identifying users by verifying credentials like passwords or phone
access like in two-factor authentication works well enough in many cases.
However, credentials and phones can be stolen and can be used by 
malicious actors to impersonate real users in order to gain access 
to restricted areas or information.

In the physical world people are often not only identified by credentials 
but also by other pieces of information like physical traits or behaviours
that are more or less unique to each person and very difficult to duplicate.
In the case of web applications we can't see the person but we have access 
to more information than just credentials, for example:

- Mouse and keyboard input patterns.
- Behaviour patterns, how the user interacts with a given application.
- Information about the computer of the user which can be given away 
by the browser.

With this information we can try to verify the identity of the user or at
least become aware of strange patterns which may alert us of a possible
impersonation.

### Components 

The project is organized in several directories:

**predictor/** : Contains the code that executes the model, it runs
in a separate process. The backend communicates with this process by using 
an RPC library called Pyro (PYthon Remote Objects). 

**training/** : Contains the training code and several jupyter notebooks
where the dataset and the model performance is explored. There is also
a utility to record mouse input from a local user etc.

**webapp/** : Contains a django web application with a backend and a frontend
so that users can interact with the model.

### Launching

- Install python 3.8
- Configure the webapp/webapp/settings.py file, the ALLOWED_HOSTS variable etc..
- Install python dependencies and run the launch script: 

```
pip install -r requirements.txt
./launch.sh
```

