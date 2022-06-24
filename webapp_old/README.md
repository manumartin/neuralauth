# Web application

This folder contains the django backend and the frontend for the web application 
that allows a user to interact with the neural network. 

It has three django apps:

deep: frontend/backend for the c&c section of the web 

chess: frontend/backend for the chess secion of the web

neuralauth: backend exposed through a REST API, exposes the service
used to analyze user input with a neural network. The neural network
runs on a process independent of the web application.
