from django.conf.urls import url

from . import views

app_name = 'neuralauth'
urlpatterns = [
	url(r'^recognize/?$', views.recognize, name='recognize'),
	url(r'^reset/?$', views.reset, name='reset'),
	url(r'^compare/?$', views.compare, name='compare'),
]
