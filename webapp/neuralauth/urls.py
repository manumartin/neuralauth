from django.urls import path

from . import views

app_name = 'neuralauth'
urlpatterns = [
	path('recognize/', views.recognize, name='recognize'),
	path('reset/', views.reset, name='reset'),
	path('compare/', views.compare, name='compare'),
]
