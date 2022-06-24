"""demo URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.10/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import include, url
from django.contrib import admin
from django.contrib import auth
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
	url(r'^$', views.index, name='index'),
	url(r'^accounts/', include('django.contrib.auth.urls'), name='accounts'),
	url(r'^polls/', include('polls.urls'), name='polls'),
	url(r'^deep/', include('deep.urls'), name='deep'),
	url(r'^chess/', include('chess.urls'), name='chess'),
	url(r'^neuralauth/', include('neuralauth.urls'), name='neuralauth'),
    url(r'^admin/', admin.site.urls, name='admin'),
# serve any other url as a static file from STATIC_ROOT
] + static('/' , document_root=settings.STATIC_ROOT)

