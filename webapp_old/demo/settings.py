"""
Django settings for demo project.

Generated by 'django-admin startproject' using Django 1.10.5.

For more information on this file, see
https://docs.djangoproject.com/en/1.10/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/1.10/ref/settings/
"""

import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/1.10/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'XXXXX'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = [ 'XXXXX' ]


# Application definition

INSTALLED_APPS = [
	'deep.apps.DeepConfig',
	'chess.apps.ChessConfig',
	'neuralauth.apps.NeuralauthConfig',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
	'django_forms_bootstrap',
	'rest_framework'
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'demo.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'demo.wsgi.application'


# Database
# https://docs.djangoproject.com/en/1.10/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}


# Password validation
# https://docs.djangoproject.com/en/1.10/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/1.10/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'Europe/Madrid'

USE_I18N = True

USE_L10N = True

USE_TZ = True

##### django rest framework ####
#REST_FRAMEWORK = {
#	'DEFAULT_AUTHENTICATION_CLASSES' : (
#		'rest_framework.authentication.SessionAuthentication',
#	),
#}


##### START CUSTOM CONFIGURATION #####

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/1.10/howto/static-files/

# Explanation:
# 1. The django staticfiles app allows the linking of static files inside
# templates using the {% static %} keyword.
# 2. Any files referenced in this way will be accesed through a url starting with app/STATIC_URL/*
# 3. In addition files referenced in root templates in this way will be searched inside /static/* (this is added
# below to STATICFILES_DIRS 
# 4. When developing, the development webserver will find the static files in the 'static' directory 
# of each application or the 'static' directory of the project
# 5. by doing this we can mantain every static file inside its own application while developing
# 6. Unfortunately we don't control every static file inclusion since we use third party js stuff that
# generates custom urls, if a js was included in app 'my_app' and wants a static 'images/image.jpg' it will
# end up requesting my_app/images/image.jpg. 
# 7. The development webserver doesnt know what to do with this, that's why we include in the project urls.py file
# a clause at the end that serves all the files inside STATIC_ROOT.
# 8. STATIC_ROOT is the place were all the static files will be collected by calling python manage.py collectstatic
# 9. The contents of STATIC_ROOT  will be deployed to a CDN or a webserver optimized for statics like nginx etc

# were to find static files
STATIC_URL = '/static/'
STATICFILES_DIRS = [
	os.path.join(BASE_DIR, "static"),
]

# This is needed to also search for statics in app directories
STATICFILES_FINDERS = (
	'django.contrib.staticfiles.finders.FileSystemFinder',
	'django.contrib.staticfiles.finders.AppDirectoriesFinder',
)

# were static files will be collected for deployment
STATIC_ROOT = 'collected_statics'

LOGIN_REDIRECT_URL = '/'

# Serve arbitrary static files, there is also a static view in the main urls.py file
#MEDIA_URL='/'
#MEDIA_ROOT='/home/manuel/directorio/proyectos/deeplearning/demo/media'

EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'