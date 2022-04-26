from django.urls import path

from .views import *
from . import views

urlpatterns = [
    path('', HomePageView.as_view(), name='home'),
    # path('post', CreatePostView.as_view(), name='add_post'),
    path('post', views.upload_pdf, name='add_post'),
    path('training', TrainingView.as_view(), name='training'),
    path('result', ResultView.as_view(), name='result'),
]