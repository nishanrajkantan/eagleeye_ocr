from django import forms
from django.forms import fields
from .models import Post
from django.forms import ClearableFileInput

class PostForm(forms.ModelForm):

    class Meta:
        model = Post
        fields = ['title', 'image']
        labels = {
        'image' : 'PDF Files (Please drag and drop your files into the box below)',
        }
        widgets = {
            'image': ClearableFileInput(attrs={'multiple': True}),
        }