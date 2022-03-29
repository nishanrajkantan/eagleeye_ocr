from django.db import models
from django.shortcuts import redirect, render
from django.views.generic import ListView, CreateView, TemplateView
from django.urls import reverse, reverse_lazy
from .forms import PostForm
from .models import Post
from django.core.files.base import ContentFile
from django.core.files import File

import os
import subprocess
import tensorflow as tf
import cv2
from object_detection.utils import label_map_util
import pikepdf
from PyPDF2 import PdfFileReader
from tqdm import tqdm
from pdf2image import convert_from_path
import pandas as pd
from inferenceutils import *

# Create your views here.
class HomePageView(ListView):
    model = Post
    template_name = 'home.html'

class CreatePostView(CreateView):
    model = Post
    form_class = PostForm
    template_name = 'post.html'
    success_url = reverse_lazy('result')

    def form_valid(self, form):
        return super().form_valid(form)

class ResultView(TemplateView):
    template_name = "result.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        obj = Post.objects.latest('id')

        zipped_model = 'inferenceutils.py'
        if not os.path.exists(zipped_model):
            subprocess.call('powershell -Command "Invoke-WebRequest https://raw.githubusercontent.com/hugozanini/object-detection/master/inferenceutils.py -OutFile .\\inferenceutils.py"')

        zipped_model = 'test_cimb\\saved_model'
        if not os.path.exists(zipped_model):
            subprocess.call("powershell Expand-Archive -Path 'test_cimb\\saved_model.zip' -DestinationPath 'test_cimb\\saved_model\\'")

        output_directory = 'inference_graph'
        labelmap_path = 'test_cimb\\saved_model\\saved_model\\label_map.pbtxt'

        category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
        tf.keras.backend.clear_session()
        model = tf.saved_model.load(f'test_cimb\\saved_model\\saved_model')

        test = pd.read_csv('test_cimb\\saved_model\\saved_model\\test_labels.csv')

        input_path = 'media\\images\\'

        if not os.path.exists(input_path):
            subprocess.call("powershell mkdir " + input_path)

        input_path1 = 'test_cimb\\saved_model\\saved_model\\raw_dataset\\'
        output_path = 'test_cimb\\saved_model\\saved_model\\predicted_coordinates.csv'

        images = os.listdir(input_path)
        images = [os.listdir(input_path)[0]]

        # for multiple images, use this code
        for image_name in images:
            image_np = load_image_into_numpy_array(input_path + image_name)
            output_dict = run_inference_for_single_image(model, image_np)
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                line_thickness=3)
            cv2.imwrite(str(image_name), image_np)
            obj = Post.objects.latest('id')
            with open(image_name, 'rb') as destination_file:
                obj.processed_image.save(image_name, File(destination_file), save=False)
            obj.save()
            
        obj = Post.objects.latest('id')
        context = {"processed_image": obj.processed_image}
        return context