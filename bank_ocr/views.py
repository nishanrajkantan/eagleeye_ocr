from django.views.generic import ListView, CreateView, TemplateView
from django.urls import reverse_lazy
from .forms import PostForm
from .models import Post
from django.shortcuts import render, redirect

# Create your views here.
class HomePageView(ListView):
    # Post.objects.all().delete()
    model = Post
    template_name = 'home.html'

# class CreatePostView(CreateView):
#     model = Post
#     form_class = PostForm
#     template_name = 'post.html'
#     success_url = reverse_lazy('home')

#     def form_valid(self, form):
#         obj = form.save(commit=False)
#         if self.request.FILES:
#             for f in self.request.FILES.getlist('image'):
#                 # obj = self.model.objects.create(image=f)
#                 obj = Post(image=f)
#                 obj.save()

#         return super(CreatePostView, self).form_valid(form)

def upload_pdf(request):
    if request.method == "POST":
        form = PostForm(request.POST, request.FILES)
        files = request.FILES.getlist('image')
        if form.is_valid():
            for f in files:
                file_instance = Post(image=f)
                file_instance.save()
        # # pec.main()
        return redirect('result')
    else:
        form = PostForm()
    return render(request, 'post.html', {'form': form})

class TrainingView(TemplateView):
    template_name = "training.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        obj = Post.objects.latest('id')

        # --------------------------------------------------------------------------------------------------------------
        # Script 1. preprocess_PDFs.py
        # --------------------------------------------------------------------------------------------------------------

        print('--------------------------------------------')
        print('Pre-processing PDF(s)')
        print('--------------------------------------------')

        import os
        from tqdm import tqdm
        from PIL import Image
        from pdf2image import convert_from_path

        # ---------------------------------------------------------------------------

        os.chdir('./')
        input_path = os.path.join(os.getcwd(), 'media\\train\\dataset\\raw_dataset\\')
        output_path = os.path.join(os.getcwd(), 'media\\train\\dataset\\jpg_dataset\\')

        # ---------------------------------------------------------------------------

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # ---------------------------------------------------------------------------

        for i, image_path in tqdm(enumerate(sorted(os.listdir(input_path)))):
            if image_path.endswith(".png"):
                image = Image.open(input_path + image_path).convert('RGB')
                image = image.resize((600, 600))
                image.save(output_path + 'jpg_image_' + str(i).zfill(3) + '.jpg')
            elif image_path.endswith(".jpeg"):
                image = Image.open(input_path + image_path).convert('RGB')
                image = image.resize((600, 600))
                image.save(output_path + 'jpg_image_' + str(i).zfill(3) + '.jpg')
            elif image_path.endswith(".jpg"):
                image = Image.open(input_path + image_path).convert('RGB')
                image = image.resize((600, 600))
                image.save(output_path + 'jpg_image_' + str(i).zfill(3) + '.jpg')
            elif image_path.endswith(".tif"):
                image = Image.open(input_path + image_path).convert('RGB')
                image = image.resize((600, 600))
                image.save(output_path + 'jpg_image_' + str(i).zfill(3) + '.jpg')
            elif image_path.endswith(".pdf"):
                images = convert_from_path(input_path + image_path, size=(600, 600))
                for i, image in enumerate(images):
                    image.save(output_path + image_path[:-4] + '_page_' + str(i) + '.jpg', 'JPEG')

        # --------------------------------------------------------------------------------------------------------------
        # Script 2. label_images.py
        # --------------------------------------------------------------------------------------------------------------

        print('--------------------------------------------')
        print('Labelling Image ROI')
        print('--------------------------------------------')

        import subprocess

        # ---------------------------------------------------------------------------

        input_path = os.path.join(os.getcwd(), 'media\\train\\dataset\\jpg_dataset\\')
        subprocess.call("labelimg " + input_path)

        # --------------------------------------------------------------------------------------------------------------
        # Script 3. augment_images.py
        # --------------------------------------------------------------------------------------------------------------

        print('--------------------------------------------')
        print('Augmenting Images')
        print('--------------------------------------------')

        import os
        import lxml.etree
        import glob
        from tqdm import tqdm
        from PIL import Image, ImageEnhance

        # ---------------------------------------------------------------------------

        input_path = os.path.join(os.getcwd(), 'media\\train\\dataset\\jpg_dataset\\')
        output_path1 = os.path.join(os.getcwd(), 'media\\train\\dataset\\augmented_dataset\\')
        output_path2 = os.path.join(os.getcwd(), 'media\\train\\dataset\\images\\')

        # ---------------------------------------------------------------------------

        if not os.path.exists(output_path1):
            os.mkdir(output_path1)

        # ---------------------------------------------------------------------------

        xml_list = glob.glob1(input_path, "*.xml")

        k = 0

        for i in tqdm(range(len(xml_list))):
            img = Image.open(input_path + xml_list[i][:-4] + '.jpg')
            width, height = img.size
            img.save(output_path1 + 'image_' + str(i).zfill(3) + ".jpg", 'JPEG')
            tree = lxml.etree.parse(input_path + xml_list[i])
            root = tree.getroot()
            for member in root.findall('object'):
                root.find('filename').text = 'image_' + str(i).zfill(3) + ".jpg"
            tree.write(output_path1 + 'image_' + str(i).zfill(3) + ".xml")
            l = 0.3

            # augment by image brightness
            for j in range(2):
                factor = 0.5 + l
                enhancer = ImageEnhance.Brightness(img)
                im_output = enhancer.enhance(factor)
                im_output.save(output_path1 + 'image_' + str(i).zfill(3) + '_brightness_' + str(k).zfill(3) + ".jpg",
                               'JPEG')
                for member in root.findall('object'):
                    root.find('filename').text = 'image_' + str(i).zfill(3) + '_brightness_' + str(k).zfill(3) + ".jpg"
                tree.write(output_path1 + 'image_' + str(i).zfill(3) + '_brightness_' + str(k).zfill(3) + ".xml")
                k += 1
                l += 0.3

            # augment by image contrast
            l = 0.3
            for j in range(2):
                factor = 0.5 + l
                enhancer = ImageEnhance.Contrast(img)
                im_output = enhancer.enhance(factor)
                im_output.save(output_path1 + 'image_' + str(i).zfill(3) + '_contrast_' + str(k).zfill(3) + ".jpg",
                               'JPEG')
                for member in root.findall('object'):
                    root.find('filename').text = 'image_' + str(i).zfill(3) + '_contrast_' + str(k).zfill(3) + ".jpg"
                tree.write(output_path1 + 'image_' + str(i).zfill(3) + '_contrast_' + str(k).zfill(3) + ".xml")
                k += 1
                l += 0.3

            # augment by image sharpness
            l = 0
            for j in range(2):
                factor = 0.05 + l
                enhancer = ImageEnhance.Sharpness(img)
                im_output = enhancer.enhance(factor)
                im_output.save(output_path1 + 'image_' + str(i).zfill(3) + '_sharpness_' + str(k).zfill(3) + ".jpg",
                               'JPEG')
                for member in root.findall('object'):
                    root.find('filename').text = 'image_' + str(i).zfill(3) + '_sharpness_' + str(k).zfill(3) + ".jpg"
                tree.write(output_path1 + 'image_' + str(i).zfill(3) + '_sharpness_' + str(k).zfill(3) + ".xml")
                k += 1
                l += 0.5

            # augment by image colour
            l = 0
            for j in range(2):
                factor = 0.05 + l
                enhancer = ImageEnhance.Color(img)
                im_output = enhancer.enhance(factor)
                im_output.save(output_path1 + 'image_' + str(i).zfill(3) + '_color_' + str(k).zfill(3) + ".jpg", 'JPEG')
                for member in root.findall('object'):
                    root.find('filename').text = 'image_' + str(i).zfill(3) + '_color_' + str(k).zfill(3) + ".jpg"
                tree.write(output_path1 + 'image_' + str(i).zfill(3) + '_color_' + str(k).zfill(3) + ".xml")
                k += 1
                l += 0.5

        total_image = sorted(glob.glob1(output_path1, "*.jpg"))
        total_xml = sorted(glob.glob1(output_path1, "*.xml"))
        print("\nThere are " + str(len(total_image)) + " images after augmentation")

        # ---------------------------------------------------------------------------

        if not os.path.exists(output_path2):
            os.mkdir(output_path2)

        # ---------------------------------------------------------------------------

        for i in tqdm(range(len(total_image))):
            img = Image.open(output_path1 + total_image[i][:-4] + '.jpg')
            img.save(output_path2 + total_image[i], 'JPEG')

        # --------------------------------------------------------------------------------------------------------------
        # Script 4. create_csv_file.py
        # --------------------------------------------------------------------------------------------------------------

        print('--------------------------------------------')
        print('Combining XML files')
        print('--------------------------------------------')

        def xml_to_csv(input_path, output_path):

            import glob
            import pandas as pd
            import xml.etree.ElementTree as ET

            xml_list = []
            for xml_file in glob.glob(input_path + '*.xml'):
                tree = ET.parse(xml_file)
                root = tree.getroot()
                for member in root.findall('object'):
                    value = (root.find('filename').text,
                             int(root.find('size')[0].text),
                             int(root.find('size')[1].text),
                             member[0].text,
                             int(member[4][0].text),
                             int(member[4][1].text),
                             int(member[4][2].text),
                             int(member[4][3].text)
                             )
                    xml_list.append(value)
            column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
            xml_df = pd.DataFrame(xml_list, columns=column_name)
            xml_df.to_csv(output_path + 'labels.csv', index=None)
            return xml_df

        input_path = os.path.join(os.getcwd(), 'media\\train\\dataset\\augmented_dataset\\')
        output_path = os.path.join(os.getcwd(), 'media\\train\\dataset\\')
        xml_to_csv(input_path, output_path)

        # --------------------------------------------------------------------------------------------------------------
        # Script 5. split_dataset.py
        # --------------------------------------------------------------------------------------------------------------

        print('--------------------------------------------')
        print('Splitting Dataset')
        print('--------------------------------------------')

        import os
        import numpy as np
        import shutil
        import glob

        # # Creating Train / Val / Test folders (One time use)
        dataset_dir = os.path.join(os.getcwd(), 'media\\train\\dataset\\')
        image_dir = 'images/'

        val_ratio = 0.15
        test_ratio = 0.05

        # Creating directories for train, val, & test
        myFileList = glob.glob1(dataset_dir + image_dir, "*.jpg")
        print("\nThere are", len(myFileList), "images read by Python")
        np.random.shuffle(myFileList)

        if not os.path.exists(dataset_dir + image_dir + 'train'):
            os.makedirs(dataset_dir + image_dir + 'train')

        if not os.path.exists(dataset_dir + image_dir + 'val'):
            os.makedirs(dataset_dir + image_dir + 'val')

        if not os.path.exists(dataset_dir + image_dir + 'test'):
            os.makedirs(dataset_dir + image_dir + 'test')

        # Creating partitions of the data after shuffeling
        np.random.shuffle(myFileList)
        train_FileNames, val_FileNames, test_FileNames = np.split(np.array(myFileList),
                                                                  [int(len(myFileList) * (1 - val_ratio + test_ratio)),
                                                                   int(len(myFileList) * (1 - test_ratio))])
        train_FileNames = [dataset_dir + image_dir + name for name in train_FileNames.tolist()]
        val_FileNames = [dataset_dir + image_dir + name for name in val_FileNames.tolist()]
        test_FileNames = [dataset_dir + image_dir + name for name in test_FileNames.tolist()]
        print('Total images: ', len(myFileList))
        print('Training: ', len(train_FileNames))
        print('Validation: ', len(val_FileNames))
        print('Testing: ', len(test_FileNames))

        # Copy-pasting images
        for name in train_FileNames:
            shutil.copy(name, dataset_dir + image_dir + '/train')
        for name in val_FileNames:
            shutil.copy(name, dataset_dir + image_dir + '/val')
        for name in test_FileNames:
            shutil.copy(name, dataset_dir + image_dir + '/test')

        # Remove images not in their specific directories
        for file in os.listdir(dataset_dir + image_dir):
            if file.endswith('.jpg'):
                os.remove(dataset_dir + image_dir + file)

        # Generate csv files for train, val, & test
        import pandas as pd

        labels_df = pd.read_csv(dataset_dir + 'labels.csv')
        train_images_path = os.listdir(dataset_dir + image_dir + '/train')
        val_images_path = os.listdir(dataset_dir + image_dir + '/val')
        test_images_path = os.listdir(dataset_dir + image_dir + '/test')

        # Split invoice_labels_train.csv into train_labels, val_labels, & test_labels
        from tqdm import tqdm

        train_list = []
        val_list = []
        test_list = []

        for i in tqdm(range(len(labels_df))):
            for j in range(len(train_images_path)):
                if labels_df['filename'][i] == train_images_path[j]:
                    train_list.append(labels_df.iloc[i][:])
            for j in range(len(val_images_path)):
                if labels_df['filename'][i] == val_images_path[j]:
                    val_list.append(labels_df.iloc[i][:])
            for j in range(len(test_images_path)):
                if labels_df['filename'][i] == test_images_path[j]:
                    test_list.append(labels_df.iloc[i][:])

        # Convert lists to dataframes
        train_df = pd.DataFrame(train_list)
        val_df = pd.DataFrame(val_list)
        test_df = pd.DataFrame(test_list)

        # Save dataframes as csv files
        train_df.to_csv(dataset_dir + 'train_labels.csv', index=False)
        val_df.to_csv(dataset_dir + 'val_labels.csv', index=False)
        test_df.to_csv(dataset_dir + 'test_labels.csv', index=False)

        # --------------------------------------------------------------------------------------------------------------
        # Script 6. create_tf_records.py
        # --------------------------------------------------------------------------------------------------------------

        print('--------------------------------------------')
        print('Preparing input for model training')
        print('--------------------------------------------')

        import subprocess
        from tqdm import tqdm
        import pandas as pd

        df = pd.read_csv(os.path.join(os.getcwd(), 'media\\train\\dataset\\labels.csv'))
        entity_names = df['class'].unique().tolist()

        with open(os.path.join(os.getcwd(), 'media\\train\\label_map.pbtxt'), 'w') as file:
            for i in tqdm(range(len(entity_names))):
                file.write('item {name: "' + entity_names[i] + '" id: ' + str(i + 1) + '}\n')

        tf_generate_record_script = os.path.join(os.getcwd(), 'media\\train\\dataset\\generate_tf_records.py')
        if not os.path.exists(tf_generate_record_script):
            subprocess.call(
                'powershell -Command "Invoke-WebRequest https://raw.githubusercontent.com/azzubair01/Bank_Statement_Digitization/main/train/dataset/generate_tf_records.py -OutFile media\\train\\dataset\\generate_tf_records.py')

        subprocess.call("python " + os.path.join(os.getcwd(),
                                                 'media\\train\\dataset\\generate_tf_records.py') + " -l " + os.path.join(
            os.getcwd(), 'media\\train\\label_map.pbtxt') + " -o " + os.path.join(os.getcwd(),
                                                                                  'media\\train\\dataset\\train.record') + " -i " + os.path.join(
            os.getcwd(), 'media\\train\\dataset\\images\\train') + " -csv " + os.path.join(os.getcwd(),
                                                                                           'media\\train\\dataset\\train_labels.csv'))
        subprocess.call("python " + os.path.join(os.getcwd(),
                                                 'media\\train\\dataset\\generate_tf_records.py') + " -l " + os.path.join(
            os.getcwd(), 'media\\train\\label_map.pbtxt') + " -o " + os.path.join(os.getcwd(),
                                                                                  'media\\train\\dataset\\val.record') + " -i " + os.path.join(
            os.getcwd(), 'media\\train\\dataset\\images\\val') + " -csv " + os.path.join(os.getcwd(),
                                                                                         'media\\train\\dataset\\val_labels.csv'))

        # --------------------------------------------------------------------------------------------------------------
        # Script 7. download_model.py
        # --------------------------------------------------------------------------------------------------------------

        print('--------------------------------------------')
        print('Downloading Tensorflow model')
        print('--------------------------------------------')

        import os
        import subprocess

        # ---------------------------------------------------------------------------

        model_name = os.path.join(os.getcwd(), 'media\\train\\frcnn_v1')
        if not os.path.exists(model_name):
            subprocess.call(
                'powershell -Command "Invoke-WebRequest http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz -OutFile media\\train\\frcnn_v1.tar.gz"')

        # ---------------------------------------------------------------------------

        zipped_model = os.path.join(os.getcwd(), 'media\\train\\frcnn_v1.tar.gz')
        unzipped_model = os.path.join(os.getcwd(), 'media\\train\\')
        if os.path.exists(zipped_model):
            subprocess.call("powershell tar -xvzf " + zipped_model + " -C " + unzipped_model)

        # ---------------------------------------------------------------------------

        model_name = os.path.join(os.getcwd(), 'media\\train\\faster_rcnn_resnet50_v1_640x640_coco17_tpu-8')
        renamed_model = os.path.join(os.getcwd(), 'media\\train\\frcnn_v1')
        if not os.path.exists(renamed_model):
            os.rename(model_name, renamed_model)

        # ---------------------------------------------------------------------------

        model_name = os.path.join(os.getcwd(), 'media\\train\\frcnn_v1.tar.gz')
        if os.path.exists(model_name):
            os.remove(model_name)

        # ---------------------------------------------------------------------------

        model_config = os.path.join(os.getcwd(), 'media\\train\\frcnn_v1.config')
        if not os.path.exists(model_config):
            subprocess.call(
                'powershell -Command "Invoke-WebRequest https://raw.githubusercontent.com/azzubair01/Bank_Statement_Digitization/main/train/frcnn_v1.config -OutFile media\\train\\frcnn_v1.config"')

        model_tf = os.path.join(os.getcwd(), 'media\\train\\models')
        if not os.path.exists(model_tf):
            os.chdir('.\\media\\train')
            subprocess.call('git clone https://github.com/tensorflow/models.git')

        # --------------------------------------------------------------------------------------------------------------
        # Script 8. configure_settings.py
        # --------------------------------------------------------------------------------------------------------------

        print('--------------------------------------------')
        print('Preparing input for model training')
        print('--------------------------------------------')

        num_classes = len(entity_names)
        num_steps = 7500
        batch_size = 1
        checkpoint_type = 'detection'

        train_record_path = 'media/train/dataset/train.record'
        val_record_path = 'media/train/dataset/val.record'
        model_dir = 'media/train/training'
        labelmap_path = 'media/train/label_map.pbtxt'

        pipeline_config_path = 'media/train/frcnn_v1.config'
        fine_tune_checkpoint = 'media/train/frcnn_v1/checkpoint/ckpt-0'

        config = """# Faster R-CNN with Resnet-50 (v1) with 640x640 input resolution
        # Trained on COCO, initialized from Imagenet classification checkpoint
        #
        # Train on TPU-8
        #
        # Achieves 29.3 mAP on COCO17 Val

        model {
          faster_rcnn {
            num_classes: """ + str(num_classes) + """
            image_resizer {
              keep_aspect_ratio_resizer {
                min_dimension: 640
                max_dimension: 640
                pad_to_max_dimension: true
              }
            }
            feature_extractor {
              type: 'faster_rcnn_resnet50_keras'
              batch_norm_trainable: true
            }
            first_stage_anchor_generator {
              grid_anchor_generator {
                scales: [0.25, 0.5, 1.0, 2.0]
                aspect_ratios: [0.5, 1.0, 2.0]
                height_stride: 16
                width_stride: 16
              }
            }
            first_stage_box_predictor_conv_hyperparams {
              op: CONV
              regularizer {
                l2_regularizer {
                  weight: 0.0
                }
              }
              initializer {
                truncated_normal_initializer {
                  stddev: 0.01
                }
              }
            }
            first_stage_nms_score_threshold: 0.0
            first_stage_nms_iou_threshold: 0.7
            first_stage_max_proposals: 300
            first_stage_localization_loss_weight: 2.0
            first_stage_objectness_loss_weight: 1.0
            initial_crop_size: 14
            maxpool_kernel_size: 2
            maxpool_stride: 2
            second_stage_box_predictor {
              mask_rcnn_box_predictor {
                use_dropout: false
                dropout_keep_probability: 1.0
                fc_hyperparams {
                  op: FC
                  regularizer {
                    l2_regularizer {
                      weight: 0.0
                    }
                  }
                  initializer {
                    variance_scaling_initializer {
                      factor: 1.0
                      uniform: true
                      mode: FAN_AVG
                    }
                  }
                }
                share_box_across_classes: true
              }
            }
            second_stage_post_processing {
              batch_non_max_suppression {
                score_threshold: 0.0
                iou_threshold: 0.6
                max_detections_per_class: 100
                max_total_detections: 300
              }
              score_converter: SOFTMAX
            }
            second_stage_localization_loss_weight: 2.0
            second_stage_classification_loss_weight: 1.0
            use_static_shapes: true
            use_matmul_crop_and_resize: true
            clip_anchors_to_image: true
            use_static_balanced_label_sampler: true
            use_matmul_gather_in_matcher: true
          }
        }

        train_config: {
          batch_size: """ + str(batch_size) + """
          sync_replicas: true
          startup_delay_steps: 0
          replicas_to_aggregate: 8
          num_steps: """ + str(num_steps) + """
          optimizer {
            momentum_optimizer: {
              learning_rate: {
                cosine_decay_learning_rate {
                  learning_rate_base: .04
                  total_steps: 25000
                  warmup_learning_rate: .013333
                  warmup_steps: 2000
                }
              }
              momentum_optimizer_value: 0.9
            }
            use_moving_average: false
          }
          fine_tune_checkpoint_version: V2
          fine_tune_checkpoint: '""" + fine_tune_checkpoint + """'
          fine_tune_checkpoint_type: '""" + checkpoint_type + """'
          data_augmentation_options {
            random_horizontal_flip {
            }
          }

          max_number_of_boxes: 100
          unpad_groundtruth_tensors: false
          use_bfloat16: true  # works only on TPUs
        }

        train_input_reader: {
          label_map_path: '""" + str(labelmap_path) + """'
          tf_record_input_reader {
            input_path: '""" + str(train_record_path) + """'
          }
        }

        eval_config: {
          metrics_set: "coco_detection_metrics"
          use_moving_averages: false
          batch_size: 1;
        }

        eval_input_reader: {
          label_map_path: '""" + str(labelmap_path) + """'
          shuffle: false
          num_epochs: 1
          tf_record_input_reader {
            input_path: '""" + str(val_record_path) + """'
          }
        }
        """

        with open(pipeline_config_path, 'w') as f:
            f.write(config)

        # --------------------------------------------------------------------------------------------------------------
        # Script 9. train.py
        # --------------------------------------------------------------------------------------------------------------

        print('--------------------------------------------')
        print('Training Model')
        print('--------------------------------------------')

        import os

        num_steps = 200
        num_eval_steps = 200

        model_dir = os.path.join(os.getcwd(), 'media\\train\\training\\')
        pipeline_config_path = os.path.join(os.getcwd(), 'media\\train\\frcnn_v1.config')

        import subprocess
        import time
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        start_time = time.time()
        subprocess.call(
            "python media/train/models/research/object_detection/model_main_tf2.py --pipeline_config_path=" + f'{pipeline_config_path}' + \
            " --model_dir=" + f'{model_dir}' + \
            " --alsologtostderr --num_train_steps=" + f'{num_steps}' + \
            " --sample_1_of_n_eval_examples=1 --num_eval_steps=" + f'{num_eval_steps}')
        print("Model training requires %s hours " % ((time.time() - start_time) / 60 / 60))

        # --------------------------------------------------------------------------------------------------------------
        # Script 12. export.py
        # --------------------------------------------------------------------------------------------------------------

        print('--------------------------------------------')
        print('Exporting Model')
        print('--------------------------------------------')

        output_directory = os.path.join(os.getcwd(), 'media/train/maybank_model')
        model_dir = os.path.join(os.getcwd(), 'media/train/training/')
        pipeline_config_path = os.path.join(os.getcwd(), 'media/train/frcnn_v1.config')

        subprocess.call("python media/train/models/research/object_detection/exporter_main_v2.py \
            --trained_checkpoint_dir " + f'{model_dir}' + "\
            --output_directory " + f'{output_directory}' + " \
            --pipeline_config_path " + f'{pipeline_config_path}')

        if not os.path.exists(os.path.join(os.getcwd(), 'media/train/maybank_model')):
            os.mkdir(os.path.join(os.getcwd(), 'media/train/maybank_model'))

        if os.path.exists(os.path.join(os.getcwd(), 'media/train/maybank_model.zip')):
            subprocess.call(
                "powershell Compress-Archive -Update -LiteralPath media\\train\\maybank_model\\saved_model\\ -DestinationPath media\\train\\maybank_model.zip")

        else:
            subprocess.call(
                "powershell Compress-Archive -LiteralPath media\\train\\maybank_model\\saved_model\\ -DestinationPath media\\train\\maybank_model.zip")

        # --------------------------------------------------------------------------------------------------------------
        # End of Python scripts
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # End of Python scripts
        # --------------------------------------------------------------------------------------------------------------

        context = {"processed_image": "processed_image"}
        return context

class ResultView(TemplateView):
    template_name = "result.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        obj = Post.objects.latest('id')

        # ---------------------------------------------------------------------------------------------------------------------
        # Script 13. test.py
        # ---------------------------------------------------------------------------------------------------------------------

        print('--------------------------------------------')
        print('Detecting Image Regions')
        print('--------------------------------------------')

        import pikepdf
        from PyPDF2 import PdfFileReader
        from tqdm import tqdm
        from pdf2image import convert_from_path
        import pandas as pd
        import subprocess
        import os
        from bank_ocr.inferenceutils import label_map_util, load_image_into_numpy_array, run_inference_for_single_image
        import tensorflow as tf

        os.chdir('./')

        zipped_model = os.path.join(os.getcwd(), 'bank_ocr\\inferenceutils.py')
        if not os.path.exists(zipped_model):
            subprocess.call(
                'powershell -Command "Invoke-WebRequest https://raw.githubusercontent.com/hugozanini/object-detection/master/inferenceutils.py -OutFile bank_ocr\\inferenceutils.py"')

        # ---------------------------------------------------------------------------------------------------------------------
        # CIMB Model
        # ---------------------------------------------------------------------------------------------------------------------

        # zipped_model = os.path.join(os.getcwd(), 'bank_ocr\\cimb_model\\saved_model')
        # if not os.path.exists(zipped_model):
        #     subprocess.call(
        #         "powershell Expand-Archive -Path 'cimb_model\\saved_model.zip' -DestinationPath 'cimb_model\\saved_model\\'")

        # labelmap_path = os.path.join(os.getcwd(), 'bank_ocr\\cimb_model\\saved_model\\saved_model\\label_map.pbtxt')

        # category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
        # tf.keras.backend.clear_session()
        # model = tf.saved_model.load(os.path.join(os.getcwd(), 'bank_ocr\\cimb_model\\saved_model\\saved_model'))


        # process_path = os.path.join(os.getcwd(), 'media\\processing_images\\cimb\\')
        # input_path1 = os.path.join(os.getcwd(), 'media\\raw_dataset\\cimb\\')
        # output_path = os.path.join(os.getcwd(), 'media\\predicted_coordinates_cimb.csv')

        # if not os.path.exists(process_path):
        #     subprocess.call("powershell mkdir " + process_path)

        # for i, pdf in tqdm(enumerate(os.listdir(input_path1))):
        #     file = pikepdf.open(input_path1 + pdf, allow_overwriting_input=True)
        #     file.save(input_path1 + pdf)

        # for i, pdf in tqdm(enumerate(os.listdir(input_path1))):
        #     input = PdfFileReader(open(input_path1 + pdf, 'rb'))
        #     width = input.getPage(0).mediaBox[2]
        #     height = input.getPage(0).mediaBox[3]
        #     images = convert_from_path(input_path1 + pdf, size=(width, height))
        #     for i, image in enumerate(images):
        #         image.save(process_path + pdf[:-4] + '_page_' + str(i) + '.jpg', 'JPEG')

        # # Getting images to test
        # images = os.listdir(process_path)

        # # for multiple images, use this code
        # rows = []
        # for image_name in images:

        #     image_np = load_image_into_numpy_array(process_path + image_name)
        #     output_dict = run_inference_for_single_image(model, image_np)

        #     # store boxes in dataframe!
        #     cut_off_scores = len(list(filter(lambda x: x >= 0.1, output_dict['detection_scores'])))

        #     for j in range(cut_off_scores):
        #         name = image_name
        #         scores = output_dict['detection_scores'][j]
        #         classes = output_dict['detection_classes'][j]
        #         for i in range(1, len(category_index) + 1):
        #             if output_dict['detection_classes'][j] == category_index[i]['id']:
        #                 classes = category_index[i]['name']
        #         ymin = output_dict['detection_boxes'][j][0]
        #         xmin = output_dict['detection_boxes'][j][1]
        #         ymax = output_dict['detection_boxes'][j][2]
        #         xmax = output_dict['detection_boxes'][j][3]

        #         row = list([name, scores, classes, ymin, xmin, ymax, xmax])
        #         rows.append(row)

        # final_df = pd.DataFrame(rows, columns=['Image', 'Scores', 'Classes', 'ymin', 'xmin', 'ymax', 'xmax'])
        # final_df.to_csv(output_path, index=False)

        # ---------------------------------------------------------------------------------------------------------------------
        # Maybank Model
        # ---------------------------------------------------------------------------------------------------------------------

        zipped_model = os.path.join(os.getcwd(),'bank_ocr\\maybank_model\\saved_model')
        if not os.path.exists(zipped_model):
            subprocess.call(
                "powershell Expand-Archive -Path 'test\\saved_model.zip' -DestinationPath 'maybank_model\\saved_model\\'")

        output_directory = 'inference_graph'
        labelmap_path = os.path.join(os.getcwd(), 'bank_ocr\\maybank_model\\saved_model\\saved_model\\label_map.pbtxt')

        category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
        tf.keras.backend.clear_session()
        model = tf.saved_model.load(os.path.join(os.getcwd(),'bank_ocr\\maybank_model\\saved_model\\saved_model'))

        process_path = os.path.join(os.getcwd(),'media\\processing_images\\mayb\\')
        input_path1 = os.path.join(os.getcwd(),'media\\images\\')
        output_path = os.path.join(os.getcwd(),'media\\predicted_coordinates_mayb.csv')

        if not os.path.exists(process_path):
            subprocess.call("powershell mkdir " + process_path)

        for i, pdf in enumerate(os.listdir(input_path1)):
            file = pikepdf.open(input_path1 + pdf, allow_overwriting_input=True)
            file.save(input_path1 + pdf)

        for i, pdf in tqdm(enumerate(os.listdir(input_path1))):
            input = PdfFileReader(open(input_path1 + pdf, 'rb'))
            width = input.getPage(0).mediaBox[2]
            height = input.getPage(0).mediaBox[3]
            images = convert_from_path(input_path1 + pdf, size=(width, height))
            for i, image in enumerate(images):
                image.save(process_path + pdf[:-4] + '_page_' + str(i) + '.jpg', 'JPEG')

        # Getting images to test
        images = os.listdir(process_path)

        # for multiple image, use this code
        rows = []
        for image_name in images:

            image_np = load_image_into_numpy_array(process_path + image_name)
            output_dict = run_inference_for_single_image(model, image_np)

            # store boxes in dataframe!
            cut_off_scores = len(list(filter(lambda x: x >= 0.1, output_dict['detection_scores'])))

            for j in range(cut_off_scores):
                name = image_name
                scores = output_dict['detection_scores'][j]
                classes = output_dict['detection_classes'][j]
                for i in range(1, len(category_index) + 1):
                    if output_dict['detection_classes'][j] == category_index[i]['id']:
                        classes = category_index[i]['name']
                ymin = output_dict['detection_boxes'][j][0]
                xmin = output_dict['detection_boxes'][j][1]
                ymax = output_dict['detection_boxes'][j][2]
                xmax = output_dict['detection_boxes'][j][3]

                row = list([name, scores, classes, ymin, xmin, ymax, xmax])
                rows.append(row)

        final_df = pd.DataFrame(rows, columns=['Image', 'Scores', 'Classes', 'ymin', 'xmin', 'ymax', 'xmax'])
        final_df.to_csv(output_path, index=False)

        # ---------------------------------------------------------------------------------------------------------------------
        # Script 14. pdf_extract_table.py
        # ---------------------------------------------------------------------------------------------------------------------

        print('--------------------------------------------')
        print('Extracting Transaction Tables')
        print('--------------------------------------------')

        import os
        import re
        import subprocess
        import camelot
        import pandas as pd
        import pikepdf
        from tqdm import tqdm

        # input_path_cimb = os.path.join(os.getcwd(),'media\\raw_dataset\\cimb\\')
        input_path_mayb = os.path.join(os.getcwd(),'media\\images\\')

        # cimb_files = sorted(os.listdir(input_path_cimb))
        mayb_files = sorted(os.listdir(input_path_mayb))

        # output_path_cimb = os.path.join(os.getcwd(),'media\\transaction_output\\cimb\\')
        output_path_mayb = os.path.join(os.getcwd(),'media\\transaction_output\\mayb\\')

        # if not os.path.exists(output_path_cimb):
        #     subprocess.call("powershell mkdir " + output_path_cimb)

        if not os.path.exists(output_path_mayb):
            subprocess.call("powershell mkdir " + output_path_mayb)

        # for i, pdf in tqdm(enumerate(cimb_files)):
        #     file = pikepdf.open(input_path_cimb + pdf, allow_overwriting_input=True)
        #     file.save(input_path_cimb + pdf)

        for i, pdf in tqdm(enumerate(mayb_files)):
            file = pikepdf.open(input_path_mayb + pdf, allow_overwriting_input=True)
            file.save(input_path_mayb + pdf)

        # for i, pdf in tqdm(enumerate(cimb_files)):
        #     table = camelot.read_pdf(input_path_cimb + pdf, pages='all', flavor='lattice')
        #     tables = pd.DataFrame()
        #     with pd.ExcelWriter(output_path_cimb + 'transaction_' + pdf[:-4] + '.xlsx', engine='xlsxwriter') as writer:
        #         tables = pd.DataFrame()
        #         for x in range(len(table)):
        #             if x == 0:
        #                 tables = tables.append(table[0].df)
        #             elif table[0].df[0][0] == table[x].df[0][0]:
        #                 tables = tables.append(table[x].df.iloc[1:])
        #         tables = tables.reset_index().drop(columns=['index'], axis=1)

        #         columns = tables[0][0].split("\n")
        #         columns = columns[:7]

        #         dates = []
        #         descriptions = []
        #         refs = []
        #         withdraws = []
        #         deposits = []
        #         taxes = []
        #         balances = []
        #         for i in range(2, len(tables)):

        #             # Get date & description
        #             # ----------------------------------------
        #             date = tables[0][i][:10]
        #             description = tables[0][i][12:]
        #             add_description = tables[1][i]
        #             ref = tables[2][i]
        #             # ----------------------------------------

        #             # Get withdrawal (debit)
        #             # ----------------------------------------
        #             withdraw = re.sub(',', '', tables[3][i])
        #             if withdraw == '':
        #                 withdraw = 0
        #             else:
        #                 withdraw = float(withdraw)
        #             # ----------------------------------------

        #             # Get deposit (credit)
        #             # ----------------------------------------
        #             deposit = re.sub(',', '', tables[4][i])
        #             if deposit == '':
        #                 deposit = 0
        #             else:
        #                 deposit = float(deposit)
        #             # ----------------------------------------

        #             # Get tax
        #             # ----------------------------------------
        #             tax = re.sub(',', '', tables[5][i])
        #             if tax == '':
        #                 tax = 0
        #             else:
        #                 tax = float(tax)
        #             # ----------------------------------------

        #             # Get account balance
        #             # ----------------------------------------
        #             balance = re.sub(',', '', tables[6][i])
        #             if balance == '':
        #                 balance = 0
        #             else:
        #                 balance = float(balance)
        #             # ----------------------------------------

        #             # Append all lists
        #             # ----------------------------------------
        #             dates.append(date)
        #             descriptions.append(description + '\n' + add_description)
        #             refs.append(ref)
        #             withdraws.append(withdraw)
        #             deposits.append(deposit)
        #             taxes.append(tax)
        #             balances.append(balance)
        #             # ----------------------------------------

        #         transactions_df = pd.DataFrame(
        #             list([dates, descriptions, refs, withdraws, deposits, taxes, balances])).T
        #         transactions_df.columns = columns
        #         transactions_df.to_excel(writer, sheet_name='transactions', index=False)

        for i, pdf in tqdm(enumerate(mayb_files)):

            table = camelot.read_pdf(input_path_mayb + pdf, pages='all', flavor='stream')
            tables = pd.DataFrame()
            with pd.ExcelWriter(output_path_mayb + 'transaction_' + pdf[:-4] + '.xlsx', engine='xlsxwriter') as writer:

                table = camelot.read_pdf(input_path_mayb + pdf, pages='all', flavor='lattice')

                tables = pd.DataFrame()
                for x in range(len(table)):
                    if x == 0:
                        tables = tables.append(table[0].df)
                    elif x != 0:
                        tables = tables.append(table[x].df.iloc[1:])
                tables = tables.reset_index().drop(columns=['index'], axis=1)
                tables1 = tables.iloc[1:].reset_index().drop(columns=['index'], axis=1)

                for i in range(len(tables.columns)):
                    multi_row_column = tables.iloc[:, i].to_list()
                    tables1[0][i] = '\n'.join(multi_row_column)

                tables1 = tables1.iloc[:1]
                tables1.columns = tables.iloc[0]

                columns = tables.iloc[0, :]

                dates = []
                descs = []
                trans = []
                balances = []

                for i in range(1, len(tables)):
                    value = tables.iloc[i, :]

                    date = value[0].split("\n")
                    description = value[1].split("\n")
                    transaction = value[2].split("\n")
                    balance = value[3].split("\n")

                    dates.extend(date)
                    descs.extend(description)
                    trans.extend(transaction)
                    balances.extend(balance)

                descriptions = []
                for i in range(len(descs)):
                    if re.search("   ", descs[i]) != None:
                        descriptions[-1] = descriptions[-1] + descs[i]
                    else:
                        descriptions.append(descs[i])
                descriptions = descriptions[1:]

                for i in range(len(descriptions)):
                    try:
                        if (descriptions[i][:22] == 'PAYMENT VIA MYDEBIT   ') and (
                                len(descriptions[i].split('   ')) == 3):
                            descriptions[i] = descriptions[i] + ('   ') + descriptions[i + 1]
                            del descriptions[i + 1]
                        elif (descriptions[i][:19] == 'FUND TRANSFER TO A/') and (
                                len(descriptions[i].split('   ')) == 3):
                            descriptions[i] = descriptions[i] + ('   ') + descriptions[i + 1]
                            del descriptions[i + 1]

                        elif (descriptions[i][:20] == 'FPX PAYMENT FR A/   ') and (
                                len(descriptions[i].split('   ')) == 3) or (len(descriptions[i].split('   ')) == 2):
                            descriptions[i] = descriptions[i] + ('   ') + descriptions[i + 1]
                            del descriptions[i + 1]

                        elif (descriptions[i] == 'ENDING BALANCE :'):
                            del descriptions[i:]
                    except:
                        pass

                trans = trans[:-3]
                balances = balances[1:]
                df = pd.DataFrame((dates, descriptions, trans, balances)).T
                df.columns = columns
                df = df.dropna()

                df['CREDIT'] = 0
                df['DEBIT'] = 0

                for i in range(len(df)):
                    if df[df.columns[2]][i].find('+') != -1:
                        df['CREDIT'][i] = re.sub(r'[,+-]+', r'', df[df.columns[2]][i])
                        df['DEBIT'][i] = 0
                        df[df.columns[3]][i] = re.sub(r',', r'', df[df.columns[3]][i])

                    elif df[df.columns[2]][i].find('-') != -1:
                        df['DEBIT'][i] = re.sub(r'[,+-]+', r'', df[df.columns[2]][i])
                        df['CREDIT'][i] = 0
                        df[df.columns[3]][i] = re.sub(r',', r'', df[df.columns[3]][i])

                df[df.columns[3]] = df[df.columns[3]].astype('float')
                df['DEBIT'] = df['DEBIT'].astype('float')
                df['CREDIT'] = df['CREDIT'].astype('float')

                df.to_excel(writer, sheet_name='transactions', index=False)

        # ---------------------------------------------------------------------------------------------------------------------
        # Script 15. extract_metadata.py
        # ---------------------------------------------------------------------------------------------------------------------

        print('--------------------------------------------')
        print('Extracting Financial Entities')
        print('--------------------------------------------')

        import os
        import subprocess
        import pytesseract
        import glob
        from PIL import Image
        import pandas as pd
        from tqdm import tqdm
        from pytesseract import Output

        # --------------------------------------------------------------------------------------------------------------
        # Prediction on CIMB transactions
        # --------------------------------------------------------------------------------------------------------------

        # coordinates_df = pd.read_csv(os.path.join(os.getcwd(),'media\\predicted_coordinates_cimb.csv'))
        # input_path = os.path.join(os.getcwd(),'media\\processing_images\\cimb\\')
        # process_path = os.path.join(os.getcwd(),'media\\cropped_images\\cimb\\')
        # output_path = os.path.join(os.getcwd(),'media\\financial_output\\cimb\\')
        # excel_file_path = os.path.join(os.getcwd(),'media\\transaction_output\\cimb\\')
        # excel_files = sorted(os.listdir(excel_file_path))

        # --------------------------------------------------------------------------------------------------------------

        # if not os.path.exists(process_path):
        #     subprocess.call("powershell mkdir " + process_path)

        # if not os.path.exists(output_path):
        #     subprocess.call("powershell mkdir " + output_path)

        # images = glob.glob1(input_path, "*.jpg")
        # tessdata_dir_config = r'-l "eng+msa" --tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata" --psm 6'
        # pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

        # --------------------------------------------------------------------------------------------------------------

        # rows = []
        # for i in tqdm(range(len(coordinates_df))):
        #     for j in range(len(images)):
        #         if coordinates_df['Image'][i] == images[j]:
        #             image = Image.open(input_path + images[j])
        #             results = pytesseract.image_to_data(image, config=tessdata_dir_config, output_type=Output.DICT)

        #             # (This is not mandatory)
        #             width, height = image.size

        #             # Setting the points for cropped image
        #             left = coordinates_df['xmin'][i] * width
        #             top = coordinates_df['ymin'][i] * height
        #             right = coordinates_df['xmax'][i] * width
        #             bottom = coordinates_df['ymax'][i] * height

        #             # Cropped image of above dimension
        #             # (It will not change orginal image)
        #             im1 = image.crop((left, top, right, bottom))
        #             im1 = im1.resize((im1.size[0] * 5, im1.size[1] * 5), resample=5)

        #             # # Shows the image in image viewer
        #             # im1.show()
        #             im1.save(process_path + images[j])
        #             text = pytesseract.image_to_string(im1, lang='eng+msa', config=tessdata_dir_config)

        #             row = list([images[j], coordinates_df['Classes'][i], text])
        #             rows.append(row)

        # raw_info = pd.DataFrame(rows, columns=['Image', 'Key', 'Value'])

        # # --------------------------------------------------------------------------------------------------------------

        # for i, excel in tqdm(enumerate(excel_files)):
        #     output_df = pd.read_excel(excel_file_path + excel)
        #     statement_name1 = excel[12:-5]
        #     with pd.ExcelWriter(output_path + 'output_' + excel[12:-5] + '.xlsx', engine='xlsxwriter') as writer:
        #         rows = []
        #         for j in range(len(raw_info)):
        #             statement_name2 = raw_info['Image'][j][:15]
        #             if (statement_name1 == statement_name2) and (raw_info['Key'][j] != 'transactions'):
        #                 row = list(raw_info.iloc[j][1:])
        #                 rows.append(row)
        #                 df = pd.DataFrame(rows, columns=['Key', 'Value'])
        #                 df = df.drop_duplicates()
        #         output_df.to_excel(writer, sheet_name='transactions', index=False)
        #         df.to_excel(writer, sheet_name='metadata', index=False)

        # --------------------------------------------------------------------------------------------------------------
        # Prediction on Maybank transactions
        # --------------------------------------------------------------------------------------------------------------

        coordinates_df2 = pd.read_csv(os.path.join(os.getcwd(),'media\\predicted_coordinates_mayb.csv'))
        input_path2 = os.path.join(os.getcwd(),'media\\processing_images\\mayb\\')
        process_path2 = os.path.join(os.getcwd(),'media\\cropped_images\\mayb\\')
        output_path2 = os.path.join(os.getcwd(),'media\\financial_output\\mayb\\')
        excel_file_path2 = os.path.join(os.getcwd(),'media\\transaction_output\\mayb\\')
        excel_files2 = sorted(os.listdir(excel_file_path2))

        # --------------------------------------------------------------------------------------------------------------

        if not os.path.exists(process_path2):
            subprocess.call("powershell mkdir " + process_path2)

        if not os.path.exists(output_path2):
            subprocess.call("powershell mkdir " + output_path2)

        images2 = glob.glob1(input_path2, "*.jpg")
        tessdata_dir_config = r'-l "eng+msa" --tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata" --psm 6'
        pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

        # --------------------------------------------------------------------------------------------------------------

        rows2 = []
        for i in tqdm(range(len(coordinates_df2))):
            for j in range(len(images2)):
                if coordinates_df2['Image'][i] == images2[j]:
                    image = Image.open(input_path2 + images2[j])
                    results = pytesseract.image_to_data(image, config=tessdata_dir_config, output_type=Output.DICT)

                    # (This is not mandatory)
                    width, height = image.size

                    # Setting the points for cropped image
                    left = coordinates_df2['xmin'][i] * width
                    top = coordinates_df2['ymin'][i] * height
                    right = coordinates_df2['xmax'][i] * width
                    bottom = coordinates_df2['ymax'][i] * height

                    # Cropped image of above dimension
                    # (It will not change orginal image)
                    im1 = image.crop((left, top, right, bottom))
                    im1 = im1.resize((im1.size[0] * 5, im1.size[1] * 5), resample=5)

                    # # Shows the image in image viewer
                    # im1.show()
                    im1.save(process_path2 + images2[j])
                    text = pytesseract.image_to_string(im1, lang='eng+msa', config=tessdata_dir_config)

                    row = list([images2[j], coordinates_df2['Classes'][i], text])
                    rows2.append(row)

        raw_info2 = pd.DataFrame(rows2, columns=['Image', 'Key', 'Value'])

        # --------------------------------------------------------------------------------------------------------------

        for i, excel in tqdm(enumerate(excel_files2)):
            output_df2 = pd.read_excel(excel_file_path2 + excel)
            statement_name1 = excel[12:-5]
            with pd.ExcelWriter(output_path2 + 'output_' + excel[12:-5] + '.xlsx', engine='xlsxwriter') as writer2:
                rows = []
                for j in range(len(raw_info2)):
                    statement_name2 = raw_info2['Image'][j][:18]
                    if (statement_name1 == statement_name2) and (raw_info2['Key'][j] != 'transactions'):
                        row = list(raw_info2.iloc[j][1:])
                        rows.append(row)
                        df2 = pd.DataFrame(rows, columns=['Key', 'Value'])
                        df2 = df2.drop_duplicates()
                output_df2.to_excel(writer2, sheet_name='transactions', index=False)
                df2.to_excel(writer2, sheet_name='metadata', index=False)

        # --------------------------------------------------------------------------------------------------------------
        # 16. generate_sender_receiver_column.py
        # --------------------------------------------------------------------------------------------------------------

        print('--------------------------------------------')
        print('Extracting Entities in Transaction Tables')
        print('--------------------------------------------')

        import pandas as pd
        import os
        import glob
        import re
        import subprocess
        from tqdm import tqdm

        input_path = os.path.join(os.getcwd(),'media\\financial_output\\mayb\\')
        file_list = glob.glob1(input_path, "*.xlsx")
        output_path = os.path.join(os.getcwd(),'media\\extract_description\\raw_files\\')

        if not os.path.exists(output_path):
            subprocess.call("powershell mkdir " + output_path)

        for j in tqdm(range(len(file_list))):

            df_transaction = pd.read_excel(input_path + file_list[j], sheet_name='transactions')
            df_metadata = pd.read_excel(input_path + file_list[j], sheet_name='metadata')
            df_metadata['Value'] = df_metadata['Value'].str.replace('\n_x000C_', '')

            senders = []
            receivers = []

            for i in tqdm(range(len(df_transaction))):
                if re.search('SVG GIRO CR',
                             df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    pattern = re.search('SVG GIRO CR',
                                        df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    description_text = df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]
                    entity = description_text[pattern.span()[1]:]

                elif re.search('SALE DEBIT',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    entity = 'ONLINE DEBIT'
                    # entity = ''

                elif re.search('DEBIT ADVICE',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    entity = 'BANK DEDUCTION'
                    # entity = ''

                elif re.search('CASH WITHDRAWAL',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    entity = 'ATM WITHDRAWAL'
                    # entity = ''

                elif re.search('IBK FUND TFR FR A/C',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    pattern1 = re.search('IBK FUND TFR FR A/C',
                                         df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    pattern2 = re.search('[*]', df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    description_text = df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]
                    entity = description_text[pattern1.span()[1]:pattern2.span()[0]]
                    # entity = ''

                elif re.search('CLEARING CHQ DEP',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    entity = 'SALARY INCOME'
                    # entity = ''

                elif re.search('TRANSFER FROM A/C',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    pattern1 = re.search('TRANSFER FROM A/C',
                                         df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    pattern2 = re.search('[*]', df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    description_text = df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]
                    entity = description_text[pattern1.span()[1]:pattern2.span()[0]]
                    # entity = ''

                elif re.search('PAYMENT VIA MYDEBIT',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    pattern1 = re.search('PAYMENT VIA MYDEBIT',
                                         df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    pattern2 = re.search('[*]', df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    description_text = df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]
                    entity = description_text[pattern1.span()[1]:]
                    # entity = ''

                elif re.search('FUND TRANSFER TO A/',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    pattern1 = re.search('FUND TRANSFER TO A/',
                                         df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    pattern2 = re.search('[*]', df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    description_text = df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]
                    entity = description_text[pattern1.span()[1]:]
                    # entity = ''

                elif re.search('FPX PAYMENT FR A/',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    # entity_list = df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i].split('   ')
                    # print(entity_list)
                    # entity = entity_list[2]
                    pattern1 = re.search('[*]', df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    pattern2 = re.search('[0-9]+', df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i][
                                                   pattern1.span()[1]:])
                    description_text = df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]
                    entity = description_text[pattern1.span()[1]:]
                    # entity = ''

                elif re.search('FUND TRANSFER TO',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    pattern1 = re.search('FUND TRANSFER TO',
                                         df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    pattern2 = re.search('[*]', df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    description_text = df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]
                    entity = description_text[pattern1.span()[1]:]
                    # entity = ''

                elif re.search('IBK FUND TFR TO A/C',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    pattern1 = re.search('IBK FUND TFR TO A/C',
                                         df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    pattern2 = re.search('[*]', df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    description_text = df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]
                    entity = description_text[pattern1.span()[1]:pattern2.span()[0]]
                    # entity = ''

                elif re.search('PYMT FROM A/C',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    pattern1 = re.search('PYMT FROM A/C',
                                         df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    pattern2 = re.search('[*]', df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    description_text = df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]
                    entity = description_text[pattern1.span()[1]:pattern2.span()[0]]
                    # entity = ''

                elif re.search('HIBAH PAID',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    pattern1 = re.search('HIBAH PAID',
                                         df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    entity = 'INSURANCE HIBAH'
                    # entity = ''

                elif re.search('CASH DEPOSIT',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    pattern1 = re.search('CASH DEPOSIT',
                                         df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    entity = 'ATM DEPOSIT'
                    # entity = ''

                else:
                    entity = ''

                if df_transaction['CREDIT'][i] != 0:
                    sender = str(entity)
                    receiver = df_metadata[df_metadata['Key'] == 'account_holder']['Value'].values[0]
                else:
                    sender = df_metadata[df_metadata['Key'] == 'account_holder']['Value'].values[0]
                    receiver = str(entity)

                senders.append(sender)
                receivers.append(receiver)

            df_new = pd.DataFrame({'SENDER': senders, 'RECEIVER': receivers})
            df_final = pd.concat([df_transaction, df_new], axis=1)
            df_final.to_excel(output_path + 'output2_' + file_list[j][7:], index=False)

        # --------------------------------------------------------------------------------------------------------------
        # 17. build_eagleyedb.py
        # --------------------------------------------------------------------------------------------------------------

        print('--------------------------------------------')
        print('Building Eagle Eye DB')
        print('--------------------------------------------')

        import pandas as pd
        import numpy as np
        import glob
        import os
        import subprocess
        from tqdm import tqdm

        # file_list = glob.glob1('C:\\Users\\DataMicron\\Desktop\\Bank_Statement_Reader\\extract_description\\raw_files\\', '*.xlsx')
        input_path = os.path.join(os.getcwd(),'media\\extract_description\\edited_files\\')
        file_list = glob.glob1(input_path, '*.xlsx')
        output_path = os.path.join(os.getcwd(),'media\\eagleyedb\\')

        if not os.path.exists(output_path):
            subprocess.call("powershell mkdir " + output_path)

        def flatten(list_of_list):
            return [item for sublist in list_of_list for item in sublist]

        def get_entity_type_id(datalake_df):
            entity_type_id = []
            for i in range(len(datalake_df.columns)):
                column_index_no = datalake_df.columns.get_loc(datalake_df.columns[i]) + 1
                entity_type_id.append(column_index_no)
            return entity_type_id

        def get_entity_type(datalake_df):
            datalake_df.columns = map(str.upper, datalake_df.columns)
            entity_type = datalake_df.columns.to_list()
            return entity_type

        def get_entity_desc(entity_type):
            entity_type_desc = entity_type.copy()
            return entity_type_desc

        def get_entity(datalake_df):
            entities = flatten(datalake_df.values.tolist())
            return entities

        def get_entity_id(entities):
            entity_ids = []
            for i in range(len(entities)):
                entity_id = i + 1
                entity_ids.append(entity_id)
            return entity_ids

        def get_factor(entity_id, entity_type):
            factor = len(entity_id) / len(entity_type)
            return int(factor)

        def get_factor2(entity_id):
            factor = len(entity_id)
            return int(factor)

        def get_entity_type2(entity_id, entity_type_id):
            entity_type = entity_type_id.copy()
            factor = get_factor(entity_id, entity_type)
            entity_type = entity_type_id.copy() * factor
            return entity_type

        def get_location(factor):
            locations = ['' for i in range(factor)]
            return locations

        def get_address(factor):
            addresses = ['' for i in range(factor)]
            return addresses

        def get_image(entity_type):
            images = []
            image = ''
            for i in range(len(entity_type)):
                if entity_type[i] == 1:
                    image = '/assets/images/money.png'
                elif entity_type[i] == 2:
                    image = '/assets/images/person.png'
                elif entity_type[i] == 3:
                    image = '/assets/images/person.png'
                else:
                    pass
                images.append(image)
            return images

        def get_doc_id(factor):
            doc_ids = [1 for i in range(factor)]
            return doc_ids

        def get_search_around(entity_type):
            searcharounds = []
            for i in range(len(entity_type)):
                if entity_type[i] == 1:
                    searcharound = ''
                elif entity_type[i] == 2:
                    searcharound = ''
                elif entity_type[i] == 3:
                    searcharound = ''
                searcharounds.append(searcharound)
            return searcharounds

        def get_search_around_url(factor):
            search_around_url = ['' for i in range(factor)]
            return search_around_url

        def get_event_name():
            event_name = ['send', '']
            return event_name

        def get_event_desc(event_name):
            event_desc = event_name.copy()
            return event_desc

        def get_event_id(event_name):
            event_ids = []
            for i in range(len(event_name)):
                event_id = i + 1
                event_ids.append(event_id)
            return event_ids

        def random_dates(start, end, n=1):
            start_u = start.value // 10 ** 9
            end_u = end.value // 10 ** 9
            return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s')

        def get_doc_id2(entity_id):
            docids = [1 for i in range(len(entity_id))]
            return docids

        def get_entity_id2(entity_id):
            entityids = entity_id.copy()
            return entityids

        def get_start_idx(entity_id):
            start_idx = [0 for i in range(len(entity_id))]
            return start_idx

        def get_end_idx(entity_id):
            end_idx = [5 for i in range(len(entity_id))]
            return end_idx

        def get_date(dates_df):
            dates_list = []
            for i in range(len(dates_df)):
                date = dates_df['TARIKH MASUK'][i]
                for j in range(3):
                    dates_list.append(date)
            return dates_list

        def get_date2(dates_df):
            dates_list = []
            for i in range(len(dates_df)):
                date = dates_df['TARIKH MASUK'][i]
                for j in range(3 - 1):
                    dates_list.append(date)
            return dates_list

        def get_entity_unique_id(docid, entity_id, start_idx, end_idx):
            df = pd.DataFrame([docid, entity_id, start_idx, end_idx]).T
            df.columns = ['docid', 'entityid', 'startidx', 'endidx']
            df['entityuniqueid'] = ''
            for i in range(len(df)):
                df['entityuniqueid'][i] = int(
                    str(df['docid'][i]) + str(df['entityid'][i]) + str(df['startidx'][i] + df['endidx'][i]))
            entityuniqueid = df['entityuniqueid'].to_list()
            return entityuniqueid

        def get_relations(entity_df, entity_mentions_df):
            merged_df = entity_df.merge(entity_mentions_df, left_on='EntityId', right_on='EntityId')

            # -----------------------------------------------------------------------------------

            # Replace entityuniqueid of similar entities
            filtered_df = merged_df[~merged_df['EntityType'].isin([6, 9, 10, 11, 12])].dropna(subset=['EntityName'])
            duplicated_df2 = filtered_df[filtered_df.duplicated(subset=['EntityName'])]
            sorted_df2 = duplicated_df2.sort_values(ascending=True, by='EntityName').reset_index(drop=True)
            for j in range(1, len(sorted_df2)):
                if sorted_df2['EntityName'][j] == sorted_df2['EntityName'][j - 1]:
                    sorted_df2['EntityUniqueId'][j] = sorted_df2['EntityUniqueId'][j - 1]
                else:
                    sorted_df2['EntityUniqueId'][j] = sorted_df2['EntityUniqueId'][j]
            for i in range(len(sorted_df2)):
                for j in range(len(merged_df)):
                    if sorted_df2['EntityName'][i] == merged_df['EntityName'][j]:
                        merged_df['EntityUniqueId'][j] = sorted_df2['EntityUniqueId'][i]

            # Split entity by entity id
            entity_type1_df = merged_df[merged_df['EntityType'] == 1].reset_index(drop=True)
            entity_type2_df = merged_df[merged_df['EntityType'] == 2].reset_index(drop=True)
            entity_type3_df = merged_df[merged_df['EntityType'] == 3].reset_index(drop=True)

            # -----------------------------------------------------------------------------------

            # Map Relations based on their entity type
            entity1ids = []
            entity2ids = []
            entity1dtypes = []
            entity2dtypes = []

            entity1d = entity_type2_df['EntityUniqueId'].tolist()
            entity1dtype = entity_type2_df['EntityType'].tolist()
            entity1ids.extend(entity1d)
            entity1dtypes.extend(entity1dtype)
            entity2d = entity_type1_df['EntityUniqueId'].tolist()
            entity2dtype = entity_type1_df['EntityType'].tolist()
            entity2ids.extend(entity2d)
            entity2dtypes.extend(entity2dtype)

            entity1d = entity_type1_df['EntityUniqueId'].tolist()
            entity1dtype = entity_type1_df['EntityType'].tolist()
            entity1ids.extend(entity1d)
            entity1dtypes.extend(entity1dtype)
            entity2d = entity_type3_df['EntityUniqueId'].tolist()
            entity2dtype = entity_type3_df['EntityType'].tolist()
            entity2ids.extend(entity2d)
            entity2dtypes.extend(entity2dtype)

            # -----------------------------------------------------------------------------------

            return entity1ids, entity2ids, entity1dtypes, entity2dtypes, merged_df

        def get_columns():
            columns = ['DocId', 'Entity1Id', 'Entity2Id', 'Entity1Type', 'Entity2Type', 'Date']
            return columns

        def get_columns2():
            columns = ['DocId', 'EventId', 'Entity1Id', 'Entity2Id', 'Date']
            return columns

        def get_event_id2(event_mentions_df):
            event_ids = []
            event_id = ''
            for i in range(len(event_mentions_df)):
                if event_mentions_df['Entity2Type'][i] == 2:
                    event_id = 1
                elif event_mentions_df['Entity2Type'][i] == 3:
                    event_id = 1
                else:
                    event_id = 1
                event_ids.append(event_id)
            return event_ids

        def clean_na(event_mentions_df2, merged_df):
            column_name = event_mentions_df2.columns.to_list()
            working_df = event_mentions_df2.merge(merged_df, left_on='Entity2Id', right_on='EntityUniqueId',
                                                  suffixes=('', '_x'))
            working_df = working_df.dropna(subset=['EntityName'])
            working_df = working_df[working_df['EntityName'] != ''].reset_index(drop=True)
            working_df2 = working_df[column_name]
            return working_df2, working_df

        def get_relations3(connected_entity_df, event_mentions_df2):
            column_name = event_mentions_df2.columns.to_list()
            connected_entity_df = connected_entity_df[connected_entity_df.duplicated(subset='EntityName', keep=False)]
            connected_entity_df['Entitiy1Id_copy'] = connected_entity_df['Entity1Id']
            connected_entity_df['Entitiy2Id_copy'] = connected_entity_df['Entity2Id']
            connected_entity_df['Entity1Id'] = connected_entity_df['Entitiy2Id_copy']
            connected_entity_df['Entity2Id'] = connected_entity_df['Entitiy1Id_copy']
            connected_entity_df['EventId'] = 2
            connected_entity_df = connected_entity_df[column_name]
            event_mentions_df2 = event_mentions_df2.append(connected_entity_df)
            event_mentions_df2 = event_mentions_df2.drop_duplicates(subset=['Entity1Id', 'Entity2Id']).reset_index(
                drop=True)

            return event_mentions_df2

        # Append all excel files into a single table
        datalake_df = pd.DataFrame()
        dates_df = pd.DataFrame()
        for i in tqdm(range(len(file_list))):
            # Read the Input database
            transaction_df = pd.read_excel(input_path + file_list[i])[['JUMLAH URUSNIAGA', 'SENDER', 'RECEIVER']]
            datalake_date = pd.read_excel(input_path + file_list[i])[['TARIKH MASUK']]
            datalake_df = datalake_df.append(transaction_df, ignore_index=True)
            dates_df = dates_df.append(datalake_date, ignore_index=True)

        # Clean column 'JUMLAH URUSNIAGA'
        datalake_df['JUMLAH URUSNIAGA'] = datalake_df['JUMLAH URUSNIAGA'].str.replace('+', '')
        datalake_df['JUMLAH URUSNIAGA'] = datalake_df['JUMLAH URUSNIAGA'].str.replace('-', '')
        datalake_df['JUMLAH URUSNIAGA'] = datalake_df['JUMLAH URUSNIAGA'].str.replace(',', '')
        datalake_df['JUMLAH URUSNIAGA'] = datalake_df['JUMLAH URUSNIAGA'].astype('float')
        datalake_df['JUMLAH URUSNIAGA'] = datalake_df['JUMLAH URUSNIAGA'].apply(lambda x: f"RM {x}")

        # Transpose Documents from Datalake into EagleyeDB
        column_name = ['DocId', 'DocName', 'DocLocation', 'Date', 'Text']
        document_df = pd.DataFrame(
            {column_name[0]: [1], column_name[1]: ['AUDIT DATABASES'], column_name[2]: ['AUDIT SYSTEM/DATABASES'],
             column_name[3]: ['1/1/2022  12:00:00 AM'], column_name[4]: [' ']})
        document_df['Date'] = document_df['Date'].astype('datetime64')

        # Transpose Entity Type from Datalake into EagleyeDB
        column_name = ['TypeId', 'TypeName', 'TypeDescription']
        entity_type = get_entity_type(datalake_df)
        entity_type_desc = get_entity_desc(entity_type)
        entity_type_id = get_entity_type_id(datalake_df)
        entity_type_df2 = pd.DataFrame([entity_type_id, entity_type, entity_type_desc]).T
        entity_type_df2.columns = column_name
        # print(entity_type_df)

        # Transpose EntityAttribute from Datalake into Eagle Eye DB
        column_name = ['EntityId', 'AttributeName', 'AttributeValue', 'DocId']
        entity_attributes_df = pd.DataFrame(
            {column_name[0]: [''], column_name[1]: [''], column_name[2]: [''], column_name[3]: ['']})

        # Transpose Entities from Datalake into Eagle Eye DB
        column_name = ['EntityId', 'EntityName', 'EntityType', 'Location', 'Address', 'Image', 'DocId']
        entities = get_entity(datalake_df)
        entity_id = get_entity_id(entities)
        factor = get_factor2(entity_id)
        entity_type2 = get_entity_type2(entity_id, entity_type_id)
        location = get_location(factor)
        address = get_address(factor)
        image = get_image(entity_type2)
        doc_id = get_doc_id(factor)
        searcharound = get_search_around(entity_type2)
        searcharoundurl = get_search_around_url(factor)
        entity_df2 = pd.DataFrame([entity_id, entities, entity_type2, location, address, image, doc_id]).T
        entity_df2.columns = column_name
        entity_df2['EntityName'] = entity_df2['EntityName'].astype('string')
        entity_df2 = entity_df2.fillna('')
        entity_df2['EntityName'] = entity_df2['EntityName'].str.replace(' 00:00:00', '')
        # print(entity_df)

        # Transpose Entity Mentions from Datalake into Eagle Eye DB
        column_name = ['DocId', 'EntityId', 'StartIndex', 'EndIndex', 'Date', 'EntityUniqueId']
        doc_id = get_doc_id2(entity_id)
        entity_ids = get_entity_id2(entity_id)
        start_idx = get_start_idx(entity_id)
        end_idx = get_end_idx(entity_id)
        date = get_date(dates_df)
        entity_unique_id = get_entity_unique_id(doc_id, entity_ids, start_idx, end_idx)
        entity_mentions_df2 = pd.DataFrame([doc_id, entity_ids, start_idx, end_idx, date, entity_unique_id]).T
        entity_mentions_df2.columns = column_name
        entity_mentions_df2['Date'] = entity_mentions_df2['Date'].astype('datetime64')
        # print(entity_mentions_df)

        # Transpose Event Types from Datalake into Eagle Eye DB
        column_name = ['EventId', 'EventName', 'EventDescription']
        event_name = get_event_name()
        event_desc = get_event_desc(event_name)
        event_id = get_event_id(event_name)
        event_types_df2 = pd.DataFrame([event_id, event_name, event_desc]).T
        event_types_df2.columns = column_name
        # print(event_types_df2)

        # Transpose Event Mentions from Datalake into Eagle Eye DB
        entity1id, entity2id, entity_type1d, entity_type2d, merged_df = get_relations(entity_df2, entity_mentions_df2)
        doc_id = get_doc_id2(entity1id)
        date = get_date2(dates_df)
        event_mentions_df2 = pd.DataFrame([doc_id, entity1id, entity2id, entity_type1d, entity_type2d, date]).T
        column_name = get_columns()
        event_mentions_df2.columns = column_name
        event_mentions_df2['Date'] = event_mentions_df2['Date'].astype('datetime64')
        event_id = get_event_id2(event_mentions_df2)
        event_mentions_df2['EventId'] = event_id
        column_name = get_columns2()
        event_mentions_df2 = event_mentions_df2[column_name]
        # event_mentions_df2

        # Auto-connect among entities with attributes
        # event_mentions_df2, connected_entity_df = clean_na(event_mentions_df2, merged_df)
        # event_mentions_df2 = get_relations3(connected_entity_df, event_mentions_df2)
        # print(event_mentions_df2)

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(output_path + 'EagleyeSampleDb.xlsx', engine='openpyxl')

        # Write each dataframe to a different worksheet.
        document_df.to_excel(writer, sheet_name='Documents', index=False)
        entity_type_df2.to_excel(writer, sheet_name='EntityTypes', index=False)
        entity_df2.to_excel(writer, sheet_name='Entities', index=False)
        entity_attributes_df.to_excel(writer, sheet_name='EntityAttributes', index=False)
        entity_mentions_df2.to_excel(writer, sheet_name='EntityMentions', index=False)
        event_types_df2.to_excel(writer, sheet_name='EventTypes', index=False)
        event_mentions_df2.to_excel(writer, sheet_name='EventMentions', index=False)

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()


        print('--------------------------------------------')
        print('DONE! Find your excel output in ' + output_path)
        print('--------------------------------------------')

        # --------------------------------------------------------------------------------------------------------------
        # End of Python scripts
        # --------------------------------------------------------------------------------------------------------------

        # obj = Post.objects.latest('id')
        # with open('file-pdf-solid-240.png', 'rb') as destination_file:
        #     obj.processed_image.save('file-pdf-solid-240.png', File(destination_file), save=False)
        # obj.save()
        #
        # obj = Post.objects.latest('id')
        context = {"processed_image": "processed_image"}
        return context