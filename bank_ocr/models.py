from django.db import models

# Create your models here.
class Post (models.Model):
    title = models.CharField(max_length=200)
    image = models.FileField(upload_to='images/')
    trainingpdf = models.FileField(upload_to='trainingpdf/', blank = True, null = True)
    processed_image = models.FileField(upload_to='processing_images/', blank = True, null = True)
    description = models.CharField(max_length=400, blank = True, null = True)

    def __str__(self) -> str:
        return self.title