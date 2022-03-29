from django.db import models

# Create your models here.
class Post (models.Model):
    title = models.CharField(max_length=200)
    image = models.ImageField(upload_to='images/')
    processed_image = models.ImageField(upload_to='processed_images/', blank = True, null = True)
    description = models.CharField(max_length=400, blank = True, null = True)

    def __str__(self) -> str:
        return self.title