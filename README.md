# dog-breeds-multiclass-image-classification-with-vit
Dog Breeds MultiClass Image Classification with ViT

Notebook first posted in my [Kaggle](https://www.kaggle.com/wesleyacheng/dog-breeds-multiclass-image-classification-w-vit).

# Model Motivation
Recently, someone asked me if you can classify dog images into their respective dog breeds instead just differentiating from cats vs dogs like my [last notebook](https://www.kaggle.com/code/wesleyacheng/cat-vs-dog-image-classification-with-cnns). I say **YES**!

Due to the complexity of the problem, we will be using the most advanced computer vision architecture released in the [2020 Google paper](https://arxiv.org/pdf/2010.11929v2.pdf), the [**Vision Transformer**](https://paperswithcode.com/methods/category/vision-transformer).

![vision-transformer](https://github.com/wesleyacheng/dog-breeds-multiclass-image-classification-with-vit/assets/15952538/abac87da-7d4d-4394-8487-a324ff0b0e0a)

The difference between the **Vision Transformer** and the traditional **Convolutional Neural Network (CNN)** is how it treats an image. In **Vision Transformers**, we take the input as a patch of the original image, say 16 x 16, and feed in into the Transformer as a sequence with positional embeddings and self-attention, while in the **CNN**, we use the same patch of original image as an input, but use convolutions and pooling layers as inductive biases. What this means is that **Vision Transformer** can use it's judgement to attend any particular patch of the image in a *global* fashion using it's self-attention mechanism without having us to guide the neural network like a **CNN** with *local* centering/cropping/bounding box our images to help its convolutions. 

![self-attention](https://github.com/wesleyacheng/dog-breeds-multiclass-image-classification-with-vit/assets/15952538/6f26184f-e088-4c2e-bda2-384c0fa423d0)

This allows the **Vision Transformer** architecture to be more flexible and scalable in nature, allowing us to create [foundation models](https://blogs.nvidia.com/blog/2023/03/13/what-are-foundation-models) in computer vision, similar to the NLP foundational models like [BERT](https://paperswithcode.com/method/bert) and [GPT](https://paperswithcode.com/method/gpt), with pre-training self-supervised/supervised on massive amount of image data that would generalize to different computer vision tasks such as *image classification, recognition, segmentation, etc.* This cross-pollination helps us move closer towards the goal of Artificial General Intelligence.

One thing about **Vision Transformers** are it has weaker inductive biases compared to **Convolutional Neural Networks** that enables it's scalability and flexibility. This feature/bug depending on who you ask will require most well-performing pre-trained models to require more data despite having less parameters compared to it's CNN counterparts.

Luckily, in this notebook, we will be using a **Vision Transformer** from [Google hosted at HuggingFace](https://huggingface.co/google/vit-base-patch16-224-in21k) pre-trained on the [ImageNet-21k dataset](https://paperswithcode.com/paper/imagenet-21k-pretraining-for-the-masses) (14 million images, 21k classes) with 16x16 patches, 224x224 resolution to bypass that data limitation. We will be fine-tuning this model to our "small" dog breeds dataset of around 20 thousand images from the [Stanford Dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) imported by Jessica Li into [Kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset) to classify dog images into 120 types of dog breeds!

# Model Description
This model is finetuned using the [Google Vision Transformer (vit-base-patch16-224-in21k)](https://huggingface.co/google/vit-base-patch16-224-in21k) on the [Stanford Dogs dataset in Kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset) to classify dog images into 120 dog breeds.

# Intended Uses & Limitations
You can use this finetuned model to classify images of dogs only and dog breeds that are in the dataset.

# How to Use
```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
import PIL
import requests

url = "https://upload.wikimedia.org/wikipedia/commons/5/55/Beagle_600.jpg"
image = PIL.Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("wesleyacheng/dog-breeds-multiclass-image-classification-with-vit")
model = AutoModelForImageClassification.from_pretrained("wesleyacheng/dog-breeds-multiclass-image-classification-with-vit")

inputs = image_processor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits

# model predicts one of the 120 Stanford dog breeds classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
```

# Model Training Metrics
| Epoch | Top-1 Accuracy |  Top-3 Accuracy | Top-5 Accuracy | Macro F1 |
|-------|----------------|-----------------|----------------|----------|
| 1     | 79.8%          | 95.1%           | 97.5%          | 77.2%    |
| 2     | 83.8%          | 96.7%           | 98.2%          | 81.9%    |
| 3     | 84.8%          | 96.7%           | 98.3%          | 83.4%    |

# Model Evaluation Metrics
| Top-1 Accuracy | Top-3 Accuracy  | Top-5 Accuracy | Macro F1 |
|----------------|-----------------|----------------|----------|
| 84.0%          | 97.1%           | 98.7%          | 83.0%    |
