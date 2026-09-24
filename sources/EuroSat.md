# EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification

Link: https://arxiv.org/pdf/1709.00029

- geo-referenced dataset EuroSAT is made
publicly available at https://github.com/phelber/eurosat 

- getting more public and continuous access to free satellite image data for Earth observation 
Reason: intention to fuel innovation in the domains of agriculture, disaster recovery, climate change, urban development, or environmental monitoring can be real-
ized [37], [2], [3], [5]

"""
In this paper, we propose a novel satellite image dataset for
the task of land use and land cover classification.
"""

## EuroSAT dataset

- consists of 27,000 labeled images with 10
different land use and land cover classes, covering 13 different spectral bands

- dataset is georeferenced and based on
openly and freely accessible Earth observation data allowing a
unique range of application

## How does patch-based land cover classification work?

- satellite scans Earth to acquire images of it
- patches extracted out of these images are used for classification
- aim is to automatically provide labels describing the represented physical land type or how the land is use
- For this
purpose, an image patch is feed into a classifier
- classifier outputs the class shown
on the image patch



In this paper CNNs are used to evaluate the performance of each spectral band

## Other benchmark datasets
 - PatternNet 
 - NWPU-RESISC45

 => atasets are based on very-high-
resolution images with a spatial resolution of up to 30 cm
per pixel

## Why so less trainings data?
- creation of a labeled dataset is extremely
time-consuming, these datasets consist likewise of only a few
hundred images per class

- Aerial Image Dataset (AID) (one of the largest datasets, etracted from Google Earth imagery)
AID consists of 30 classes with
200 to 400 images per class

## Convolutional Neural Networks (CNNs)

- are type of Neural Networks
- became with the impressive results
on image classification challenges
- pretrained networks
proved to be suitable for the classification of remotely sensed
images 
- The presented work extensively evaluated all
proposed machine learning methods and concluded that that
deep CNNs outperform non-deep learning approaches on the
considered dataset
- Between the classes, there is a low positive correlation.
The classes most common to each other are the two presented
agricultural classes and the two classes representing residential
and industrial building.


## used CNN models
- GoogleNet 
- ResNet-5

- deep CNNs perform better than shallow CNNs

## Conclusion
- Paper ddressed the challenge of land
use and land cover classification
- The
RGB band combination with an overall classification accuracy
of 98.57% outperformed the shortwave-infrared and the color-
infrared band combination and leads to a better classification
accuracy than all single-band evaluation
- The proposed dataset can be leveraged for
multiple real-world Earth observation applications