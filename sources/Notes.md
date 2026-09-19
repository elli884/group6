# Definitions

## NAVI-Orbital 
- is from the NASA and Loft Orbital developed Softwaresystem, which was in April 2026 of the Satellite YAM-9 the worldwide first sucessful In-Orbit-demonstration of an autonomous VLM

- autonomous picture analysing: system uses a compact version of Googles open Gemma 3 (4B-Parameter, 4-Bit-quantisiert) to discribe recorded pictures directly in the space

- semantic compression: instead of sending enormous data to earch, NAVI-Orbital compress the picture information semantically and return directly the text description

- agent architecture: the steering takes place over a graph-based state machine (LangGraph) with specialized agents for the object recognition and the operator driven diaglog

- Hardware: system runs on edge-computing hardware in the orbit (Nvidia Jetson Orin AGX)

## Open Earth-observation (EO) datasets

###  AID (Aerial Image Dataset):
-  contains 10,000 aerial images divided into 30 distinct scene classes, designed to test remote sensing and scene classification algorithms

- every image in the dataset is a three-channel RGB color image with a fixed resolution of 600 × 600 pixels

- relies on high-resolution Google Earth images

### EuroSat
- it is built using imagery from the Sentinel-2 satellite constellation, which trades raw spatial resolution for deep spectral coverage and precise geographic referencing

- Total Images: 27,000 labeled and geo-referenced image patches

- Image Resolution: Small 64 × 64 pixels per patch

- Unlike AID, which only provides standard RGB images, EuroSAT is unique because it is offered in two distinct flavors depending on your machine learning goals:
EuroSAT_MS (Multi-Spectral): Contains all 13 spectral bands captured by Sentinel-2 (including Visible, Near-Infrared, and Short-Wave Infrared). These are distributed as .tif files with raw float values. [1] (https://www.kaggle.com/datasets/apollo2506/eurosat-dataset), [2] (https://zenodo.org/records/7711810), [3] (https://www.researchgate.net/publication/319463676_EuroSAT_A_Novel_Dataset_and_Deep_Learning_Benchmark_for_Land_Use_and_Land_Cover_Classification), [4] (https://www.tensorflow.org/datasets/catalog/eurosat)EuroSAT_RGB: A standard 3-channel optical RGB version encoded as JPEG images, making it easy to drop directly into standard computer vision models

## LoRA adapter

- LoRA stands for Low-Rank Adaptation

- A LoRA adapter is a small set of trainable parameters that lets you fine-tune a large AI model without retraining the whole model

- a LoRA adapter isn't a model by itself. It's an add-on that normally needs a compatible base model

- The interesting part is that image LoRAs can often be very small compared with the base model, and you can control their influence with a LoRA weight

- There are also important differences between Stable Diffusion LoRAs and FLUX LoRAs, particularly regarding what parts of the model they modify, training methods, and compatibility.

### The basic idea

Suppose you have a large model with billions of parameters:

Base model → already knows language, reasoning, coding, etc.

Instead of changing all those billions of parameters, LoRA keeps the base model frozen and learns a small additional set of parameters:

Base model + LoRA adapter → specialized model