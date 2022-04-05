# Luna VisionLabs

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/9a84be56ae864b09a667dcf1a2c400f8)](https://www.codacy.com/manual/VisionLabs/lunasdk?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=matemax/lunasdk&amp;utm_campaign=Badge_Grade)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A python interface to VisionLabs LUNA product.

We implemented only interface to LUNA C++ SDK. It is face detection and recognition library.

## Install

The main dependencies is Luna SDK library and an FaceEnginePythonBindings. If these dependencies is installed
 add to *PATH* a path to *FaceEngineSDK.dll* or *FaceEngineSDK.so*.

You can install this package several path:

-   install from github:
  
    ```console
    pip install git+https://github.com/matemax/lunasdk.git
    ```

## Setup

We get a path to folder with neural network models and path to a FaceEngine configuration file  from environment 
variable *FSDK_ROOT* (*{FSDK_ROOT}/data*, *{FSDK_ROOT}/data/faceengine.conf*) by default. You can also specify these paths 
within *VLFaceEngine* initialization. 

## Getting Started

Attributes estimation example:

``` python
>>> from lunavl.sdk.luna_faces import VLFaceDetector
>>> from lunavl.sdk.luna_faces import VLImage
>>> detector = VLFaceDetector()
>>> image = VLImage.load(
...         url='https://cdn1.savepice.ru/uploads/2019/4/15/aa970957128d9892f297cdfa5b3fda88-full.jpg')
>>> detection = detector.detectOne(image)
>>> detection.basicAttributes.age
16.0
>>> detection.emotions.predominateEmotion.name
'Happiness'
```
