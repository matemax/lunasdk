# Luna VisionLabs

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/d24e9508258849c2b40760fce3448c6b)](https://www.codacy.com/app/matemax/lunasdk?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=matemax/lunasdk&amp;utm_campaign=Badge_Grade)

A python interface to VisionLabs LUNA product.

We implemented only interface to LUNA C++ SDK. It is face detection and recognition library.

## Install

The main dependencies is Luna SDK library and an FaceEnginePythonBindings. If these dependencies is installed
 add to *PATH* a path to *FaceEngineSDK.dll* or *FaceEngineSDK.so*. 
 
You can install this package several path:

- install from github: *pip install  git+https://github.com/matemax/lunasdk.git*
- you can load archive from github, unpack it and make a command: *python setup.py install*.

## Setup

We get a path to folder with neural network models and path to a FaceEngine configuration file  from environment 
variable *FSDK_ROOT* (*{FSDK_ROOT}/data*, *{FSDK_ROOT}/data/faceengine.conf*) by default. You can specify these path's 
in initialization *VLFaceEngine*. 
