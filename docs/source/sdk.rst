LUNA SDK
========

.. toctree::
   :maxdepth: 2
   :caption: Contents:


   sdk/modules

Getting Started
---------------

Attributes estimation example:

>>> from lunavl.sdk.faceengine.engine import VLFaceEngine
>>> from lunavl.sdk.luna_faces import VLFaceDetector
>>> from lunavl.sdk.luna_faces import VLImage
>>> VLFaceDetector.initialize(VLFaceEngine())
>>> detector = VLFaceDetector()
>>> image = VLImage.load(
...         url='https://cdn1.savepice.ru/uploads/2019/4/15/aa970957128d9892f297cdfa5b3fda88-full.jpg')
>>> detection = detector.detectOne(image)
>>> detection.basicAttributes.age
16.0
>>> detection.emotions.predominateEmotion.name
'Happiness'
