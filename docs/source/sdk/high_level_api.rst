High level api
==============

Face detection and attributes estimation
----------------------------------------

High level api is represented *VLFaceDetection* and *VLFaceDetector* classes.

*VLFaceDetection* is container for a face detection. It has properties which are corresponding face estimations.
Every property is a singleton which estimates corresponding attribute once and remembers it for next calling.

*VLFaceDetector* is a wraper on *FaceDetector* which convert *FaceDetection* to *VLFaceDetection*. Class contains
a face estimator collection as a class attribute for a detections attributes estimation. If you want specify an unique
*VLFaceEngine* for the instance of *VLFaceDetector* you should set the FaceEngine in the *init* of the class.

Example
~~~~~~~

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