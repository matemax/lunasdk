Face detection and landmarks
============================

Luna VL provides methods for faces detection on images and find landmarks on it.

Face detection
--------------

Detectors
~~~~~~~~~

There are 3 face detectors: *FACE_DET_V1, FACE_DET_V2, FACE_DET_V3*. *FaceDetV1* detector is more precise and
*FaceDetV2* works two times faster. *FaceDetV1* and *FaceDetV2* performance depends on number of faces on image and
image complexity. *FaceDetV3* performance depends only on target image resolution. *FACE_DET_V3* is the latest and most
precise detector. In terms of performance it is similar to *FaceDetV1* detector. *FaceDetV3*  may be slower then
*FaceDetV1* on images with one face and much more faster on images with many faces.


You should create a face detector using the method *createFaceDetector* of class *VLFaceEngine* for faces detection.
You should set detector type when creating detector. Once initialize detector can be using as many times as you like.

.. warning::

    We don’t recommend create often new detector because it is very slowly operation.


Face alignment
--------------

Face alignment is the process of special key points (called "landmarks") detection on a face. FaceEngine does landmark
detection at the same time as the face detection since some of the landmarks are by products of that detection.


Landmarks5
~~~~~~~~~~

At the very minimum, just 5 landmarks are required: two for eyes, one for a nose tip and two for mouth corners. Using
these coordinates, one may warp the source photo image (see Chapter“Image warping”) for use with all other FaceEngine
algorithms. All detector may provide 5 landmarks for each detection without additional computations.

Landmarks68
~~~~~~~~~~~

More advanced 68-points face alignment is also implemented. Use this when you need precise information about face and
its parts. The 68 landmarks require additional computation time, so don’t use it if you don’t need precise information
about a face. If you use 68 landmarks , 5 landmarks will be reassigned to more precise subset of 68 landmarks.
