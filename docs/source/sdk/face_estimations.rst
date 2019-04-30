Face Estimations
================

Head pose
---------

This estimator is designed to determine camera-space head pose. Since 3D head translation is hard to determine
reliably without camera-specific calibration,only 3D rotation component is estimated.
It estimate Tait–Bryan angles for head (https://en.wikipedia.org/wiki/Euler_angles#Tait–Bryan_angles). Zero position
corresponds to a face placed orthogonally to camera direction, with the axis of symmetry parallel to the vertical
camera axis.

There are two head pose estimation method available: estimate by 68 face-aligned landmarks and estimate by original
input image in RGB format. Estimation by image is more precise. If you have already extracted 68 landmarks for another
facilities you may save time, and use fast estimator from 68landmarks.

Emotions
--------

This estimator aims to determine whether a face depicted on an image expresses the following emotions:

    - Anger
    - Disgust
    - Fear
    - Happiness
    - Surprise
    - Sadness
    - Neutrality

You can pass only warped images with detected faces to the estimator interface. Better image quality leads to better
results.

Emotions estimation presents emotions a snormalized float values in the range of [0..1] where 0 is lack of
a specific emotion and 1st is the maximum intensity of an emotion.


.. automodule:: lunavl.sdk.estimators.head_pose
    :members:

.. automodule:: lunavl.sdk.estimators.emotion
    :members:
