Face attributes estimations
===========================

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


Mouth state
-----------


This estimator is designed to determine smile/mouth/occlusion probability using warped image. Smile estimation
structure consists of:

    - Smile score
    - Mouth score
    - Occlusion score

Sum of scores always equals 1. Each score means probability of corresponding state. Smile score prevails in cases where
smile was successfully detected. If there is any object on photo that hides mouth occlusion score prevails.
Mouth score prevails in cases where neither a smile nor an occlusion was detected.

Eyes estimation
---------------

This estimator aims to determine:

    - eye state: open, closed, occluded
    - precise eye iris location as an array of landmarks
    - recise eyelid location as an array of landmarks


Iris landmarks are presented with a template structure Landmarks that is specialized for 32points. Eyelid landmarks
are presented with a template structure Landmarks that is specialized for 6points.


You can only pass warped image with detected face to the estimator interface. Better image quality leads to better
results.

.. note::

    Orientation terms “left” and “right” refer to the way you see the image as it is show non the screen. It means
    that left eye is not necessarily left from the person’s point of view, but is on the left side of the screen.
    Consequently, right eye is the one on the right side of the screen. More formally, the label “left” refers to
    subject’s left eye (and similarly for the right eye), such that *xright<xleft*.

Gaze direction estimation
-------------------------

This estimator is designed to determine gaze direction relatively to head pose estimation. Zero position corresponds to
a gaze direction orthogonally to face plane, with the axis of symmetry parallel to the vertical camera axis

.. note::

  Roll angle is not estimated, prediction precision decreases as a rotation angle increases.

.. automodule:: lunavl.sdk.estimators.base_estimation
    :members:

.. automodule:: lunavl.sdk.estimators.head_pose
    :members:

.. automodule:: lunavl.sdk.estimators.emotions
    :members:

.. automodule:: lunavl.sdk.estimators.mouth_state
    :members:

.. automodule:: lunavl.sdk.estimators.eyes
    :members:
