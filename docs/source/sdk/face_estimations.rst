Face attributes estimations
===========================

.. _`head pose`:

Head pose
---------

This estimator is designed to determine camera-space head pose. Since 3D head translation is hard to determine
reliably without camera-specific calibration,only 3D rotation component is estimated.
It estimate `Tait–Bryan`_ angles for head. Zero position
corresponds to a face placed orthogonally to camera direction, with the axis of symmetry parallel to the vertical
camera axis.

There are two head pose estimation method available: estimate by 68 face-aligned landmarks and estimate by original
input image in RGB format. Estimation by image is more precise. If you have already extracted 68 landmarks for another
facilities you may save time, and use fast estimator from 68landmarks.

.. _Tait–Bryan: https://en.wikipedia.org/wiki/Euler_angles#Tait–Bryan_angles

.. emotions_:

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

.. `mouth state`_:

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

.. _eyes:

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

.. _`gaze direction`:

Gaze direction estimation
-------------------------

This estimator is designed to determine gaze direction relatively to head pose estimation. Zero position corresponds to
a gaze direction orthogonally to face plane, with the axis of symmetry parallel to the vertical camera axis

.. note::

  Roll angle is not estimated, prediction precision decreases as a rotation angle increases.

.. _`basic attributes`:

Basic attribute estimation
--------------------------

The Attribute estimator determines face basic attributes. Currently, the following attributes are available:

    - age: determines person’s age
    - gender: determines person’s gender
    - ethnicity: determines ethnicity of a person

Before using attribute estimator, user is free to decide whether to estimateor not some specific attributes
listed above through

Output structure, which consists of optional fields describing results of user requested attributes:

    - age is reported in years (float in range [0, 100])
    - for gender estimation 1 means male, 0 means female. Estimation precision in cooperative mode is 99.81%
      with the threshold 0.5. Estimation precision in non-cooperative mode is 92.5%.
    - ethnicity estimation returns 4 float normalized values, each value describes probability of person’s ethnicity.
      The following ethnicity's are available:

         - asian
         - caucasian
         - african american
         - indian

Classes and methods
-------------------


.. automodule:: lunavl.sdk.estimators.base_estimation
    :members:

.. automodule:: lunavl.sdk.estimators.face_estimators.head_pose
    :members:

.. automodule:: lunavl.sdk.estimators.face_estimators.emotions
    :members:

.. automodule:: lunavl.sdk.estimators.face_estimators.mouth_state
    :members:

.. automodule:: lunavl.sdk.estimators.face_estimators.eyes
    :members:

.. automodule:: lunavl.sdk.estimators.face_estimators.basic_attributes
    :members:

.. automodule:: lunavl.sdk.estimators.basic_attributes
    :members:
