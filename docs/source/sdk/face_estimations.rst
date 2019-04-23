Face Estimations
================

Head pose
=========

This estimator is designed to determine camera-space head pose. Since 3D head translation is hard to determine
reliably without camera-specific calibration,only 3D rotation component is estimated.
It estimate Tait–Bryan angles for head (https://en.wikipedia.org/wiki/Euler_angles#Tait–Bryan_angles). Zero position
corresponds to a face placed orthogonally to camera direction, with the axis of symmetry parallel to the vertical
camera axis.

There are two head pose estimation method available: estimate by 68 face-aligned landmarks and estimate by original
input image in RGB format. Estimation by image is more precise. If you have already extracted 68 landmarks for another
facilities you may save time, and use fast estimator from 68landmarks.
