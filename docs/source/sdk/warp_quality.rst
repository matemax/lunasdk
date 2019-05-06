Warp quality
============

This estimator aims to predict visual quality of an image. It is trained specifically on pre-warped human face images
and will produce lower factor if:

 - Image is blurred;
 - Image is under-exposured (i.e., too dark);
 - Image is over-exposured (i.e., too light);
 - Image color variation is low (i.e., image is monochrome or close to monochrome).

 The quality factor is a value in range [0..1] where 0 corresponds to low quality and 1 to high quality.

.. automodule:: lunavl.sdk.estimators.warp_quality
    :members:
