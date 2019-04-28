Warping
=======

.. _`warping`:

Warping is the process of face image normalization. It requires landmarks and face detection to operate.
The warp has the following properties:

    - its size is always 250x250 pixels
    - it's always in RGB color format
    - it always contains just a single face
    - the face is always centered and rotated so that imaginary line between the eyes is horizontal.

The purpose of the process is to:

    - compensate image plane rotation (roll angle);
    - center the image using eye positions;
    - properly crop the image.



.. automodule:: lunavl.sdk.faceengine.warper
    :members:
