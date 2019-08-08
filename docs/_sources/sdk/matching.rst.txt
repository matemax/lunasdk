Descriptor matching
===================

.. _`face descriptors matching`:

Face descriptor matching
------------------------

It is possible to match a pair (or more) previously extracted descriptors to find out their similarity. With this
information, it is possible to implement face search and other analysis applications.

It is possible to match a pair of descriptors with each other or a single descriptor with a descriptor batch.

A simple rule to help you decide which storage to opt for:

    #) when searching among less than a hundred descriptors use separate descriptor object;

    #) when searching among bigger number of descriptors use a batch.


.. automodule:: lunavl.sdk.faceengine.descriptors
    :members:

.. automodule:: lunavl.sdk.faceengine.matcher
    :members:

