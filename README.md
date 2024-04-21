# Face API / Face Matching with ChromaDB

This module provides the basics for identifying and recognizing faces using the [Face API](https://github.com/vladmandic/face-api/tree/master).
We will leverage ChromaDB as well.

## Docker Container Setup

You will need to install some more packages to use TensorFlow for face detection.

* `apt install -y libhdf5-serial-dev`
* `pip install h5py`
* `pip install tensorflow`

Then run `npm install` from this directory.

Then:
* `npm rebuild @tensorflow/tfjs-node --build-from-source`
* `bash ./download-models.sh`
