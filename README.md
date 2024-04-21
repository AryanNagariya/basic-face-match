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

## Images Download

Now we'll make use of the following:

* [names.csv](https://penn-cis545-files.s3.amazonaws.com/names.csv), which is the original list of actor names from the homeworks. You can update this with whatever IMDB actors you have.
* [IMDB-Face.csv](https://penn-cis545-files.s3.amazonaws.com/IMDb-Face.csv), from https://github.com/fwang91/IMDb-Face.

You can simply:

`wget ` to get a zipfile.

```
wget https://penn-cis545-files.s3.amazonaws.com/imdb-faces.zip
cd images
unzip ../imdb-faces.zip
cd ..
```

## Creating Your ChromaDB Instance

Launch ChromaDB in Docker, via `chroma run --host 0.0.0.0`.

In a separate Docker Terminal, from the same directory, run `node app.js` and let it index everything!

Take a look at the functions in [app.js](app.js), to see examples of computing an embedding from an image, opening a connection to Chroma, indexing in Chroma, and looking up an entry in Chroma.

