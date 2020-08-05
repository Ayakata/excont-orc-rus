# -*- coding: utf-8 -*-

from sys import argv
from os import listdir
from os.path import isfile, join
from io import BytesIO
from keras_ocr.pipeline import Pipeline

imagePath = './TEMP/images'
weights = './TEMP/weights/recognizer_2020-07-27T13 14 33.213487.h5'

def readBytes(name):
    with open(name, 'rb') as fin:
        return BytesIO(fin.read())    

def make_ocr():

    onlyfiles = [join(imagePath, f) for f in listdir(imagePath) if isfile(join(imagePath, f))]

    # Get a set of three example images
    images = list( map( lambda name: readBytes(name), onlyfiles ))

    # keras-ocr will automatically download pretrained
    # weights for the detector and recognizer.
    pipeline = Pipeline()

    pipeline.recognizer.model.load_weights(weights)

    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples.
    prediction_groups = pipeline.recognize(images)
    pass


if __name__ == '__main__':

    make_ocr()
    