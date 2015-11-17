"""expand_mnist.py
~~~~~~~~~~~~~~~~~~

Take the 50,000 MNIST training images, and create an expanded set of
250,000 images, by displacing each training image up, down, left and
right, by one pixel.  Save the resulting file to
../data/mnist_expanded.pkl.gz.

Note that this program is memory intensive, and may not run on small
systems.

"""

from __future__ import print_function

#### Libraries

# Standard library
import _pickle as cPickle
import gzip
import io
import os.path
import random

# Third-party libraries
import numpy as np

print("Expanding the MNIST training set")
def load(filename="../data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, __, ___ = cPickle.load(f, encoding='latin1')
    f.close()
    f = open("r0-999")
    outputsStr = io.StringIO(f.read()[1:-2])
    f.close()
    outputs = np.loadtxt(outputsStr, delimiter=',', dtype="int32")
    test_data = (training_data[0][900:1000], training_data[1][900:1000])
    validation_data = (training_data[0][800:900], training_data[1][800:900])
    training_data = (training_data[0][0:800], training_data[1][0:800])
    print(len(training_data[0]))
    print(len(validation_data[0]))
    print(len(test_data[0]))
    return (training_data, validation_data, test_data)

def save(data, filename="../data/mnist_holes.pkl.gz"):
    print("Saving data. This may take a few minutes.")
    f = gzip.open(filename, "w")
    cPickle.dump(data, f)
    f.close()


save(load())
