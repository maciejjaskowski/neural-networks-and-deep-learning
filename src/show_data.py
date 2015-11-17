# Standard library
import _pickle as cPickle
import gzip

import numpy as np


def load_data_shared(filename="../data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = cPickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

training_data, validation_data, test_data = load_data_shared()

import matplotlib.pyplot as plt
import matplotlib

def n_holes(ans):
  if (ans == 0 or ans == 6 or ans == 9):
   return 1
  elif (ans == 8):
   return 2
  else:
   return 0

def show_to_choose(td, ans,i):
  plt.ion()
  plt.imshow(np.reshape(td, (28, 28)), cmap='Greys')
  plt.show()
  nb = input("{}: Choose number [{}]:".format(i, n_holes(ans)))
  if (nb == '§'):
    nb = 0
  else:
    nb = int(nb)
  plt.close()
  plt.ioff()
  return nb


sub = range(500,1000)
result = [show_to_choose(img,ans,i) for img,ans,i in zip(training_data[0][sub], training_data[1][sub], sub)]
print(result)
