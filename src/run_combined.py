import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

#training_data_h, validation_data_h, test_data_h = network3.load_data_shared("../data/mnist_holes.pkl.gz")
training_data_h, validation_data_h, test_data_h = network3.load_data_shared("../data/mnist_holes_expanded.pkl.gz")

training_data, validation_data, test_data = network3.load_data_shared("../data/mnist.pkl.gz")

mini_batch_size = 10


conv1 = ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                      filter_shape=(20, 1, 5, 5), 
                      poolsize=(2, 2))

f1 = FullyConnectedLayer(n_in=20*12*12, n_out=100)
s1 = SoftmaxLayer(n_in=100, n_out=10)
net = Network([conv1, f1, s1], mini_batch_size)

f2 = FullyConnectedLayer(n_in=20*12*12, n_out=100)
s2 = SoftmaxLayer(n_in=100, n_out=10)
net_h = Network([conv1, f2, s2], mini_batch_size)


for i in range(0, 60):
  print("Epoch {}, net".format(i))
  net.SGD(training_data, 1, mini_batch_size, 0.1, 
            validation_data, test_data) 
  print("Epoch {}, net_h".format(i))
  net_h.SGD(training_data_h, 1, mini_batch_size, 0.1,
            validation_data_h, test_data_h)
