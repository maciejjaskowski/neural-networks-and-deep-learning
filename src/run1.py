import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 10


conv1 = ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                      filter_shape=(20, 1, 5, 5), 
                      poolsize=(2, 2))

f1 = FullyConnectedLayer(n_in=20*12*12, n_out=100)

s1 = SoftmaxLayer(n_in=100, n_out=10)
net = Network([conv1, f1, s1], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, 
            validation_data, test_data) 


net2 = Network([conv1, f2, s2], mini_batch_size)
net2.SGD(training_data2, 60, mini_batch_size, 0.1,
             validation_data2, test_data2)
