import network3
import cPickle
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
from network3 import SGD

training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 10

#cpl = ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
#              filter_shape=(20, 1, 5, 5),
#              poolsize=(2, 2))

#fcl = FullyConnectedLayer(n_in=20*12*12, n_out=100)
#sl =  SoftmaxLayer(n_in=100, n_out=10)

#net = Network([cpl, fcl, sl], mini_batch_size)

#net.SGD(training_data, 1, mini_batch_size, 0.1,
#            validation_data, test_data)

#f = file('obj.save', 'wb')
#cPickle.dump([cpl, fcl, sl], f, protocol=cPickle.HIGHEST_PROTOCOL)
#f.close()

f = file('obj.save', 'rb')
cpl, fcl, sl = cPickle.load(f)
f.close()

net = Network([cpl, fcl, sl], mini_batch_size)
SGD(net, training_data, mini_batch_size, 0.1,
            validation_data, test_data)
