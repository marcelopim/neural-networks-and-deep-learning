import mnist_loader
# import network2_mod as network
# import network2 as network
import network_py36 as network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10])
# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)

net.SGD_niin(training_data, 30, 10, 10.0, lmbda = 100.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True)
# net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0,evaluation_data=validation_data, monitor_evaluation_accuracy=True)

# net.save('pesos')