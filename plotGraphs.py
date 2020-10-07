import matplotlib.pyplot as plt

x = ['MNIST_D', 'MNIST_F', 'CIFAR_10', 'CIFAR_100_F', 'CIFAR_100_C']
y = [99.04, 92.12, 72.95, 40.32, 51.95]

plt.bar(x, y)
plt.xlabel("Dataset")
plt.ylabel("Accuracy Percentage")
plt.title("CNN: Accuracy % vs Dataset")
plt.show()




y = [98.13, 88.04, 10.00, 1.00, 5.00]
plt.bar(x, y)
plt.xlabel("Dataset")
plt.ylabel("Accuracy Percentage")
plt.title("ANN: Accuracy % vs Dataset")
plt.show()

