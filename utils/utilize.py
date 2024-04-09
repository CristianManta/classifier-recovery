# For CIFAR10
from main import get_data_loader, ModifiedResNet18, train_model, infer

cifar_train_loader = get_data_loader('CIFAR10')
cifar_model = ModifiedResNet18()
train_model(cifar_model, cifar_train_loader)
cifar_infer_loader = get_data_loader('CIFAR10', train=False)
cifar_output = infer(cifar_model, cifar_infer_loader, before_softmax=True)
print("--- Cifar10 Output before Softmax ---")
print(cifar_output)
print()

# For MNIST
mnist_train_loader = get_data_loader('MNIST')
mnist_model = ModifiedResNet18()
train_model(mnist_model, mnist_train_loader)
mnist_infer_loader = get_data_loader('MNIST', train=False)
mnist_output = infer(mnist_model, mnist_infer_loader, before_softmax=True)
print("--- MNIST Output before Softmax ---")
print(mnist_output)
print()
