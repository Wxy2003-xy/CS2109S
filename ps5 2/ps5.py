# RUN THIS CELL FIRST
import math
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np
from numpy import allclose, isclose

from collections.abc import Callable

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Create a tensor with requires_grad set to True
x = torch.tensor([2.0], requires_grad=True)

# Compute the gradient of a simple expression using backward
y = x**2 + 2 * x
y.backward()

# Print the derivative value of y i.e dy/dx = 2x + 2 = 6.0.
print("Gradient of y with respect to x:", x.grad)

# Detach the gradient of x
x = x.detach()

# Print the gradient of x after detachment
print("Gradient of x after detachment:", x.grad)

# Extract the scalar value of a tensor as a Python number
x_value = x.item()
print("Value of x as a Python number:", x_value)

# This is a demonstration: You just need to run this cell without editing.

x = torch.linspace(-math.pi, math.pi, 1000) # Task 1.1: What is torch.linspace?
y_true = torch.sin(x)

plt.plot(x, y_true, linestyle='solid', label='sin(x)')
plt.axis('equal')
plt.title('Original function to fit')
plt.legend()
plt.show()

# Run this cell to explore what the FIRST 10 VALUES of x has been assigned to.
# By default, each cell will always print the output of the last expression in the cell
# You can explore what x is by modifying the expression e.g. x.max(), x.shape
x[:10]

### Task 1.1 - What is `torch.linspace`?

# This is a demonstration: You just need to run this cell without editing.

# Set learning rate
learning_rate = 1e-6

# Initialize weights to 0
a = torch.tensor(0.)
b = torch.tensor(0.)
c = torch.tensor(0.)
d = torch.tensor(0.)

print('iter', 'loss', '\n----', '----', sep='\t')
for t in range(1, 5001): # 5000 iterations
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x**2 + d * x**3

    # Compute MSE loss
    loss = torch.mean(torch.square(y_pred - y_true))
    if t % 1000 == 0:
        print(t, loss.item(), sep='\t')

    # Backpropagation
    grad_y_pred = 2.0 * (y_pred - y_true) / y_pred.shape[0]
    
    # Compute gradients of a, b, c, d with respect to loss
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

# print fitted polynomial
equation = f'{a:.5f} + {b:.5f} x + {c:.5f} x^2 + {d:.5f} x^3'

y_pred = a + b * x + c * x**2 + d * x**3
plt.plot(x, y_true, linestyle='solid', label='sin(x)')
plt.plot(x, y_pred, linestyle='dashed', label=f'{equation}')
plt.axis('equal')
plt.title('3rd degree poly fitted to sine (MSE loss)')
plt.legend()
plt.show()

# This is a demonstration: You just need to run this cell without editing.

# Set learning rate
learning_rate = 1e-6

# Initialize weights to 0
a = torch.tensor(0., requires_grad=True)
b = torch.tensor(0., requires_grad=True)
c = torch.tensor(0., requires_grad=True)
d = torch.tensor(0., requires_grad=True)

print('iter', 'loss', '\n----', '----', sep='\t')
for t in range(1, 5001):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute MAE loss
    loss = torch.mean(torch.abs(y_pred - y_true))
    if t % 1000 == 0:
        print(t, loss.item(), sep='\t')

    # Automatically compute gradients
    loss.backward()

    # Update weights using gradient descent
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad
        a.grad.zero_() # reset gradients !important
        b.grad.zero_() # reset gradients !important
        c.grad.zero_() # reset gradients !important
        d.grad.zero_() # reset gradients !important
        # What happens if you don't reset the gradients?

# print fitted polynomial
equation = f'{a:.5f} + {b:.5f} x + {c:.5f} x^2 + {d:.5f} x^3'

y_pred = a + b * x + c * x ** 2 + d * x ** 3
plt.plot(x, y_true, linestyle='solid', label='sin(x)')
plt.plot(x, y_pred.detach().numpy(), linestyle='dashed', label=f'{equation}')
plt.axis('equal')
plt.title('3rd degree poly fitted to sine (MAE loss)')
plt.legend()
plt.show()

### Task 1.2 - Polyfit model

def polyfit(x: torch.Tensor, y: torch.Tensor, loss_fn: Callable, n: int, lr: float, n_iter: int):
    """
    Parameters
    ----------
        x : A tensor of shape (1, n)
        y : A tensor of shape (1, n)
        loss_fn : Function to measure loss
        n : The nth-degree polynomial
        lr : Learning rate
        n_iter : The number of iterations of gradient descent
        
    Returns
    -------
        Near-optimal coefficients of the nth-degree polynomial as a tensor of shape (1, n+1) after `n_iter` epochs.
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_1_2():
    x = torch.linspace(-math.pi, math.pi, 10)
    y = torch.sin(x)
    
    def mse(y_true: torch.Tensor, y_pred: torch.Tensor):
        assert y_true.shape == y_pred.shape, f"Your ground truth and predicted values need to have the same shape {y_true.shape} vs {y_pred.shape}"
        return torch.mean(torch.square(y_pred - y_true))
    
    def mae(y_true: torch.Tensor, y_pred: torch.Tensor):
        assert y_true.shape == y_pred.shape, f"Your ground truth and predicted values need to have the same shape {y_true.shape} vs {y_pred.shape}"
        return torch.mean(torch.abs(y_pred - y_true))
    
    test1 = polyfit(x, x, mse, 1, 1e-1, 100).tolist()
    test2 = polyfit(x, x**2, mse, 2, 1e-2, 2000).tolist()
    test3 = polyfit(x, y, mse, 3, 1e-3, 5000).tolist()
    test4 = polyfit(x, y, mae, 3, 1e-3, 5000).tolist()
    
    assert allclose(test1, [0.0, 1.0], atol=1e-6)
    assert allclose(test2, [0.0, 0.0, 1.0], atol=1e-5)
    assert allclose(test3, [0.0, 0.81909, 0.0, -0.08469], atol=1e-3)
    assert allclose(test4, [0.0, 0.83506, 0.0, -0.08974], atol=1e-3)

### Task 1.3 - Observations on different model configurations

# You may use this cell to run your observations


### Task 2.1 - Forward pass

def forward_pass(x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor, activation_fn: Callable):
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_2_1():
    w0 = torch.tensor([[-1., 1.], [1., -1.]], requires_grad=True)
    w1 = torch.tensor([[0.], [1.], [1.]], requires_grad=True)
    
    output0 = forward_pass(torch.linspace(0,1,50).reshape(-1, 1), w0, w1, torch.relu)
    x_sample = torch.linspace(-2, 2, 5).reshape(-1, 1)
    test1 = forward_pass(x_sample, w0, w1, torch.relu).tolist()
    output1 = [[3.], [2.], [1.], [0.], [1.]]
    
    assert output0.shape == torch.Size([50, 1])
    assert test1 == output1

### Task 2.2 - Backward propagation

"""
paste your w0, w1 and loss values here
w0 = ...
w1 = ...
loss = ...
"""
""" YOUR CODE HERE """
raise NotImplementedError
""" YOUR CODE END HERE """

### Task 2.3 - Different random seeds

# Define a linear layer using nn.Module
class LinearLayer(nn.Module):
    """
    Linear layer as a subclass of `nn.Module`.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = nn.Parameter(torch.randn(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight) + self.bias
    
class SineActivation(nn.Module):
    """
    Sine activation layer.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)

class Model(nn.Module):
    """
    Neural network created using `LinearLayer` and `SineActivation`.
    """
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(Model, self).__init__()
        self.l1 = LinearLayer(input_size, hidden_size)
        self.act = SineActivation()
        self.l2 = LinearLayer(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        return x
    
input_size = 1
hidden_size = 1
num_classes = 1

model = Model(input_size, hidden_size, num_classes)

x = torch.tensor([[1.0]])
output = model(x)
print("Original value: ", x)
print("Value after being processed by Model: ", output)


class Squared(nn.Module):
    """
    Module that returns x**2.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.x = x
        return x**2

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        grad_input = 2 * self.x * grad_output
        return grad_input

x_sample = torch.linspace(-2, 2, 100)
sigmoid_output = nn.Sigmoid()(x_sample).detach().numpy()
tanh_output = nn.Tanh()(x_sample).detach().numpy()
relu_output = nn.ReLU()(x_sample).detach().numpy()

f = plt.figure()
f.set_figwidth(6)
f.set_figheight(6)
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.title("Input: 100 x-values between -2 to 2 \n\n Output: Corresponding y-values after passed through each activation function\n", fontsize=16)
plt.axvline(x=0, color='r', linestyle='dashed')
plt.axhline(y=0, color='r', linestyle='dashed')
plt.plot(x_sample, sigmoid_output)
plt.plot(x_sample, tanh_output)
plt.plot(x_sample, relu_output)
plt.legend(["","","Sigmoid Output", "Tanh Output", "ReLU Output"])
plt.show()

### Task 3.1 - Forward pass (NN)

class MyFirstNeuralNet(nn.Module):
    def __init__(self): # set the arguments you'd need
        super().__init__()
        self.l1 = nn.Linear(1, 2) # bias included by default
        self.l2 = nn.Linear(2, 1) # bias included by default
        self.relu = nn.ReLU()
 
    # Task 3.1: Forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass to process input through two linear layers and ReLU activation function.

        Parameters
        ----------
        x : A tensor of of shape (n, 1) where n is the number of training instances

        Returns
        -------
            Tensor of shape (n, 1)
        '''
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """

def test_task_3_1():
    x_sample = torch.linspace(-2, 2, 5).reshape(-1, 1)
    
    model = MyFirstNeuralNet()
    
    state_dict = OrderedDict([
        ('l1.weight', torch.tensor([[1.],[-1.]])),
        ('l1.bias',   torch.tensor([-1., 1.])),
        ('l2.weight', torch.tensor([[1., 1.]])),
        ('l2.bias',   torch.tensor([0.]))
    ])
    
    model.load_state_dict(state_dict)
    
    student1 = model.forward(x_sample).detach().numpy()
    output1 = [[3.], [2.], [1.], [0.], [1.]]
    
    assert allclose(student1, output1, atol=1e-5)

x = torch.tensor([1.0], requires_grad=True)

#Loss function
y = x ** 2 + 2 * x

# Define an optimizer, pass it our tensor x to update
optimiser = torch.optim.SGD([x], lr=0.1)

# Perform backpropagation
y.backward()

print("Value of x before it is updated by optimiser: ", x)
print("Gradient stored in x after backpropagation: ", x.grad)

# Call the step function on the optimizer to update weight
optimiser.step()

#Weight update, x = x - lr * x.grad = 1.0 - 0.1 * 4.0 = 0.60
print("Value of x after it is updated by optimiser: ", x)

# Set gradient of weight to zero
optimiser.zero_grad()
print("Gradient stored in x after zero_grad is called: ", x.grad)

torch.manual_seed(6) # Set seed to some fixed value

epochs = 10000

model = MyFirstNeuralNet()
# the optimizer controls the learning rate
optimiser = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0)
loss_fn = nn.MSELoss()

x = torch.linspace(-10, 10, 1000).reshape(-1, 1)
y = torch.abs(x-1)

print('Epoch', 'Loss', '\n-----', '----', sep='\t')
for i in range(1, epochs+1):
    # reset gradients to 0
    optimiser.zero_grad()
    # get predictions
    y_pred = model(x)
    # compute loss
    loss = loss_fn(y_pred, y)
    # backpropagate
    loss.backward()
    # update the model weights
    optimiser.step()

    if i % 1000 == 0:
        print (f"{i:5d}", loss.item(), sep='\t')

y_pred = model(x)
plt.plot(x, y, linestyle='solid', label='|x-1|')
plt.plot(x, y_pred.detach().numpy(), linestyle='dashed', label='perceptron')
plt.axis('equal')
plt.title('Fit NN on y=|x-1| function')
plt.legend()
plt.show()

### Task 3.2 - Model weights

"""
state_dict = OrderedDict([]) # paste the output in
"""
""" YOUR CODE HERE """
raise NotImplementedError
""" YOUR CODE END HERE """

def get_loss(model: nn.Module) -> int | float:
    model.load_state_dict(state_dict)
    x = torch.linspace(-10, 10, 1000).reshape(-1, 1)
    y = torch.abs(x-1)
    loss_fn = nn.MSELoss()
    y_pred = model.forward(x)
    return loss_fn(y_pred, y).item()

assert model.load_state_dict(state_dict)
assert get_loss(model) < 1

# DO NOT REMOVE THIS CELL – THIS DOWNLOADS THE MNIST DATASET
# RUN THIS CELL BEFORE YOU RUN THE REST OF THE CELLS BELOW
from torchvision import datasets

# This downloads the MNIST datasets ~63MB
mnist_train = datasets.MNIST("./", train=True, download=True)
mnist_test  = datasets.MNIST("./", train=False, download=True)

x_train = mnist_train.data.reshape(-1, 784) / 255
y_train = mnist_train.targets
    
x_test = mnist_test.data.reshape(-1, 784) / 255
y_test = mnist_test.targets

### Task 3.3 - Define the model architecture and implement the forward pass

class DigitNet(nn.Module):
    def __init__(self, input_dimensions: int, num_classes: int): # set the arguments you'd need
        super().__init__()
        """
        YOUR CODE HERE
        - DO NOT hardcode the input_dimensions, use the parameter in the function
        - Your network should work for any input and output size 
        - Create the 3 layers (and a ReLU layer) using the torch.nn layers API
        """
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for the network.
        
        Parameters
        ----------
        x : Input tensor (batch size is the entire dataset)

        Returns
        -------
            The output of the entire 3-layer model.
        """
        
        """
        YOUR CODE
        
        - Pass the inputs through the sequence of layers
        - Run the final output through the Softmax function on the right dimension!
        """
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """

def test_task_3_3():
    model = DigitNet(784, 10)
    assert [layer.detach().numpy().shape for name, layer in model.named_parameters()] \
            == [(512, 784), (512,), (128, 512), (128,), (10, 128), (10,)]

### Task 3.4 - Training Loop

def train_model(x_train: torch.Tensor, y_train: torch.Tensor, epochs: int = 20):
    """
    Trains the model for 20 epochs/iterations
    
    Parameters
    ----------
        x_train : A tensor of training features of shape (60000, 784)
        y_train : A tensor of training labels of shape (60000, 1)
        epochs  : Number of epochs, default of 20
        
    Returns
    -------
        The final model 
    """
    model = DigitNet(784, 10)

    optimiser = torch.optim.Adam(model.parameters()) # use Adam
    loss_fn = nn.CrossEntropyLoss()   # use CrossEntropyLoss

    for i in range(epochs):
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """

    return model

def test_task_3_4():
    x_train_new = torch.rand(5, 784, requires_grad=True)
    y_train_new = ones = torch.ones(5, dtype=torch.uint8)
    
    assert type(train_model(x_train_new, y_train_new)) == DigitNet

# This is a demonstration: You can use this cell for exploring your trained model

idx = 0 # try on some index

scores = digit_model(x_test[idx:idx+1])
_, predictions = torch.max(scores, 1)
print("true label:", y_test[idx].item())
print("pred label:", predictions[0].item())

plt.imshow(x_test[idx].numpy().reshape(28, 28), cmap='gray')
plt.axis("off")
plt.show()

### Task 3.5 - Evaluate the model

def get_accuracy(scores: torch.Tensor, labels: torch.Tensor) -> int | float:
    """
    Helper function that returns accuracy of model
    
    Parameters
    ----------
        scores : The raw softmax scores of the network
        labels : The ground truth labels
        
    Returns
    -------
        Accuracy of the model. Return a number in range [0, 1].
        0 means 0% accuracy while 1 means 100% accuracy
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_3_5():
    scores = torch.tensor([[0.4118, 0.6938, 0.9693, 0.6178, 0.3304, 0.5479, 0.4440, 0.7041, 0.5573,
             0.6959],
            [0.9849, 0.2924, 0.4823, 0.6150, 0.4967, 0.4521, 0.0575, 0.0687, 0.0501,
             0.0108],
            [0.0343, 0.1212, 0.0490, 0.0310, 0.7192, 0.8067, 0.8379, 0.7694, 0.6694,
             0.7203],
            [0.2235, 0.9502, 0.4655, 0.9314, 0.6533, 0.8914, 0.8988, 0.3955, 0.3546,
             0.5752],
            [0,0,0,0,0,0,0,0,0,1]])
    y_true = torch.tensor([5, 3, 6, 4, 9])
    acc_true = 0.4
    assert isclose(get_accuracy(scores, y_true),acc_true) , "Mismatch detected"
    print("passed")

## Task 4.1: Convolution Under The Hood

torch.manual_seed(0)

def conv2d(img: torch.Tensor, kernel: torch.Tensor):
    """
    PARAMS
        img: the 2-dim image with a specific height and width
        kernel: a 2-dim kernel (smaller than image dimensions) that convolves the given image
    
    RETURNS
        the convolved 2-dim image
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_4_1():
    x1 = torch.tensor([
        [4, 9, 3, 0, 3],
        [9, 7, 3, 7, 3],
        [1, 6, 6, 9, 8],
        [6, 6, 8, 4, 3],
        [6, 9, 1, 4, 4]
    ])
    k1 = torch.ones((2, 2))
    o1 = torch.tensor([
        [29., 22., 13., 13.],
        [23., 22., 25., 27.],
        [19., 26., 27., 24.],
        [27., 24., 17., 15.]
    ])
    
    x2 = torch.tensor([
        [1, 9, 9, 9, 0, 1],
        [2, 3, 0, 5, 5, 2],
        [9, 1, 8, 8, 3, 6],
        [9, 1, 7, 3, 5, 2],
        [1, 0, 9, 3, 1, 1],
        [0, 3, 6, 6, 7, 9]
    ])
    k2 = torch.tensor([
        [6, 3, 4, 5],
        [0, 8, 2, 8],
        [2, 7, 5, 0],
        [0, 8, 1, 9]
    ])
    o2 = torch.tensor([
        [285., 369., 286.],
        [230., 317., 257.],
        [306., 374., 344.]
    ])
    
    # TEST YOUR conv2d FUNCTION HERE
    c1 = conv2d(x1, k1)
    print(c1, torch.all(torch.eq(c1, o1)).item())
    c2 = conv2d(x2, k2)
    print(c2, torch.all(torch.eq(c2, o2)).item())

## Task 4.2: Max Pooling Under The Hood

torch.manual_seed(0)

def maxpool2d(img: torch.Tensor, size: int):
    """
    PARAMS
        img: the 2-dim image with a specific height and width
        size: an integer corresponding to the window size for Max Pooling
    
    RETURNS
        the 2-dim output after Max Pooling
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_4_2():
    x1 = torch.tensor([
        [4, 9, 3, 0, 3],
        [9, 7, 3, 7, 3],
        [1, 6, 6, 9, 8],
        [6, 6, 8, 4, 3],
        [6, 9, 1, 4, 4]
    ])
    k1 = 2
    o1 = torch.tensor([
        [9., 9., 7., 7.],
        [9., 7., 9., 9.],
        [6., 8., 9., 9.],
        [9., 9., 8., 4.]
    ])
    
    x2 = torch.tensor([
        [1, 9, 9, 9, 0, 1],
        [2, 3, 0, 5, 5, 2],
        [9, 1, 8, 8, 3, 6],
        [9, 1, 7, 3, 5, 2],
        [1, 0, 9, 3, 1, 1],
        [0, 3, 6, 6, 7, 9]
    ])
    k2 = 3
    o2 = torch.tensor([
        [9., 9., 9., 9.],
        [9., 8., 8., 8.],
        [9., 9., 9., 8.],
        [9., 9., 9., 9.]
    ])
    
    # TEST YOUR maxpool2d FUNCTION HERE
    m1 = maxpool2d(x1, k1)
    print(m1, torch.all(torch.eq(m1, o1)).item())
    m2 = maxpool2d(x2, k2)
    print(m2, torch.all(torch.eq(m2, o2)).item())

# do not remove this cell
# run this before moving on

T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

"""
Note: If you updated the path to the directory containing `MNIST` 
directory, please update it here as well.
"""
mnist_train = datasets.MNIST("./", train=True, download=False, transform=T)
mnist_test = datasets.MNIST("./", train=False, download=False, transform=T)

"""
if you feel your computer can't handle too much data, you can reduce the batch
size to 64 or 32 accordingly, but it will make training slower. 

We recommend sticking to 128 but do choose an appropriate batch size that your
computer can manage. The training phase tends to require quite a bit of memory.
"""
train_loader = torch.utils.data.DataLoader(mnist_train, shuffle=True, batch_size=256)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=10000)

# no need to code
# run this before moving on

train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()
print(f"Label: {label}")

## Task 5.1: Building a Vanilla ConvNet

class RawCNN(nn.Module):
    """
    CNN model using Conv2d and MaxPool2d layers.
    """
    def __init__(self, classes: int):
        super().__init__()
        """
        classes: integer that corresponds to the number of classes for MNIST
        """
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """
        
        x = x.view(-1, 64*5*5) # Flattening – do not remove this line

        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """

def test_task_5_1():
    # Test your network's forward pass
    num_samples, num_channels, width, height = 20, 1, 28, 28
    x = torch.rand(num_samples, num_channels, width, height)
    net = RawCNN(10)
    y = net(x)
    print(y.shape) # torch.Size([20, 10])

## Task 5.2: Building a ConvNet with Dropout

class DropoutCNN(nn.Module):
    """
    CNN that uses Conv2d, MaxPool2d, and Dropout layers.
    """
    def __init__(self, classes: int, drop_prob: float = 0.5):
        super().__init__()
        """
        classes: integer that corresponds to the number of classes for MNIST
        drop_prob: probability of dropping a node in the neural network
        """
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """
        
        x = x.view(-1, 64*5*5) # Flattening – do not remove

        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """

def test_task_5_2():
    # Test your network's forward pass
    num_samples, num_channels, width, height = 20, 1, 28, 28
    x = torch.rand(num_samples, num_channels, width, height)
    net = DropoutCNN(10)
    y = net(x)
    print(y.shape) # torch.Size([20, 10])

## Task 5.3: Training your Vanilla and Dropout CNNs

def train_model(loader: torch.utils.data.DataLoader, model: nn.Module):
    """
    PARAMS
    loader: the data loader used to generate training batches
    model: the model to train
  
    RETURNS
        the final trained model and losses
    """

    """
    YOUR CODE HERE
    
    - create the loss and optimizer
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """
    epoch_losses = []
    for i in range(10):
        epoch_loss = 0
        model.train()
        for idx, data in enumerate(loader):
            x, y = data
            """
            YOUR CODE HERE
            
            - reset the optimizer
            - perform forward pass
            - compute loss
            - perform backward pass
            """
            """ YOUR CODE HERE """
            raise NotImplementedError
            """ YOUR CODE END HERE """

        epoch_loss = epoch_loss / len(loader)
        epoch_losses.append(epoch_loss)
        print("Epoch: {}, Loss: {}".format(i, epoch_loss))
        

    return model, epoch_losses

# do not remove – nothing to code here
# run this cell before moving on
# but ensure get_accuracy from task 3.5 is defined

with torch.no_grad():
    vanilla_model.eval()
    for i, data in enumerate(test_loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        pred_vanilla = vanilla_model(x)
        acc = get_accuracy(pred_vanilla, y)
        print(f"vanilla acc: {acc}")
        
    do_model.eval()
    for i, data in enumerate(test_loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        pred_do = do_model(x)
        acc = get_accuracy(pred_do, y)
        print(f"drop-out (0.5) acc: {acc}")
        
"""
The network with Dropout might under- or outperform the network without
Dropout. However, in terms of generalisation, we are assured that the Dropout
network will not overfit – that's the guarantee of Dropout.

A very nifty trick indeed!
"""

## Task 5.4: Observing Effects of Dropout

%%time 
# do not remove – nothing to code here
# run this before moving on

print("======Training Dropout Model with Dropout Probability 0.10======")
do10_model, do10_losses = train_model(train_loader, DropoutCNN(10, 0.10))
print("======Training Dropout Model with Dropout Probability 0.95======")
do95_model, do95_losses = train_model(train_loader, DropoutCNN(10, 0.95))

# do not remove – nothing to code here
# run this cell before moving on
# but ensure get_accuracy from task 3.5 is defined

with torch.no_grad():
    do10_model.eval()
    for i, data in enumerate(test_loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        pred_do = do10_model(x)
        acc = get_accuracy(pred_do, y)
        print(acc)

    do95_model.eval()
    for i, data in enumerate(test_loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        pred_do = do95_model(x)
        acc = get_accuracy(pred_do, y)
        print(acc)

cifar_train = datasets.CIFAR10("./", train=True, download=True, transform=transforms.ToTensor())
cifar_train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=128, shuffle=True)

train_features, train_labels = next(iter(cifar_train_loader))
img = train_features[0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,7))
transform = transforms.Compose([transforms.RandomHorizontalFlip()
                                # YOUR CODE HERE
                                ]) # add in your own transformations to test
tensor_img = transform(img)
ax1.imshow(img.permute(1,2,0))
ax1.axis("off")
ax1.set_title("Before Transformation")
ax2.imshow(tensor_img.permute(1, 2, 0))
ax2.axis("off")
ax2.set_title("After Transformation")
plt.show()

## Task 6.1: Picking Data Augmentations

def get_augmentations() -> transforms.Compose:
    T = transforms.Compose([
        transforms.ToTensor(),
        # YOUR CODE HERE
    ])
    
    return T

# do not remove this cell
# run this before moving on

T = get_augmentations()

cifar_train = datasets.CIFAR10("./", train=True, download=True, transform=T)
cifar_test = datasets.CIFAR10("./", train=False, download=True, transform=T)

"""
if you feel your computer can't handle too much data, you can reduce the batch
size to 64 or 32 accordingly, but it will make training slower. 

We recommend sticking to 128 but dochoose an appropriate batch size that your
computer can manage. The training phase tends to require quite a bit of memory.

CIFAR-10 images have dimensions 3x32x32, while MNIST is 1x28x28
"""
cifar_train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=128, shuffle=True)
cifar_test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=10000)

densenet = nn.Sequential(
                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
                nn.Softmax(1) # softmax dimension
            )

x = torch.rand(15, 784) # a batch of 15 MNIST images
y = densenet(x) # here we simply run the sequential densenet on the `x` tensor
print(y.shape) # a batch of 15 predictions

convnet = nn.Sequential(
                nn.Conv2d(1, 32, (3,3)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (3,3)),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(36864, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
                nn.Softmax(1) # softmax dimension
            )

x = torch.rand(15, 1, 28, 28) # a batch of 15 MNIST images
y = convnet(x) # here we simply run the sequential convnet on the `x` tensor
print (y.shape) # a batch of 15 predictions

## Task 6.2: Build a ConvNet for CIFAR-10

class CIFARCNN(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        """
        classes: integer that corresponds to the number of classes for CIFAR-10
        """
        self.conv = nn.Sequential(
                        # YOUR CODE HERE
                    )

        self.fc = nn.Sequential(
                        # YOUR CODE HERE
                    )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """
        x = x.view(x.shape[0], 64, 6*6).mean(2) # GAP – do not remove this line
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """
        return out

%%time
# do not remove – nothing to code here
# run this cell before moving on

cifar10_model, losses = train_model(cifar_train_loader, CIFARCNN(10))

# do not remove – nothing to code here
# run this cell before moving on
# but ensure get_accuracy from task 3.5 is defined

with torch.no_grad():
    cifar10_model.eval()
    for i, data in enumerate(cifar_test_loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        pred = cifar10_model(x)
        acc = get_accuracy(pred, y)
        print(f"cifar accuracy: {acc}")
        
# don't worry if the CIFAR-10 accuracy is low, it's a tough dataset to crack.
# as long as you get something shy of 50%, you should be alright!

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import matplotlib.pyplot as plt

# Set seeds for reproducibility
torch.manual_seed(2109)
np.random.seed(2109)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

## Task 7.1: RNN Cell

def rnn_cell_forward(xt, h_prev, Wxh, Whh, Why, bh, by):
    """
    Implements a single forward step of the RNN-cell

    Args:
        xt: 2D tensor of shape (nx, m)
            Input data at timestep "t"
        h_prev: 2D tensor of shape (nh, m)
            Hidden state at timestep "t-1"
        Wxh: 2D tensor of shape (nx, nh)
            Weight matrix multiplying the input
        Whh: 2D tensor of shape (nh, nh)
            Weight matrix multiplying the hidden state
        Why: 2D tensor of shape (nh, ny)
            Weight matrix relating the hidden-state to the output
        bh: 1D tensor of shape (nh, 1)
            Bias relating to next hidden-state
        by: 2D tensor of shape (ny, 1)
            Bias relating the hidden-state to the output

    Returns:
        yt_pred -- prediction at timestep "t", tensor of shape (ny, m)
        h_next -- next hidden state, of shape (nh, m)
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_7_1():
    public_paras = {
        'xt': torch.tensor([[1., 1., 2.], [2., 1., 3.], [3., 5., 3.]]),
        'h_prev': torch.tensor([[5., 3., 2.], [1., 3., 2.]]),
        'Wxh': torch.tensor([[2., 2.], [3., 4.], [4., 3.]]),
        'Whh': torch.tensor([[2., 4.], [2., 3.]]),
        'Why': torch.tensor([[3., 5.], [5., 4.]]),
        'bh': torch.tensor([[1.], [2.]]),
        'by': torch.tensor([[3.], [1.]]),
    }
    
    expected_yt_pred = torch.tensor([[0.7311, 0.7311, 0.7311], [0.2689, 0.2689, 0.2689]])
    expected_h_next = torch.tensor([[1., 1., 1.], [1., 1., 1.]])
    
    actual_yt_pred, actual_h_next = rnn_cell_forward(**public_paras)
    assert torch.allclose(actual_yt_pred, expected_yt_pred, atol=1e-4)
    assert torch.allclose(actual_h_next, expected_h_next, atol=1e-4)

## Task 7.2: Generate Sine Wave Data

def generate_sine_wave(num_time_steps):
    """
    Generates a sine wave data

    Args:
        num_time_steps: int
            Number of time steps
    Returns:
        data: 1D tensor of shape (num_time_steps,)
            Sine wave data with corresponding time steps
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_7_2():
    num_time_steps_public = 5
    expected_data_public = torch.tensor([0.0000e+00, 1.7485e-07, 3.4969e-07, 4.7700e-08, 6.9938e-07])
    actual_data = generate_sine_wave(num_time_steps_public)
    
    assert torch.allclose(actual_data, expected_data_public)

num_time_steps = 500
sine_wave_data = generate_sine_wave(num_time_steps)

# Plot the sine wave
plt.plot(sine_wave_data)
plt.title('Sine Wave')
plt.show()

## Task 7.3: Create sequences

def create_sequences(sine_wave, seq_length):
    """
    Create overlapping sequences from the input time series and generate labels 
    Each label is the value immediately following the corresponding sequence.
    
    Args:
        sine_wave: A 1D tensor representing the time series data (e.g., sine wave).
        seq_length: int. The length of each sequence (window) to be used as input to the RNN.

    Returns: 
        windows: 2D tensor where each row is a sequence (window) of length `seq_length`.
        labels: 1D tensor where each element is the next value following each window.
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_7_3():
    seq_length_test = 2
    sine_wave_test = torch.tensor([0., 1., 2., 3.])
    expected_sequences = torch.tensor([[0., 1.], [1., 2.]])
    expected_labels = torch.tensor([2., 3.])
    
    actual_sequences, actual_labels = create_sequences(sine_wave_test, seq_length_test)
    assert torch.allclose(actual_sequences, expected_sequences)
    assert torch.allclose(actual_labels, expected_labels)

# Create sequences and labels
seq_length = 20
sequences, labels = create_sequences(sine_wave_data, seq_length)
# Add extra dimension to match RNN input shape [batch_size, seq_length, num_features]
sequences = sequences.unsqueeze(-1)
sequences.shape

# Split the sequences into training data (first 80%) and test data (remaining 20%) 
train_size = int(len(sequences) * 0.8)
train_seqs, train_labels = sequences[:train_size], labels[:train_size]
test_seqs, test_labels = sequences[train_size:], labels[train_size:]

## Task 7.4: Building RNN Model

class SineRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the SineRNN model.

        Args:
            input_size (int): The number of input features per time step (typically 1 for univariate time series).
            hidden_size (int): The number of units in the RNN's hidden layer.
            output_size (int): The size of the output (usually 1 for predicting a single value).
        """
        super(SineRNN, self).__init__()
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """
        
    def forward(self, x):
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """

def test_task_7_4():
    input_size = output_size = 1
    hidden_size = 50
    model = SineRNN(input_size, hidden_size, output_size).to(device)
    assert [layer.detach().numpy().shape for _, layer in model.named_parameters()]\
          == [(50, 1), (50, 50), (50,), (50,), (1, 50), (1,)]

# Define loss function, and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(train_seqs)
    loss = criterion(outputs.squeeze(), train_labels)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

# Predict on unseen data
model.eval()
y_pred = []
input_seq = test_seqs[0]  # Start with the first testing sequence

with torch.no_grad():
    for _ in range(len(test_seqs)):
        output = model(input_seq)
        y_pred.append(output.item())
        
        # Use the predicted value as the next input sequence
        next_seq = torch.cat((input_seq[1:, :], output.unsqueeze(0)), dim=0)
        input_seq = next_seq

# Plot the true sine wave and predictions
plt.plot(sine_wave_data, c='gray', label='Actual data')
plt.scatter(np.arange(seq_length + len(train_labels)), sine_wave_data[:seq_length + len(train_labels)], marker='.', label='Train')
x_axis_pred = np.arange(len(sine_wave_data) - len(test_labels), len(sine_wave_data))
plt.scatter(x_axis_pred, y_pred, marker='.', label='Predicted')
plt.legend(loc="lower left")
plt.show()


if __name__ == '__main__':
    test_task_1_2()
    test_task_2_1()
    test_task_3_1()
    test_task_3_3()
    test_task_3_4()
    test_task_3_5()
    test_task_4_1()
    test_task_4_2()
    test_task_5_1()
    test_task_5_2()
    test_task_7_1()
    test_task_7_2()
    test_task_7_3()
    test_task_7_4()