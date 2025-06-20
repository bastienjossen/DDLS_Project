# coding: utf-8
###
 # @file   models.py
 # @author John Stephan <john.stephan@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2023 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Collection of models.
###

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------- #
# Simple fully-connected model, for MNIST
class fc_mnist(nn.Module):
	""" Simple, small fully connected model."""

	def __init__(self):
		""" Model parameter constructor. """
		super().__init__()
		# Build parameters
		self._f1 = nn.Linear(28 * 28, 100)
		self._f2 = nn.Linear(100, 10)

	def forward(self, x):
		""" Model's forward pass. """
		x = F.relu(self._f1(x.view(-1, 28 * 28)))
		x = F.log_softmax(self._f2(x), dim=1)
		return x

# ---------------------------------------------------------------------------- #
# Simple convolutional model, for MNIST
class cnn_mnist(nn.Module):
	""" Simple, small convolutional model."""

	def __init__(self):
		""" Model parameter constructor. """
		super().__init__()
		# Build parameters
		self._c1 = nn.Conv2d(1, 20, 5, 1)
		self._c2 = nn.Conv2d(20, 50, 5, 1)
		self._f1 = nn.Linear(800, 500)
		self._f2 = nn.Linear(500, 10)

	def forward(self, x):
		""" Model's forward pass. """
		x = F.relu(self._c1(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self._c2(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self._f1(x.view(-1, 800)))
		x = F.log_softmax(self._f2(x), dim=1)
		return x

# ---------------------------------------------------------------------------- #
# Simple logistic regression model for MNIST
class logreg_mnist(nn.Module):
	""" Simple logistic regression model."""

	def __init__(self):
		""" Model parameter constructor. """
		super().__init__()
		# Build parameters
		self._linear = nn.Linear(784, 10)

	def forward(self, x):
		""" Model's forward pass. """
		return torch.sigmoid(self._linear(x.view(-1, 784)))

# ---------------------------------------------------------------------------- #
#JS: Simple logistic regression model (for phishing)
class logreg_phishing(nn.Module):
	""" Simple logistic regression model."""

	def __init__(self, din, dout=1):
		""" Model parameter constructor.
		Args:
			din  Number of input dimensions
			dout Number of output dimensions
		"""
		super().__init__()
		# Store model parameters
		self._din  = din
		self._dout = dout
		# Build parameters
		self._linear = nn.Linear(din, dout)

	def forward(self, x):
		""" Model's forward pass. """
		return torch.sigmoid(self._linear(x.view(-1, self._din)))


# ---------------------------------------------------------------------------- #
#JS: Simple convolutional model, for CIFAR-10/100 (3 input channels)
class cnn_cifar_old(nn.Module):
  """ Simple, small convolutional model."""

  def __init__(self):
    """ Model parameter constructor."""
    super().__init__()
    # Build parameters
    self._c1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    self._b1 = nn.BatchNorm2d(self._c1.out_channels)
    self._c2 = nn.Conv2d(self._c1.out_channels, 64, kernel_size=3, padding=1)
    self._b2 = nn.BatchNorm2d(self._c2.out_channels)
    self._m1 = nn.MaxPool2d(2)
    self._d1 = nn.Dropout(p=0.25)
    self._c3 = nn.Conv2d(self._c2.out_channels, 128, kernel_size=3, padding=1)
    self._b3 = nn.BatchNorm2d(self._c3.out_channels)
    self._c4 = nn.Conv2d(self._c3.out_channels, 128, kernel_size=3, padding=1)
    self._b4 = nn.BatchNorm2d(self._c4.out_channels)
    self._m2 = nn.MaxPool2d(2)
    self._d2 = nn.Dropout(p=0.25)
    self._d3 = nn.Dropout(p=0.25)
    self._f1 = nn.Linear(8192, 128)
    self._f2 = nn.Linear(self._f1.out_features, 10)

  def forward(self, x):
    """ Model's forward pass. """
    activation = F.relu
    flatten    = lambda x: x.view(x.shape[0], -1)
    logsoftmax = F.log_softmax
    # Forward pass
    x = self._c1(x)
    x = activation(x)
    x = self._b1(x)
    x = self._c2(x)
    x = activation(x)
    x = self._b2(x)
    x = self._m1(x)
    x = self._d1(x)
    x = self._c3(x)
    x = activation(x)
    x = self._b3(x)
    x = self._c4(x)
    x = activation(x)
    x = self._b4(x)
    x = self._m2(x)
    x = self._d2(x)
    x = flatten(x)
    x = self._f1(x)
    x = activation(x)
    x = self._d3(x)
    x = self._f2(x)
    x = logsoftmax(x, dim=1)
    return x


class cifar_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 200, 5)
        self.fc1 = nn.Linear(200 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Test(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 200, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(200, 1000, 5)
        self.fc1 = nn.Linear(1000 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class cnn_cifar(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, padding=2)
        self.conv2 = nn.Conv2d(self.conv1.out_channels, 100, 5, padding=2)
        self.conv3 = nn.Conv2d(self.conv2.out_channels, 200, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(self.conv3.out_channels * 4 * 4, 512)
        self.fc2 = nn.Linear(self.fc1.out_features, 256)
        self.fc3 = nn.Linear(self.fc2.out_features, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


#################################################################################################
#JS: Resnet models

class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 
 
class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 
 
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
 
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
 
    def forward(self, x, out_feature=False):
        x = self.conv1(x)

        x = self.bn1(x)
        out = F.relu(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)
        if out_feature == False:
            return out
        else:
            return out,feature
 
 
def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)
 
def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes)
 
def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes)
 
def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3,4,23,3], num_classes)
 
def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3,8,36,3], num_classes)