import sys
import numpy as np
from math import exp, sqrt
from random import randint
import matplotlib
from matplotlib import pyplot as plt

# compute kernel
def linear_kernel(x1, x2):
  return np.dot(x1.T, x2)

def gaussian_kernel(x1, x2, sigma):
  if sigma == 0:
    return 0
  else:
    t = np.sum((x1-x2)*(x1-x2))
    return exp(-t/(2*sigma*sigma)) 

# Do optimize to find alphas
def takeStep(i1, i2, kernel, alphas, b):
  alpha1 = alphas[i1]
  alpha2 = alphas[i2]
  y1 = y[i1]
  y2 = y[i2]
  E1 = 0
  E2 = 0
  for i in range(N):
    E1 += alphas[i] * y[i] * kernel(X[i], X[i1])
    E2 += alphas[i] * y[i] * kernel(X[i], X[i2])
  E1 = E1 + b[0] - y1
  E2 = E2 + b[0] - y2
  s = y1 * y2
  # Compute L and H
  if y1 != y2:
    L = max(0, alpha2 - alpha1)
    H = min(C, C + alpha2 - alpha1)
  else:
    L = max(0, alpha2 + alpha1 - C)
    H = min(C, alpha2 + alpha1)
  if L == H:
    return 0
  # Compute eta
  eta = kernel(X[i1], X[i1]) + kernel(X[i2], X[i2]) - 2*kernel(X[i1], X[i2])
  if eta <= 0:
    return 0
  a2 = alpha2 + y2*(E1-E2)/eta
  # Clip
  a2 = min(H, a2)
  a2 = max(L, a2)
  # Check the change in alpha is significant
  eps = 0.00001 # 10^-5
  if abs(a2 - alpha2) < eps*(a2+alpha2+eps):
    return 0
  # Update a1
  a1 = alpha1 + s*(alpha2 - a2)
  # Update threshold b
  b1 = b[0] - E1 - y1*(a1-alpha1)*kernel(X[i1], X[i1]) - y2*(a2-alpha2)*kernel(X[i1], X[i2])
  b2 = b[0] - E2 - y1*(a1-alpha1)*kernel(X[i1], X[i2]) - y2*(a2-alpha2)*kernel(X[i2], X[i2])
  if (0 < a1 and a1 < C):
    b[0] = b1
  elif (0 < a2 and a2 < C):
    b[0] = b2
  else:
    b[0] = (b1 + b2) / 2
  # Store a1 and a2 in alphas array
  alphas[i1] = a1
  alphas[i2] = a2
  return 1
  
 
# Examine each training example
def examineExample(i2, kernel, alphas, b):
  y2 = y[i2]
  alpha2 = alphas[i2]
  E2 = 0
  for i in range(N):
    # SVM output on point i2
    E2 += alphas[i] * y[i] * kernel(X[i], X[i2])
  E2 = E2 + b[0] - y2
  r2 = E2 * y2
  if (r2 < -tol and alpha2 < C) or (r2 > tol and alpha2 > 0):
    # get random i1
    i1 = randint(0, N-1)
    while i1 == i2:
      i1 = randint(0, N-1) # make sure i1 different i2
    # do optimize to find alphas
    if takeStep(i1, i2, kernel, alphas, b):
      return 1
  return 0

def svm_train(alphas, b):
  numChanged = 0
  examineAll = 1

  while (numChanged > 0 or examineAll == 1):
    numChanged = 0
    if examineAll == 1:
      for i in range(len(X)):
        numChanged += examineExample(i, linear_kernel, alphas, b)
    else:
      for i in range(len(alphas)):
	if alphas[i] != 0 and alphas[i] != C:
	  numChanged += examineExample(i, linear_kernel, alphas, b)
    if examineAll == 1:
      examineAll = 0
    elif numChanged == 0:
      examineAll = 1
  # Print result
  #print(b)
  #print(alphas)

# Calculate accuracy on testset
def cal_accuracy(testset, actual):
  correct = 0
  predicted = np.zeros(testset.shape[0])
  # loop through all testset examples
  for i in range(testset.shape[0]):
    # calculate svm output by w'.x + b
    p = np.dot(w.T, testset[i,:]) + b
    if p >= 0:
      predicted[i] = 1.0
    else:
      predicted[i] = -1.0
  for i in range(len(actual)):
    if predicted[i] == actual[i]:
      correct += 1
  return correct / float(len(actual)) * 100.0

# Compute boundary
def cal_boundary():
  return 2.0/sqrt(np.dot(w.T, w))

# Compute the decision cost of the dataset
def cal_decision_cost(dataset):
  cost = 0.0
  for i in range(dataset.shape[0]):
    cost += np.dot(w.T, dataset[i, :]) + b
  return cost

# Plot the decision boundary
def plot_decision_boundary():
  fig = plt.figure()
  # split dataset into 2 classes
  cls1 = list()
  cls2 = list()
  for i in range(N):
    if y[i] == 1:
      cls1.append(X[i])
    else:
      cls2.append(X[i])
  class1 = np.array(cls1)
  class2 = np.array(cls2)
  plt.scatter(class1[:,0],class1[:,1], marker='+', c='blue')
  plt.scatter(class2[:,0],class2[:,1], marker='o', c='red')

  # plot decision bound based on w and alpha
  a = -w[0] / w[1]
  xx = np.linspace(0, 5)
  yy = a * xx - (b[0]) / w[1]

  plt.plot(xx, yy, 'k-')
  plt.show()

# Main program
print('Linear SVM. Usage: python linear.py <C: default = 1>')
print('Reading file/data-ex1.txt')
file = open('../file/data-ex1.txt', 'r')
x_data = list()
y_data = list()
for line in file:
  items = line.split()
  features = list()
  features.append(float(items[1].split(':')[1]))
  features.append(float(items[2].split(':')[1]))
  x_data.append(features)
  y_data.append(float(items[0]))
file.close()

# Training SVM
C = 1
if len(sys.argv) > 1:
  C = float(sys.argv[1])
tol = 0.001
X = np.array(x_data)
y = np.array(y_data)
N = X.shape[0]
alphas = np.zeros(N)
b = [0.0]

# Train
print('Training SVM with C = %.1f' % C)
svm_train(alphas, b)
print('Bias b = %.3f' % b[0])

# Compute w
w = np.zeros(X.shape[1])
for i in range(N):
  w += X[i,:] * (y[i] * alphas[i])
print('Weights:')
print(w)

# Decision boundary
boundary = cal_boundary()
print("Boundary = %.3f" % (boundary))

# Accuracy
print("Accuracy on training set = %.3f" % cal_accuracy(X, y))

# Decision cost
print("Decision cost on dataset = %.3f" % cal_decision_cost(X))

# Plot decision boundary
print('Plotting decision boundary ...')
plot_decision_boundary()
