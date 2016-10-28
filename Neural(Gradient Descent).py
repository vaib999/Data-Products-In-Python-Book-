import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class AdalineGD(object):
      """ADAptive LInear NEuron classifier.
      Parameters
      ------------
      eta : float
      Learning rate (between 0.0 and 1.0)
      n_iter : int
      Passes over the training dataset.

      Attributes
      -----------
      w_ : 1d-array
      Weights after fitting.
      errors_ : list
      Number of misclassifications in every epoch.
      """
      def __init__(self, eta=0.01, n_iter=50):
           self.eta = eta
           self.n_iter = n_iter
      def fit(self, X, y):
           """ Fit training data.
           Parameters
           ----------
           X : {array-like}, shape = [n_samples, n_features]
           Training vectors,
           where n_samples is the number of samples and
           n_features is the number of features.
           y : array-like, shape = [n_samples]
           Target values.
           Returns
           -------
           self : object
           """
           self.w_ = np.zeros(1 + X.shape[1])
           self.cost_ = []
           for i in range(self.n_iter):
                output = self.net_input(X)#[  4.3287   4.2357   4.0029   4.236    4.2822........]
                errors = (y - output)#[-5.3287 -5.2357 -5.0029 -5.236.........]
                self.w_[1:] += self.eta * X.T.dot(errors)#Taking transpose
                self.w_[0] += self.eta * errors.sum()#[ -6.545091  -36.1128963 -19.1878272]
                cost = (errors**2).sum() / 2.0#2230.85396025
                self.cost_.append(cost)
           return self
      def net_input(self, X):
           """Calculate net input"""
           return np.dot(X, self.w_[1:]) + self.w_[0]
      def activation(self, X):
           """Compute linear activation"""
           return self.net_input(X)
      def predict(self, X):
           """Return class label after unit step"""
           return np.where(self.activation(X) >= 0.0, 1, -1)

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail()

y = df.iloc[0:100, 4].values#o/p:['Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa'....]
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
#[[ 5.1  1.4]
#[ 4.9  1.4]
#[ 4.7  1.3]....]

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1),np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')
ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1),ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()
