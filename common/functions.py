import numpy as np

def softmax(a):
  exp_a = np.exp(a)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a

  return y

def cross_entropy_error(y, t):
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)
  
  batch_size = y.shape[0]
  delta = 1e-7  # 0을 넣으면 -무한대 나올 수 있으니
  # return -np.sum(t * np.log(y + delta)) / batch_size
  return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size

def sigmoid(x):
  return 1 / (1 + np.exp(-x))