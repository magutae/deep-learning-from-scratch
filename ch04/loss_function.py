import numpy as np

def mean_squared_error(y, t):
  return 0.5 * np.sum((y = t) ** 2)

def cross_entropy_error(y, t):
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)
  
  batch_size = y.shape[0]
  delta = 1e-7  # 0을 넣으면 -무한대 나올 수 있으니
  # return -np.sum(t * np.log(y + delta)) / batch_size
  return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size

# 중심차분, 중앙차분
def numerical_diff(f, x):
  h = 1e-4
  return (f(x+h) - f(x-h)) / (2*h)

